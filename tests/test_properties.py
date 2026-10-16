"""
Property tests: invariants that must hold regardless of how storage is valued.

These are deliberately value-free. They pin relationships (continuity, capacity limits,
table shapes, determinism) rather than specific flows or prices, so they survive the
open question about storage valuation in multi-period mode and act as the regression
net for refactoring work. See test_sim_hourly_rolling_equiv.py for that question.
"""

import sqlite3

import pandas as pd
import pytest

import powergama

TOL = 1e-6

RESULT_TABLES = [
    "Res_ObjFunc",
    "Res_Nodes",
    "Res_Branches",
    "Res_BranchesSens",
    "Res_DcBranches",
    "Res_Generators",
    "Res_Storage",
    "Res_Pumping",
    "Res_FlexibleLoad",
]


def _solve(data, dbfile, mode="hourly", **kwargs):
    """Solve the case in the given objective mode and return the Results object."""
    if mode == "multi_period":
        kwargs.setdefault("horizon_hours", 24)
        kwargs.setdefault("commit_hours", 24)
    lp = powergama.LpProblem(data, objective_mode=mode, **kwargs)
    res = powergama.Results(data, dbfile, replace=True)
    lp.solve(res, solver="appsi_highs", solve_args={})
    return res


def _tables(res):
    """Read every result table into DataFrames, ordered deterministically."""
    out = {}
    with sqlite3.connect(res.db.filename) as con:
        for name in RESULT_TABLES:
            cols = [r[1] for r in con.execute(f"PRAGMA table_info({name})").fetchall()]
            order = ", ".join(c for c in ("timestep", "indx") if c in cols)
            out[name] = pd.read_sql_query(f"SELECT * FROM {name} ORDER BY {order}", con)
    return out


@pytest.fixture(params=["hourly", "multi_period"])
def solved_case(request, tmp_path, testcase_9bus_data):
    """The 9-bus case solved in each objective mode, with its input data."""
    data = testcase_9bus_data
    res = _solve(data, tmp_path / f"{request.param}.sqlite", mode=request.param)
    return request.param, data, res


def test_result_table_shapes(solved_case):
    """Every result table must have one row per entity per timestep.

    A table that silently reports nothing (or reports a constant) still has the right
    shape, but a table missing rows entirely does not -- this is the cheapest check
    that catches a mode writing no results for some entity class.
    """
    _mode, data, res = solved_case
    tables = _tables(res)
    n_ts = len(res.timerange)

    expected = {
        "Res_ObjFunc": n_ts,
        "Res_Nodes": len(data.node) * n_ts,
        "Res_Branches": len(data.branch) * n_ts,
        "Res_DcBranches": len(data.dcbranch) * n_ts,
        "Res_Generators": len(data.generator) * n_ts,
        "Res_Storage": len(data.getIdxGeneratorsWithStorage()) * n_ts,
        "Res_Pumping": len(data.getIdxGeneratorsWithPumping()) * n_ts,
        "Res_FlexibleLoad": len(data.getIdxConsumersWithFlexibleLoad()) * n_ts,
    }
    for name, n_rows in expected.items():
        assert len(tables[name]) == n_rows, f"{name}: expected {n_rows} rows, got {len(tables[name])}"

    # Sensitivities are only stored for branches that carry a capacity constraint.
    assert len(tables["Res_BranchesSens"]) == len(data.getIdxBranchesWithFlowConstraints()) * n_ts


def test_generation_within_installed_capacity(solved_case):
    """0 <= output <= pmax for every generator and timestep.

    The effective upper bound is tighter than pmax for inflow-limited and storage
    generators, so this is the loosest bound that holds universally.
    """
    _mode, data, res = solved_case
    gen = _tables(res)["Res_Generators"]
    pmax = data.generator["pmax"]

    assert gen["output"].min() >= -TOL, "negative generation"
    worst = (gen["output"] - gen["indx"].map(pmax)).max()
    assert worst <= TOL, f"generation exceeds pmax by {worst}"


def test_branch_flow_within_capacity(solved_case):
    """|flow| <= capacity on every AC branch with a finite capacity."""
    _mode, data, res = solved_case
    flows = _tables(res)["Res_Branches"]
    capacity = data.branch["capacity"]

    cap = flows["indx"].map(capacity)
    finite = cap.notna() & (cap != float("inf"))
    excess = (flows["flow"].abs() - cap)[finite].max()
    assert excess <= TOL, f"branch flow exceeds capacity by {excess}"


def test_storage_continuity(solved_case):
    """storage[t] == storage[t-1] + (inflow + pump*eff - output)*dt - spilled.

    Holds in both modes by construction but by different mechanisms: the hourly path
    clips the energy balance after each solve and records the overflow as
    inflow_spilled, while multi-period makes storage an LP variable with an explicit
    spill term. Both must satisfy the same identity.
    """
    _mode, data, res = solved_case
    tables = _tables(res)
    storage = tables["Res_Storage"].set_index(["timestep", "indx"])["storage"]
    gen = tables["Res_Generators"].set_index(["timestep", "indx"])
    dt = res.grid.timeDelta

    for i in data.getIdxGeneratorsWithStorage():
        row = data.generator.loc[i]
        inflow = data.profiles[row["inflow_ref"]] * row["inflow_fac"] * row["pmax"]
        for t in list(res.timerange)[1:]:
            expected = (
                storage[(t - 1, i)]
                + inflow[t] * dt
                - gen.loc[(t, i), "output"] * dt
                - gen.loc[(t, i), "inflow_spilled"]
            )
            assert storage[(t, i)] == pytest.approx(expected, abs=1e-4), (
                f"storage continuity violated for generator {i} at timestep {t}"
            )


def test_objective_is_finite(solved_case):
    """A non-finite objective means an unbounded or unsolved LP slipped through."""
    _mode, _data, res = solved_case
    obj = _tables(res)["Res_ObjFunc"]["value"]
    assert obj.notna().all(), "objective contains NaN"
    assert (obj.abs() < 1e12).all(), "objective is implausibly large"


@pytest.mark.parametrize("mode", ["hourly", "multi_period"])
def test_run_twice_is_deterministic(tmp_path, testcase_9bus_data, mode):
    """The same inputs must produce bit-identical results across runs.

    Golden-number tests are only meaningful if this holds; a degenerate LP whose
    optimal basis varies between runs would make pinned duals meaningless.
    """
    data = testcase_9bus_data
    first = _tables(_solve(data, tmp_path / "run1.sqlite", mode=mode))
    second = _tables(_solve(data, tmp_path / "run2.sqlite", mode=mode))

    for name in RESULT_TABLES:
        pd.testing.assert_frame_equal(first[name], second[name], check_exact=True, obj=name)


# ---------------------------------------------------------------------------
# Known defects, pinned as strict xfails.
#
# These assert the behaviour we want, not the behaviour we have, so that a fix
# flips them green. strict=True means an accidental fix fails loudly as XPASS
# rather than passing silently.
# ---------------------------------------------------------------------------


def _with_flexible_load(data):
    """Give consumer 0 a flexible-load share with a storage reservoir."""
    data.consumer.loc[0, "flex_basevalue"] = 10
    data.consumer.loc[0, "flex_fraction"] = 0.1
    data.consumer.loc[0, "flex_on_off"] = 0.2
    data.consumer.loc[0, "flex_storage"] = 10
    data.consumer.loc[0, "flex_storagelevel_init"] = 0.5
    data.consumer.loc[0, "flex_storval_filling"] = "flexdemand"
    data.consumer.loc[0, "flex_storval_time"] = "const"
    return data


def test_flexible_load_storage_evolves_hourly(tmp_path, testcase_9bus_data):
    """Reference case: in hourly mode flexible-load storage does move."""
    data = _with_flexible_load(testcase_9bus_data)
    res = _solve(data, tmp_path / "flex_hourly.sqlite", mode="hourly")
    flex = _tables(res)["Res_FlexibleLoad"]

    assert not flex.empty, "fixture produced no flexible load rows"
    assert flex["storage"].nunique() > 1, "flexible-load storage never changed"


def test_flexible_load_storage_evolves_multi_period(tmp_path, testcase_9bus_data):
    """Flexible load must be priced, dispatched and accounted for in multi_period too.

    Asserting only that storage moved would also pass if the reservoir merely drained
    with no charging at all, which is what the inert version did once the write-back
    was added. So this also requires non-zero demand and a non-zero price.
    """
    data = _with_flexible_load(testcase_9bus_data)
    res = _solve(data, tmp_path / "flex_multi.sqlite", mode="multi_period")
    flex = _tables(res)["Res_FlexibleLoad"]

    assert not flex.empty, "fixture produced no flexible load rows"
    assert flex["storage"].nunique() > 1, "flexible-load storage never changed"
    assert flex["demand"].max() > TOL, "flexible load was never dispatched"
    assert flex["value"].abs().max() > TOL, "flexible load was never priced"


def test_flexible_load_reservoir_obeys_continuity_multi_period(tmp_path, testcase_9bus_data):
    """The reservoir must follow level[t] = level[t-1] + (charge - drain) * dt.

    In this mode the level is an LP variable over the window with a hard capacity,
    rather than a state clipped after each solve, so it must also stay at or below
    capacity -- including across the window boundary, where it is carried forward
    through self._storage_flexload.
    """
    data = _with_flexible_load(testcase_9bus_data)
    res = _solve(data, tmp_path / "flex_continuity.sqlite", mode="multi_period")
    flex = _tables(res)["Res_FlexibleLoad"]

    for indx, rows in flex.groupby("indx"):
        rows = rows.sort_values("timestep")
        drain = data.consumer.loc[indx, "flex_fraction"] * data.consumer.loc[indx, "demand_avg"]
        capacity = data.consumer.loc[indx, "flex_storage"] * drain

        assert rows["storage"].max() <= capacity + TOL, f"consumer {indx}: reservoir exceeded capacity"
        step = rows["storage"].diff().iloc[1:]
        expected = (rows["demand"] - drain).iloc[1:]
        assert (step - expected).abs().max() < 1e-6, f"consumer {indx}: reservoir continuity broken"


def _branches_at_capacity(res, data):
    """Rows where an AC branch sits at its rating, joined to its stored sensitivity.

    Res_Branches is indexed by branch, Res_BranchesSens by position within
    getIdxBranchesWithFlowConstraints(). They coincide only when every branch has a
    finite capacity, so build the mapping rather than merging on indx.
    """
    tables = _tables(res)
    constrained = data.getIdxBranchesWithFlowConstraints()
    sens = tables["Res_BranchesSens"].copy()
    sens["branch"] = sens["indx"].map(dict(enumerate(constrained)))

    flows = tables["Res_Branches"].rename(columns={"indx": "branch"})
    df = flows.merge(sens.drop(columns="indx"), on=["timestep", "branch"])
    df["capacity"] = df["branch"].map(data.branch["capacity"])
    return df[(df["flow"].abs() >= df["capacity"] - 1e-4) & df["capacity"].notna()]


def test_branch_capacity_sensitivity_is_reported_multi_period(tmp_path, testcase_9bus_data):
    """Reference case: multi_period does price a congested branch."""
    data = testcase_9bus_data
    res = _solve(data, tmp_path / "sens_multi.sqlite", mode="multi_period")
    binding = _branches_at_capacity(res, data)

    assert not binding.empty, "no branch reached its rating; the case cannot test congestion pricing"
    assert binding["cap_sensitivity"].abs().max() > TOL, "every congested branch reported a zero shadow price"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Branch capacity sensitivities are identically zero in hourly mode. Across the "
        "9-bus and two-area cases, under both glpk and appsi_highs and all three loss "
        "methods, Res_BranchesSens contains nothing but zeros -- while the same 9-bus "
        "case in multi_period mode does produce non-zero values. Branches sit at their "
        "rating for whole runs with nodal price differences of up to 990 EUR/MWh across "
        "them, so the constraint is plainly binding. Rewriting cMaxFlowAc from one "
        "two-sided ranged row into two one-sided rows does not recover the dual, so the "
        "constraint form is not the cause."
    ),
)
def test_branch_capacity_sensitivity_is_reported_hourly(tmp_path, testcase_9bus_data):
    """A branch held at its rating by congestion must carry a non-zero shadow price."""
    data = testcase_9bus_data
    res = _solve(data, tmp_path / "sens_hourly.sqlite", mode="hourly")
    binding = _branches_at_capacity(res, data)

    assert not binding.empty, "no branch reached its rating; the case cannot test congestion pricing"
    assert binding["cap_sensitivity"].abs().max() > TOL, "every congested branch reported a zero shadow price"
