"""Inter-area rules in RT: the DA summed flow of a rule, its balancing penalty and its lock.

``inter_area_rules`` has three optional columns read by ``RtExtension`` only:
``da_flow_ref`` (profile column with the DA summed flow, MW, area_from -> area_to),
``balancing_penalty`` (EUR/MWh of deviation of the summed flow from it) and ``lock_share``
(lambda: the summed flow stays within DA -/+ (1 - lambda) x NTC). ``prepare_rt(...,
inter_area=InterAreaFlows(...))`` writes the ``da_flow_<k>`` columns and fills them in.

On ``two_area`` the one rule AREA_A -> AREA_B is crossed by the AC branch a2-b1 and the
HVDC link a1-b2, both from A to B.
"""

import sqlite3

import numpy as np
import pandas as pd
import pytest
from rt_case import N_STEPS, read_table, series, two_area_data

import powergama
from powergama.lp.rt import (
    InterAreaFlows,
    RtExtension,
    combine_columns,
    da_flow_columns,
    prepare_rt,
    summarize_channels,
)

BRANCH_AB = 1  # a2-b1
DCBRANCH_AB = 0  # a1-b2


def _solve(data, dbfile, extensions=()):
    lp = powergama.LpProblem(data, extensions=list(extensions))
    res = powergama.Results(data, dbfile, replace=True)
    lp.solve(res, solver="appsi_highs", solve_args={})
    return lp, res


@pytest.fixture(scope="module")
def da(tmp_path_factory):
    return _solve(two_area_data(), tmp_path_factory.mktemp("da") / "da.sqlite")[1]


# --- the columns and prepare_rt (no LP change) -------------------------------------------


def test_grid_data_has_the_rule_columns_with_defaults():
    rules = two_area_data().inter_area_rules
    assert list(rules.columns[-3:]) == ["balancing_penalty", "lock_share", "da_flow_ref"]
    assert rules.at[0, "balancing_penalty"] == 0.0
    assert np.isnan(rules.at[0, "lock_share"])
    assert rules.at[0, "da_flow_ref"] == ""


def test_da_flow_column_is_the_stored_summed_flow(da, tmp_path):
    prepared = prepare_rt(da, tmp_path, inter_area=InterAreaFlows(balancing_penalty=5.0, lock_share=0.8))
    ac = series(read_table(da, "Res_Branches"), BRANCH_AB, "flow")
    dc = series(read_table(da, "Res_DcBranches"), DCBRANCH_AB, "flow")
    np.testing.assert_array_equal(prepared.da_solution["da_flow_0"].to_numpy(), (ac + dc).to_numpy())
    rules = prepared.tables["inter_area_rules"]
    assert rules.loc[0, ["da_flow_ref", "balancing_penalty", "lock_share"]].tolist() == ["da_flow_0", 5.0, 0.8]

    # the files read back into an RT grid with the columns
    data = powergama.GridData()
    data.readGridData(**prepared.grid_files())
    assert data.inter_area_rules.loc[0, ["da_flow_ref", "balancing_penalty", "lock_share"]].tolist() == ["da_flow_0", 5.0, 0.8]


def test_da_flow_sign_follows_the_rule_direction(da):
    """A rule B -> A gets the same flow with the opposite sign."""
    rules = da.grid.inter_area_rules.copy()
    rules.loc[1] = rules.loc[0]
    rules.loc[1, ["area_from", "area_to"]] = ["AREA_B", "AREA_A"]
    refs = da_flow_columns(da.get_dispatch_frames(), rules, grid=da.grid)
    np.testing.assert_array_equal(refs.profiles["da_flow_1"].to_numpy(), -refs.profiles["da_flow_0"].to_numpy())
    assert refs.inter_area_rules["da_flow_ref"].tolist() == ["da_flow_0", "da_flow_1"]
    assert "balancing_penalty" not in refs.inter_area_rules.columns  # None keeps the table's value


def test_per_rule_values_and_defaults_kept(da, tmp_path):
    """Per-rule values by row; None keeps the table's own balancing penalty and lock share."""
    data = two_area_data()
    rules = da.grid.inter_area_rules.copy()
    rules.loc[0, ["balancing_penalty", "lock_share"]] = [7.0, 0.5]
    refs = da_flow_columns(da.get_dispatch_frames(), rules, grid=data, balancing_penalty={0: 3.0})
    assert refs.inter_area_rules.loc[0, "balancing_penalty"] == 3.0
    assert "lock_share" not in refs.inter_area_rules.columns
    prepared = prepare_rt(da, tmp_path, inter_area=InterAreaFlows())
    assert prepared.tables["inter_area_rules"].loc[0, ["balancing_penalty", "da_flow_ref"]].tolist() == [0.0, "da_flow_0"]
    assert np.isnan(prepared.tables["inter_area_rules"].at[0, "lock_share"])


def test_rule_without_branches_is_skipped_unless_selected(da):
    rules = da.grid.inter_area_rules.copy()
    rules.loc[1] = rules.loc[0]
    rules.loc[1, ["area_from", "area_to"]] = ["AREA_A", "AREA_C"]
    frames = da.get_dispatch_frames()
    refs = da_flow_columns(frames, rules, grid=da.grid)
    assert list(refs.profiles.columns) == ["da_flow_0"]
    with pytest.raises(ValueError, match="rule 1 .*no branch across its boundary"):
        da_flow_columns(frames, rules, grid=da.grid, mask=pd.Series([False, True], index=rules.index))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"balancing_penalty": -1.0}, "balancing_penalty"),
        ({"lock_share": 1.5}, "lock_share"),
        ({"lock_share": {3: 0.5}}, "rule row"),
    ],
)
def test_bad_values_raise(da, kwargs, match):
    with pytest.raises(ValueError, match=match):
        da_flow_columns(da.get_dispatch_frames(), da.grid.inter_area_rules, grid=da.grid, **kwargs)


def test_nan_flow_raises(da):
    frames = da.get_dispatch_frames()
    frames["dcbranch_flow"].iloc[3, DCBRANCH_AB] = np.nan
    with pytest.raises(ValueError, match="NaN in the DA flow of branch 0"):
        da_flow_columns(frames, da.grid.inter_area_rules, grid=da.grid)


def test_prepare_without_inter_area_table_raises(da, tmp_path, monkeypatch):
    monkeypatch.setattr(da.grid, "inter_area_rules", None)
    with pytest.raises(ValueError, match="no inter-area table"):
        prepare_rt(da, tmp_path, inter_area=InterAreaFlows())


def test_summary_lists_the_rule(da, tmp_path):
    prepared = prepare_rt(
        da, tmp_path, inter_area=InterAreaFlows(balancing_penalty=5.0, lock_share=1.0, source="flows from the test")
    )
    channels = pd.read_csv(prepared.files["channels"])
    row = channels.set_index("name").loc["da_flow_0"]
    assert (row["family"], row["da_row"], row["usages"], row["rt_rows"]) == (
        "da_flow",
        0,
        "flow_ref",
        "inter_area_rules:0",
    )
    assert row["area"] == "AREA_A->AREA_B"
    assert row["desc"] == "balancing_penalty 5, lock_share 1"
    text = prepared.files["summary"].read_text(encoding="utf-8")
    assert "## `da_flow_<k>`: DA summed flow of an inter-area rule (MW) (1 channel)" in text
    assert "Source: AC and DC branch flows from DA run `da.sqlite`; flows from the test." in text
    assert "  - rule 0 (AREA_A->AREA_B): balancing_penalty 5, lock_share 1" in text
    assert "inter_area 1 of 1 rules" in text


def test_summary_reports_rules_without_flow(da):
    frames = da.get_dispatch_frames()
    rules = da.grid.inter_area_rules.copy()
    rules.loc[1] = rules.loc[0]
    rules.loc[1, ["area_from", "area_to"]] = ["AREA_A", "AREA_C"]
    refs = combine_columns(da_flow_columns(frames, rules, grid=da.grid))
    gen, cons = da.grid.generator, da.grid.consumer
    text, channels = summarize_channels(refs, gen, cons, inter_area_rules=rules)
    assert channels["name"].tolist() == ["da_flow_0"]
    assert "- 1 inter-area rule without any channel:" in text
    assert "no `da_flow_ref` (no balancing penalty or lock in RT)" in text
    with pytest.raises(ValueError, match="no inter-area table"):
        summarize_channels(refs, gen, cons)


def test_da_run_ignores_the_rule_columns(da, tmp_path):
    """A DA run with balancing penalty, lock share and da_flow_ref set gets the same results."""
    data = two_area_data()
    steps = list(data.timerange)
    data.profiles["da_flow_0"] = 100.0
    data.inter_area_rules.loc[0, ["balancing_penalty", "lock_share", "da_flow_ref"]] = [5.0, 1.0, "da_flow_0"]
    _, res = _solve(data, tmp_path / "da_cols.sqlite")
    for table in ["Res_Generators", "Res_Branches", "Res_DcBranches", "Res_Nodes"]:
        pd.testing.assert_frame_equal(read_table(res, table), read_table(da, table), obj=table)
    assert len(steps) == N_STEPS


# --- RT: the penalty on the summed flow ---------------------------------------------


def _rt_data(da, *, penalty=0.0, lock_share=np.nan):
    """two_area as the RT grid, rule 0 held to the DA summed flow (no other target)."""
    data = two_area_data()
    frames = da.get_dispatch_frames()
    data.profiles["da_flow_0"] = (frames["branch_flow"][BRANCH_AB] + frames["dcbranch_flow"][DCBRANCH_AB]).to_numpy()
    data.inter_area_rules.loc[0, ["da_flow_ref", "balancing_penalty", "lock_share"]] = ["da_flow_0", penalty, lock_share]
    return data


def _transfer(res):
    ac = series(read_table(res, "Res_Branches"), BRANCH_AB, "flow")
    dc = series(read_table(res, "Res_DcBranches"), DCBRANCH_AB, "flow")
    return (ac + dc).to_numpy(), ac.to_numpy(), dc.to_numpy()


def test_penalty_is_on_the_sum_not_per_branch(da, tmp_path):
    """The HVDC link is out in RT: its DA flow moves to the parallel AC branch at no cost.

    A per-branch penalty would have charged the moved flow twice (once per branch); on the
    summed flow nothing deviates.
    """
    data = _rt_data(da, penalty=50.0)
    data.dcbranch.loc[DCBRANCH_AB, "capacity"] = 0.0
    lp, res = _solve(data, tmp_path / "rt.sqlite", [RtExtension()])
    transfer, ac, dc = _transfer(res)
    da_dc = da.get_dispatch_frames()["dcbranch_flow"][DCBRANCH_AB].to_numpy()
    assert da_dc.max() > 100.0  # the link carried flow in DA
    np.testing.assert_allclose(dc, 0.0, atol=1e-6)
    np.testing.assert_allclose(transfer, data.profiles["da_flow_0"].to_numpy(), atol=1e-6)
    np.testing.assert_allclose(ac, transfer, atol=1e-6)
    assert set(lp.model.cRtRuleFlowTracking) == {0}
    assert lp.model.varRtRuleDevPos[0].value == pytest.approx(0.0, abs=1e-6)
    assert lp.model.varRtRuleDevNeg[0].value == pytest.approx(0.0, abs=1e-6)


def test_penalty_prices_the_deviation_of_the_sum(da, tmp_path):
    """A lower RT NTC forces the summed flow below DA; each MWh short costs the penalty."""
    data = _rt_data(da, penalty=7.0)
    data.inter_area_rules.loc[0, "ntc_forward"] = 200.0
    lp, res = _solve(data, tmp_path / "rt.sqlite", [RtExtension()])
    transfer, _, _ = _transfer(res)
    short = data.profiles["da_flow_0"].to_numpy() - transfer
    assert short.max() == pytest.approx(50.0)  # DA ran at the 250 MW NTC
    assert transfer.max() <= 200.0 + 1e-6
    # the last timestep is the model's state
    m = lp.model
    assert m.varRtRuleDevNeg[0].value - m.varRtRuleDevPos[0].value == pytest.approx(short[-1], abs=1e-6)
    with sqlite3.connect(res.db.filename) as con:
        objective = pd.read_sql_query("SELECT value FROM Res_ObjFunc ORDER BY timestep", con)["value"]
    nodes = read_table(res, "Res_Nodes").groupby("timestep")[["loadshed", "dumpload"]].sum().sum(axis=1)
    slack = powergama.constants.loadshedcost * nodes.to_numpy()
    np.testing.assert_allclose(objective.to_numpy() - slack, 7.0 * short, atol=1e-5)  # nothing else is priced


def test_rule_without_da_flow_ref_gets_no_penalty(da):
    data = two_area_data()
    data.inter_area_rules.loc[0, ["balancing_penalty", "lock_share"]] = [50.0, 1.0]
    lp = powergama.LpProblem(data, extensions=[RtExtension()])
    assert len(lp.model.s_rt_rule) == 0 and len(lp.model.cRtRuleFlowTracking) == 0
    assert lp._extensions[0].ctx.rules == {}


def test_rule_with_da_flow_ref_but_no_penalty_or_lock_is_not_held(da):
    lp = powergama.LpProblem(_rt_data(da), extensions=[RtExtension()])
    assert len(lp.model.s_rt_rule) == 0


@pytest.mark.parametrize(
    "change, match",
    [
        ({"da_flow_ref": "no_such_column"}, "da_flow_ref 'no_such_column' is not a profile column"),
        ({"balancing_penalty": -1.0}, "balancing_penalty must be >= 0"),
        ({"lock_share": 1.5}, r"lock_share must be within \[0, 1\]"),
        ({"area_to": "AREA_C"}, "no branch across its boundary"),
    ],
)
def test_bad_rule_inputs_raise(da, change, match):
    data = _rt_data(da, penalty=5.0)
    for col, value in change.items():
        data.inter_area_rules[col] = data.inter_area_rules[col].astype(object)
        data.inter_area_rules.loc[0, col] = value
    with pytest.raises(ValueError, match=match):
        powergama.LpProblem(data, extensions=[RtExtension()])


def test_nan_in_da_flow_raises(da):
    data = _rt_data(da, penalty=5.0)
    data.profiles.loc[data.profiles.index[3], "da_flow_0"] = np.nan
    with pytest.raises(ValueError, match="profile 'da_flow_0' .*has NaN"):
        powergama.LpProblem(data, extensions=[RtExtension()])


# --- RT: the lock on the summed flow ------------------------------------------------

GEN_GAS_A = 0  # gas_a, the only thermal unit in AREA_A
# DA exports at the 250 MW forward NTC every hour; the lock windows at lambda = 0.8:
WIDTH_FORWARD, WIDTH_BACKWARD = 0.2 * 250.0, 0.2 * 150.0


def _outage_data(da, **rule):
    """RT with gas_a out: AREA_A cannot keep exporting, and coal_b in AREA_B covers it unless locked."""
    data = _rt_data(da, **rule)
    data.generator.loc[GEN_GAS_A, "pmax"] = 0.0
    return data


def _objective(res):
    with sqlite3.connect(res.db.filename) as con:
        return pd.read_sql_query("SELECT value FROM Res_ObjFunc ORDER BY timestep", con)["value"].to_numpy()


def _shed(res):
    nodes = read_table(res, "Res_Nodes").groupby("timestep")[["loadshed", "dumpload"]].sum().sum(axis=1)
    return nodes.to_numpy()


def test_lock_binds_on_the_summed_flow(da, tmp_path):
    """Exporting less is the free alternative to shedding in AREA_A; the lock holds it at DA - (1 - lambda) x NTC."""
    da_flow = _rt_data(da).profiles["da_flow_0"].to_numpy()
    _, res_free = _solve(_outage_data(da), tmp_path / "free.sqlite", [RtExtension()])
    free, _, _ = _transfer(res_free)
    lp, res = _solve(_outage_data(da, lock_share=0.8), tmp_path / "locked.sqlite", [RtExtension()])
    locked, _, _ = _transfer(res)

    under = free < da_flow - WIDTH_BACKWARD - 1e-6
    assert under.any()  # without the lock the export falls below the window
    assert np.all(locked >= da_flow - WIDTH_BACKWARD - 1e-6)
    assert np.all(locked <= da_flow + WIDTH_FORWARD + 1e-6)
    np.testing.assert_allclose(locked[under], da_flow[under] - WIDTH_BACKWARD, atol=1e-6)
    assert _shed(res)[under].min() > _shed(res_free)[under].max()  # what the lock holds in is shed instead
    assert set(lp.model.cRtRuleLockForward) == set(lp.model.cRtRuleLockBackward) == {0}
    assert len(lp.model.cRtRuleFlowTracking) == 0  # a lock without a penalty prices nothing
    m = lp.model  # the last timestep is the model's state
    assert m.p_rt_rule_lock_ub[0].value == pytest.approx(da_flow[-1] + WIDTH_FORWARD)
    assert m.p_rt_rule_lock_lb[0].value == pytest.approx(da_flow[-1] - WIDTH_BACKWARD)


def test_full_lock_share_pins_the_summed_flow_to_da(da, tmp_path):
    """lambda = 1 is a window of width 0: the summed flow is the DA summed flow."""
    data = _outage_data(da, lock_share=1.0)
    _, res = _solve(data, tmp_path / "pinned.sqlite", [RtExtension()])
    transfer, _, _ = _transfer(res)
    np.testing.assert_allclose(transfer, data.profiles["da_flow_0"].to_numpy(), atol=1e-6)


def test_penalty_and_lock_on_one_rule(da, tmp_path):
    """Each MWh of deviation costs the penalty, and the window still bounds it."""
    data = _outage_data(da, penalty=7.0, lock_share=0.8)
    lp, res = _solve(data, tmp_path / "both.sqlite", [RtExtension()])
    transfer, _, _ = _transfer(res)
    dev = transfer - data.profiles["da_flow_0"].to_numpy()
    assert dev.min() == pytest.approx(-WIDTH_BACKWARD)  # shedding costs more than the penalty
    assert np.all(dev >= -WIDTH_BACKWARD - 1e-6)
    m = lp.model
    assert m.varRtRuleDevPos[0].value - m.varRtRuleDevNeg[0].value == pytest.approx(dev[-1], abs=1e-6)
    slack = powergama.constants.loadshedcost * _shed(res)
    np.testing.assert_allclose(_objective(res) - slack, 7.0 * np.abs(dev), atol=1e-5)
