"""
Integration tests for a two-area case exercising the features the 9-bus case lacks.

The 9-bus fixture has no DC branch, no pumping, no flexible load and a single area, so
inter-area NTC, pumping and flexible load were effectively untested -- and the 9-bus NTC
fixture file references an area with no nodes, so every NTC constraint was skipped.
Adding a second area changes the B matrix and the reference node, so this has to be its
own case rather than a variant of 9-bus.
"""

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

import powergama

datapath = Path(__file__).parent / "test_data/two_area"
TOL = 1e-6
NTC_FORWARD = 250.0
NTC_BACKWARD = 150.0


@pytest.fixture
def two_area_data() -> powergama.GridData:
    data = powergama.GridData()
    data.readGridData(
        nodes=datapath / "nodes.csv",
        ac_branches=datapath / "branches.csv",
        dc_branches=datapath / "dcbranches.csv",
        generators=datapath / "generators.csv",
        consumers=datapath / "consumers.csv",
        inter_area_ntc=datapath / "inter_area_ntc.csv",
    )
    data.readProfileData(
        filename=datapath / "profiles.csv",
        storagevalue_filling=datapath / "profiles_storval_filling.csv",
        storagevalue_time=datapath / "profiles_storval_time.csv",
        timerange=range(0, 48),
        timedelta=1.0,
    )
    return data


def _tables(res):
    with sqlite3.connect(res.db.filename) as con:
        return {
            name: pd.read_sql_query(f"SELECT * FROM {name} ORDER BY timestep, indx", con)
            for name in (
                "Res_Branches",
                "Res_DcBranches",
                "Res_Generators",
                "Res_Storage",
                "Res_Pumping",
                "Res_FlexibleLoad",
                "Res_Nodes",
            )
        }


def test_fixture_exercises_the_intended_features(two_area_data):
    """Fail loudly if the fixture stops covering what it exists to cover."""
    data = two_area_data
    assert sorted(data.node["area"].unique()) == ["AREA_A", "AREA_B"]
    assert len(data.dcbranch) == 1, "no DC branch"
    assert len(data.getIdxGeneratorsWithPumping()) >= 1, "no pumping generator"
    assert len(data.getIdxGeneratorsWithStorage()) >= 1, "no storage generator"
    assert len(data.getIdxConsumersWithFlexibleLoad()) >= 1, "no flexible load"
    assert data.inter_area_ntc is not None and len(data.inter_area_ntc) == 1


def test_inter_area_ntc_constraints_are_built_and_bind(tmp_path, two_area_data):
    """The NTC rows must map onto real branches and actually constrain the transfer.

    The 9-bus NTC fixture produced zero constraints because its `OTHER_AREA` has no
    nodes, so both rules hit Constraint.Skip. Asserting the rows exist is not enough;
    this also checks the limit is reached, otherwise the test proves nothing.
    """
    data = two_area_data
    lp = powergama.LpProblem(data)
    res = powergama.Results(data, tmp_path / "two_area.sqlite", replace=True)

    # The NTC bookkeeping must have found border branches in both directions.
    assert hasattr(lp, "_inter_area_ntc_rows"), "NTC constraint bookkeeping was never built"
    rows = lp._inter_area_ntc_rows
    assert len(rows) == 1
    row = rows[0] if isinstance(rows, list) else next(iter(rows.values()))
    border = sum(len(row.get(k, [])) for k in ("ac_pos", "ac_neg", "dc_pos", "dc_neg"))
    assert border > 0, f"NTC row matched no border branches: {row}"

    lp.solve(res, solver="appsi_highs", solve_args={})
    tables = _tables(res)

    # Net A->B transfer is the AC interconnector (a2->b1, index 1) plus the HVDC (a1->b2).
    ac = tables["Res_Branches"]
    dc = tables["Res_DcBranches"]
    ac_border = ac[ac["indx"] == 1].set_index("timestep")["flow"]
    dc_border = dc[dc["indx"] == 0].set_index("timestep")["flow"]
    transfer = ac_border + dc_border

    assert (transfer <= NTC_FORWARD + TOL).all(), f"forward NTC exceeded: max {transfer.max()}"
    assert (transfer >= -NTC_BACKWARD - TOL).all(), f"backward NTC exceeded: min {transfer.min()}"
    assert transfer.max() > NTC_FORWARD - 1.0, (
        f"NTC never came close to binding (max transfer {transfer.max()} vs limit {NTC_FORWARD}); "
        "the test would pass even with the constraint removed"
    )


def test_pumping_and_storage_are_active(tmp_path, two_area_data):
    """Pumping must actually be used, and the reservoir must cycle."""
    data = two_area_data
    lp = powergama.LpProblem(data)
    res = powergama.Results(data, tmp_path / "pump.sqlite", replace=True)
    lp.solve(res, solver="appsi_highs", solve_args={})
    tables = _tables(res)

    pumping = tables["Res_Pumping"]["output"]
    storage = tables["Res_Storage"]["storage"]
    assert not pumping.empty, "no pumping rows stored"
    assert pumping.max() > TOL, "pumping capacity was never used"
    assert storage.nunique() > 1, "reservoir level never changed"


def test_flexible_load_is_active(tmp_path, two_area_data):
    """The flexible share must be dispatched and its reservoir must move."""
    data = two_area_data
    lp = powergama.LpProblem(data)
    res = powergama.Results(data, tmp_path / "flex.sqlite", replace=True)
    lp.solve(res, solver="appsi_highs", solve_args={})
    flex = _tables(res)["Res_FlexibleLoad"]

    assert not flex.empty, "no flexible load rows stored"
    assert flex["demand"].max() > TOL, "flexible demand never dispatched"
    assert flex["storage"].nunique() > 1, "flexible-load storage never changed"


def test_dc_branch_within_capacity(tmp_path, two_area_data):
    """HVDC flow must respect its rating, and the link must carry power."""
    data = two_area_data
    lp = powergama.LpProblem(data)
    res = powergama.Results(data, tmp_path / "dc.sqlite", replace=True)
    lp.solve(res, solver="appsi_highs", solve_args={})

    dc = _tables(res)["Res_DcBranches"]
    capacity = data.dcbranch.loc[0, "capacity"]
    assert (dc["flow"].abs() <= capacity + TOL).all(), "DC flow exceeds capacity"
    assert dc["flow"].abs().max() > TOL, "DC link carried no power"


# ---------------------------------------------------------------------------
# Multi-period mode. The day-joint cNtcFwd/cNtcBwd and cPumpRoom/cPumpCap rules had
# never been executed by any test, because the 9-bus case has no NTC and no pumping.
# ---------------------------------------------------------------------------


@pytest.fixture
def two_area_multi_period_res(tmp_path, two_area_data):
    data = two_area_data
    lp = powergama.LpProblem(data, objective_mode="multi_period", horizon_hours=24, commit_hours=24)
    res = powergama.Results(data, tmp_path / "two_area_mp.sqlite", replace=True)
    lp.solve(res, solver="appsi_highs", solve_args={})
    return data, res


def test_multi_period_respects_inter_area_ntc(two_area_multi_period_res):
    data, res = two_area_multi_period_res
    tables = _tables(res)
    ac_border = tables["Res_Branches"].query("indx == 1").set_index("timestep")["flow"]
    dc_border = tables["Res_DcBranches"].query("indx == 0").set_index("timestep")["flow"]
    transfer = ac_border + dc_border

    assert (transfer <= NTC_FORWARD + TOL).all(), f"forward NTC exceeded: max {transfer.max()}"
    assert (transfer >= -NTC_BACKWARD - TOL).all(), f"backward NTC exceeded: min {transfer.min()}"
    assert transfer.max() > NTC_FORWARD - 1.0, "NTC never came close to binding"
    assert len(data.inter_area_ntc) == 1


def test_multi_period_pumping_and_storage_are_active(two_area_multi_period_res):
    _data, res = two_area_multi_period_res
    tables = _tables(res)

    assert tables["Res_Pumping"]["output"].max() > TOL, "pumping capacity was never used"
    assert tables["Res_Storage"]["storage"].nunique() > 1, "reservoir level never changed"


def test_multi_period_flexible_load_is_active(two_area_multi_period_res):
    """Same assertion as the hourly case, now that this mode prices flexible load."""
    _data, res = two_area_multi_period_res
    flex = _tables(res)["Res_FlexibleLoad"]

    assert not flex.empty, "no flexible load rows stored"
    assert flex["demand"].max() > TOL, "flexible demand never dispatched"
    assert flex["storage"].nunique() > 1, "flexible-load storage never changed"


def test_multi_period_rolling_window_with_lookahead(tmp_path, two_area_data):
    """A 48h horizon committing 24h at a time -- commit_hours < horizon_hours.

    True rolling horizon was never exercised: the day-ahead test uses 24/24 and the ramp
    tests use 1/1, so the window-truncation path at the end of the run was dead code.
    """
    data = two_area_data
    lp = powergama.LpProblem(data, objective_mode="multi_period", horizon_hours=48, commit_hours=24)
    res = powergama.Results(data, tmp_path / "rolling.sqlite", replace=True)
    lp.solve(res, solver="appsi_highs", solve_args={})

    tables = _tables(res)
    n_ts = len(data.timerange)
    assert len(tables["Res_Generators"]) == len(data.generator) * n_ts
    assert len(tables["Res_Nodes"]) == len(data.node) * n_ts
    assert tables["Res_Storage"]["storage"].nunique() > 1
