import sqlite3

import pandas as pd
import pyomo.environ as pyo
import pytest

import powergama


def test_simulate_glpk(tmp_path, testcase_9bus_data):
    """Test simulation execution"""
    data = testcase_9bus_data
    lp = powergama.LpProblem(data)
    res = powergama.Results(data, tmp_path / "temp_testcase_9bus.sqlite3", replace=True)
    lp.solve(res, solver="glpk", solve_args=dict())


@pytest.mark.skipif(not pyo.SolverFactory("cbc").available(), reason="Skipping test because CBC is not available.")
def test_simulate_cbc(tmp_path, testcase_9bus_data):
    """Test simulation execution"""
    data = testcase_9bus_data
    lp = powergama.LpProblem(data)
    res = powergama.Results(data, tmp_path / "temp_testcase_9bus.sqlite3", replace=True)
    lp.solve(res, solver="cbc", solve_args=dict())


def test_simulate_highs(tmp_path, testcase_9bus_data):
    """Test simulation execution"""
    data = testcase_9bus_data
    lp = powergama.LpProblem(data)
    res = powergama.Results(data, tmp_path / "temp_testcase_9bus.sqlite3", replace=True)
    lp.solve(res, solver="appsi_highs", solve_args=dict())


def test_simulate_highs_continue(tmp_path, testcase_9bus_data):
    """Test that continuing previously stopped simulation works"""
    data = testcase_9bus_data
    # add flex consumer to check that loadflex storage is well behaved
    data.consumer.loc[0, "flex_basevalue"] = 10
    data.consumer.loc[0, "flex_fraction"] = 0.1
    data.consumer.loc[0, "flex_on_off"] = 0.2
    data.consumer.loc[0, "flex_storage"] = 10
    data.consumer.loc[0, "flex_storagelevel_init"] = 0.5
    data.consumer.loc[0, "flex_storval_filling"] = "flexdemand"
    data.consumer.loc[0, "flex_storval_time"] = "const"

    lp = powergama.LpProblem(data)
    # 1 run all in one go:
    data.timerange = range(0, 10)
    res1 = powergama.Results(data, tmp_path / "temp_testcase_9bus_1.sqlite3", replace=True)
    lp.solve(res1, solver="appsi_highs", solve_args=dict())
    with sqlite3.connect(res1.db.filename) as con:
        res1_branch = pd.read_sql("SELECT * FROM Res_Branches", con)
        res1_storage = pd.read_sql("SELECT * FROM Res_Storage", con)
        res1_loadflex = pd.read_sql("SELECT * FROM Res_FlexibleLoad", con)

    # 2. run in two steps:
    data.timerange = range(0, 6)  # make it stop after step=5
    res2 = powergama.Results(data, tmp_path / "temp_testcase_9bus_2.sqlite3", replace=True)
    lp2 = powergama.LpProblem(data)
    lp2.solve(res2, solver="appsi_highs", solve_args=dict())

    data.timerange = range(0, 10)  # should make it continue from 6
    res2 = powergama.Results(data, tmp_path / "temp_testcase_9bus_2.sqlite3", replace=False)
    lp2 = powergama.LpProblem(data)
    lp2.solve(res2, solver="appsi_highs", solve_args=dict(), continue_from_last=True)
    with sqlite3.connect(res2.db.filename) as con:
        res2_branch = pd.read_sql("SELECT * FROM Res_Branches", con)
        res2_storage = pd.read_sql("SELECT * FROM Res_Storage", con)
        res2_loadflex = pd.read_sql("SELECT * FROM Res_FlexibleLoad", con)

    # another alternative to continue simulation:
    data.timerange = range(0, 6)  # make it stop after step=5
    res3 = powergama.Results(data, tmp_path / "temp_testcase_9bus_3.sqlite3", replace=True)
    lp3 = powergama.LpProblem(data)
    lp3.solve(res3, solver="appsi_highs", solve_args=dict())

    res3 = powergama.Results.from_existing(tmp_path / "temp_testcase_9bus_3.sqlite3", timedelta=1.0)
    data3 = res3.grid
    # extend profiles:
    data3.profiles = data.profiles
    data3.timerange = range(0, 10)
    lp3 = powergama.LpProblem(data3)
    lp3.solve(res3, solver="appsi_highs", solve_args=dict(), continue_from_last=True)
    with sqlite3.connect(res3.db.filename) as con:
        res3_branch = pd.read_sql("SELECT * FROM Res_Branches", con)
        res3_storage = pd.read_sql("SELECT * FROM Res_Storage", con)
        res3_loadflex = pd.read_sql("SELECT * FROM Res_FlexibleLoad", con)

    assert res2_branch.sum().sum() == pytest.approx(res1_branch.sum().sum(), rel=1e-6)

    assert res1_branch.shape == res2_branch.shape

    # branch flows are the same
    pd.testing.assert_frame_equal(res1_branch, res2_branch, check_exact=False, rtol=1e-6)
    pd.testing.assert_frame_equal(res1_branch, res3_branch, check_exact=False, rtol=1e-6)

    # storage properties are the same (CSP generator with storage)
    pd.testing.assert_frame_equal(res1_storage, res2_storage, check_exact=False, rtol=1e-6)
    pd.testing.assert_frame_equal(res1_storage, res3_storage, check_exact=False, rtol=1e-6)

    # storage properties are the same (CSP generator with storage)
    pd.testing.assert_frame_equal(res1_loadflex, res2_loadflex, check_exact=False, rtol=1e-6)
    pd.testing.assert_frame_equal(res1_loadflex, res3_loadflex, check_exact=False, rtol=1e-6)


def _read_result_tables(res):
    with sqlite3.connect(res.db.filename) as con:
        return {
            "branch": pd.read_sql("SELECT * FROM Res_Branches ORDER BY timestep, indx", con),
            "gen": pd.read_sql("SELECT * FROM Res_Generators ORDER BY timestep, indx", con),
            "storage": pd.read_sql("SELECT * FROM Res_Storage ORDER BY timestep, indx", con),
        }


def _run_oneshot_vs_resumed(data, tmp_path, tag, lossmethod):
    """Solve 0..10 in one go, and as 0..6 then resumed to 10. Return both result sets."""
    data.timerange = range(0, 10)
    res1 = powergama.Results(data, tmp_path / f"{tag}_1.sqlite3", replace=True)
    powergama.LpProblem(data, lossmethod=lossmethod).solve(res1, solver="appsi_highs", solve_args=dict())
    one_shot = _read_result_tables(res1)

    data.timerange = range(0, 6)
    res2 = powergama.Results(data, tmp_path / f"{tag}_2.sqlite3", replace=True)
    powergama.LpProblem(data, lossmethod=lossmethod).solve(res2, solver="appsi_highs", solve_args=dict())

    data.timerange = range(0, 10)
    res2 = powergama.Results(data, tmp_path / f"{tag}_2.sqlite3", replace=False)
    powergama.LpProblem(data, lossmethod=lossmethod).solve(
        res2, solver="appsi_highs", solve_args=dict(), continue_from_last=True
    )
    return one_shot, _read_result_tables(res2)


def test_continue_from_last_with_ramp_limits(tmp_path, testcase_9bus_data):
    """Resuming must reproduce a one-shot run when ramp limits are active.

    Covers the `_gen_prev` restore in solve(), which only runs when ramp limits are set
    and was otherwise untested -- the existing continue test has no ramp limits.
    """
    data = testcase_9bus_data
    data.generator.loc[1, "pmax"] = 250  # avoid load shedding
    data.generator["ramp_up_pu"] = 0.2
    data.generator["ramp_down_pu"] = 0.2

    one_shot, resumed = _run_oneshot_vs_resumed(data, tmp_path, "ramp", lossmethod=0)
    for key in ("branch", "gen", "storage"):
        pd.testing.assert_frame_equal(one_shot[key], resumed[key], check_exact=False, rtol=1e-6, obj=key)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "continue_from_last is not exactly resumable with lossmethod=1 or 2. The loss "
        "linearisation is taken around the previous timestep's flows, which the hourly "
        "path carries implicitly in Var.value rather than persisting. On resume the "
        "model is fresh, so the pre-solve in solve() seeds the operating point instead "
        "of the real preceding solution, and results diverge from the first resumed "
        "timestep onward (~0.04 MW on 9-bus, and it propagates). Storage and _gen_prev "
        "restore correctly because they are read back from the database. The fix is the "
        "same: seed the linearisation from Res_Branches, which already holds the flows."
    ),
)
def test_continue_from_last_with_losses(tmp_path, testcase_9bus_data):
    """Resuming should reproduce a one-shot run when losses are modelled."""
    data = testcase_9bus_data
    data.branch["resistance"] = 0.01
    data.generator.loc[1, "pmax"] = 250  # avoid load shedding

    one_shot, resumed = _run_oneshot_vs_resumed(data, tmp_path, "loss", lossmethod=1)
    for key in ("branch", "gen", "storage"):
        pd.testing.assert_frame_equal(one_shot[key], resumed[key], check_exact=False, rtol=1e-6, obj=key)
