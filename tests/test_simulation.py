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
