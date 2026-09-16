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


def test_daily_joint_preparation_uses_hourly_input_helpers(testcase_9bus_data, monkeypatch):
    """The joint extension must use the hourly model's input calculations."""
    data = testcase_9bus_data
    lp = powergama.LpProblem(data, objective_mode="daily_24h")
    calls = {"generator": 0, "demand": 0, "storage": 0}

    original_generator = lp._get_generator_profile_values
    original_demand = lp._get_demand_at_timestep
    original_storage = lp._get_storage_dispatch_costs

    def generator_values(*args, **kwargs):
        calls["generator"] += 1
        return original_generator(*args, **kwargs)

    def demand_value(*args, **kwargs):
        calls["demand"] += 1
        return original_demand(*args, **kwargs)

    def storage_costs(*args, **kwargs):
        calls["storage"] += 1
        return original_storage(*args, **kwargs)

    monkeypatch.setattr(lp, "_get_generator_profile_values", generator_values)
    monkeypatch.setattr(lp, "_get_demand_at_timestep", demand_value)
    monkeypatch.setattr(lp, "_get_storage_dispatch_costs", storage_costs)

    lp._updateLpProblem(0)
    _, pre_data = lp._build_day_joint_model([0])

    assert calls == {
        "generator": 2 * len(lp.s_gen),
        "demand": 2 * len(lp.s_load),
        "storage": 1 + len(lp.s_gen_storage),
    }
    for i in lp.s_gen:
        inflow, capacity, pmin = original_generator(0, i)
        assert pre_data["inflow_h"][(0, i)] == pytest.approx(inflow)
        if i in lp.s_gen_storage:
            assert lp.p_gen_pmax[i].value == pytest.approx(min(inflow + lp._storage[i] / lp.timeDelta, capacity))
        else:
            assert lp.p_gen_pmax[i].value == pytest.approx(inflow)
            assert lp.p_gen_pmin[i].value == pytest.approx(max(min(inflow, pmin), 0))
    for i in lp.s_load:
        assert lp.p_demand[i].value == pytest.approx(original_demand(0, i))


def test_storage_pump_credit_uses_stored_energy_value(testcase_9bus_data):
    """Pumping is credited only for energy that reaches storage."""
    data = testcase_9bus_data
    storage_gen = 4
    data.generator.loc[storage_gen, "pump_cap"] = 100.0
    data.generator.loc[storage_gen, "pump_efficiency"] = 0.8
    data.generator.loc[storage_gen, "pump_deadband"] = 2.0

    lp = powergama.LpProblem(data, objective_mode="daily_24h")
    storagevalue, pump_credit = lp._get_storage_dispatch_costs(0, storage_gen)

    assert pump_credit == pytest.approx(storagevalue * 0.8 - 2.0)
    lp._updateLpProblem(0)
    assert lp.p_genpump_cost[storage_gen].value == pytest.approx(pump_credit)
    _, pre_data = lp._build_day_joint_model([0])
    assert pre_data["pump_cost_h"][(0, storage_gen)] == pytest.approx(pump_credit)


def test_rt_storage_tracks_da_generation_and_charge(testcase_9bus_data):
    """RT storage deviations use signed DA generation and charging values."""
    data = testcase_9bus_data
    storage_gen = 4
    efficiency = 0.8
    deadband = 2.0
    fee = 1.0
    data.generator.loc[storage_gen, "pump_cap"] = 100.0
    data.generator.loc[storage_gen, "pump_efficiency"] = efficiency
    data.generator.loc[storage_gen, "pump_deadband"] = deadband
    data.profiles["rt_generation_target"] = 0.0
    data.profiles["rt_pump_target"] = 0.0
    data.generator.loc[storage_gen, "rt_target_ref"] = "rt_generation_target"
    data.generator.loc[storage_gen, "rt_pump_target_ref"] = "rt_pump_target"

    lp = powergama.LpProblem(data, is_rt=True, rt_balancing_fee_eur_per_mwh=fee)
    lp._updateLpProblem(0)

    storagevalue = lp.p_gen_cost[storage_gen].value
    pump_credit = lp.p_genpump_cost[storage_gen].value
    assert lp.p_rt_deviation_price_gen_pos[storage_gen].value == pytest.approx(fee + storagevalue)
    assert lp.p_rt_deviation_price_pump_pos[storage_gen].value == pytest.approx(fee - pump_credit)
    zero_storage_cycle_cost = (
        efficiency * lp.p_rt_deviation_price_gen_pos[storage_gen].value
        + lp.p_rt_deviation_price_pump_pos[storage_gen].value
    )
    assert zero_storage_cycle_cost == pytest.approx(fee * (1.0 + efficiency) + deadband)


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
