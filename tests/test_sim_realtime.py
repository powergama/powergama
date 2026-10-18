"""
Real-time (balancing) mode: tracking of day-ahead targets and application of day-ahead locks.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import powergama
from powergama.lp.project_locks import LOCK_ATTRS

datapath = Path(__file__).parent / "test_data/data_9bus"
datapath_two_area = Path(__file__).parent / "test_data/two_area"

N_STEPS = 24
GEN_LOCKED = 0  # gas_a: hard-locked to its baseline output
GEN_TARGET = 3  # coal_b: soft dispatch target
GEN_STORAGE = 2  # hydro_pump: soft storage-filling target
CONS_LOCKED = 0  # consumer at a2: demand locked
CONS_FLEX = 1  # consumer at b2: flexible load with a soft target
BRANCH_BORDER = 1  # a2-b1, the AC branch between the two areas
DCBRANCH_BORDER = 0  # a1-b2 HVDC link
FLOW_WINDOW = 30.0  # MW either side of the baseline border flow
LOCKED_DEMAND = 140.0  # MW, against a profile-driven demand_avg of 150


def _two_area_data():
    data = powergama.GridData()
    data.readGridData(
        nodes=datapath_two_area / "nodes.csv",
        ac_branches=datapath_two_area / "branches.csv",
        dc_branches=datapath_two_area / "dcbranches.csv",
        generators=datapath_two_area / "generators.csv",
        consumers=datapath_two_area / "consumers.csv",
        inter_area_ntc=datapath_two_area / "inter_area_ntc.csv",
    )
    data.readProfileData(
        filename=datapath_two_area / "profiles.csv",
        storagevalue_filling=datapath_two_area / "profiles_storval_filling.csv",
        storagevalue_time=datapath_two_area / "profiles_storval_time.csv",
        timerange=range(0, N_STEPS),
        timedelta=1.0,
    )
    return data


def _read(res, table):
    with sqlite3.connect(res.db.filename) as con:
        return pd.read_sql_query(f"SELECT * FROM {table} ORDER BY timestep, indx", con)  # nosec B608


def _series(df, indx, col):
    return df[df["indx"] == indx].set_index("timestep")[col]


def _write_lock(path, frame):
    frame.reset_index().to_parquet(path)
    return str(path)


def build_rt_case(tmp_path):
    """A two-step case: a baseline (day-ahead) run, then RT inputs derived from it.

    Returns the RT GridData, with soft targets as profile columns and hard locks as
    parquet files on the grid object, the way the external prepare-rt step supplies them.
    """
    base = _two_area_data()
    lp = powergama.LpProblem(base)
    res = powergama.Results(base, tmp_path / "da.sqlite", replace=True)
    lp.solve(res, solver="appsi_highs", solve_args={})
    gen = _read(res, "Res_Generators")
    sto = _read(res, "Res_Storage")
    flex = _read(res, "Res_FlexibleLoad")
    ac = _read(res, "Res_Branches")
    dc = _read(res, "Res_DcBranches")

    data = _two_area_data()
    steps = list(data.timerange)

    # Soft targets, as fractions the way the profile columns are interpreted.
    data.profiles["rt_target_coal"] = (
        _series(gen, GEN_TARGET, "output").reindex(steps).values / data.generator.loc[GEN_TARGET, "pmax"]
    )
    data.profiles["rt_storage_hydro"] = (
        _series(sto, GEN_STORAGE, "storage").reindex(steps).values / data.generator.loc[GEN_STORAGE, "storage_cap"]
    )
    data.profiles["rt_target_flex"] = (
        _series(flex, CONS_FLEX, "demand").reindex(steps).values / data.consumer.loc[CONS_FLEX, "demand_avg"]
    )
    data.generator["rt_target_ref"] = ""
    data.generator.loc[GEN_TARGET, "rt_target_ref"] = "rt_target_coal"
    data.generator["rt_storage_target_ref"] = ""
    data.generator.loc[GEN_STORAGE, "rt_storage_target_ref"] = "rt_storage_hydro"
    data.consumer["rt_target_ref"] = ""
    data.consumer.loc[CONS_FLEX, "rt_target_ref"] = "rt_target_flex"

    # Hard locks and border-flow windows/targets, as parquet artefacts.
    idx = pd.MultiIndex.from_product([steps, [GEN_LOCKED]], names=["timestep", "indx"])
    gen_lock = pd.DataFrame({"output": _series(gen, GEN_LOCKED, "output").reindex(steps).values}, index=idx)
    idx = pd.MultiIndex.from_product([steps, [CONS_LOCKED]], names=["timestep", "indx"])
    cons_lock = pd.DataFrame({"demand": LOCKED_DEMAND}, index=idx)
    ac_flow = _series(ac, BRANCH_BORDER, "flow").reindex(steps).values
    dc_flow = _series(dc, DCBRANCH_BORDER, "flow").reindex(steps).values
    idx_ac = pd.MultiIndex.from_product([steps, [BRANCH_BORDER]], names=["timestep", "indx"])
    idx_dc = pd.MultiIndex.from_product([steps, [DCBRANCH_BORDER]], names=["timestep", "indx"])
    idx_sto = pd.MultiIndex.from_product([steps, [GEN_STORAGE]], names=["timestep", "indx"])
    data.foreign_gen_lock_parquet = _write_lock(tmp_path / "gen_lock.parquet", gen_lock)
    data.foreign_consumer_lock_parquet = _write_lock(tmp_path / "cons_lock.parquet", cons_lock)
    data.border_ac_flow_lock_parquet = _write_lock(
        tmp_path / "ac_lock.parquet", pd.DataFrame({"flow": ac_flow}, index=idx_ac)
    )
    data.border_dc_flow_lock_parquet = _write_lock(
        tmp_path / "dc_lock.parquet", pd.DataFrame({"flow": dc_flow}, index=idx_dc)
    )
    data.border_ac_flow_bounds_parquet = _write_lock(
        tmp_path / "ac_bounds.parquet",
        pd.DataFrame({"flow_lb": ac_flow - FLOW_WINDOW, "flow_ub": ac_flow + FLOW_WINDOW}, index=idx_ac),
    )
    data.border_dc_flow_bounds_parquet = _write_lock(
        tmp_path / "dc_bounds.parquet",
        pd.DataFrame({"flow_lb": dc_flow - FLOW_WINDOW, "flow_ub": dc_flow + FLOW_WINDOW}, index=idx_dc),
    )
    data.da_storage_marginalprice_parquet = _write_lock(
        tmp_path / "storage_price.parquet", pd.DataFrame({"marginalprice": 20.0}, index=idx_sto)
    )

    # The real-time deviation: less wind than the day-ahead run planned for.
    data.profiles["wind"] = 0.7 * data.profiles["wind"]
    return data, {"ac_flow": ac_flow, "dc_flow": dc_flow, "gen_lock": gen_lock["output"].values}


def solve_rt_case(data, dbfile):
    lp = powergama.LpProblem(data, is_rt=True, rt_balancing_fee_eur_per_mwh=10, rt_xborder_flow_penalty_eur_per_mwh=5)
    res = powergama.Results(data, dbfile, replace=True)
    lp.solve(res, solver="appsi_highs", solve_args={})
    return lp, res


def test_rt_applies_locks_and_targets(tmp_path):
    data, da = build_rt_case(tmp_path)
    lp, res = solve_rt_case(data, tmp_path / "rt.sqlite")

    gen = _read(res, "Res_Generators")
    np.testing.assert_allclose(_series(gen, GEN_LOCKED, "output").values, da["gen_lock"], atol=1e-6)

    ac = _series(_read(res, "Res_Branches"), BRANCH_BORDER, "flow").values
    assert np.all(ac >= da["ac_flow"] - FLOW_WINDOW - 1e-6)
    assert np.all(ac <= da["ac_flow"] + FLOW_WINDOW + 1e-6)
    dc = _series(_read(res, "Res_DcBranches"), DCBRANCH_BORDER, "flow").values
    assert np.all(dc >= da["dc_flow"] - FLOW_WINDOW - 1e-6)
    assert np.all(dc <= da["dc_flow"] + FLOW_WINDOW + 1e-6)

    # The locked demand reached the power balance parameter.
    assert lp.model.p_demand[CONS_LOCKED].value == pytest.approx(LOCKED_DEMAND)

    # Every kind of target produced a tracking constraint.
    assert set(lp.model.cRtTargetTracking) == {GEN_TARGET}
    assert set(lp.model.cRtStorageTargetTracking) == {GEN_STORAGE}
    assert set(lp.model.cRtFlexLoadTargetTracking) == {CONS_FLEX}
    assert set(lp.model.cRtIoTargetTrackingAc) == {BRANCH_BORDER}
    assert set(lp.model.cRtIoTargetTrackingDc) == {DCBRANCH_BORDER}


def test_rt_tracks_dispatch_target_when_deviations_are_expensive(tmp_path, testcase_9bus_data):
    data = testcase_9bus_data
    data.timerange = range(7)
    data.profiles = data.profiles.loc[:6]

    ind_coal_plant = 2

    # This was computed from a normal simulation (coal plant dispatch)
    expected_dispatch = [10.0, 10.0, 10.0, 10.0, 10.0, 277.1186940355519, 282.1072831247453]

    # modification for real-time
    data.profiles["wind"] = data.profiles["wind"] + 0.1 * 0

    data.profiles["rt_lock_coal"] = expected_dispatch / data.generator.loc[ind_coal_plant, "pmax"]
    data.generator.loc[ind_coal_plant, "rt_target_ref"] = "rt_lock_coal"

    lp = powergama.LpProblem(
        data,
        is_rt=True,
        rt_balancing_fee_eur_per_mwh=100,
    )
    res = powergama.Results(data, tmp_path / "test_rt.sqlite")
    lp.solve(res, solver="appsi_highs", solve_args={})

    gen_coal_plant = res.db.getResultGeneratorPower(ind_coal_plant, timeMaxMin=[0, 1e9])

    assert gen_coal_plant == pytest.approx(expected_dispatch)


def test_rt_target_can_be_violated_if_economic(tmp_path, testcase_9bus_data):
    data = testcase_9bus_data
    data.timerange = range(7)
    data.profiles = data.profiles.loc[:6]

    ind_coal_plant = 2

    # This was computed from a normal simulation (coal plant dispatch)
    expected_dispatch = [10.0, 10.0, 10.0, 10.0, 10.0, 277.1186940355519, 282.1072831247453]

    # modification for real-time
    data.profiles["wind"] = data.profiles["wind"] + 0.1 * 0

    data.profiles["rt_lock_coal"] = expected_dispatch / data.generator.loc[ind_coal_plant, "pmax"]
    data.generator.loc[ind_coal_plant, "rt_target_ref"] = "rt_lock_coal"

    lp = powergama.LpProblem(
        data,
        is_rt=True,
        rt_balancing_fee_eur_per_mwh=1,
    )
    res = powergama.Results(data, tmp_path / "test_rt.sqlite")
    lp.solve(res, solver="appsi_highs", solve_args={})

    gen_coal_plant = res.db.getResultGeneratorPower(ind_coal_plant, timeMaxMin=[0, 1e9])

    # Generator does not follow the target dispatch
    assert gen_coal_plant[5] < expected_dispatch[5]
    assert gen_coal_plant[6] < expected_dispatch[6]


@pytest.mark.parametrize("mode", ["single_period", "multi_period"])
def test_non_rt_run_ignores_rt_inputs(tmp_path, mode):
    """Without is_rt, RT targets and DA locks are neither built nor applied.

    The day-ahead locks are inputs to the RT step only. A grid carrying them in a
    normal run gets a warning, and the same results as without them.
    """
    data, _ = build_rt_case(tmp_path)
    with pytest.warns(UserWarning, match="ignored"):
        lp = powergama.LpProblem(data, objective_mode=mode)
    res = powergama.Results(data, tmp_path / "with_locks.sqlite", replace=True)
    lp.solve(res, solver="appsi_highs", solve_args={})

    rt_names = [c.name for c in lp.model.component_objects() if c.name.startswith(("p_rt_", "varRt", "cRt"))]
    assert rt_names == []

    for attr in LOCK_ATTRS:
        setattr(data, attr, "")
    lp_plain = powergama.LpProblem(data, objective_mode=mode)
    res_plain = powergama.Results(data, tmp_path / "without_locks.sqlite", replace=True)
    lp_plain.solve(res_plain, solver="appsi_highs", solve_args={})
    for table in ["Res_Generators", "Res_Branches", "Res_DcBranches", "Res_Nodes"]:
        pd.testing.assert_frame_equal(_read(res, table), _read(res_plain, table), obj=table)
