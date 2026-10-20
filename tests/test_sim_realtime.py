"""
Real-time (balancing) mode: tracking of day-ahead targets and application of day-ahead locks.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rt_case import (
    BRANCH_BORDER,
    CONS_FLEX,
    CONS_LOCKED,
    DCBRANCH_BORDER,
    FLOW_WINDOW,
    GEN_LOCKED,
    GEN_STORAGE,
    GEN_TARGET,
    LOCKED_DEMAND,
    build_rt_case,
    read_table,
    series,
    solve_rt_case,
)

import powergama
from powergama.lp.rt import RtExtension
from powergama.lp.rt.locks import LOCK_ATTRS

datapath = Path(__file__).parent / "test_data/data_9bus"


def test_rt_applies_locks_and_targets(tmp_path):
    data, da = build_rt_case(tmp_path)
    lp, res = solve_rt_case(data, tmp_path / "rt.sqlite")

    gen = read_table(res, "Res_Generators")
    np.testing.assert_allclose(series(gen, GEN_LOCKED, "output").values, da["gen_lock"], atol=1e-6)

    ac = series(read_table(res, "Res_Branches"), BRANCH_BORDER, "flow").values
    assert np.all(ac >= da["ac_flow"] - FLOW_WINDOW - 1e-6)
    assert np.all(ac <= da["ac_flow"] + FLOW_WINDOW + 1e-6)
    dc = series(read_table(res, "Res_DcBranches"), DCBRANCH_BORDER, "flow").values
    assert np.all(dc >= da["dc_flow"] - FLOW_WINDOW - 1e-6)
    assert np.all(dc <= da["dc_flow"] + FLOW_WINDOW + 1e-6)

    # The locked demand reached the power balance parameter.
    assert lp.model.p_demand[CONS_LOCKED, 0].value == pytest.approx(LOCKED_DEMAND)

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
        extensions=[RtExtension(balancing_fee_eur_per_mwh=100)],
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
        extensions=[RtExtension(balancing_fee_eur_per_mwh=1)],
    )
    res = powergama.Results(data, tmp_path / "test_rt.sqlite")
    lp.solve(res, solver="appsi_highs", solve_args={})

    gen_coal_plant = res.db.getResultGeneratorPower(ind_coal_plant, timeMaxMin=[0, 1e9])

    # Generator does not follow the target dispatch
    assert gen_coal_plant[5] < expected_dispatch[5]
    assert gen_coal_plant[6] < expected_dispatch[6]


@pytest.mark.parametrize("mode", ["single_period", "multi_period"])
def test_run_without_rt_extension_ignores_rt_inputs(tmp_path, mode):
    """Without RtExtension, RT targets and DA locks are neither built nor applied.

    The day-ahead locks and targets are inputs to the RT step only. A grid carrying
    them in a normal run gets the same results as one without them.
    """
    data, _ = build_rt_case(tmp_path)
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
        pd.testing.assert_frame_equal(read_table(res, table), read_table(res_plain, table), obj=table)
