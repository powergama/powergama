"""
A multi-period window of one timestep is the single-period problem.

Both formulations value storage the same way: no cost per MWh drawn, and the
energy left in storage at the end valued at the storage value (issue #40). Both
keep a flexible-load reservoir within its bounds, and both apply the same ramp
guard to a window's first timestep (issue #42). So with horizon_hours=1 they must
give the same dispatch, storage, flows and prices.
"""

from pathlib import Path

import pandas as pd
import pytest

import powergama

datapath_two_area = Path(__file__).parent / "test_data/two_area"


def _solve_both(data, tmp_path):
    lp_single = powergama.LpProblem(data, objective_mode="single_period")
    lp_multi = powergama.LpProblem(data, objective_mode="multi_period", horizon_hours=1, commit_hours=1)
    res_single = powergama.Results(data, tmp_path / "single_period.sqlite")
    res_multi = powergama.Results(data, tmp_path / "multi_period.sqlite")
    lp_single.solve(res_single, solver="appsi_highs", solve_args={})
    lp_multi.solve(res_multi, solver="appsi_highs", solve_args={})
    return res_single, res_multi


def _assert_equal(res_single, res_multi, data):
    for t in res_single.timerange:
        assert res_single.db.getResultGeneratorPowerAll(t) == pytest.approx(
            res_multi.db.getResultGeneratorPowerAll(t), abs=1e-6
        ), f"generation differs at t={t}"
        assert res_single.db.getResultStorageFillingAll(t) == pytest.approx(
            res_multi.db.getResultStorageFillingAll(t), abs=1e-6
        ), f"storage differs at t={t}"
        assert res_single.db.getResultFlexloadStorageFillingAll(t) == pytest.approx(
            res_multi.db.getResultFlexloadStorageFillingAll(t), abs=1e-6
        ), f"flexible-load storage differs at t={t}"
    for flows_single, flows_multi in zip(res_single.getAverageBranchFlows(), res_multi.getAverageBranchFlows()):
        assert flows_single == pytest.approx(flows_multi, abs=1e-6)
    for n in range(len(data.node)):
        assert res_single.getNodalPrices(n) == pytest.approx(res_multi.getNodalPrices(n), abs=1e-6)


def test_single_period_equals_multi_period_horizon_1(tmp_path, testcase_9bus_data):
    """Test if rolling optimisation with horizon=1 matches single-period optimisation"""
    data = testcase_9bus_data
    _assert_equal(*_solve_both(data, tmp_path), data)


def _two_area():
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
        timerange=range(48),
        timedelta=1.0,
    )
    return data


def test_single_period_equals_multi_period_horizon_1_two_area(tmp_path):
    """The same on two_area, which has pumping, ramp limits, flexible load, an HVDC link
    and a binding NTC."""
    data = _two_area()
    assert len(data.getIdxGeneratorsWithPumping()) > 0
    assert len(data.getIdxConsumersWithFlexibleLoad()) > 0
    assert not pd.isna(data.generator["ramp_up_pu"]).all()
    assert not pd.isna(data.inter_area_ntc["ntc_forward"]).all()

    _assert_equal(*_solve_both(data, tmp_path), data)
