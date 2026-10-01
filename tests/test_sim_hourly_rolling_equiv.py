"""
Integration test: IEEE 9 bus system
"""

from pathlib import Path

import pytest

import powergama

datapath = Path(__file__).parent / "test_data/data_9bus"


def test_hourly_equals_daily_horizon_1(tmp_path, testcase_9bus_data):
    """Test if rolling optimisation with horizon=1 matches hourly optimiation"""

    data = testcase_9bus_data

    lp_hourly = powergama.LpProblem(
        data,
        objective_mode="hourly",
    )

    lp_daily = powergama.LpProblem(
        data,
        objective_mode="daily_24h",
        objective_day_horizon_hours=1,
        objective_day_commit_hours=1,
    )

    res_hourly = powergama.Results(data, tmp_path / "hourly.sqlite")
    res_daily = powergama.Results(data, tmp_path / "daily.sqlite")

    lp_hourly.solve(res_hourly, solver="appsi_highs", solve_args={})
    lp_daily.solve(res_daily, solver="appsi_highs", solve_args={})

    print(tmp_path)
    assert res_hourly.getAverageBranchFlows()[0] == pytest.approx(res_daily.getAverageBranchFlows()[0])
    assert res_hourly.getAverageBranchFlows()[1] == pytest.approx(res_daily.getAverageBranchFlows()[1])
    assert res_hourly.getNodalPrices(0) == pytest.approx(res_daily.getNodalPrices(0))
