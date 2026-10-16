"""
Integration test: IEEE 9 bus system
"""

from pathlib import Path

import pytest

import powergama

datapath = Path(__file__).parent / "test_data/data_9bus"


@pytest.mark.xfail(
    reason=(
        "Storage draw is priced twice in multi_period mode. The objective charges "
        "gen_cost_h (= storval) per MWh generated AND subtracts terminal_storval * "
        "varStorage[i, H-1]. Substituting the storage continuity equality "
        "varStorage[i,h] == prev + (inflow - gen + pump*eff - spill)*dt makes the "
        "effective marginal cost of drawing from storage storval[h] + storval[last]*dt, "
        "against storval in hourly mode. At H=1 that is exactly storval*(1+dt). "
        "This affects every horizon, not only H=1, so test_sim_dayhead's pinned values "
        "record current behaviour rather than correct behaviour. A fix must price either "
        "throughput or terminal state, not both -- see the storage valuation issue."
    )
)
def test_hourly_equals_multi_period_horizon_1(tmp_path, testcase_9bus_data):
    """Test if rolling optimisation with horizon=1 matches hourly optimiation"""

    data = testcase_9bus_data

    lp_hourly = powergama.LpProblem(
        data,
        objective_mode="hourly",
    )

    lp_multi = powergama.LpProblem(
        data,
        objective_mode="multi_period",
        horizon_hours=1,
        commit_hours=1,
    )

    res_hourly = powergama.Results(data, tmp_path / "hourly.sqlite")
    res_multi = powergama.Results(data, tmp_path / "multi_period.sqlite")

    lp_hourly.solve(res_hourly, solver="appsi_highs", solve_args={})
    lp_multi.solve(res_multi, solver="appsi_highs", solve_args={})

    print(tmp_path)
    assert res_hourly.getAverageBranchFlows()[0] == pytest.approx(res_multi.getAverageBranchFlows()[0])
    assert res_hourly.getAverageBranchFlows()[1] == pytest.approx(res_multi.getAverageBranchFlows()[1])
    assert res_hourly.getNodalPrices(0) == pytest.approx(res_multi.getNodalPrices(0))
