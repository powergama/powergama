"""
Check that ramp rate limitation is reset between days.
"""

from pathlib import Path

import pytest

import powergama

datapath = Path(__file__).parent / "test_data/data_9bus"


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
