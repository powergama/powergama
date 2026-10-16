"""
Integration test: IEEE 9 bus system
"""

from pathlib import Path

import pytest

import powergama

datapath = Path(__file__).parent / "test_data/data_9bus"


def test_multi_period_regression(tmp_path, testcase_9bus_data):
    data = testcase_9bus_data

    lp = powergama.LpProblem(
        data,
        objective_mode="multi_period",
        horizon_hours=24,
        commit_hours=24,
    )
    print(data.profiles.shape)

    res = powergama.Results(data, tmp_path / "multi_period.sqlite")
    lp.solve(res, solver="appsi_highs")

    EXPECTED_FLOWS_AVG_12 = [
        31.232478057817005,
        3.935302982637881,
        0,
        238.76169705207744,
        99.92035344729948,
        6.524508479994189,
        0,
        149.92035344729953,
        0,
    ]

    EXPECTED_PRICES_0 = [
        10,
        10,
        10,
        16.231167,
        16.231167,
        16.231167,
        16.231167,
        16.231167,
        16.231167,
        16.231167,
        10,
        10,
        10,
        10,
        16.231167,
        16.231167,
        20,
        20,
        20,
        20,
        20,
        16.231167,
        16.231167,
        10,
        10,
        10,
        10,
        10,
        15.837619,
        15.837619,
        15.837619,
        15.837619,
        15.837619,
        15.837619,
        10,
        10,
        10,
        10,
        15.837619,
        15.837619,
        20,
        20,
        20,
        20,
        20,
        15.837619,
        15.837619,
        10,
    ]
    EXPECTED_STORAGE_0 = [2009.2212215864724]

    assert res.getAverageBranchFlows()[0] == pytest.approx(EXPECTED_FLOWS_AVG_12)
    assert res.getNodalPrices(0) == pytest.approx(EXPECTED_PRICES_0)
    assert res.getGeneratorStorageAll(timestep=23) == pytest.approx(EXPECTED_STORAGE_0)
