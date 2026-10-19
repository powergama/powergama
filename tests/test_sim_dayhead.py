"""
Integration test: IEEE 9 bus system
"""

import sqlite3
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

    # Re-derived when storage stopped being priced twice (issue #40). The CSP plant
    # starts each window valued at its filling level then: 50% full gives
    # 9 EUR/MWh * 0.978 = 8.8 EUR/MWh, below coal's 10, so it displaces coal until
    # it is empty by the end of the first day. Prices are therefore set by coal (10),
    # gas (20) or congestion between them, never by storage. The old values had a
    # price of about twice the storage value and storage held back, because each
    # MWh drawn was charged its storage value twice.
    # Each window's total objective, rather than its flows: the window LPs have
    # several equally cheap dispatches, so the flows depend on how the solver breaks
    # the tie, and changed when the formulations were merged (#42) while every
    # window's objective, price and end-of-day storage stayed the same. The totals
    # are of the objective as stored per timestep, which does not depend on the path
    # storage takes within the window either.
    EXPECTED_WINDOW_OBJECTIVES = [90754.21, 100196.076]

    EXPECTED_PRICES_0 = [
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        20.0,
        20.0,
        20.0,
        20.0,
        20.0,
        15.319899,
        10.0,
        10.0,
        10.0,
        10.0,
        15.319899,
        15.319899,
        20.0,
        20.0,
        20.0,
        20.0,
        20.0,
        15.319899,
        15.319899,
        10.0,
    ]
    EXPECTED_STORAGE_0 = [0.0]

    with sqlite3.connect(res.db.filename) as con:
        objective = [v for (v,) in con.execute("SELECT value FROM Res_ObjFunc ORDER BY timestep")]
    assert [sum(objective[:24]), sum(objective[24:])] == pytest.approx(EXPECTED_WINDOW_OBJECTIVES, abs=1e-2)
    assert res.getNodalPrices(0) == pytest.approx(EXPECTED_PRICES_0)
    assert res.getGeneratorStorageAll(timestep=23) == pytest.approx(EXPECTED_STORAGE_0, abs=1e-6)
