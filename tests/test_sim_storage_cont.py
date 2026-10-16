"""
Check that storage level evolves consistently with inflow, output(discharging) and pumping(charging)
"""

from pathlib import Path

import pytest

import powergama

datapath = Path(__file__).parent / "test_data/data_9bus"


def test_storage_continuity(tmp_path, testcase_9bus_data):
    data = testcase_9bus_data
    n_hours = data.profiles.shape[0]

    lp = powergama.LpProblem(data)
    res = powergama.Results(data, tmp_path / "storage.sqlite")
    lp.solve(res, solver="appsi_highs", solve_args={})

    gen_ind = 4
    generation = res.db.getResultGeneratorPower(gen_ind, timeMaxMin=[0, 9999])
    # no pump capacity in this test system so pumping is zero
    # pumping = res.db.getResultPumpPower(gen_ind, timeMaxMin=[0, 9999])
    pumping = [0] * len(generation)
    storage = res.db.getResultStorageFilling(gen_ind, timeMaxMin=[0, 9999])
    inflow_ref = data.generator.loc[gen_ind, "inflow_ref"]
    inflow_fac = data.generator.loc[gen_ind, "inflow_fac"]
    pmax = data.generator.loc[gen_ind, "pmax"]
    inflow = data.profiles.loc[:, inflow_ref] * inflow_fac * pmax

    for t in range(1, n_hours):
        expected = max(0, storage[t - 1] + inflow[t] + pumping[t] - generation[t])
        print(t, storage[t], expected, storage[t] - expected)
        assert storage[t] == pytest.approx(expected)
