"""
Integration test: IEEE 9 bus system
"""
from pathlib import Path

import pytest

import powergama

datapath = Path(__file__).parent / "test_data/data_9bus"


def test_powerflow_matrices():

    data = powergama.GridData()
    data.readGridData(
        nodes=datapath / "9busmod_nodes.csv",
        ac_branches=datapath / "9busmod_branches.csv",
        dc_branches=None,
        generators=datapath / "9busmod_generators.csv",
        consumers=datapath / "9busmod_consumers.csv",
    )
    Bbus, DA = data.compute_power_flow_matrices(base_Z=1)

    expected_DA = {
        (0, 0): -17.36111,
        (6, 1): 16.0,
        (3, 2): -17.06484,
        (0, 3): 17.36111,
        (1, 3): -10.86956,
        (8, 3): 11.76470,
        (1, 4): 10.86956,
        (2, 4): -5.88235,
        (2, 5): 5.88235,
        (3, 5): 17.06484,
        (4, 5): -9.92063,
        (4, 6): 9.92063,
        (5, 6): -13.88888,
        (5, 7): 13.88888,
        (6, 7): -16.0,
        (7, 7): -6.21118,
        (7, 8): 6.21118,
        (8, 8): -11.76470,
    }
    expected_Bbus = {
        (0, 0): 17.36111,
        (3, 0): -17.36111,
        (1, 1): 16.0,
        (7, 1): -16.0,
        (2, 2): 17.06484,
        (5, 2): -17.06484,
        (0, 3): -17.36111,
        (3, 3): 39.99538,
        (4, 3): -10.86956,
        (8, 3): -11.76470,
        (3, 4): -10.86956,
        (4, 4): 16.75191,
        (5, 4): -5.88235,
        (2, 5): -17.06484,
        (4, 5): -5.88235,
        (5, 5): 32.86783,
        (6, 5): -9.92063,
        (5, 6): -9.92063,
        (6, 6): 23.80952,
        (7, 6): -13.88888,
        (1, 7): -16.0,
        (6, 7): -13.88888,
        (7, 7): 36.10006,
        (8, 7): -6.21118,
        (3, 8): -11.76470,
        (7, 8): -6.21118,
        (8, 8): 17.97588,
    }

    assert dict(DA.todok()) == pytest.approx(expected_DA)
    assert dict(Bbus.todok()) == pytest.approx(expected_Bbus)


def test_integration_9bus():
    timerange = range(24 * 100, 24 * 101)
    data = powergama.GridData()

    data.readGridData(
        nodes=datapath / "9busmod_nodes.csv",
        ac_branches=datapath / "9busmod_branches.csv",
        dc_branches=None,
        generators=datapath / "9busmod_generators.csv",
        consumers=datapath / "9busmod_consumers.csv",
    )
    data.readProfileData(
        filename=datapath / "9busmod_profiles.csv",
        storagevalue_filling=datapath / "9busmod_profiles_storval_filling.csv",
        storagevalue_time=datapath / "9busmod_profiles_storval_time.csv",
        timerange=timerange,
        timedelta=1.0,
    )

    lp = powergama.LpProblem(data)
    res = powergama.Results(data, "example_9busmod.sqlite")
    lp.solve(res, solver="glpk")

    # Check if results are as expected

    expected_average_branch_flow = [
        [
            10.0,
            0,
            31.07313626378624,
            62.618337703595806,
            63.07366016488877,
            2.6546450300468116,
            0,
            113.07366016488875,
            0,
        ],
        [0, 112.42014886386117, 30.61781380249325, 0, 0, 39.580984865158094, 150.0, 0, 136.92633983511118],
        [
            10.0,
            112.42014886386117,
            61.690950066279484,
            62.618337703595806,
            63.07366016488877,
            42.23562989520491,
            150.0,
            113.07366016488875,
            136.92633983511118,
        ],
    ]
    average_branch_flow = res.getAverageBranchFlows()
    for i in range(len(expected_average_branch_flow)):
        assert expected_average_branch_flow[i] == pytest.approx(average_branch_flow[i])

    expected_nodal_prices_0 = [
        8.8,
        8.96,
        9.16,
        9.36,
        9.64,
        10.0,
        10.32795615,
        10.36075176,
        10.26236492,
        10.0,
        9.72,
        9.28,
        8.8,
        8.32,
        7.88,
        7.6,
        7.52,
        7.68,
        7.10258823,
        8.72,
        9.32,
        9.84,
        10.19677369,
        10.0,
    ]
    assert res.getNodalPrices(0).tolist() == pytest.approx(expected_nodal_prices_0)

    # TODO check more results
