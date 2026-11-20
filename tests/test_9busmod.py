"""
Integration test: IEEE 9 bus system
"""

from pathlib import Path

import pytest

import powergama


def test_powerflow_matrices(testcase_9bus_data):
    data = testcase_9bus_data

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


def test_integration_9bus(tmp_path, testcase_9bus_data):
    timerange = range(24 * 100, 24 * 101)
    data = testcase_9bus_data

    # Read profile data explicitly from file because default data only includes range(0,48)
    datapath = Path(__file__).parent / "test_data/data_9bus"
    data.readProfileData(
        filename=datapath / "9busmod_profiles.csv",
        storagevalue_filling=datapath / "9busmod_profiles_storval_filling.csv",
        storagevalue_time=datapath / "9busmod_profiles_storval_time.csv",
        timerange=timerange,
        timedelta=1.0,
    )

    lp = powergama.LpProblem(data)
    res = powergama.Results(data, tmp_path / "example_9busmod.sqlite")
    lp.solve(res, solver="appsi_highs")

    # Check if results are as expected

    # Pinned with HiGHS. GLPK finds an equally cheap dispatch at a tie, and the different
    # storage level it leaves carries through to later hours, so its values differ.
    expected_average_branch_flow = [
        [
            10.0,
            0,
            31.33868838724942,
            62.264406838964355,
            62.98528142372052,
            2.5810442529153543,
            0,
            112.98528142372054,
            0,
        ],
        [0, 112.50852760502944, 30.61781380249326, 0, 0, 39.595762829194825, 150.0, 0, 137.01471857627942],
        [
            10.0,
            112.50852760502944,
            61.95650218974268,
            62.264406838964355,
            62.98528142372052,
            42.17680708211018,
            150.0,
            112.98528142372054,
            137.01471857627942,
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
        9.44,
        9.04,
        8.56,
        8.08,
        7.64,
        7.32,
        7.24,
        7.44,
        6.73270589,
        8.44,
        9.04,
        9.56,
        10.0,
        10.0,
    ]
    assert res.getNodalPrices(0).tolist() == pytest.approx(expected_nodal_prices_0)

    # TODO check more results
