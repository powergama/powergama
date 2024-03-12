from pathlib import Path

import pytest

import powergama

datapath = Path(__file__).parent / "test_data/data_9bus"
timerange = range(0, 48)


@pytest.fixture
def testcase_9bus_data() -> powergama.GridData:
    """9 bus data"""
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
    return data


@pytest.fixture
def testcase_9bus_res(testcase_9bus_data) -> powergama.Results:
    """Result object for 9 bus data case"""

    data = testcase_9bus_data
    lp = powergama.LpProblem(data)
    res = powergama.Results(data, "testcase_9bus.sqlite3", replace=True)
    lp.solve(res, solver="glpk")
    return res
