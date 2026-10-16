import copy
from pathlib import Path

import pytest

import powergama

datapath = Path(__file__).parent / "test_data/data_9bus"
timerange = range(0, 48)


def _load_9bus() -> powergama.GridData:
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


@pytest.fixture(scope="session")
def _9bus_source() -> powergama.GridData:
    """Parse the 9-bus CSVs once per session (the profiles are ~670 kB)."""
    return _load_9bus()


@pytest.fixture
def testcase_9bus_data(_9bus_source) -> powergama.GridData:
    """9 bus data.

    A deep copy per test: several tests mutate this (pmax, ramp limits, flexible-load
    columns, extra profile columns), so it must not be shared.
    """
    return copy.deepcopy(_9bus_source)


@pytest.fixture(scope="session")
def _9bus_solved_db(tmp_path_factory) -> Path:
    """Solve the baseline 9-bus case once per session and return the database path.

    Six tests consume the solved baseline read-only, which previously meant six
    identical 48-timestep solves. Only the database is cached here, not the Results
    object, so that `testcase_9bus_res` can still bind it to each test's own GridData.
    """
    dbfile = tmp_path_factory.mktemp("baseline") / "testcase_9bus.sqlite3"
    data = _load_9bus()
    lp = powergama.LpProblem(data)
    res = powergama.Results(data, dbfile, replace=True)
    lp.solve(res, solver="glpk")
    return dbfile


@pytest.fixture
def testcase_9bus_res(testcase_9bus_data, _9bus_solved_db) -> powergama.Results:
    """Result object for 9 bus data case.

    Reuses the session-cached solve, but is constructed against this test's own
    GridData so that `res.grid is testcase_9bus_data` still holds -- several tests
    request both fixtures and compare across them.
    """
    return powergama.Results(testcase_9bus_data, _9bus_solved_db, replace=False)
