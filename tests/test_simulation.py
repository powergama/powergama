import pyomo.environ as pyo
import pytest

import powergama


def test_simulate_glpk(testcase_9bus_data):
    """Test simulation execution"""
    data = testcase_9bus_data
    lp = powergama.LpProblem(data)
    res = powergama.Results(data, "temp_testcase_9bus.sqlite3", replace=True)
    lp.solve(res, solver="glpk", solve_args=dict())


@pytest.mark.skipif(not pyo.SolverFactory("cbc").available(), reason="Skipping test because CBC is not available.")
def test_simulate_cbc(testcase_9bus_data):
    """Test simulation execution"""
    data = testcase_9bus_data
    lp = powergama.LpProblem(data)
    res = powergama.Results(data, "temp_testcase_9bus.sqlite3", replace=True)
    lp.solve(res, solver="cbc", solve_args=dict())


def test_simulate_highs(testcase_9bus_data):
    """Test simulation execution"""
    data = testcase_9bus_data
    lp = powergama.LpProblem(data)
    res = powergama.Results(data, "temp_testcase_9bus.sqlite3", replace=True)
    lp.solve(res, solver="appsi_highs", solve_args=dict())
