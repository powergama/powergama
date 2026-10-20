"""
Both modes solve the same LP over a window of timesteps (issue #42).

Single-period is a window of one timestep. Multi-period windows are built once per
window length and rewritten for every window, through the same solve loop -- so a
window cut short at the end of the run, and any solver, work as in single-period.
"""

import sqlite3

import pyomo.environ as pyo
import pytest

import powergama


def _objective(res):
    with sqlite3.connect(res.db.filename) as con:
        return [v for (v,) in con.execute("SELECT value FROM Res_ObjFunc ORDER BY timestep")]


def test_window_cut_short_at_the_end(tmp_path, testcase_9bus_data):
    """48 timesteps in windows of 36 committing 24: the second window has only 24 left.

    It gets a model of its own, and the configured model is left in place afterwards.
    """
    data = testcase_9bus_data
    lp = powergama.LpProblem(data, objective_mode="multi_period", horizon_hours=36, commit_hours=24)
    full_model = lp.model
    res = powergama.Results(data, tmp_path / "tail.sqlite")
    lp.solve(res, solver="appsi_highs")

    assert len(_objective(res)) == 48
    assert sorted(lp._window_models) == [24, 36]
    assert lp.model is full_model
    assert len(lp.model.s_h) == 36


def test_single_period_is_a_window_of_one(testcase_9bus_data):
    lp = powergama.LpProblem(testcase_9bus_data)
    assert list(lp.model.s_h) == [0]
    assert len(lp.model.cRampDn) == 0 and len(lp.model.cGenUbStorAvail) == 0


@pytest.mark.skipif(not pyo.SolverFactory("glpk").available(), reason="GLPK is not available.")
def test_multi_period_with_another_solver(tmp_path, testcase_9bus_data):
    """Multi-period used to build its own HiGHS solver for every window and gave zero
    prices with any other solver; it now uses the requested one, duals included."""
    data = testcase_9bus_data
    kwargs = dict(objective_mode="multi_period", horizon_hours=24, commit_hours=24)
    res_highs = powergama.Results(data, tmp_path / "highs.sqlite")
    powergama.LpProblem(data, **kwargs).solve(res_highs, solver="appsi_highs")
    res_glpk = powergama.Results(data, tmp_path / "glpk.sqlite")
    powergama.LpProblem(data, **kwargs).solve(res_glpk, solver="glpk")

    assert sum(_objective(res_glpk)) == pytest.approx(sum(_objective(res_highs)))
    assert res_glpk.getNodalPrices(0) == pytest.approx(res_highs.getNodalPrices(0))
    assert max(res_glpk.getNodalPrices(0)) > 0
