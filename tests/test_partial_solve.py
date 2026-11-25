"""Solving part of the time range from a given state: solve(timesteps, state), SimulationState, MemoryResults."""

import copy
import sqlite3

import numpy as np
import pandas as pd
import pytest

import powergama
from powergama import MemoryResults, SimulationState


def _with_flex_and_ramps(data):
    """9-bus data with a flexible load and ramp limits, so all of the state matters."""
    data.consumer.loc[0, "flex_basevalue"] = 10
    data.consumer.loc[0, "flex_fraction"] = 0.1
    data.consumer.loc[0, "flex_on_off"] = 0.2
    data.consumer.loc[0, "flex_storage"] = 10
    data.consumer.loc[0, "flex_storagelevel_init"] = 0.5
    data.consumer.loc[0, "flex_storval_filling"] = "flexdemand"
    data.consumer.loc[0, "flex_storval_time"] = "const"
    data.generator["ramp_up_pu"] = 0.1
    data.generator["ramp_down_pu"] = 0.1
    data.timerange = range(0, 10)
    return data


def _db_frame(res, table, column):
    with sqlite3.connect(res.db.filename) as con:
        df = pd.read_sql(f"SELECT timestep, indx, {column} FROM {table}", con)  # nosec B608
    return df.pivot(index="timestep", columns="indx", values=column).rename_axis(index="timestep", columns=None)


@pytest.fixture
def full_run(tmp_path, testcase_9bus_data):
    data = _with_flex_and_ramps(testcase_9bus_data)
    res = powergama.Results(data, tmp_path / "full.sqlite3", replace=True)
    powergama.LpProblem(data).solve(res, solver="appsi_highs")
    return data, res


def test_memory_results_match_database(full_run):
    data, res = full_run
    mem = MemoryResults(data)
    powergama.LpProblem(data).solve(mem, solver="appsi_highs", show_progress=False)
    assert mem.timesteps == list(range(10))
    for name, table, column in [
        ("generator_power", "Res_Generators", "output"),
        ("storage", "Res_Storage", "storage"),
        ("flexload_storage", "Res_FlexibleLoad", "storage"),
        ("loadshed_power", "Res_Nodes", "loadshed"),
    ]:
        pd.testing.assert_frame_equal(mem.frame(name), _db_frame(res, table, column), check_dtype=False, rtol=1e-6)


def test_partial_solve_from_state_reproduces_full_run(full_run):
    """Re-solving timesteps 6..9 from the full run's state at 5 gives the full run's results."""
    data, res = full_run
    mem = MemoryResults(data)
    state = SimulationState.from_results(res, timestep=5)
    powergama.LpProblem(data).solve(mem, timesteps=range(6, 10), state=state, show_progress=False)
    assert mem.timesteps == [6, 7, 8, 9]
    expected = _db_frame(res, "Res_Generators", "output").loc[6:9]
    pd.testing.assert_frame_equal(mem.frame("generator_power"), expected, check_dtype=False, rtol=1e-6)
    expected = _db_frame(res, "Res_Storage", "storage").loc[6:9]
    pd.testing.assert_frame_equal(mem.frame("storage"), expected, check_dtype=False, rtol=1e-6)


def test_get_state_matches_from_results(full_run):
    data, res = full_run
    lp = powergama.LpProblem(data)
    lp.solve(MemoryResults(data), timesteps=range(0, 6), show_progress=False)
    got = lp.get_state()
    want = SimulationState.from_results(res, timestep=5)
    stores = data.getIdxGeneratorsWithStorage()
    np.testing.assert_allclose(got.storage.iloc[stores], want.storage.iloc[stores], rtol=1e-6)
    np.testing.assert_allclose(got.storage_flexload, want.storage_flexload, rtol=1e-6)
    np.testing.assert_allclose(got.gen_prev, want.gen_prev, rtol=1e-6, atol=1e-6)


def test_repeated_solves_reuse_solver_and_repeat_results(full_run):
    """The same short solve twice on one LpProblem: same results, solver kept."""
    data, res = full_run
    lp = powergama.LpProblem(data)
    state = SimulationState.from_results(res, timestep=3)
    first, second = MemoryResults(data), MemoryResults(data)
    lp.solve(first, timesteps=[4, 5], state=state, show_progress=False)
    opt = lp._window_models[1][2]
    lp.solve(second, timesteps=[4, 5], state=state, show_progress=False)
    assert lp._window_models[1][2] is opt
    pd.testing.assert_frame_equal(first.frame("generator_power"), second.frame("generator_power"))


def test_state_with_modified_grid(full_run):
    """The intended use: the full run's state, a generator taken out, a few timesteps."""
    data, res = full_run
    outaged = copy.deepcopy(data)
    outaged.generator.loc[0, ["pmax", "pmin"]] = 0.0
    mem = MemoryResults(outaged)
    state = SimulationState.from_results(res, timestep=5)
    powergama.LpProblem(outaged).solve(mem, timesteps=range(6, 8), state=state, show_progress=False)
    assert (mem.frame("generator_power")[0] == 0).all()


def test_timesteps_and_state_are_checked(full_run):
    data, res = full_run
    lp = powergama.LpProblem(data)
    mem = MemoryResults(data)
    with pytest.raises(ValueError, match="consecutive"):
        lp.solve(mem, timesteps=[1, 3])
    with pytest.raises(ValueError, match="outside"):
        lp.solve(mem, timesteps=range(8, 12))
    with pytest.raises(ValueError, match="empty"):
        lp.solve(mem, timesteps=[])
    with pytest.raises(ValueError, match="continue_from_last"):
        lp.solve(res, continue_from_last=True, state=SimulationState.from_results(res, 5))
    with pytest.raises(ValueError, match="storage values"):
        SimulationState.from_results(res, timestep=99)


def test_progress_bar_shown_or_hidden(capfd, testcase_9bus_data):
    data = testcase_9bus_data
    data.timerange = range(0, 3)
    lp = powergama.LpProblem(data)
    capfd.readouterr()
    lp.solve(MemoryResults(data), show_progress=False)
    assert "it/s" not in capfd.readouterr().err
    lp.solve(MemoryResults(data), show_progress=True)
    assert "3/3" in capfd.readouterr().err  # text bar outside a notebook
