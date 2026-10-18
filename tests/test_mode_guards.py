"""
Guards against silently-unsupported combinations of simulation modes.

Each combination below used to run and produce plausible-looking but wrong results
rather than failing. The tests assert that they now raise instead.
"""

import pytest

import powergama
from powergama.lp.rt import RtExtension


def test_rt_with_multi_period_raises(testcase_9bus_data):
    """The multi-period model has no extension hooks, so RT there must raise.

    It would build none of the p_rt_* parameters, deviation variables or deviation
    objective, and silently solve a cost-minimising LP instead.
    """
    with pytest.raises(NotImplementedError, match="RtExtension"):
        powergama.LpProblem(testcase_9bus_data, objective_mode="multi_period", extensions=[RtExtension()])


@pytest.mark.parametrize("lossmethod", [1, 2])
def test_lossmethod_with_multi_period_raises(testcase_9bus_data, lossmethod):
    """The multi-period power balance has no loss terms and reported losses as zero."""
    with pytest.raises(NotImplementedError, match="lossmethod"):
        powergama.LpProblem(testcase_9bus_data, objective_mode="multi_period", lossmethod=lossmethod)


def test_old_hourly_mode_spelling_is_rejected(testcase_9bus_data):
    """The mode was renamed "hourly" -> "single_period" with no alias.

    A timestep need not be an hour, so "hourly" was never the right name. The
    rename is a clean break rather than a deprecation because objective_mode has
    never appeared in a release -- the latest tag is v1.5.2 and this parameter
    arrived after it. Pinned so the break stays deliberate: a silently-accepted
    "hourly" would be worse than the error.
    """
    with pytest.raises(ValueError, match="single_period"):
        powergama.LpProblem(testcase_9bus_data, objective_mode="hourly")


def test_lossmethod_with_single_period_mode_is_allowed(testcase_9bus_data):
    """Guard must not catch the supported combination."""
    lp = powergama.LpProblem(testcase_9bus_data, objective_mode="single_period", lossmethod=1)
    assert lp is not None


def test_daily_reset_with_partial_commit_window_raises(testcase_9bus_data):
    """ramp_daily_reset only means "midnight" when the commit window is a whole day.

    The multi-period path releases the ramp constraint at every commit-window boundary;
    with commit_hours=1 that is every hour, which left the ramp constraints empty.
    """
    data = testcase_9bus_data
    data.generator["ramp_daily_reset"] = True
    data.generator["ramp_up_pu"] = 0.05
    data.generator["ramp_down_pu"] = 0.05

    with pytest.raises(NotImplementedError, match="ramp_daily_reset"):
        powergama.LpProblem(
            data,
            objective_mode="multi_period",
            horizon_hours=24,
            commit_hours=1,
        )


def test_daily_reset_with_whole_day_commit_is_allowed(testcase_9bus_data):
    """Guard must not catch the supported combination."""
    data = testcase_9bus_data
    data.generator["ramp_daily_reset"] = True
    data.generator["ramp_up_pu"] = 0.05

    lp = powergama.LpProblem(
        data,
        objective_mode="multi_period",
        horizon_hours=24,
        commit_hours=24,
    )
    assert lp is not None


def test_daily_reset_guard_ignores_all_false_column(testcase_9bus_data):
    """A present-but-all-False ramp_daily_reset column must not trip the guard."""
    data = testcase_9bus_data
    data.generator["ramp_daily_reset"] = False

    lp = powergama.LpProblem(
        data,
        objective_mode="multi_period",
        horizon_hours=1,
        commit_hours=1,
    )
    assert lp is not None


def test_continue_from_last_on_empty_database_raises(tmp_path, testcase_9bus_data):
    """Resuming from an empty database used to fail as TypeError on None + 1."""
    data = testcase_9bus_data
    lp = powergama.LpProblem(data)
    res = powergama.Results(data, tmp_path / "empty.sqlite", replace=True)

    with pytest.raises(ValueError, match="continue_from_last"):
        lp.solve(res, solver="appsi_highs", solve_args={}, continue_from_last=True)
