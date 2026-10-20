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


def test_daily_reset_with_partial_commit_window_releases_only_at_midnight(tmp_path, testcase_9bus_data):
    """ramp_daily_reset means midnight, whatever the commit window (issue #42).

    It used to be released at every commit-window boundary, which with commit_hours=1
    was every hour, and so this combination was rejected. Now the limit into a midnight
    hour is lifted wherever in a window that hour falls.
    """
    data = testcase_9bus_data
    mask = ~data.generator["type"].isin(["wind", "csp", "nuclear"])
    data.generator["ramp_daily_reset"] = True
    data.generator["ramp_up_pu"] = float("nan")
    data.generator["ramp_down_pu"] = float("nan")
    data.generator.loc[mask, "ramp_up_pu"] = 0.05
    data.generator.loc[mask, "ramp_down_pu"] = 0.05

    lp = powergama.LpProblem(data, objective_mode="multi_period", horizon_hours=24, commit_hours=1)
    res = powergama.Results(data, tmp_path / "reset.sqlite")
    lp.solve(res, solver="appsi_highs")

    limit = 0.05 * data.generator["pmax"]
    p_now = res.db.getResultGeneratorPowerAll(0)
    for t in range(1, data.profiles.shape[0]):
        p_prev, p_now = p_now, res.db.getResultGeneratorPowerAll(t)
        if t % 24 == 0:
            continue
        for i in data.generator.index[mask]:
            assert abs(p_now[i] - p_prev[i]) <= limit[i] + 1e-6, f"generator {i} broke its ramp limit at t={t}"


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
