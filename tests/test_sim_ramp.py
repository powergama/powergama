import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rt_case import read_table, series, two_area_data

import powergama

datapath = Path(__file__).parent / "test_data/data_9bus"
TOL = 1e-6
CONFLICT = "inconsistent caps"


def test_generator_ramp_limit(tmp_path, testcase_9bus_data):
    """Check that ramp rates are observed in daily optimisation mode.

    Ramp-up limits always hold. A ramp-down limit gives way only where the generator
    runs out of energy: the first hour of each window applies the single-period guard,
    so a storage plant that empties (here the CSP plant, at timestep 7) generates what
    it has left rather than making the window infeasible. Any breach must therefore
    be at the generator's availability.
    """
    data = testcase_9bus_data

    # set ramp rate limits (change in pu from one timestep to the next)
    data.generator["ramp_up_pu"] = 0.1
    data.generator["ramp_down_pu"] = 0.1

    print(data.generator)

    lp = powergama.LpProblem(data, objective_mode="multi_period", commit_hours=1, horizon_hours=1)
    res = powergama.Results(data, tmp_path / "ramp.sqlite")
    lp.solve(res, solver="appsi_highs", solve_args={})

    n_hours = data.profiles.shape[0]
    storage_gens = data.getIdxGeneratorsWithStorage()
    p_now = res.db.getResultGeneratorPowerAll(0)
    breaches = 0

    for t in range(1, n_hours):
        p_prev = p_now
        p_now = res.db.getResultGeneratorPowerAll(t)
        storage_prev = dict(zip(storage_gens, res.db.getResultStorageFillingAll(timestep=t - 1)))
        for i, gen in data.generator.iterrows():
            assert p_now[i] - p_prev[i] - gen["ramp_up_pu"] * gen["pmax"] <= TOL
            if p_now[i] - p_prev[i] + gen["ramp_down_pu"] * gen["pmax"] < -TOL:
                breaches += 1
                available = gen["pmax"] * gen["inflow_fac"] * data.profiles.loc[t, gen["inflow_ref"]]
                available += storage_prev.get(i, 0.0) / data.timeDelta
                assert p_now[i] == pytest.approx(available, abs=1e-4), (
                    f"generator {i} broke its ramp-down limit at t={t} without running out of energy"
                )
    assert breaches > 0, "fixture no longer exercises the energy-limited case"


def test_generator_ramp_limit_single_period(tmp_path, testcase_9bus_data):
    """Check that ramp rates are observed in single-period optimisation mode.

    The single-period path implements ramp limits by tightening the generator bounds in
    _updateLpProblem against the previous timestep's dispatch, which is a completely
    separate implementation from the multi-period cRampUp/cRampDn constraints that
    test_generator_ramp_limit exercises. Nothing compared the two before.

    Only energy-unlimited generators are checked. For an inflow- or storage-limited
    generator the availability bound can fall below the ramp-down floor (a CSP plant at
    sunset cannot keep generating), and the `pmin_now = min(pmin_now, pmax_now)` guard in
    _updateLpProblem then drops the ramp floor rather than making the LP infeasible. That
    is deliberate, so "ramp limits are always respected" is not an invariant in single-period
    mode -- note the multi-period path has no such escape hatch.
    """
    data = testcase_9bus_data

    data.generator["ramp_up_pu"] = 0.1
    data.generator["ramp_down_pu"] = 0.1

    lp = powergama.LpProblem(data, objective_mode="single_period")
    res = powergama.Results(data, tmp_path / "ramp_single_period.sqlite")
    lp.solve(res, solver="appsi_highs", solve_args={})

    # Generators whose availability bound cannot bind: no storage, and a flat inflow
    # profile at full installed capacity.
    dispatchable = [
        i
        for i, gen in data.generator.iterrows()
        if gen["storage_cap"] == 0 and data.profiles[gen["inflow_ref"]].nunique() == 1 and gen["inflow_fac"] == 1
    ]
    assert dispatchable, "fixture has no energy-unlimited generators to test"

    n_hours = data.profiles.shape[0]
    p_now = res.db.getResultGeneratorPowerAll(0)
    binding = 0

    for t in range(1, n_hours):
        p_prev = p_now
        p_now = res.db.getResultGeneratorPowerAll(t)
        for i in dispatchable:
            gen = data.generator.loc[i]
            delta = p_now[i] - p_prev[i]
            max_up = gen["ramp_up_pu"] * gen["pmax"]
            max_down = gen["ramp_down_pu"] * gen["pmax"]
            assert delta - max_up <= TOL, f"gen {i} t={t}: ramp up {delta} exceeds {max_up}"
            assert delta + max_down >= -TOL, f"gen {i} t={t}: ramp down {delta} exceeds {max_down}"
            if abs(abs(delta) - max_up) < 1e-3:
                binding += 1

    # Guard against a vacuous pass: if no ramp limit ever binds, the assertions above
    # would hold for a completely unconstrained dispatch too.
    assert binding > 0, "no ramp limit ever became binding, so this test proves nothing"


def test_ramp_limit_holds_across_midnight(tmp_path, testcase_9bus_data):
    """There is no daily reset: the ramp limit holds into midnight as at any other hour.

    Day-long windows (commit 24) put midnight at a window's first hour, where the ramp is
    a bound around the previous dispatch; one 48-hour window puts it inside the window
    (h=24), where it is cRampUp/cRampDn. (Windows always start at multiples of
    commit_hours, so a 24-hour horizon never has midnight inside a window.)
    Energy-unlimited generators only, as in the other ramp tests.
    """
    data = testcase_9bus_data
    mask = ~data.generator["type"].isin(["wind", "csp", "nuclear"])
    data.generator["ramp_up_pu"] = float("nan")
    data.generator["ramp_down_pu"] = float("nan")
    data.generator.loc[mask, "ramp_up_pu"] = 0.05
    data.generator.loc[mask, "ramp_down_pu"] = 0.05

    limit = 0.05 * data.generator["pmax"]
    n = data.profiles.shape[0]
    for horizon in (24, n):
        lp = powergama.LpProblem(data, objective_mode="multi_period", commit_hours=horizon, horizon_hours=horizon)
        res = powergama.Results(data, tmp_path / f"ramp_{horizon}.sqlite")
        lp.solve(res, solver="appsi_highs", solve_args={})

        p_now = res.db.getResultGeneratorPowerAll(0)
        for t in range(1, n):
            p_prev, p_now = p_now, res.db.getResultGeneratorPowerAll(t)
            for i in data.generator.index[mask]:
                assert abs(p_now[i] - p_prev[i]) <= limit[i] + TOL, f"generator {i} broke its ramp limit at t={t}"


def test_ramp_down_slack_keeps_window_feasible(tmp_path, testcase_9bus_data):
    """A ramp-down limit that cannot be kept after a window's first hour costs, but does not fail.

    The CSP plant starts at 600 MW with 100 MWh left and no sun. Hour 0 applies the
    single-period guard and generates the 100 MWh; from hour 1 the storage is empty, yet
    the ramp-down limit (30 MW/h) asks for at least 70 MW. Without the slack the window
    is infeasible.
    """
    data = testcase_9bus_data
    (csp,) = data.getIdxGeneratorsWithStorage()
    data.generator["ramp_down_pu"] = float("nan")
    data.generator.loc[csp, "ramp_down_pu"] = 0.05
    data.profiles = data.profiles.iloc[:4]
    data.timerange = range(4)
    assert data.profiles.loc[:3, data.generator.loc[csp, "inflow_ref"]].max() == 0, "fixture needs a night"

    lp = powergama.LpProblem(data, objective_mode="multi_period", horizon_hours=4, commit_hours=4)
    storage = lp.storage.copy()
    storage[csp] = 100.0
    lp.set_storage_state(storage=storage)
    lp._gen_prev[csp] = 600.0
    res = powergama.Results(data, tmp_path / "ramp_slack.sqlite")
    lp.solve(res, solver="appsi_highs", solve_args={})

    output = [res.db.getResultGeneratorPowerAll(t)[csp] for t in range(4)]
    assert output[0] == pytest.approx(100.0)
    assert output[1:] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)


# coal_b in two_area: 300 MW, ramp 0.25 pu/h (75 MW/h) both ways, fuelcost 40.
COAL = 3
COAL_PMAX = 300.0
COAL_RAMP = 75.0


def _coal_caps(data, floor, ceiling):
    """Cap coal_b with pmin_ref/pmax_ref profiles (fractions of its pmax)."""
    assert data.generator.loc[COAL, "pmax"] == COAL_PMAX
    assert data.generator.loc[COAL, "ramp_up_pu"] * COAL_PMAX == COAL_RAMP
    data.profiles["coal_floor"] = floor
    data.profiles["coal_ceiling"] = ceiling
    data.generator.loc[COAL, ["pmin_ref", "pmax_ref"]] = ["coal_floor", "coal_ceiling"]


def _step(data, before, after, at=3):
    return pd.Series([before] * at + [after] * (len(data.profiles) - at), index=data.profiles.index)


def _coal_output(res):
    return series(read_table(res, "Res_Generators"), COAL, "output").values


def test_cap_floor_rising_faster_than_ramp_wins(tmp_path):
    """Single period: the floor jumps from 0 to 270 MW while the unit is held at 30 MW or less
    and can ramp up by 75 MW/h. Caps win over ramps, so it goes to the floor, not to 105 MW."""
    data = two_area_data()
    _coal_caps(data, floor=_step(data, 0.0, 0.9), ceiling=_step(data, 0.1, 1.0))

    lp = powergama.LpProblem(data)
    res = powergama.Results(data, tmp_path / "floor.sqlite")
    with pytest.warns(UserWarning, match=rf"(?s){CONFLICT}.*t=3, generator 3 \(coal_b\): floor rises"):
        lp.solve(res, solver="appsi_highs", solve_args={})

    out = _coal_output(res)
    assert out[2] <= 30.0 + TOL
    assert out[3] == pytest.approx(270.0, abs=TOL)
    ((t, i, kind, mw),) = lp.cap_ramp_conflicts
    assert (t, i, kind) == (3, COAL, "floor_up")
    assert mw == pytest.approx(270.0 - out[2] - COAL_RAMP)


def test_cap_ceiling_falling_faster_than_ramp_wins(tmp_path):
    """Single period: the ceiling drops from 300 to 30 MW while the unit is held at 270 MW or
    more and can ramp down by 75 MW/h. It goes to the ceiling."""
    data = two_area_data()
    _coal_caps(data, floor=_step(data, 0.9, 0.0), ceiling=_step(data, 1.0, 0.1))

    lp = powergama.LpProblem(data)
    res = powergama.Results(data, tmp_path / "ceiling.sqlite")
    with pytest.warns(UserWarning, match=rf"(?s){CONFLICT}.*t=3, generator 3 \(coal_b\): ceiling falls"):
        lp.solve(res, solver="appsi_highs", solve_args={})

    out = _coal_output(res)
    assert out[2] >= 270.0 - TOL
    assert out[3] == pytest.approx(30.0, abs=TOL)
    ((t, i, kind, mw),) = lp.cap_ramp_conflicts
    assert (t, i, kind) == (3, COAL, "ceiling_down")
    assert mw == pytest.approx(out[2] - COAL_RAMP - 30.0)


def _solve_window(data, tmp_path, name, hours=6, match=CONFLICT):
    data.profiles = data.profiles.iloc[:hours]
    data.timerange = range(hours)
    lp = powergama.LpProblem(data, objective_mode="multi_period", horizon_hours=hours, commit_hours=hours)
    res = powergama.Results(data, tmp_path / f"{name}.sqlite")
    with pytest.warns(UserWarning, match=match) as caught:
        lp.solve(res, solver="appsi_highs", solve_args={})
    assert len([w for w in caught if CONFLICT in str(w.message)]) == 1, "one warning per window"
    return lp, _coal_output(res)


def _ramp_limits(m, param):
    return {h: param[COAL, h].value for h in range(1, len(m.s_h))}


def test_cap_floor_rising_inside_a_window_lifts_the_ramp_up_limit(tmp_path):
    """Multi period: the floor rises from 0 to 270 MW at h=3 while the unit is capped at 30 MW
    before. A hard 75 MW/h ramp-up would make the window infeasible; the ramp-up limit into h=3
    is lifted to the rise of the floor (270 MW), so the unit reaches the floor at no cost.

    Unlike the first hour's clip, the lifted limit is a step size, not a target: from 30 MW the
    unit may go up to 300 MW at h=3, and here it goes a little above the floor."""
    data = two_area_data()
    _coal_caps(data, floor=_step(data, 0.0, 0.9), ceiling=_step(data, 0.1, 1.0))
    lp, out = _solve_window(data, tmp_path, "floor_window")
    m = lp.model

    assert out[2] <= 30.0 + TOL
    assert 270.0 - TOL <= out[3] <= out[2] + 270.0 + TOL
    assert _ramp_limits(m, m.p_ramp_up_mw) == {1: COAL_RAMP, 2: COAL_RAMP, 3: 270.0, 4: COAL_RAMP, 5: COAL_RAMP}
    assert _ramp_limits(m, m.p_ramp_dn_mw) == {h: COAL_RAMP for h in range(1, 6)}
    assert all(v.value == pytest.approx(0.0, abs=1e-6) for v in m.varRampSlack.values())
    assert lp.cap_ramp_conflicts == [(3, COAL, "floor_up", pytest.approx(270.0 - COAL_RAMP))]


def test_cap_ceiling_falling_inside_a_window_lifts_the_ramp_down_limit(tmp_path):
    """Multi period: the ceiling falls from 300 to 30 MW at h=3 while the floor holds the unit
    at 270 MW or more before. The ramp-down limit into h=3 is lifted to the fall (270 MW), so the
    unit goes to the ceiling without the priced ramp-down slack."""
    data = two_area_data()
    _coal_caps(data, floor=_step(data, 0.9, 0.0), ceiling=_step(data, 1.0, 0.1))
    lp, out = _solve_window(data, tmp_path, "ceiling_window")
    m = lp.model

    assert out[2] >= 270.0 - TOL
    assert out[3] == pytest.approx(30.0, abs=TOL)
    assert _ramp_limits(m, m.p_ramp_dn_mw) == {1: COAL_RAMP, 2: COAL_RAMP, 3: 270.0, 4: COAL_RAMP, 5: COAL_RAMP}
    assert _ramp_limits(m, m.p_ramp_up_mw) == {h: COAL_RAMP for h in range(1, 6)}
    assert m.varRampSlack[COAL, 3].value == pytest.approx(0.0, abs=1e-6)
    assert lp.cap_ramp_conflicts == [(3, COAL, "ceiling_down", pytest.approx(270.0 - COAL_RAMP))]


@pytest.mark.parametrize("seed", range(5))
def test_lifted_ramp_limits_never_make_a_window_infeasible(tmp_path, seed):
    """Floor and ceiling jumping at random, both at once included (the floor rising while the
    ceiling falls): the unit stays inside its caps every hour, the ramp-down slack is never
    needed, and every step is within the effective ramp limits."""
    rng = np.random.default_rng(seed)
    data = two_area_data()
    hours = 12
    a, b = rng.uniform(0, 1, (2, len(data.profiles)))
    floor, ceiling = np.minimum(a, b), np.maximum(a, b)
    floor[4], ceiling[4] = 0.0, 1.0  # wide open, then both caps close in at once
    floor[5], ceiling[5] = 0.45, 0.55
    _coal_caps(data, floor=floor, ceiling=ceiling)
    lp, out = _solve_window(data, tmp_path, f"random_{seed}", hours=hours)
    m = lp.model

    assert np.all(out >= floor[:hours] * COAL_PMAX - 1e-6)
    assert np.all(out <= ceiling[:hours] * COAL_PMAX + 1e-6)
    assert all(v.value == pytest.approx(0.0, abs=1e-6) for v in m.varRampSlack.values())
    up, dn = _ramp_limits(m, m.p_ramp_up_mw), _ramp_limits(m, m.p_ramp_dn_mw)
    for h in range(1, hours):
        assert up[h] == pytest.approx(max(COAL_RAMP, (floor[h] - floor[h - 1]) * COAL_PMAX))
        assert -dn[h] - 1e-6 <= out[h] - out[h - 1] <= up[h] + 1e-6
    # Every lifted limit is a reported conflict, and nothing else is.
    rise = (floor[1:hours] - floor[: hours - 1]) * COAL_PMAX - COAL_RAMP
    fall = (ceiling[: hours - 1] - ceiling[1:hours]) * COAL_PMAX - COAL_RAMP
    expected = [(h + 1, COAL, "floor_up", x) for h, x in enumerate(rise) if x > 1e-6]
    expected += [(h + 1, COAL, "ceiling_down", x) for h, x in enumerate(fall) if x > 1e-6]
    assert [c[:3] for c in lp.cap_ramp_conflicts] == [c[:3] for c in sorted(expected)]
    assert [c[3] for c in lp.cap_ramp_conflicts] == pytest.approx([c[3] for c in sorted(expected)])


def test_cap_ramp_warning_lists_at_most_ten_conflicts(tmp_path):
    """Caps jumping every hour of a 12-hour window: one warning lists ten conflicts and
    counts the rest; lp.cap_ramp_conflicts has them all."""
    data = two_area_data()
    up = np.arange(len(data.profiles)) % 2 == 1
    _coal_caps(data, floor=np.where(up, 0.9, 0.0), ceiling=np.where(up, 1.0, 0.1))
    lp, _ = _solve_window(data, tmp_path, "every_hour", hours=12, match=r"(?s)in 11 place\(s\).*\.\.\. and 1 more")
    assert len(lp.cap_ramp_conflicts) == 11


def test_ordinary_ramps_report_no_conflicts(tmp_path, testcase_9bus_data):
    """Ramp limits that only bind, and a storage plant running empty (the CSP plant), are not
    cap/ramp conflicts: no warning, nothing recorded, in either mode."""
    data = testcase_9bus_data
    data.generator["ramp_up_pu"] = 0.1
    data.generator["ramp_down_pu"] = 0.1
    for mode, horizon in (("single_period", 1), ("multi_period", 24)):
        lp = powergama.LpProblem(data, objective_mode=mode, horizon_hours=horizon, commit_hours=horizon)
        res = powergama.Results(data, tmp_path / f"{mode}.sqlite")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            lp.solve(res, solver="appsi_highs", solve_args={})
        assert not [w for w in caught if CONFLICT in str(w.message)]
        assert lp.cap_ramp_conflicts == []
