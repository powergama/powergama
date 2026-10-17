from pathlib import Path

import pandas as pd

import powergama

datapath = Path(__file__).parent / "test_data/data_9bus"
TOL = 1e-6


def test_generator_ramp_limit(tmp_path, testcase_9bus_data):
    """Check that ramp rates are observed in daily optimisation mode."""
    data = testcase_9bus_data

    # set ramp rate limits (change in pu from one timestep to the next)
    data.generator["ramp_up_pu"] = 0.1
    data.generator["ramp_down_pu"] = 0.1

    print(data.generator)

    lp = powergama.LpProblem(data, objective_mode="multi_period", commit_hours=1, horizon_hours=1)
    res = powergama.Results(data, tmp_path / "ramp.sqlite")
    lp.solve(res, solver="appsi_highs", solve_args={})

    n_hours = data.profiles.shape[0]
    p_now = res.db.getResultGeneratorPowerAll(0)

    for t in range(1, n_hours):
        p_prev = p_now
        p_now = res.db.getResultGeneratorPowerAll(t)
        for i, gen in data.generator.iterrows():
            assert p_now[i] - p_prev[i] - gen["ramp_up_pu"] * gen["pmax"] <= TOL
            assert p_now[i] - p_prev[i] + gen["ramp_down_pu"] * gen["pmax"] >= -TOL


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


def test_daily_reset_releases_ramp_constraint(tmp_path, testcase_9bus_data):
    """Test that ramp rate limit reset at midnight works"""
    data = testcase_9bus_data

    # set ramp rate limits (change in pu from one timestep to the next)

    data.generator["ramp_daily_reset"] = True
    mask = ~data.generator["type"].isin(["wind", "csp", "nuclear"])
    data.generator.loc[mask, "ramp_up_pu"] = 0.05
    data.generator.loc[mask, "ramp_down_pu"] = 0.05

    print(data.generator)

    # commit_hours must be 24 here: the multi-period path releases the ramp constraint at
    # each commit-window boundary, so only whole-day windows make "daily reset" mean
    # midnight. With commit_hours=1 the reset fired every hour, which left cRampUp/cRampDn
    # empty for the whole run and made this test vacuous.
    lp = powergama.LpProblem(data, objective_mode="multi_period", commit_hours=24, horizon_hours=24)
    res = powergama.Results(data, tmp_path / "ramp.sqlite")
    lp.solve(res, solver="appsi_highs", solve_args={})

    max_up = (data.generator["ramp_up_pu"] * data.generator["pmax"]).to_dict()
    max_down = (data.generator["ramp_down_pu"] * data.generator["pmax"]).to_dict()

    df = pd.DataFrame(data.generator["type"])
    df["ramp_up_max"] = max_up
    df["ramp_down_max"] = max_down

    for tstep in [12, 24]:
        p_prev = res.db.getResultGeneratorPowerAll(timestep=tstep - 1)
        p_now = res.db.getResultGeneratorPowerAll(timestep=tstep)
        ramp = {i: p_now[i] - p_prev[i] for i in range(len(p_now))}
        df[tstep - 1] = p_prev
        df[tstep] = p_now
        df[f"ramp_{tstep}"] = ramp

        # print("MAX RAMP:  UP: ", max_up, " DOWN: ", max_down)
        # print(f"t={tstep} RAMP: ", ramp)

        ramp_jump_up = {i: ramp[i] - max_up[i] for i in ramp}
        ramp_jump_down = {i: ramp[i] + max_down[i] for i in ramp}
        # print("Ramp - max (violation if > 0): ", ramp_jump_up, " => max = ", max(ramp_jump_up.values()))
        # print("Ramp + min (violation if < 0): ", ramp_jump_down, " => min = ", min(ramp_jump_down.values()))

        print(df)

        if tstep % 24 == 0:
            # Violation _IS_ allowed, and must actually occur: a genuine breach of the
            # down-ramp limit at the reset boundary is what proves the reset released the
            # constraint. A merely-negative value would also hold for any ramp-feasible
            # dispatch, which is why this previously passed with no ramp constraints built.
            assert min(ramp_jump_down.values()) < -TOL, "expected the daily reset to permit a ramp breach"
        else:
            # no violation allowed
            assert max(ramp_jump_up.values()) <= TOL, "ramp up exceeding max value"
            assert min(ramp_jump_down.values()) >= -TOL, "ramp down exceeding max value"
