from pathlib import Path

import pandas as pd

import powergama

datapath = Path(__file__).parent / "test_data/data_9bus"


def test_generator_ramp_limit(tmp_path, testcase_9bus_data):
    """Check that ramp rates are observed in daily optimisation mode."""
    data = testcase_9bus_data

    # set ramp rate limits (change in pu from one timestep to the next)
    data.generator["ramp_up_pu"] = 0.1
    data.generator["ramp_down_pu"] = 0.1

    print(data.generator)

    lp = powergama.LpProblem(
        data, objective_mode="daily_24h", objective_day_commit_hours=1, objective_day_horizon_hours=1
    )
    res = powergama.Results(data, tmp_path / "ramp.sqlite")
    lp.solve(res, solver="appsi_highs", solver_path={})

    n_hours = data.profiles.shape[0]
    p_now = res.db.getResultGeneratorPowerAll(0)

    for t in range(1, n_hours):
        p_prev = p_now
        p_now = res.db.getResultGeneratorPowerAll(t)
        for i, gen in data.generator.iterrows():
            assert p_now[i] - p_prev[i] <= gen["ramp_up_pu"] * gen["pmax"]
            assert p_now[i] - p_prev[i] >= -gen["ramp_down_pu"] * gen["pmax"]


def test_daily_reset_releases_ramp_constraint(tmp_path, testcase_9bus_data):
    """Test that ramp rate limit reset at midnight works"""
    data = testcase_9bus_data

    # set ramp rate limits (change in pu from one timestep to the next)

    data.generator["ramp_daily_reset"] = True
    mask = ~data.generator["type"].isin(["wind", "csp", "nuclear"])
    data.generator.loc[mask, "ramp_up_pu"] = 0.05
    data.generator.loc[mask, "ramp_down_pu"] = 0.05

    print(data.generator)

    lp = powergama.LpProblem(
        data, objective_mode="daily_24h", objective_day_commit_hours=1, objective_day_horizon_hours=1
    )
    res = powergama.Results(data, tmp_path / "ramp.sqlite")
    lp.solve(res, solver="appsi_highs", solver_path={})

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
            # violation _IS_ allowed
            # assert max(ramp_jump_up.values()) > 0
            assert min(ramp_jump_down.values()) < 0  # expecting value -37.7 for coal down-ramp
        else:
            # no violation allowed
            assert max(ramp_jump_up.values()) <= 0, "ramp up exceeding max value"
            assert min(ramp_jump_down.values()) >= 0, "ramp down exceeding max value"
