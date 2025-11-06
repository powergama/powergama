import sqlite3

import pandas as pd
import pytest

import powergama


def test_lossmethods(tmp_path, testcase_9bus_data):
    """Test simulation execution"""
    data = testcase_9bus_data

    # set resistance to a non-zero value to get non-zero losses:
    data.branch["resistance"] = 0.01

    # increased nuclear capacity to avoid loadshedding
    data.generator.loc[1, "pmax"] = 250

    # lossmethod = 0
    lp = powergama.LpProblem(data, lossmethod=0)
    res = powergama.Results(data, tmp_path / "temp_testcase_9bus_loss0.sqlite3", replace=True)
    lp.solve(res, solver="appsi_highs", solve_args=dict())
    with sqlite3.connect(res.db.filename) as con:
        res_branch = pd.read_sql("SELECT * FROM Res_Branches", con)
        res_gen = pd.read_sql("SELECT * FROM Res_Generators", con)

    gen_mean = res_gen.groupby("timestep")["output"].sum().mean()
    loss_mean = res_branch.groupby("timestep")["loss"].sum().mean()
    assert res_branch["flow"].abs().mean() == pytest.approx(97.93493, abs=1e-5)
    assert loss_mean == 0.0
    assert gen_mean == pytest.approx(500.0, abs=1e-5)

    # lossmethod = 1
    lp = powergama.LpProblem(data, lossmethod=1)
    res = powergama.Results(data, tmp_path / "temp_testcase_9bus_loss1.sqlite3", replace=True)
    lp.solve(res, solver="appsi_highs", solve_args=dict())
    with sqlite3.connect(res.db.filename) as con:
        res_branch = pd.read_sql("SELECT * FROM Res_Branches", con)
        res_gen = pd.read_sql("SELECT * FROM Res_Generators", con)

    gen_mean = res_gen.groupby("timestep")["output"].sum().mean()
    loss_mean = res_branch.groupby("timestep")["loss"].sum().mean()
    assert res_branch["flow"].abs().mean() == pytest.approx(99.04444, abs=1e-5)
    assert loss_mean == pytest.approx(14.07895, abs=1e-5)
    assert gen_mean == pytest.approx(514.07895, abs=1e-5)

    # lossmethod = 2
    lp = powergama.LpProblem(data, lossmethod=2)
    res = powergama.Results(data, tmp_path / "temp_testcase_9bus_loss2.sqlite3", replace=True)
    lp.solve(res, solver="appsi_highs", solve_args=dict())
    with sqlite3.connect(res.db.filename) as con:
        res_branch = pd.read_sql("SELECT * FROM Res_Branches", con)
        res_gen = pd.read_sql("SELECT * FROM Res_Generators", con)

    gen_mean = res_gen.groupby("timestep")["output"].sum().mean()
    loss_mean = res_branch.groupby("timestep")["loss"].sum().mean()
    assert res_branch["flow"].abs().mean() == pytest.approx(98.99539, abs=1e-5)
    assert loss_mean == pytest.approx(14.90379, abs=1e-5)
    assert gen_mean == pytest.approx(514.90379, abs=1e-5)
