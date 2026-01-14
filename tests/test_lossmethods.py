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


def test_lossmethod1_flows():
    """Minimal test case showing issues with lossmethod 1 and unphysical two-way flow


    Grid: triangle A-B-C case with generation in A+C and demand in B.
    Impedance directs flow A->C->B, but C-B capacity is limiting (18 MW).
    This also limits flow A->B because of impedances.
    To supply all the load in B, optimisation adds high loss on A-B branch, thereby allowing
    more flow also on A->C branch.
    """

    pgdata3 = powergama.GridData()
    pg3_node = pd.DataFrame(columns=powergama.GridData.keys_powergama["node"])
    pg3_br = pd.DataFrame(columns=powergama.GridData.keys_powergama["branch"])
    pg3_dc = pd.DataFrame(columns=powergama.GridData.keys_powergama["dcbranch"])
    pg3_cons = pd.DataFrame(columns=powergama.GridData.keys_powergama["consumer"])
    pg3_gen = pd.DataFrame(columns=powergama.GridData.keys_powergama["generator"])
    pg3_node["id"] = ["A", "B", "C"]
    pg3_node["area"] = ["AR", "AR", "AR"]
    pg3_node["lat"] = [60, 60, 61]
    pg3_node["lon"] = [2, 4, 3]
    pg3_br["node_from"] = ["A", "A", "C"]
    pg3_br["node_to"] = ["B", "C", "B"]
    pg3_br["capacity"] = [100, 70, 18]
    pg3_br["reactance"] = [1, 2, 2]
    pg3_br["resistance"] = [0.05, 0.05, 0.05]
    pg3_gen["node"] = ["A", "B"]  # trying gen at B instead of C
    pg3_gen["type"] = "gen"
    pg3_gen["pmax"] = [200, 20]
    pg3_gen["pmin"] = 0
    pg3_gen["fuelcost"] = [10, 30]
    pg3_gen["inflow_fac"] = 1
    pg3_gen["inflow_ref"] = "const"
    pg3_cons["node"] = ["B"]
    pg3_cons["demand_avg"] = [100]
    pg3_cons["demand_ref"] = "const"
    pg3_profiles = pd.DataFrame({"const": [1] * 11})
    pg3_storval_time = None
    pg3_storval_fill = None
    data_dict = {
        "node": pg3_node,
        "branch": pg3_br,
        "dcbranch": pg3_dc,
        "consumer": pg3_cons,
        "generator": pg3_gen,
        "profiles": pg3_profiles,
        "storval_time": pg3_storval_time,
        "storval_filling": pg3_storval_fill,
    }
    pgdata3.from_dict(data_dict=data_dict, timedelta=1.0)
    pgdata3._fillEmptyCells(powergama.GridData.keys_powergama)

    # Create and solve the case:
    lp3 = powergama.LpProblem(pgdata3, lossmethod=1)
    pgres3 = powergama.Results(pgdata3, "tmp_testcase.sqlite", replace=True)
    lp3.solve(pgres3, solver="appsi_highs", solve_args={})

    # Extract results:
    with sqlite3.connect(pgres3.db.filename) as con:
        res3_branch = pd.read_sql("SELECT * FROM Res_Branches", con)
        res3_gen = pd.read_sql("SELECT * FROM Res_Generators", con)
        res3_node = pd.read_sql("SELECT * FROM Res_Nodes", con)
    con.close()
    res3_branch = res3_branch.set_index(["indx", "timestep"])
    res3_gen = res3_gen.set_index(["indx", "timestep"])
    res3_node = res3_node.set_index(["indx", "timestep"])
    df_flows = pd.DataFrame(
        {
            "flow12": lp3.varAcBranchFlow12.extract_values(),
            "flow21": lp3.varAcBranchFlow21.extract_values(),
            "loss12": lp3.varLossAc12.extract_values(),
            "loss21": lp3.varLossAc21.extract_values(),
        }
    )

    # Check
    # flow12 and flow21 are both >0. Physically, only one of them should be
    # But in the circumstances of this test case, with the present modelling, this behaviour is as expected
    assert df_flows.loc[1, "flow12"] == pytest.approx(70)
    assert df_flows.loc[1, "flow21"] == pytest.approx(51.34715)
    # flow minus loss equals capacity of last segment
    assert df_flows.loc[1, "flow12"] - df_flows.loc[1, "loss12"] - df_flows.loc[1, "flow21"] == pytest.approx(18)
