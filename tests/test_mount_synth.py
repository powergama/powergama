from pathlib import Path

import numpy as np
import pytest

import powergama

"""
A set of tests based on a small use case, where storage is required to avoid load shedding.
"""

datapath = Path(__file__).parent / "test_data/mount_synth"
data_prefix = "mount_"


def load_gridmodel(name):
    if name == "shed":
        generators_file = "generators.csv"
        storagevalue_filling_file = None
        storagevalue_time_file = None
    elif name == "battery":
        generators_file = "generators_storage.csv"
        storagevalue_filling_file = datapath / (data_prefix + "storval_filling.csv")
        storagevalue_time_file = datapath / (data_prefix + "storval_time.csv")
    elif name == "zero_eff":
        generators_file = "generators_zero_eff_storage.csv"
        storagevalue_filling_file = datapath / (data_prefix + "storval_filling.csv")
        storagevalue_time_file = datapath / (data_prefix + "storval_time.csv")

    gridmodel = powergama.GridData()
    gridmodel.readGridData(
        nodes=datapath / (data_prefix + "nodes.csv"),
        ac_branches=datapath / (data_prefix + "branches.csv"),
        dc_branches=datapath / (data_prefix + "hvdc.csv"),
        generators=datapath / (data_prefix + generators_file),
        consumers=datapath / (data_prefix + "consumers.csv"),
    )
    gridmodel.readProfileData(
        filename=datapath / (data_prefix + "profiles.csv"),
        storagevalue_filling=storagevalue_filling_file,
        storagevalue_time=storagevalue_time_file,
        timerange=range(0, 3),
        timedelta=1.0,
    )
    return gridmodel


def test_warn_zero_eff_storage():
    """Check that a warning is triggered if a generator is specified with
    non-zero capacity but zero efficiency.
    """
    with pytest.warns(UserWarning, match="very low pump efficiency"):
        load_gridmodel("zero_eff")


def solve_problem(gridmodel, name):
    lp = powergama.LpProblem(gridmodel, lossmethod=1)

    res = powergama.Results(gridmodel, datapath / ("test-results-%s.sqlite3" % name), replace=True)
    solver = "glpk"

    lp.solve(res, solver=solver)
    return res


exp_load_shedding = {"shed": np.array([3, 0, 0]), "battery": np.zeros(3)}
exp_system_cost = {"shed": 2.7, "battery": 3.6}
exp_generation = {"shed": [[12, 10, 5], []], "battery": np.array([[15, 10, 5], [0, 0, 3]])}
exp_DA = {(0, 0): -20, (0, 1): 20, (1, 0): -20, (1, 2): 20}
exp_Bbus = {
    (0, 2): -20.0,
    (0, 1): -20.0,
    (0, 0): 40.0,
    (1, 1): 20.0,
    (1, 0): -20.0,
    (2, 2): 20.0,
    (2, 0): -20.0,
}


def test_power_flow_matrices():
    """Check that the power flow matrices are as expected."""
    gridmodel = load_gridmodel("shed")
    Bbus, DA = gridmodel.compute_power_flow_matrices(base_Z=1)
    assert exp_DA == pytest.approx(dict(DA.todok())), "Unexpected DA matrix"
    assert exp_Bbus == pytest.approx(dict(Bbus.todok())), "Unexpected Bbus matrix"


@pytest.mark.parametrize("scenario_name", ["shed", "battery"])
def test_shedding(scenario_name):
    """
    Runs two versions of the scenario, one with and one without storage. We expect the latter to require load shedding.
    """
    gridmodel = load_gridmodel(scenario_name)
    res = solve_problem(gridmodel, scenario_name)

    assert exp_load_shedding[scenario_name] == pytest.approx(res.getLoadsheddingPerNode()), "Unexpected load shedding"
    assert exp_system_cost[scenario_name] == pytest.approx(res.getSystemCost()["mount"]), "Unexpected system cost"
    for gen in range(len(exp_generation[scenario_name])):
        assert exp_generation[scenario_name][gen] == pytest.approx(
            res.db.getResultGeneratorPower(gen, [0, 3])
        ), "Unexpected generation"
