from pathlib import Path

import numpy as np
import pytest

import powergama

datapath = Path(__file__).parent / "test_data/mount_synth"
data_prefix = "mount_"


gridmodel_shed = powergama.GridData()
gridmodel_shed.readGridData(
    nodes=datapath / (data_prefix + "nodes.csv"),
    ac_branches=datapath / (data_prefix + "branches.csv"),
    dc_branches=datapath / (data_prefix + "hvdc.csv"),
    generators=datapath / (data_prefix + "generators.csv"),
    consumers=datapath / (data_prefix + "consumers.csv"),
)
gridmodel_shed.readProfileData(filename=datapath / (data_prefix + "profiles.csv"), timerange=range(0, 3), timedelta=1.0)

gridmodel_battery = powergama.GridData()
gridmodel_battery.readGridData(
    nodes=datapath / (data_prefix + "nodes.csv"),
    ac_branches=datapath / (data_prefix + "branches.csv"),
    dc_branches=datapath / (data_prefix + "hvdc.csv"),
    generators=datapath / (data_prefix + "generators_storage.csv"),
    consumers=datapath / (data_prefix + "consumers.csv"),
)
gridmodel_battery.readProfileData(
    filename=datapath / (data_prefix + "profiles.csv"),
    storagevalue_filling=datapath / (data_prefix + "storval_filling.csv"),
    storagevalue_time=datapath / (data_prefix + "storval_time.csv"),
    timerange=range(0, 3),
    timedelta=1.0,
)

gridmodels = {"shed": gridmodel_shed, "battery": gridmodel_battery}


def solve_problem(gridmodel, name):
    lp = powergama.LpProblem(gridmodel, lossmethod=1)

    res = powergama.Results(gridmodel, datapath / ("test-results-%s.sqlite3" % name), replace=True)
    solver = "glpk"

    lp.solve(res, solver=solver)
    return res


exp_load_shedding = {"shed": np.array([3, 0, 0]), "battery": np.zeros(3)}

exp_generation = {"shed": [[12, 10, 5], []], "battery": np.array([[15, 10, 5], [0, 0, 3]])}


@pytest.mark.parametrize("scenario_name", ["shed", "battery"])
def test_shedding(scenario_name):
    """
    Runs two versions of the scenario, one with and one without storage. Expect the former to require load shedding.
    """
    res = solve_problem(gridmodels[scenario_name], scenario_name)

    assert exp_load_shedding[scenario_name] == pytest.approx(res.getLoadsheddingPerNode())
    for gen in range(len(exp_generation[scenario_name])):
        print(scenario_name, gen)
        assert exp_generation[scenario_name][gen] == pytest.approx(res.db.getResultGeneratorPower(gen, [0, 3]))
