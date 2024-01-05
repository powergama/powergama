import copy
import os
from pathlib import Path

import pandas as pd
import pytest

import powergama
import powergama.scenarios

# Set up and empty a folder for temporary files
tmppath = Path(__file__).parent / "tmp_files"
os.makedirs(tmppath, exist_ok=True)
for ff in os.listdir(tmppath):
    os.remove(tmppath / ff)

# Initialise a gridmodel, using the 9bus example
datapath = Path(__file__).parent / "test_data/data_9bus"
data_prefix = "9busmod_"

gridmodel = powergama.GridData()
gridmodel.readGridData(
    nodes=datapath / (data_prefix + "nodes.csv"),
    ac_branches=datapath / (data_prefix + "branches.csv"),
    dc_branches=None,
    generators=datapath / (data_prefix + "generators.csv"),
    consumers=datapath / (data_prefix + "consumers.csv"),
)
gridmodel.readProfileData(
    filename=datapath / (data_prefix + "profiles.csv"),
    storagevalue_filling=datapath / (data_prefix + "profiles_storval_filling.csv"),
    storagevalue_time=datapath / (data_prefix + "profiles_storval_time.csv"),
    timerange=range(24 * 100, 24 * 101),
    timedelta=1.0,
)

# Save the scenario to a file
scenario_file = tmppath / "scenario.csv"
powergama.scenarios.saveScenario(gridmodel, scenario_file=scenario_file)

# Keep a copy of the original gridmodel
original_model = copy.deepcopy(gridmodel)

empty_file = datapath / "scenario_empty.csv"


@pytest.mark.parametrize("scenario_loading_file", [empty_file, scenario_file])
def test_old_and_new_scenarios_same(scenario_loading_file):
    loaded_scenario = powergama.scenarios.newScenario(
        copy.deepcopy(gridmodel), scenario_file=scenario_loading_file, newfile_prefix=str(tmppath / "new_")
    )

    # Check that the gridmodels have the same number of properties
    assert len(original_model.__dict__.keys()) == len(
        loaded_scenario.__dict__.keys()
    ), "Expect the same entries in the two models"
    # Compare the stored values
    for key in original_model.__dict__.keys():
        if type(original_model.__dict__[key]) == pd.core.frame.DataFrame:
            assert (original_model.__dict__[key] == loaded_scenario.__dict__[key]).all().all(), (
                "Expect the gridmodel entries for %s to match" % key
            )
        else:
            assert original_model.__dict__[key] == loaded_scenario.__dict__[key]
