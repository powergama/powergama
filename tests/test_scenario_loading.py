import copy
from pathlib import Path

import pandas as pd
import pytest

import powergama
import powergama.scenarios

# Initialise a gridmodel, using the 9bus example
datapath = Path(__file__).parent / "test_data/data_9bus"


@pytest.mark.parametrize("scenario_loading_file", ["empty", "saved"])
def test_old_and_new_scenarios_same(
    testcase_9bus_data,
    scenario_loading_file,
    tmp_path,
):
    original_model = copy.deepcopy(testcase_9bus_data)

    if scenario_loading_file == "saved":
        scenario_file = tmp_path / "scenario.csv"
        powergama.scenarios.saveScenario(
            original_model,
            scenario_file=scenario_file,
        )
    else:
        scenario_file = datapath / "scenario_empty.csv"

    loaded_scenario = powergama.scenarios.newScenario(
        copy.deepcopy(testcase_9bus_data),
        scenario_file=scenario_file,
        newfile_prefix=str(tmp_path / "new_"),
    )

    assert set(original_model.__dict__) == set(loaded_scenario.__dict__)

    for key, value in original_model.__dict__.items():
        other = loaded_scenario.__dict__[key]

        if isinstance(value, pd.DataFrame):
            pd.testing.assert_frame_equal(value, other)
        else:
            assert value == other
