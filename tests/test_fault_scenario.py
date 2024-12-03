"""
A set of tests for fault scenario analysis.
"""
import copy
import pathlib

import pandas as pd
import pyomo.environ as pyo
import pytest

import powergama
import powergama.fault_scenarios as fs


def test_fault_scenario_creation():
    """Test to check that creation of a fault scenario works"""
    pass


@pytest.mark.skipif(not pyo.SolverFactory("glpk").available(), reason="Skipping test because GLPK is not available.")
def test_faultsimulation(testcase_9bus_data, testcase_9bus_res, tmp_path):
    """9 bus test to check that fault scenario simulation works, and gives expected result"""

    gridmodel_base = testcase_9bus_data

    # Fault specs:
    fault_spec_generators = {"fault_rate": 0.05, "fault_duration": 4, "fault_sizes": {None: 100}}
    scenario_seeds = [1, 2]  # two fault scenarios

    full_profiles = fs.FullProfiles()
    full_profiles.stored_profiles = gridmodel_base.profiles
    full_profiles.timedelta = gridmodel_base.timeDelta
    full_profiles.stored_storagevalue_time = gridmodel_base.storagevalue_time
    full_profiles.storagevalue_filling = gridmodel_base.storagevalue_filling

    lp_base = powergama.LpProblem(gridmodel_base, lossmethod=1)

    # temporary folder for files created during test:
    dirname = tmp_path

    base_db_loc = pathlib.Path(dirname).absolute() / "base_powergama.sqlite3"
    print("Base database file:", base_db_loc.absolute())

    # 1. Run baseline simulation and save result to database
    res_base = powergama.Results(gridmodel_base, base_db_loc, replace=True)
    lp_base.solve(res_base, solver="glpk")

    # 2. Create fault scenario files:
    fault_spec = fs.FaultSpec(spec_generators=fault_spec_generators)
    failure_dir = pathlib.Path(dirname).absolute() / "faults"
    print("failure_dir", failure_dir)
    fs.create_fault_scenarios(gridmodel_base, failure_dir, fault_spec=fault_spec, seed_list=scenario_seeds)

    # 3 Run simulations with faults (and save results to sql-files)
    print("Running simulations with faults...")
    for scen in scenario_seeds:
        fault_scenario_dir = failure_dir / f"fault_scenario_{scen}"
        fault_scenario_dir.mkdir(exist_ok=True)
        fault_scenario_file = failure_dir / f"fault_scenario_{scen}.txt"
        fs.run_fault_simulation(
            gridmodel_base,
            full_profiles,
            fault_scenario_file=fault_scenario_file,
            failure_dir=fault_scenario_dir,
            db_base=base_db_loc,
            solver="glpk",
        )

    # 4 Inspect results
    print("Inspecting results...")
    res_nodes_base = fs.specify_storage.load_table_from_res(base_db_loc, "Res_Nodes")
    base_node = copy.deepcopy(res_nodes_base)
    # combine results from all fault situations:
    loadshedding_all = pd.DataFrame()
    for scen in scenario_seeds:
        print(f"fault scenario {scen}")
        fault_scenario_dir = failure_dir / f"fault_scenario_{scen}"
        for subfolder in pathlib.Path(fault_scenario_dir).glob("failure_*/"):
            res_table = fs.specify_storage.load_table_from_res(subfolder / "failure_case_combined.sqlite3", "Res_Nodes")
            if (res_table["loadshed"] > 0).any():
                # update result with data from this fault situation:
                mask = base_node["timestep"].isin(res_table["timestep"])
                base_node.loc[mask, :] = res_table.drop(columns="fault_start").set_index(base_node.loc[mask].index)
        loadshedding_all[scen] = base_node["loadshed"]

    loadshedding_sums = loadshedding_all.sum()

    assert loadshedding_sums[1] == pytest.approx(815.8665322112945)

    print("Done.")
