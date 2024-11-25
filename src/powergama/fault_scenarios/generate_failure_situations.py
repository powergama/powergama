import pathlib
import time

import numpy as np

from . import failure_situation, specify_storage


def create_fault_scenarios(gridmodel_base, fault_path, seed_list, fault_spec):
    """Create files specifying fault scenario, saved in specified folder

    gridmodel_base : powergama.GridData - the grid model used as base
    fault_path : pathlib.Path where fault scenario files will be stored
    seed_list : list of random seeds used for fault scenario creation. The lengh of this list
               determines how many scenarios are created.
    fault_spec : FaultSpec - specificatino for faults
    """

    fault_duration = fault_spec.generator_fault_duration

    # Make folder if it does not exist:
    pathlib.Path(fault_path).mkdir(exist_ok=True)

    N_steps = gridmodel_base.timerange[-1]

    for seed in seed_list:
        switches_off = failure_situation.build_switches_off_array(
            N_steps=N_steps, data=gridmodel_base, fault_spec=fault_spec, seed=seed
        )
        fault_situation_list = failure_situation.build_off_states_list(
            switches_off, N_steps=N_steps, generator_fault_duration=fault_duration
        )
        print(seed, len(fault_situation_list) * fault_duration / N_steps)

        failure_situation.write_fault_sits_to_file(fault_path / f"fault_scenario_{seed}.txt", fault_situation_list)
    print(f"Finished writing {len(seed_list)} scenarios to folder")
    print(fault_path)


def run_fault_simulation(gridmodel_base, full_profiles, fault_scenario_file, failure_dir, db_base, solver):
    """Run simulation with faults, as specified in fault scenario file

    gridmodel_base : powergama.GridData - base grid model
    full_profiles : FullProfiles
    fault_scenario_file : name of fault scenario file
    failure_dir : pathlib.Path - where to save sql file with fault scenario results
    db_base : sqlite file with base model results
    solver : name of solver to use (glpk, cbc, gurobi_persistent, gurobi,...)
    """

    # Load base results and model
    base_case_storage = specify_storage.load_storage_states_from_res(db_base)
    base_case_flexload = specify_storage.load_flexload_states_from_res(db_base)

    N_steps = gridmodel_base.timerange[-1]
    print(f"N_steps: {N_steps}")

    # Read in all fault situations from fault scenario file
    fault_situation_list = failure_situation.read_fault_sits_from_file(fault_scenario_file)

    print("Number of fault situations: %s" % len(fault_situation_list))
    # print('Fraction of hours to be resimulated: %s' %(len(fault_situation_list)*FAULT_DUR/N_steps))
    start_all = time.time()

    # For each fault situation in our timeline, run the simulation
    for ii in range(len(fault_situation_list)):
        if ii % 1000 == 0:
            print(f"{ii}/{len(fault_situation_list)}")
        failure_situation.collect_res_failure_situation(
            fault_situation_list[ii],
            full_profiles=full_profiles,
            gridmodel_base=gridmodel_base,
            failure_dir=failure_dir / f"failure_{ii}",
            base_case_storage=base_case_storage,
            base_case_flexload=base_case_flexload,
            solver=solver,
        )
    end_all = time.time()
    print("Total simulation time: %s" % (np.round(end_all - start_all, 2)))
