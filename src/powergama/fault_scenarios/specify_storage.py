"""
This file has a bunch of useful scripts for loading data as well as running fault cases.
"""
import pathlib
import sqlite3 as db

import numpy as np
import pandas as pd

import powergama

from .LpFaultProblem import FaultResults, LpFaultProblem


def set_problem_startpoint(lpproblem, storage_init, storage_flexload_init):
    """Update storage of lpproblem
    Based on  _storeResultsAndUpdateStorage here
    https://github.com/sighellan/powergama/blob/main/src/powergama/LpProblemPyomo.py#L591
    """

    # 1. Update generator storage:
    storagecapacity = lpproblem._grid.generator["storage_cap"]
    lpproblem._storage = np.maximum(0, np.minimum(storagecapacity, storage_init))

    # 2. Update flexible load storage
    # TODO: Assert that values are allowed.
    lpproblem._storage_flexload = storage_flexload_init


class ProblemState:
    def __init__(self, lpproblem=None, storage=None, flex_load=None):
        if lpproblem is not None:
            self.storage = lpproblem._storage
            self.storage_flexload = lpproblem._storage_flexload
        elif (storage is None) and (flex_load is None):
            self.storage = storage
            self.storage_flexload = flex_load
        else:
            raise ValueError("can only specify in one way")

    def __str__(self):
        return "storage:\n%s\n\nstorage_flexload:\n%s" % (self.storage, self.storage_flexload)


def load_table_from_res(db_filename, tablename):
    if tablename == "Res_Nodes":
        query = "SELECT * FROM Res_Nodes "
    elif tablename == "Res_Storage":
        query = "SELECT * FROM Res_Storage "
    elif tablename == "Res_FlexibleLoad":
        query = "SELECT * FROM Res_FlexibleLoad "
    elif tablename == "Res_Generators":
        query = "SELECT * FROM Res_Generators "
    else:
        raise Exception(f"Unknown SQL database table {tablename}")
    con = db.connect(db_filename)
    with con:
        cur = con.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        col_names = [desc[0] for desc in cur.description]
        df = pd.DataFrame(rows, columns=col_names)
    # explicitly close to avoid error when running test in temporary directory
    con.close()
    return df


def load_flexload_states_from_res(db_filename):
    return load_table_from_res(db_filename, "Res_FlexibleLoad")


def load_storage_states_from_res(db_filename):
    return load_table_from_res(db_filename, "Res_Storage")


def load_shedding_table(res_file, max_ts):
    res_nodes = load_table_from_res(res_file, "Res_Nodes")
    load_shedding = np.ones((max_ts, np.max(res_nodes["indx"]) + 1)) * np.nan
    for _, rr in res_nodes.iterrows():
        load_shedding[int(rr.timestep), int(rr.indx)] = rr.loadshed
    return load_shedding


def solve_retry(cur_gridmodel, cur_db_loc, solver, storage_init, flexload_init):
    try:
        lp_cur = powergama.LpProblem(cur_gridmodel, lossmethod=1)
        if storage_init is not None:
            set_problem_startpoint(lp_cur, storage_init, flexload_init)
        res_cur = powergama.Results(cur_gridmodel, cur_db_loc, replace=True)
        lp_cur.solve(res_cur, solver=solver)
    except Exception:
        # TODO: print log saying we've relaxed problem
        print("Trying again without minimum power generation constraints.")
        # TODO: How do we track this really?
        # Presumably if less than Pmin, it really should be no power?
        # Relax Pmin constraints
        cur_gridmodel.generator["pmin"] = np.zeros(len(cur_gridmodel.generator))
        lp_cur = powergama.LpProblem(cur_gridmodel, lossmethod=1)
        if storage_init is not None:
            set_problem_startpoint(lp_cur, storage_init, flexload_init)
        res_cur = powergama.Results(cur_gridmodel, cur_db_loc, replace=True)
        lp_cur.solve(res_cur, solver=solver)
    return lp_cur, res_cur


def run_failure_case_LpFaultProblem(
    failure_dir,
    data_dir,
    base_case_storage,
    base_case_flexload,
    get_gridmodel,
    timesteps,
    solver,
    num_steps=1,
    lossmethod=1,
):
    """Run simulation with fault situation

    failure_dir : pathlib.Path

    """
    res_file = pathlib.Path(failure_dir) / "failure_case_combined.sqlite3"
    max_timestep = np.max(timesteps)
    full_timerange = range(max_timestep + num_steps)
    cur_gridmodel = get_gridmodel(data_dir, full_timerange)

    # TODO: Make a version similar to solve_retry?
    lp_fault = LpFaultProblem(
        cur_gridmodel,
        base_case_storage,
        base_case_flexload,
        lossmethod=lossmethod,
        timesteps=timesteps,
        fault_duration=num_steps,
    )
    res_fault = FaultResults(cur_gridmodel, res_file, replace=True)
    lp_fault.solve(res_fault, solver=solver)
    return lp_fault, res_file
