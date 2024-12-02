"""
This file has functions to set up fault simulations and saving and reading from file, as well as
collecting results for a fault situation.
"""

import pathlib

import numpy as np

from .full_profiles import get_gridmodel_failure
from .specify_storage import run_failure_case_LpFaultProblem


class FaultSpec:
    """Specification of faults"""

    def __init__(self, spec_generators=None, spec_branches=None):
        """Specify fault

        spec_generators : dict {fault_rate: float, fault_duration: float, fault_sizes : dict {type:value, None:default}}

        """
        if spec_generators:
            self.generator_fault_rate = spec_generators["fault_rate"]
            self.generator_fault_duration = spec_generators["fault_duration"]
            self.generator_fault_size = spec_generators["fault_sizes"]
        else:
            # default values if none specified:
            self.generator_fault_rate = 0.5665 / 8760
            self.generator_fault_duration = 21
            self.generator_fault_sizes = {None: 1000}

        # TODO: specify for branches

    def get_unit_size(self, gen_type):
        if gen_type in self.generator_fault_size:
            return self.generator_fault_size[gen_type]
        else:
            return self.generator_fault_size[None]


def collect_res_failure_situation(
    fail_situation, full_profiles, gridmodel_base, failure_dir, base_case_storage, base_case_flexload, solver
):
    """For a specific failure situation, run the simulation for the specified duration

    fail_situtaion: tuple - (start, end, failed_generators, failed_branches)
    failure_dir : pathlib.Path - where files are
    """
    start, end, failed_generators, failed_branches = fail_situation

    def get_gridmodel_specific_failure(unused, timerange):
        return get_gridmodel_failure(
            full_profiles,
            gridmodel_base,
            timerange,
            failed_generators=failed_generators,
            failed_branches=failed_branches,
        )

    test_timesteps = [start]
    pathlib.Path(failure_dir).mkdir(exist_ok=True)
    run_failure_case_LpFaultProblem(
        failure_dir,
        None,
        base_case_storage,
        base_case_flexload,
        get_gridmodel_specific_failure,
        test_timesteps,
        num_steps=end - start,
        solver=solver,
        lossmethod=0,
    )


def build_switches_off_array(N_steps, data, fault_spec, seed=None, verbose=False, exclude_nodes=None):
    """

    N_steps : int
    data : powergama.GridModel object - grid model
    fault_spec : FaultSpec object - fault situation specification
    """
    fault_rate = fault_spec.generator_fault_rate

    if seed is not None:
        np.random.seed(seed)
    N_gen = data.numGenerators()
    # Array specifying whether a generator turns off at that time
    # Given in the percentage of switch-off, so a value between 0 and 1
    switches_off = np.zeros((N_gen, N_steps))

    # Iterate through generators and sample when and how much generators fail
    for gg in range(N_gen):
        gen = data.generator.iloc[gg]
        if exclude_nodes and (gen.node in exclude_nodes):
            # amount_off = 0  # We don't simulate generator faults at these nodes
            continue
        unit_size = fault_spec.get_unit_size(gen.type)
        if gen.pmax <= unit_size:
            # If there is only one we check whether it is on or off
            switches_off[gg, :] = np.random.binomial(n=1, p=fault_rate, size=N_steps)
        else:
            # We have num_gens generators, of which num_full_gens are of unit_size and the
            # potential remaining one is of remaining size.
            num_gens = np.ceil(gen.pmax / unit_size)
            if num_gens * unit_size > gen.pmax:
                num_full_gens = num_gens - 1
                remaining = gen.pmax - num_full_gens * unit_size
            else:
                num_full_gens = num_gens
                remaining = 0
            # Sample whether the generators are on or off
            # Done independently for each generator and each timestep
            switches_off_full = np.random.binomial(n=num_full_gens, p=fault_rate, size=N_steps)
            switches_off_rem = np.random.binomial(n=1, p=fault_rate, size=N_steps)
            if verbose and (switches_off_full.any() or switches_off_rem.any()):
                print(gen)
                print(switches_off_full, switches_off_rem)
            switches_off[gg, :] = switches_off_full * unit_size / gen.pmax + switches_off_rem * remaining / gen.pmax
    return switches_off


def build_off_states_list(switches_off, N_steps, generator_fault_duration):
    # List of tuples of (start, end, generators_fail, branches_fail)
    # TODO: branches_fail
    # generators_fail: list of tuples of (gg_idx, amount_off)
    # branches_fail: TODO
    fault_situation_list = []
    tt = 0

    while tt < N_steps:
        if switches_off[:, tt].any():
            start = tt
            end = min(tt + generator_fault_duration, N_steps)  # not inclusive
            # TODO: N_steps, or plus or minus one?
            generators_amount = []
            # Find any other ones that start within this area and also set them to start at tt
            to_turn_off = np.nonzero(switches_off[:, start:end].any(1))[0]
            # Set any that start at tt to be off until tt + duration
            for gg in to_turn_off:
                amount_off = np.max(switches_off[gg, start:end])
                # HACK Need to add int(...) and float(...) to avoid getting np.int64(...) in created files
                # Code should be improved to avoid such things
                generators_amount.append((int(gg), float(amount_off)))
            fault_situation_list.append((start, end, generators_amount, []))
            # jump to tt + duration
            tt += generator_fault_duration
        else:
            tt += 1
    return fault_situation_list


def build_off_states_array(fault_situation_list, data, N_steps):
    N_gen = data.numGenerators()
    off_states = np.zeros((N_gen, N_steps))
    for fault in fault_situation_list:
        start, end, failed_generators, failed_branches = fault

        for gg, amount in failed_generators:
            off_states[gg, start:end] = amount
    # TODO: failed_branches
    return off_states


def write_fault_sits_to_file(filename, fault_situation_list):
    ff = open(filename, "w")
    for fault in fault_situation_list:
        start, end, generators, lines = fault
        ff.write(f"{start}, {end}\n")
        for gg_tup in generators:
            ff.write(f"{gg_tup}, ")
        ff.write("\n")
        for ll_tup in lines:
            ff.write(f"{ll_tup}, ")
        ff.write("\n")
        ff.write("\n")
    ff.close()


def parse_comps(gg_str):
    # Read in a sequence of tuples in a string and return a list of tuples (int, float)
    gg_list = []
    if not (gg_str.split("), ")[-1] == "\n"):
        raise Exception("Error in parse_comps")
    # Get a list of the tuple strings
    gg_tup_list = gg_str.split("), ")[:-1]
    # For each tuple string, parse it into a tuple of (int, float)
    for ii in range(len(gg_tup_list)):
        gg_tup_ex = gg_tup_list[ii]
        gg_idx_ex, gg_frac = gg_tup_ex.split(", ")
        gg_idx = gg_idx_ex.split("(")[1]
        gg_list.append((int(gg_idx), float(gg_frac)))
    return gg_list


def read_fault_sits_from_file(filename):
    ff = open(filename, "r")
    rows = ff.readlines()
    ff.close()

    new_fault_list = []
    max_ii = len(rows) - 1
    ii = 0
    while True:
        start, end_ex = rows[ii].split(", ")
        end = end_ex.split("\n")[0]
        gg_list = parse_comps(rows[ii + 1])
        ll_list = parse_comps(rows[ii + 2])
        if not (rows[ii + 3] == "\n"):
            raise Exception("There should be an empty line between each fault situation")
        new_fault_list.append((int(start), int(end), gg_list, ll_list))
        ii += 4
        if ii > max_ii:
            break

    return new_fault_list
