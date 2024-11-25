import copy

import numpy as np
import pyomo.opt

import powergama

"""
This file includes versions of LpProblem, Results and DataBase that are adjusted to running fault simulations.
"""


def extract_init(base_case, N_init, ii):
    storage_init = np.zeros(N_init)
    # Need to process the data from database into same format
    # Off by one (ii-1) since these are storage levels as stored as results for next step
    for _, rr in base_case[base_case.timestep == ii - 1].iterrows():
        storage_init[int(rr.indx)] = rr.storage
    return storage_init


class LpFaultProblem(powergama.LpProblem):
    """
    Modification of powergama LpProblem, to be run for non-sequential time-steps and
    import storage levels from previous runs
    """

    def __init__(self, grid, storage_base, flexload_base, timesteps=None, lossmethod=0, fault_duration=1):
        self._storage_base = storage_base
        self._flexload_base = flexload_base
        self._timesteps = timesteps
        self._fault_duration = fault_duration
        self._prev_ts = None
        self._dur_counter = 0
        self._counter = 0

        if lossmethod not in [0, 1]:
            raise ValueError("Not implemented for lossmethod 2 yet")
        super().__init__(grid, lossmethod)

    def _starting_timesteps_to_solve(self):
        if self._timesteps is None:
            numTimesteps = len(self._grid.timerange)
            return range(numTimesteps)
        else:
            return self._timesteps

    def _get_timesteps_to_solve(self):
        starting_timesteps = self._starting_timesteps_to_solve()
        max_ts = self._grid.timerange[-1] + 1
        return [st + delay for st in starting_timesteps for delay in range(self._fault_duration) if st + delay < max_ts]

    def _get_fault_start(self, timestep):
        # dur_counter has already been incremented in _update_progress hence + 1
        return timestep - self._dur_counter + 1

    def reset_from_ts(self, timestep, storage_base, flexload_base):
        if timestep > 0:
            storage_init = extract_init(storage_base, len(self._grid.generator), timestep)
            flexload_init = extract_init(flexload_base, len(self._grid.consumer), timestep)

            storagecapacity = self._grid.generator["storage_cap"]
            self._storage = np.maximum(0, np.minimum(storagecapacity, storage_init))
            self._storage_flexload = flexload_init

    def _updateLpProblem(self, timestep):
        end_ts = self._grid.timerange[-1]
        if (self._prev_ts is None) or (self._dur_counter == self._fault_duration):
            reverting = True
        elif self._prev_ts == end_ts:  # last timestep reached end of possible simulation
            if not (timestep <= end_ts):
                raise Exception("timestep<=end_ts")
            reverting = True
        else:
            if not (self._prev_ts + 1 == timestep):
                raise Exception("self._prev_ts + 1 == timestep")
            reverting = False
        if reverting:
            self.reset_from_ts(timestep, self._storage_base, self._flexload_base)
            self._dur_counter = 0
        super()._updateLpProblem(timestep)

    def _update_progress(self, cur_step, num_steps):

        self._dur_counter += 1
        self._prev_ts = cur_step
        self._counter += 1
        super()._update_progress(self._counter - 1, num_steps)

    def _relax_and_retry(self, opt, warmstart, count, solve_args):
        print("Trying again without minimum power generation constraints.")
        old_pmin = copy.deepcopy(self._grid.generator["pmin"])
        N_gen = len(self._grid.generator)
        self._grid.generator["pmin"] = np.zeros(N_gen)
        old_p_gen_pmin = copy.deepcopy(self.p_gen_pmin)
        for i in range(N_gen):
            self.p_gen_pmin[i] = 0

        warmstart_now = warmstart and (count > 0)
        if warmstart_now:
            res = opt.solve(self, warmstart=warmstart_now, **solve_args)
        else:
            res = opt.solve(self, **solve_args)
        self.solver_res = res

        if (res.solver.status != pyomo.opt.SolverStatus.ok) or (
            res.solver.termination_condition == pyomo.opt.TerminationCondition.infeasible
        ):
            raise Exception("No solution after relaxing problem.")

        # Reset settings
        self._grid.generator["pmin"] = old_pmin
        for i in range(N_gen):
            self.p_gen_pmin[i] = old_p_gen_pmin[i]


class FaultDatabase(powergama.database.DatabaseBaseClass):
    def __init__(self, filename):
        super().__init__(filename)
        self.timestep_str = "timestep INT, fault_start INT"
        self.timestep_qs = "?,?"

    def timestep_tuple(self, timestep, fault_start):
        print(timestep, fault_start)
        return (timestep, fault_start)

    def extra_something(self):
        print("Bonus!")

    # TODO: might need updating?
    # def getTimerange(self):


class FaultResults(powergama.ResultsBaseClass):
    def _init_database(self, databasefile):
        self.db = FaultDatabase(databasefile)
