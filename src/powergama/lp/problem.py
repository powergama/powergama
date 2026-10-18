"""
Module containing PowerGAMA LpProblem class

 Power flow equations:

 Linearised ("DC") power flow equation
 Pinj - Bprime * theta = 0
           Bprime = (N-1)x(N-1) matrix (removed ref.bus row/column)
           theta = phase angles (at N-1 buses)
           Pinj = generation - load at node (cf makeSbus)

 Relationship between angles and power flow
 Pb - (D x A) x theta = 0
           theta_j = phase angle node j (excluding ref. node)
           Pb_k = power flow branch k
           D = diag(-b_k) (negative of susceptance on branch k)
           A = Mx(N-1) node-branch incidence (adjacency) matrix
"""

import warnings

import networkx as nx
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pyomo.opt
from pyomo.contrib import appsi
from tqdm import tqdm

from . import solver as solver_mod
from . import storage_value
from .config import CONFIG_ATTRS, LpConfig
from .context import CONTEXT_ATTRS, GridContext
from .multi_period import build as mp_build
from .multi_period import data as mp_data
from .multi_period import extract as mp_extract
from .objective_trace import write_objective_trace
from .single_period import build as sp_build
from .single_period import data as sp_data
from .single_period import extract as sp_extract


class LpProblem:
    """LP problem formulation

    Holds the Pyomo model as `self.model` rather than being one. The simulation
    state that makes the problem sequential -- storage levels, previous dispatch,
    the loss linearisation operating point -- lives on the LpProblem, and the
    model is rebuilt or mutated against it. See the note above
    `_create_sets_and_parameters` for the rule every construction rule follows.

    Attribute access falls through to the model, so `lp.varGeneration` still
    resolves for existing callers. New code should say `lp.model.varGeneration`.

    Parameters
    ==========
    grid : GridData
        grid data object
    lossmethod : int
        loss method; 0=no losses, 1=linearised losses, 2=added as load
    penalty_twoway_flow : float
        penalty factor to discourage flow in both directions on a branch with losses
    objective_mode : str
        "single_period" (default) solves one LP per timestep in sequence, carrying storage
        state forward between solves. "multi_period" solves a joint LP over a window
        of `horizon_hours` with storage continuity inside the optimisation.
        These are not equivalent at horizon_hours=1 -- see
        tests/test_sim_single_period_rolling_equiv.py.
    horizon_hours : int
        Length of the optimisation window, in hours. "multi_period" mode only.
    commit_hours : int
        How many hours at the start of each window are committed and stored before the
        window advances. Equal to `horizon_hours` for non-overlapping windows; smaller
        for a rolling horizon with lookahead (e.g. 48h horizon committing 24h).
        Capped at `horizon_hours`. "multi_period" mode only.
    storage_initial_fill_scale : float
        Scales the initial storage filling level of hydro-with-pumping and battery
        generators, relative to `storage_ini` in the grid data.
    extensions : list of ModelExtension
        Optional additions to the single-period model, e.g.
        :class:`powergama.lp.rt.RtExtension` for balancing ("real-time") simulation.
        See :mod:`powergama.lp.extension`.

    Notes
    =====
    Unsupported mode combinations raise NotImplementedError at construction:
    an extension with "multi_period", lossmethod != 0 with "multi_period", and
    ramp_daily_reset with commit_hours != 24.
    """

    def __init__(
        self,
        grid,
        lossmethod=0,
        penalty_twoway_flow=0,
        objective_mode="single_period",
        horizon_hours=24,
        commit_hours=24,
        storage_initial_fill_scale=1.0,
        extensions=None,
    ):
        # 1. The Pyomo model. Its identity must be preserved for the whole of solve():
        #    APPSI's incremental update and gurobi_persistent.set_instance both key off
        #    the object, so rebinding self.model mid-run would silently desynchronise
        #    the solver from the model it was given.
        self.model = pyo.ConcreteModel()

        # 2. Compute matrices used in power flow equaions
        print("Computing B and DA matrices...")
        self._Bbus, self._DA = grid.compute_power_flow_matrices()

        print("Initialising LP problem...")

        # Find synchronous areas and specify reference node in each area
        G = nx.Graph()
        G.add_nodes_from(grid.node["id"])
        G.add_edges_from(zip(grid.branch["node_from"], grid.branch["node_to"]))
        G_subs = (G.subgraph(c) for c in nx.connected_components(G))
        self.refnodes = []
        for gr in G_subs:
            refnode = list(gr.nodes)[0]
            self.refnodes.append(refnode)
            print("Found synchronous area (size = {}), using ref node = {}".format(gr.order(), refnode))
        # use first node as voltage angle reference

        # Caller-supplied settings, validated once. Readers still say
        # self._lossmethod / self._objective_mode and so on; __getattr__ resolves those
        # against the config.
        self._cfg = LpConfig.from_args(
            lossmethod=lossmethod,
            penalty_twoway_flow=penalty_twoway_flow,
            objective_mode=objective_mode,
            horizon_hours=horizon_hours,
            commit_hours=commit_hours,
            storage_initial_fill_scale=storage_initial_fill_scale,
        )
        self._extensions = tuple(extensions or ())
        self._cfg.check_extensions(self._extensions)
        self._exempt_pinned_from_ramp = any(ext.exempt_pinned_from_ramp for ext in self._extensions)
        self._grid = grid
        self.timeDelta = grid.timeDelta
        self._timestep_day_hour: dict[int, tuple[int, int]] = {}
        self._daily_objective_trace: dict[int, float] = {}
        self._hourly_objective_trace: list[tuple[int, int, int, float]] = []
        self._solver_persistent = False
        # Everything the grid implies, derived once. Readers still say
        # self._idx_generatorsWithStorage and so on; __getattr__ resolves those
        # against the context.
        self._ctx = GridContext.from_grid(grid, self._Bbus, self._DA, self.refnodes)
        self._cfg.check_ramp_daily_reset(self._ctx.ramp_daily_reset)
        # Previous-timestep generation dispatch; NaN signals first timestep (no ramp constraint).
        self._gen_prev = np.full(len(grid.generator), np.nan)
        # self._fancy_progressbar = False

        # Initial values of marginal costs, storage and storage values
        # Apply optional start-fill scaling only to pumped hydro and batteries.
        _ini_fill_scale = self._cfg.storage_initial_fill_scale
        _gen_type = (
            grid.generator["type"].astype(str).str.lower()
            if "type" in grid.generator.columns
            else pd.Series("", index=grid.generator.index)
        )
        _pump_cap = pd.to_numeric(grid.generator.get("pump_cap", 0.0), errors="coerce").fillna(0.0)
        _storage_cap = pd.to_numeric(grid.generator.get("storage_cap", 0.0), errors="coerce").fillna(0.0)
        _eligible_scaled_fill = ((_gen_type == "hydro") & (_pump_cap > 0.0)) | (_gen_type == "battery")
        _fill_scale = pd.Series(1.0, index=grid.generator.index, dtype=float)
        _fill_scale.loc[_eligible_scaled_fill] = _ini_fill_scale
        _storage_ini = pd.to_numeric(grid.generator.get("storage_ini", 0.0), errors="coerce").fillna(0.0)
        self._storage = (_fill_scale * _storage_ini * _storage_cap).fillna(0)
        self._storage_flexload = (
            grid.consumer["flex_storagelevel_init"]
            * grid.consumer["flex_storage"]
            * grid.consumer["flex_fraction"]
            * grid.consumer["demand_avg"]
        ).fillna(0)
        self._energyspilled = grid.generator["storage_cap"].copy(deep=True)
        self._energyspilled[:] = 0

        # 3. Extensions derive their own data, now that the context and the initial
        #    simulation state exist.
        for ext in self._extensions:
            ext.prepare(self, grid)

        # 4. Build the Pyomo model. Order is load-bearing -- see single_period.build.
        sp_build.build_model(self.model, grid, self._ctx, self._cfg, self._extensions)

    def __getattr__(self, name):
        """Fall through to the config, then the grid context, then the Pyomo model.

        Settled configuration lives on `self._cfg` and grid derivations on
        `self._ctx`, but the rest of the code --
        including extensions, whose per-timestep code reads dozens of attributes off
        the problem -- spells it `self._lossmethod`, `self._objective_mode` and so on. Resolving those names
        here keeps every reader working while the values live in one frozen object.
        Only names in CONFIG_ATTRS and CONTEXT_ATTRS are redirected, and none of them
        is mutated after construction,
        so there is no counterpart to the write hazard described below.

        Keeps `lp.varAcBranchFlow12.extract_values()` and the like working for callers
        written when LpProblem *was* the model. Python only calls this when normal
        lookup fails, so it never shadows real attributes.

        Reads only. Assigning `lp.cFoo = pyo.Constraint(...)` sets a plain attribute on
        the LpProblem and the component is never added to the model -- construction
        must say `self.model.cFoo` explicitly. That is why there is no __setattr__
        counterpart: a write passthrough would make the mistake invisible, whereas
        without one the constraint simply goes missing from the LP.
        """
        # __getattr__ runs before __init__ assigns self.model during unpickling and on
        # any failed lookup inside __init__ itself; guard against recursing forever.
        if name in ("model", "_cfg", "_ctx"):
            raise AttributeError(name)
        state = self.__dict__
        if name.startswith("_"):
            bare = name[1:]
            if bare in CONFIG_ATTRS:
                cfg = state.get("_cfg")
                if cfg is not None:
                    return getattr(cfg, bare)
            if bare in CONTEXT_ATTRS:
                ctx = state.get("_ctx")
                if ctx is not None:
                    return getattr(ctx, bare)
        try:
            return getattr(state["model"], name)
        except KeyError:
            raise AttributeError(name) from None

    @property
    def storage(self):
        """Generator storage filling levels in MWh, indexed like `grid.generator`."""
        return self._storage

    @property
    def storage_flexload(self):
        """Flexible-load reservoir levels in MWh, indexed like `grid.consumer`."""
        return self._storage_flexload

    def set_storage_state(self, storage=None, storage_flexload=None):
        """Seed the simulation state that makes the single-period problem sequential.

        Both are held as pandas Series indexed like the grid data they describe, and
        this is the only place that guarantees it. It matters because
        `solve(continue_from_last=True)` assigns into them with `.loc` and
        `_storeResultsAndUpdateStorage` selects from them by generator and consumer
        index: a bare numpy array happens to behave the same way while the grid uses a
        default RangeIndex, and stops doing so without warning if it ever does not.

        A Series argument is aligned on its index; an array is taken positionally.

        Generator storage is clipped into `[0, storage_cap]`. Flexible-load levels are
        taken as given, because the single-period path lets them leave their range on purpose
        and prices that through `constants.flexload_outside_cost`.
        """
        if storage is not None:
            values = pd.Series(storage, index=self._grid.generator.index, dtype=float)
            self._storage = values.clip(lower=0.0, upper=self._grid.generator["storage_cap"])
        if storage_flexload is not None:
            self._storage_flexload = pd.Series(storage_flexload, index=self._grid.consumer.index, dtype=float)

    # ------------------------------------------------------------------
    # Model construction.
    #
    # Invariant for every constraint, objective and bound rule below: Pyomo
    # components (p_*, s_*, var*, c*, OBJ) are read off the rule's `model`
    # argument, never off `self`; everything else -- grid data, configuration,
    # the precomputed adjacency and matrix structures -- is closed over from
    # `self` at build time. The model is `self.model`, a separate object, so a
    # component read off `self` now resolves through __getattr__ to the same
    # place by accident rather than by design, and would stop resolving the
    # moment an LpProblem carried more than one model. Enforced by
    # tests/test_model_invariants.py.
    #
    # Note the asymmetry on writes: `self.model.cFoo = pyo.Constraint(...)`
    # registers the component, `self.cFoo = ...` quietly does not. Construction
    # must always name the model.
    # ------------------------------------------------------------------

    def _get_timesteps_to_solve(self, continue_from_last=False, results=None):
        numTimesteps = len(self._grid.timerange)
        time_steps = range(numTimesteps)
        if continue_from_last:
            timestep_last = results.get_last_timestep_in_results()
            if timestep_last is None:
                # Empty results database: there is nothing to continue from. Results()
                # with replace=False only prints a notice in this case, so without an
                # explicit check this fails later as a TypeError on None + 1.
                raise ValueError(
                    "continue_from_last=True requires a results database with at least one "
                    "stored timestep, but the database is empty. Run a simulation first, or "
                    "use continue_from_last=False to start from the beginning."
                )
            # first step is last plus one
            time_steps = range(timestep_last + 1, numTimesteps)
            print(f"Continue from timestep {timestep_last}.")
        return time_steps

    def _build_day_hour_index(self, timesteps_to_solve):
        """Build mapping timestep -> (day_index, hour_in_day) for diagnostics and horizon grouping."""
        self._timestep_day_hour = {}
        for offset, ts in enumerate(timesteps_to_solve):
            day = int(offset) // int(self._horizon_hours)
            hour = int(offset) % int(self._horizon_hours)
            self._timestep_day_hour[int(ts)] = (day, hour)

    # ------------------------------------------------------------------
    # 24-hour joint optimisation
    # ------------------------------------------------------------------

    def _solve_and_store_day_joint(self, day_timesteps, results, solver_name, commit_hours=None):
        """Build, solve, and store results for one rolling-horizon LP window.

        Creates a fresh APPSI HiGHS instance for the day model, respecting
        the same environment variable seeds/threads as the single-period path.
        """
        ts0 = day_timesteps[0]
        if commit_hours is None:
            commit_hours = len(day_timesteps)
        commit_hours = int(max(1, min(int(commit_hours), len(day_timesteps))))
        target_horizon = int(max(1, int(self._horizon_hours)))
        solve_h = int(len(day_timesteps))
        tail_note = ""
        if solve_h < target_horizon:
            tail_note = f", tail-window truncated from target {target_horizon}h"
        print(
            f"\n[24h joint] window starting at timestep {ts0} ({solve_h}h solve, {commit_hours}h commit{tail_note}) ..."
        )
        wd = mp_data.prepare_window(self, day_timesteps)
        m = mp_build.build_window_model(self, wd)
        print(
            f"  Model built: {len(list(m.s_gen)) * len(day_timesteps)} gen×h vars, "
            f"{len(list(m.s_node)) * len(day_timesteps)} node×h power-balance constraints"
        )

        day_opt = solver_mod.create_window_solver(solver_name)

        res = day_opt.solve(m)
        if res.termination_condition != appsi.base.TerminationCondition.optimal:
            raise RuntimeError(f"[24h joint] non-optimal at timestep {ts0}: {res.termination_condition}")
        day_opt.load_vars()
        duals = day_opt.get_duals()
        print("  Solved OK. Extracting results ...")
        mp_extract.extract_and_store(self, m, wd, results, duals, commit_hours=commit_hours)

    # ------------------------------------------------------------------

    def _relax_and_retry(self, opt, warmstart, count, solve_args):
        raise NotImplementedError

    # TODO: Update to allow persistent model solving
    # (remove+add constraint instead of mutable parameters)
    def _updateLpProblem(self, timestep):
        """Rewrite the model's parameters and bounds for *timestep*.

        Kept as a method because it is an extension seam, not merely a long
        function: LpFaultProblem overrides it, reseeds storage from a baseline
        database inside the override, then calls super(). It must stay callable
        exactly once per timestep, in order.
        """
        sp_data.update_timestep(self, timestep)

    def _update_persistent_model(self, opt):
        """Update objective function, constraints and bounds in persistent model"""
        # Mutable parameters is not enough
        #
        # TODO: Code for persistent solver
        # https://pyomo.readthedocs.io/en/latest/solvers/persistent_solvers.html
        # to use pwersistent solvers, probably have to set instance at the
        # start, and then modify it in each iteration rather than giving
        # it as an argument to opt.solve:
        #    opt.set_instance(self.model)
        # and then use opt.solve()
        # To modify e.g. a constraint between solves, remove and add, e.g.:
        #    opt.remove_constraint(m.c)
        #    del m.c
        #    m.c = pe.Constraint(expr=m.y <= m.x)
        #    opt.add_constraint(m.c)
        # Variables can be updated without removing/adding
        #    m.x.setlb(1.0)
        #    opt.update_var(m.x)

        # TODO Check that it is correct - speed up possible?
        # TODO check if deleting and recreating constraint expression is necessary
        # (seems not)

        # 1. p_gen_pmin, p_gen_pmax => gen maxmin
        # 2. p_demand, p_branch_ac_power_loss, p_branch_dc_power_loss => powerbalance
        # 3. p_gen_cost, p_genpump_cost, p_loadflex_cost => OBJ

        # 1.
        for c in self.model.cGenMaxLimit.values():
            opt.remove_constraint(c)
        for c in self.model.cGenMinLimit.values():
            opt.remove_constraint(c)
        # del self.model.cGenMaxLimit
        # del self.model.cGenMinLimit
        # self._create_constraint_generator_output()
        for c in self.model.cGenMaxLimit.values():
            opt.add_constraint(c)
        for c in self.model.cGenMinLimit.values():
            opt.add_constraint(c)

        # 2.
        for c in self.model.cPowerbalance.values():
            opt.remove_constraint(c)
        # del self.model.cPowerbalance
        # self._create_constraint_powerbalance(self._grid)
        for c in self.model.cPowerbalance.values():
            opt.add_constraint(c)

        # 3.
        # no remove_object?
        # del self.model.OBJ
        # self._create_objective(self._grid)
        opt.set_objective(self.model.OBJ)

    # def _compute_transmission_loss(self, branch_flows, aclossmultiplier=1):
    #    """Compute power losses based on power flow"""
    #    for b in self.model.s_branch_ac:
    #        r = self._grid.branch.loc[b, "resistance"]
    #        branch_flows = self.model.p_branch_ac_powerflow[b]
    #        lossMVA = r * branch_flows[b] ** 2 / const.baseMVA
    #        # A multiplication factor to account for reactive current losses
    #        lossMVA = lossMVA * aclossmultiplier

    def _update_params_powerlosses(self, aclossmultiplier=1, dclossmultiplier=1):
        sp_data.update_loss_parameters(self, aclossmultiplier, dclossmultiplier)

    def _flexload_storage_value(self, i, storage_level, timestep):
        return storage_value.flexload_storage_value(self._grid, i, storage_level, timestep)

    def _get_fault_start(self, timestep):
        # Used by LpFaultProblem
        return None

    def _storeResultsAndUpdateStorage(self, timestep, results):
        sp_extract.store_results_and_update_storage(self, timestep, results)

    def solve(
        self,
        results,
        solver="cbc",
        solver_path=None,
        warmstart=False,
        savefiles=False,
        aclossmultiplier=1,
        dclossmultiplier=1,
        solve_args=None,
        continue_from_last=False,
    ):
        """
        Solve LP problem for each time step in the time range

        Parameters
        ----------
        results : Results
            PowerGAMA Results object reference
        solver : string (optional)
            name of solver to use ("cbc" or "gurobi"). Gurobi uses python
            interface, whilst CBC uses command line executable
        solver_path :string (optional, only relevant for cbc)
            path for solver executable
        warmstart : Boolean
            Use warmstart option (only some solvers, e.g. gurobi)
        savefiles : Boolean
            Save Pyomo model file and LP problem MPS file for each timestep
            This may be useful for debugging.
        aclossmultiplier : float
            Multiplier factor to scale computed AC losses, used with method 1
        dclossmultiplier : float
            Multiplier factor to scale computed DC losses, used with method 1
        logfile : string
            Name of log file for LP solver. Will keep only last iteration
        solve_args : dict
            Arguments passed on to pyomo.solve(...) in each iteration
        continue_from_last : bool
            Whether to continue from last saved result (useful if simulation
            was interrupted)

        Returns
        -------
        results : Results
            PowerGAMA Results object reference
        """
        if solve_args is None:
            solve_args = {
                "tee": False,  # stream the solver output
                "keepfiles": False,  # print the LP file for examination
                "symbolic_solver_labels": True,  # use human readable names
                "logfile": "lpsolver_log.txt",
            }

        if "_persistent" in solver:
            self._solver_persistent = True

        # Initalise solver, and check it is available
        opt = solver_mod.create_solver(solver, self.model, solver_path=solver_path, solve_args=solve_args)

        # Enable access to dual values.
        self.dual = solver_mod.create_dual_handle(opt, self.model)

        timesteps_to_solve = self._get_timesteps_to_solve(continue_from_last=continue_from_last, results=results)
        self._build_day_hour_index(timesteps_to_solve)
        for ext in self._extensions:
            ext.on_solve_start(self, timesteps_to_solve, continue_from_last, results)

        if continue_from_last:
            self._storage.loc[self._idx_generatorsWithStorage] = results.db.getResultStorageFillingAll(
                timestep=timesteps_to_solve[0] - 1
            )
            self._storage_flexload.loc[self._idx_consumersWithFlexLoad] = results.db.getResultFlexloadStorageFillingAll(
                timestep=timesteps_to_solve[0] - 1
            )
            if self._ramp_up_pu is not None or self._ramp_down_pu is not None:
                prev_gen = results.db.getResultGeneratorPowerAll(timestep=timesteps_to_solve[0] - 1)
                for i in self.model.s_gen:
                    self._gen_prev[i] = prev_gen.get(i, 0.0)

        if self._objective_mode == "multi_period":
            # ── true 24h joint LP path ────────────────────────────────
            # (lossmethod != 0 is rejected in __init__; losses are not modelled here)

            # Rolling-horizon solve: optimise over objective horizon, commit only first 24h.
            # This enables next-day lookahead (e.g. 48h solve / 24h commit).
            solve_h = int(max(1, int(self._horizon_hours)))
            commit_h = int(max(1, min(int(self._commit_hours), solve_h)))
            windows = []
            for start in range(0, len(timesteps_to_solve), commit_h):
                win = list(timesteps_to_solve[start : start + solve_h])
                if win:
                    windows.append(win)

            print(f"Solving (24h joint LP) — {len(windows)} window(s): {solve_h}h lookahead / {commit_h}h commit ...")
            for win in windows:
                # For daily_reset generators, clear _gen_prev at commit boundary (same rule intent as single-period)
                if self._ramp_daily_reset is not None:
                    for i in self.model.s_gen:
                        if self._disable_nuclear_ramp_profile_conflict[int(i)]:
                            continue
                        if self._ramp_daily_reset[i]:
                            self._gen_prev[i] = np.nan
                self._solve_and_store_day_joint(win, results, solver, commit_hours=commit_h)

            write_objective_trace(self)
            return results

        if self._lossmethod in [1, 2]:
            print("Computing losses in first timestep")
            self._updateLpProblem(timestep=timesteps_to_solve[0])
            res = opt.solve(self.model)
            if isinstance(res, appsi.solvers.highs.HighsResults):
                if res.termination_condition != appsi.base.TerminationCondition.optimal:
                    raise RuntimeError(
                        f"APPSI HIGHS non-optimal termination before timestep loop: {res.termination_condition}"
                    )
                opt.load_vars()
            # Now, power flow values are computed for the first timestep, and
            # power losses can be computed.

        print("Solving...")
        count = 0
        warmstart_now = False
        self._daily_objective_trace = {}
        self._hourly_objective_trace = []
        for timestep in tqdm(timesteps_to_solve):
            # update LP problem (inflow, storage, profiles)
            self._updateLpProblem(timestep)
            self._update_params_powerlosses(aclossmultiplier, dclossmultiplier)
            if self._solver_persistent:
                self._update_persistent_model(opt=opt)

            # solve the LP problem
            if savefiles:
                # self.concretemodel.pprint('concretemodel_{}.txt'.format(timestep))
                self.model.write(
                    "LPproblem_{}.mps".format(timestep),
                    io_options={"symbolic_solver_labels": True},
                )
                # self.concretemodel.write("LPproblem_{}.nl".format(timestep))

            if warmstart:
                if opt.warm_start_capable():
                    # warmstart available (does not work with cbc)
                    if count > 0:
                        warmstart_now = warmstart
                    count = count + 1
                    solve_args["warmstart"] = warmstart_now
                else:
                    raise Exception("Solver ({}) is not capable of warm start".format(opt.name))

            try:
                res = opt.solve(self.model, **solve_args)
            except Exception as ex:
                # do something ()
                print(f"SOLVE ERROR at timestep={timestep}. Tries to solve again")
                if solver == "appsi_highs":
                    # APPsi/HiGHS may emit cascading low-level row-bound update
                    # errors on immediate retry; preserve original failure signal.
                    raise
                try:
                    res = opt.solve(self.model, **solve_args)
                except Exception:
                    # re-raise exception:
                    raise ex

            # store result for inspection if necessary
            self.solver_res = res

            # debugging:
            if False:
                print(
                    "Solver status = {}. Termination condition = {}".format(
                        res.solver.status, res.solver.termination_condition
                    )
                )

            if isinstance(res, appsi.solvers.highs.HighsResults):
                if res.termination_condition != appsi.base.TerminationCondition.optimal:
                    raise RuntimeError(
                        f"APPSI HIGHS non-optimal termination at timestep={timestep}: {res.termination_condition}"
                    )
                opt.load_vars()
                self.dual = opt.get_duals()
            elif res.solver.status != pyomo.opt.SolverStatus.ok:
                warnings.warn("Something went wrong with LP solver: {}".format(res.solver.status))
                try:
                    self._relax_and_retry(opt, warmstart, count, solve_args)
                except NotImplementedError:
                    raise Exception("Something went wrong with LP solver: {}".format(res.solver.status))
            elif res.solver.termination_condition == pyomo.opt.TerminationCondition.infeasible:
                warnings.warn("t={}: No feasible solution found.".format(timestep))
                try:
                    self._relax_and_retry(opt, warmstart, count, solve_args)
                except NotImplementedError:
                    raise Exception("t={}: No feasible solution found.".format(timestep))

            # This call is required for fault scenario simulation. Does nothing here.
            self._update_progress(timestep, len(timesteps_to_solve))

            if int(timestep) in self._timestep_day_hour:
                day_idx, hour_in_day = self._timestep_day_hour[int(timestep)]
                obj_value = float(self.model.OBJ())
                self._daily_objective_trace[day_idx] = self._daily_objective_trace.get(day_idx, 0.0) + obj_value
                self._hourly_objective_trace.append((int(timestep), int(day_idx), int(hour_in_day), obj_value))

            for ext in self._extensions:
                ext.after_timestep(self, timestep)

            # store results and update storage levels
            self._storeResultsAndUpdateStorage(timestep, results)

        if self._objective_mode == "multi_period" and self._daily_objective_trace:
            print("Daily objective trace (single-period-path accumulation):")
            for d in sorted(self._daily_objective_trace):
                print(f"  day={d}: objective_sum={self._daily_objective_trace[d]:.6f}")

        write_objective_trace(self)

        return results

    def _update_progress(self, n=None, maxn=None):
        return
