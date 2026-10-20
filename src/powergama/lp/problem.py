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
from .formulation import build as formulation_build
from .formulation import data as formulation_data
from .formulation import extract as formulation_extract
from .objective_trace import write_objective_trace


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
        of `horizon_hours` with storage continuity inside the optimisation. Both are
        the same LP (lp.formulation): single-period is a window of one timestep.
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
        Optional additions to the model, e.g.
        :class:`powergama.lp.rt.RtExtension` for balancing ("real-time") simulation.
        See :mod:`powergama.lp.extension`.

    Notes
    =====
    Unsupported combinations raise NotImplementedError at construction: an
    extension, or lossmethod != 0, with windows longer than one timestep.
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
        # 1. The Pyomo model. APPSI's incremental update and
        #    gurobi_persistent.set_instance both key off the object, so a solver must
        #    only ever be handed the model it was created for. self.model is the model
        #    for windows of the configured length; a window cut short at the end of a
        #    multi-period run gets a model and solver of its own (_use_window_model).
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
        self.set_storage_state(
            storage=(_fill_scale * _storage_ini * _storage_cap).fillna(0),
            storage_flexload=(
                grid.consumer["flex_storagelevel_init"]
                * grid.consumer["flex_storage"]
                * grid.consumer["flex_fraction"]
                * grid.consumer["demand_avg"]
            ).fillna(0),
        )
        self._energyspilled = grid.generator["storage_cap"].copy(deep=True)
        self._energyspilled[:] = 0

        # 3. Extensions derive their own data, now that the context and the initial
        #    simulation state exist.
        for ext in self._extensions:
            ext.prepare(self, grid)

        # 4. Build the Pyomo model. Order is load-bearing -- see formulation.build.
        #    The objective's terms per hour tell result extraction what to report, and
        #    whether an extension removed the storage value from the objective.
        H = self._cfg.window_hours
        self._objective_by_hour = formulation_build.build_model(
            self.model, grid, self._ctx, self._cfg, H, self._extensions
        )
        self._objective_term_names = self._objective_by_hour[-1][0]
        # model, objective by hour, solver, dual handle -- per window length
        self._window_models = {H: [self.model, self._objective_by_hour, None, None]}

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
        taken as given; the LP starts from them moved inside the reservoir's bounds (see
        `GridContext.flex_storage_seed`).
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

    def _windows(self, timesteps_to_solve):
        """The windows to solve, in order: lists of timesteps.

        A window of `horizon_hours` timesteps starts every `commit_hours`; in
        single-period mode both are 1. Windows at the end of the run are cut short.
        """
        H, commit = self._cfg.window_hours, self._cfg.window_commit
        timesteps = list(timesteps_to_solve)
        return [timesteps[start : start + H] for start in range(0, len(timesteps), commit)]

    def _use_window_model(self, H, solver, solver_path, solve_args):
        """Make the model for windows of *H* timesteps the active one, building it if needed.

        Only windows cut short at the end of a multi-period run have another length
        than the model built in __init__. Each length keeps its own model and its own
        solver for the whole of solve(): APPSI's incremental update and
        gurobi_persistent.set_instance key off the model object, so a solver must only
        ever be handed the one model it was created for.
        """
        if H not in self._window_models:
            model = pyo.ConcreteModel()
            objective = formulation_build.build_model(model, self._grid, self._ctx, self._cfg, H, self._extensions)
            self._window_models[H] = [model, objective, None, None]
        entry = self._window_models[H]
        if entry[2] is None:
            opt = solver_mod.create_solver(solver, entry[0], solver_path=solver_path, solve_args=solve_args)
            entry[2], entry[3] = opt, solver_mod.create_dual_handle(opt, entry[0])
        self.model, objective, opt, self.dual = entry
        self._objective_by_hour = objective
        self._objective_term_names = objective[-1][0]
        return opt

    def _relax_and_retry(self, opt, warmstart, count, solve_args):
        raise NotImplementedError

    def _updateLpProblem(self, timesteps):
        """Rewrite the model's parameters for the window of *timesteps*.

        Kept as a method because it is an extension seam, not merely a long
        function: LpFaultProblem overrides it, reseeds storage from a baseline
        database inside the override, then calls super(). It must stay callable
        exactly once per window, in order.
        """
        formulation_data.update_window(self, timesteps)

    def _update_persistent_model(self, opt):
        """Update objective function, constraints and bounds in persistent model"""
        # Mutable parameters is not enough: a persistent solver keeps its own copy of
        # every constraint, so each one with a mutable parameter is removed and added
        # again, and the objective is set again.
        # https://pyomo.readthedocs.io/en/latest/solvers/persistent_solvers.html
        for name in (
            "cGenMaxLimit",  # p_gen_pmax
            "cGenMinLimit",  # p_gen_pmin
            "cPump",  # p_pump_pmax
            "cStorCont",  # p_storage_prev, p_storage_inflow
            "cSpillInflow",  # p_storage_inflow
            "cFlexStorCont",  # p_flex_storage_prev
            "cPowerbalance",  # p_demand, p_branch_*_power_loss
            "cLossAc12",  # p_branch_ac_powerflow12/21
            "cLossAc21",
            "cLossDc12",  # p_branch_dc_powerflow12/21
            "cLossDc21",
            "cGenUbStorAvail",  # p_storage_inflow
            "cRampUp",  # p_ramp_up_mw
            "cRampDn",  # p_ramp_dn_mw
        ):
            con = self.model.component(name)
            if con is None:
                continue
            for c in con.values():
                opt.remove_constraint(c)
                opt.add_constraint(c)
        # p_storage_value, p_loadflex_cost => OBJ
        opt.set_objective(self.model.OBJ)

    def _update_params_powerlosses(self, aclossmultiplier=1, dclossmultiplier=1):
        formulation_data.update_loss_parameters(self, aclossmultiplier, dclossmultiplier)

    def _flexload_storage_value(self, i, storage_level, timestep):
        return storage_value.flexload_storage_value(self._grid, i, storage_level, timestep)

    def _get_fault_start(self, timestep):
        # Used by LpFaultProblem
        return None

    def _storeResultsAndUpdateStorage(self, timesteps, commit, objective_values, results):
        formulation_extract.store_window(self, timesteps, commit, objective_values, results)

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

        In single-period mode, one LP per timestep. In multi-period mode, one LP per
        window of `horizon_hours` timesteps, of which the first `commit_hours` are
        stored before the next window starts.

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
            Save Pyomo model file and LP problem MPS file for each window
            This may be useful for debugging.
        aclossmultiplier : float
            Multiplier factor to scale computed AC losses, used with method 1
        dclossmultiplier : float
            Multiplier factor to scale computed DC losses, used with method 1
        logfile : string
            Name of log file for LP solver. Will keep only last iteration
        solve_args : dict
            Arguments passed on to pyomo.solve(...) in each iteration. The default
            suits the SolverFactory solvers; "appsi_highs" takes none.
        continue_from_last : bool
            Whether to continue from last saved result (useful if simulation
            was interrupted)

        Returns
        -------
        results : Results
            PowerGAMA Results object reference
        """
        if solve_args is None and solver == "appsi_highs":
            # The APPSI interface takes no per-solve keyword arguments.
            solve_args = {}
        elif solve_args is None:
            solve_args = {
                "tee": False,  # stream the solver output
                "keepfiles": False,  # print the LP file for examination
                "symbolic_solver_labels": True,  # use human readable names
                "logfile": "lpsolver_log.txt",
            }

        if "_persistent" in solver:
            self._solver_persistent = True

        # Initalise solver, and check it is available. Each window length gets its own;
        # the full-length one is created now so that a missing solver fails early.
        for entry in self._window_models.values():
            entry[2] = entry[3] = None
        opt = self._use_window_model(self._cfg.window_hours, solver, solver_path, solve_args)

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
                for i in self._ctx.gen_ids:
                    self._gen_prev[i] = prev_gen.get(i, 0.0)

        # Losses are only modelled in windows of one timestep (see LpConfig).
        if self._lossmethod in [1, 2] and continue_from_last:
            # Losses are linearised around the preceding timestep's flows. When resuming,
            # that solution is in the database, so seed from it rather than a pre-solve:
            # anything else changes every result from here on (issue #41).
            prev = timesteps_to_solve[0] - 1
            window = [prev, prev + 1]
            formulation_data.seed_loss_operating_point(
                self,
                ac_flows={indx: flow for _, indx, flow in results.db.getResultBranchFlowAll(window, "ac")},
                dc_flows={indx: flow for _, indx, flow in results.db.getResultBranchFlowAll(window, "dc")},
            )
        elif self._lossmethod in [1, 2]:
            print("Computing losses in first timestep")
            self._updateLpProblem([timesteps_to_solve[0]])
            res = opt.solve(self.model)
            if isinstance(res, appsi.solvers.highs.HighsResults):
                if res.termination_condition != appsi.base.TerminationCondition.optimal:
                    raise RuntimeError(
                        f"APPSI HIGHS non-optimal termination before timestep loop: {res.termination_condition}"
                    )
                opt.load_vars()
            # Now, power flow values are computed for the first timestep, and
            # power losses can be computed.

        windows = self._windows(timesteps_to_solve)
        if self._cfg.window_hours > 1:
            print(
                f"Solving {len(windows)} window(s) of {self._cfg.window_hours} timesteps, "
                f"committing {self._cfg.window_commit} ..."
            )
        else:
            print("Solving...")
        count = 0
        warmstart_now = False
        self._daily_objective_trace = {}
        self._hourly_objective_trace = []
        for window in tqdm(windows):
            timestep = window[0]
            opt = self._use_window_model(len(window), solver, solver_path, solve_args)
            # update LP problem (inflow, storage, profiles)
            self._updateLpProblem(window)
            self._update_params_powerlosses(aclossmultiplier, dclossmultiplier)
            if self._solver_persistent:
                self._update_persistent_model(opt=opt)
            if self._cfg.window_hours > 1:
                solver_mod.discard_warm_start(opt)

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

            if isinstance(res, appsi.solvers.highs.HighsResults):
                if res.termination_condition != appsi.base.TerminationCondition.optimal:
                    raise RuntimeError(
                        f"APPSI HIGHS non-optimal termination at timestep={timestep}: {res.termination_condition}"
                    )
                opt.load_vars()
                self.dual = self._window_models[len(window)][3] = opt.get_duals()
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

            # The objective of each committed hour, evaluated once: on a large grid the
            # objective expression is slow to evaluate.
            commit = min(self._cfg.window_commit, len(window))
            objective_values = [float(pyo.value(expr)) for _, expr in self._objective_by_hour[:commit]]
            for ts, obj_value in zip(window[:commit], objective_values):
                # This call is required for fault scenario simulation. Does nothing here.
                self._update_progress(ts, len(timesteps_to_solve))

                if int(ts) in self._timestep_day_hour:
                    day_idx, hour_in_day = self._timestep_day_hour[int(ts)]
                    self._daily_objective_trace[day_idx] = self._daily_objective_trace.get(day_idx, 0.0) + obj_value
                    self._hourly_objective_trace.append((int(ts), int(day_idx), int(hour_in_day), obj_value))

                for ext in self._extensions:
                    ext.after_timestep(self, ts)

            # store results and update storage levels
            self._storeResultsAndUpdateStorage(window, commit, objective_values, results)

        # Leave the full-length model as self.model, whatever the last window was.
        if self.model is not self._window_models[self._cfg.window_hours][0]:
            self._use_window_model(self._cfg.window_hours, solver, solver_path, solve_args)

        write_objective_trace(self)

        return results

    def _update_progress(self, n=None, maxn=None):
        return
