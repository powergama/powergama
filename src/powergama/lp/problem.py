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
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pyomo.opt
from pyomo.contrib import appsi
from tqdm import tqdm

from .. import constants as const
from ..timestep_result import TimestepResult, aggregate_to_nodes
from . import project_locks, storage_value
from . import solver as solver_mod
from .config import CONFIG_ATTRS, LpConfig
from .context import CONTEXT_ATTRS, GridContext
from .rt import diagnostics as rt_diagnostics
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
    is_rt : bool
        Solve the balancing ("real-time") problem: minimise deviation from a day-ahead
        schedule rather than generation cost. Requires objective_mode="single_period".
    rt_balancing_fee_eur_per_mwh : float
        Flat fee per MWh of real-time redispatch, to discourage marginal deviations.
    rt_p_res_curtailment_factor : float
        Penalty multiplier on real-time curtailment of variable renewable generation.
    rt_xborder_flow_penalty_eur_per_mwh : float
        Penalty per MWh of deviation from day-ahead cross-border flows.
    rt_dispatch_objective_mode : str
        Ignored when is_rt=True, which forces "deviation".

    Notes
    =====
    Unsupported mode combinations raise NotImplementedError at construction:
    is_rt with "multi_period", lossmethod != 0 with "multi_period", and
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
        rt_dispatch_objective_mode="deviation",
        rt_balancing_fee_eur_per_mwh=1.0,
        rt_p_res_curtailment_factor=2.0,
        rt_xborder_flow_penalty_eur_per_mwh=0.0,
        is_rt=False,
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
        # self._lossmethod / self._is_rt and so on; __getattr__ resolves those
        # against the config.
        self._cfg = LpConfig.from_args(
            lossmethod=lossmethod,
            penalty_twoway_flow=penalty_twoway_flow,
            objective_mode=objective_mode,
            horizon_hours=horizon_hours,
            commit_hours=commit_hours,
            storage_initial_fill_scale=storage_initial_fill_scale,
            rt_dispatch_objective_mode=rt_dispatch_objective_mode,
            rt_balancing_fee_eur_per_mwh=rt_balancing_fee_eur_per_mwh,
            rt_p_res_curtailment_factor=rt_p_res_curtailment_factor,
            rt_xborder_flow_penalty_eur_per_mwh=rt_xborder_flow_penalty_eur_per_mwh,
            is_rt=is_rt,
        )
        self._grid = grid
        self.timeDelta = grid.timeDelta
        self._rt_solver_debug_path: Path | None = None
        self._rt_debug_session_id: str = ""
        rt_diagnostics.initialize_debug_stream(self, grid)
        self._timestep_day_hour: dict[int, tuple[int, int]] = {}
        self._daily_objective_trace: dict[int, float] = {}
        self._hourly_objective_trace: list[tuple[int, int, int, float]] = []
        self._solver_persistent = False
        # Optional sparse DA locks (parquet long format) injected by prepare-rt.
        # Loaded before the context, which needs the border branch sets they
        # identify. The rest are unpacked onto self rather than held as one
        # object, because rt_diagnostics and both model builders read these names.
        _locks = project_locks.load_da_locks(grid)
        self._foreign_gen_lock = _locks.foreign_gen_lock
        self._foreign_cons_lock = _locks.foreign_cons_lock
        self._border_ac_flow_lock = _locks.border_ac_flow_lock
        self._border_dc_flow_lock = _locks.border_dc_flow_lock
        self._border_ac_flow_lb = _locks.border_ac_flow_lb
        self._border_ac_flow_ub = _locks.border_ac_flow_ub
        self._border_dc_flow_lb = _locks.border_dc_flow_lb
        self._border_dc_flow_ub = _locks.border_dc_flow_ub
        self._da_storage_marginalprice = _locks.da_storage_marginalprice

        # Everything the grid implies, derived once. Readers still say
        # self._idx_generatorsWithStorage and so on; __getattr__ resolves those
        # against the context.
        self._ctx = GridContext.from_grid(grid, self._Bbus, self._DA, self.refnodes, _locks)
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

        # 3. Build the Pyomo model. Order is load-bearing -- see single_period.build.
        sp_build.build_model(self.model, grid, self._ctx, self._cfg)

    def __getattr__(self, name):
        """Fall through to the config, then the grid context, then the Pyomo model.

        Settled configuration lives on `self._cfg` and grid derivations on
        `self._ctx`, but the rest of the code --
        including rt_diagnostics, which reads some 77 attributes off the problem --
        spells it `self._lossmethod`, `self._is_rt` and so on. Resolving those names
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

    def _build_day_joint_model(self, day_timesteps):
        """Build a multi-period Pyomo LP covering all timesteps in *day_timesteps*.

        KEY new constraint (storage continuity):
            varStorage[i,h] = storage_prev
                              + (inflow[h,i] - varGen[i,h]
                                 + varPump[i,h]*eff - varSpill[i,h]) * dt
        varStorage is an optimisation variable, so the solver sees the full
        day's inflow/demand profile and can pre-charge/pre-dispatch storage.

        Everything else (power balance, DC flow-angle, branch limits, ramp
        rates, BE border locks) is the same physics as the single-period model,
        now indexed by (entity, hour) instead of (entity) alone.

        Returns (model, pre_data_dict).
        """
        H = len(day_timesteps)
        grid = self._grid
        dt = float(self.timeDelta)
        node_list = list(self.model.s_node)
        branch_ac_list = list(self.model.s_branch_ac)
        # Positions for the Bbus and DA rows, mapped once rather than scanned per rule
        # call -- the rules below run once per (entity, hour), so a linear scan there is
        # quadratic in the entity count and linear again in the horizon.
        node_position = {n: i for i, n in enumerate(node_list)}
        branch_ac_position = {b: i for i, b in enumerate(branch_ac_list)}
        gen_storage_set = frozenset(self._idx_generatorsWithStorage)
        gen_pump_set = frozenset(self._idx_generatorsWithPumping)
        flex_set = frozenset(self._idx_consumersWithFlexLoad)

        # ── pre-compute storage capacity and pump parameters ──────────
        storage_ini = {int(i): float(self._storage[i]) for i in self._idx_generatorsWithStorage}
        storage_cap = {
            int(i): max(0.0, float(grid.generator.loc[i, "storage_cap"])) for i in self._idx_generatorsWithStorage
        }
        spill_frac_series = pd.to_numeric(grid.generator.get("spill_cap_frac", 1.0), errors="coerce")
        if not isinstance(spill_frac_series, pd.Series):
            spill_frac_series = pd.Series(spill_frac_series, index=grid.generator.index)
        spill_cap_mw = {}
        for i in self._idx_generatorsWithStorage:
            raw_frac = spill_frac_series.loc[i] if i in spill_frac_series.index else 1.0
            frac = 1.0 if pd.isna(raw_frac) else float(raw_frac)
            frac = max(0.0, min(1.0, frac))
            pmax_i = max(0.0, float(grid.generator.loc[i, "pmax"]))
            spill_cap_mw[int(i)] = frac * pmax_i
        pump_eff_g = {int(i): float(grid.generator.loc[i, "pump_efficiency"]) for i in self._idx_generatorsWithPumping}
        pump_cap_raw = {int(i): float(grid.generator.loc[i, "pump_cap"]) for i in self._idx_generatorsWithPumping}

        # ── pre-compute storval for each storage gen (BOD filling level) ─
        gen_cost_h: dict = {}  # (h, i) → total cost including storval
        pump_cost_h: dict = {}  # (h, i) → pump credit
        # Terminal storval: opportunity cost of 1 MWh left in storage at end of day.
        # Evaluated at end-of-day time reference so the solver values conservation of
        # stored energy and won't drain storage just because the day is ending.
        terminal_storval: dict = {}  # i → €/MWh

        for i in self._idx_generatorsWithStorage:
            filling_ref = grid.generator.loc[i, "storval_filling_ref"]
            time_ref = grid.generator.loc[i, "storval_time_ref"]
            cap_i = storage_cap[i]
            fill_frac = (storage_ini[i] / cap_i) if cap_i > 0 else 0.0
            fill_frac = max(0.0, min(1.0, fill_frac))
            fill_col = int(round(fill_frac * 100))
            for hi, ts in enumerate(day_timesteps):
                storval = (
                    float(grid.generator.loc[i, "storage_price"])
                    * float(grid.storagevalue_filling.loc[fill_col, filling_ref])
                    * float(grid.storagevalue_time.loc[ts, time_ref])
                )
                gen_cost_h[(hi, i)] = storval
                if i in gen_pump_set:
                    gen_cost_h[(hi, i)] = storval  # same, so gen is expensive when full
                    pump_cost_h[(hi, i)] = storval - float(grid.generator.loc[i, "pump_deadband"])
            # Terminal value = storval at end-of-day timestep (last hour of the day)
            terminal_storval[i] = (
                float(grid.generator.loc[i, "storage_price"])
                * float(grid.storagevalue_filling.loc[fill_col, filling_ref])
                * float(grid.storagevalue_time.loc[day_timesteps[-1], time_ref])
            )

        # Non-storage generators: fuelcost is constant within the day.
        for i in self.model.s_gen:
            if i not in gen_storage_set:
                base_cost = float(pyo.value(self.model.p_gen_cost[i]))
                for hi in range(H):
                    gen_cost_h[(hi, i)] = base_cost

        # ── pre-compute inflow / pmax / pmin per (hour, gen) ─────────
        P_max_base = grid.generator["pmax"]
        P_min_base = grid.generator["pmin"]
        inflow_h: dict = {}  # (hi, i) → inflow MW
        capacity_h: dict = {}  # (hi, i) → installed pmax × availability
        pmin_h: dict = {}  # (hi, i) → pmin lower bound

        for hi, ts in enumerate(day_timesteps):
            for i in self.model.s_gen:
                inflow_fac = float(grid.generator.loc[i, "inflow_fac"])
                inflow_ref = grid.generator.loc[i, "inflow_ref"]
                pmax_fac = 1.0
                if self._pmax_ref is not None:
                    pr = self._pmax_ref.iloc[i]
                    if isinstance(pr, str) and pr in grid.profiles.columns:
                        pmax_fac = float(grid.profiles.loc[ts, pr])
                cap = max(0.0, float(P_max_base[i]) * pmax_fac)
                infl = cap * inflow_fac * float(grid.profiles.loc[ts, inflow_ref])
                pmin = float(P_min_base[i])
                if self._pmin_ref is not None:
                    pr = self._pmin_ref.iloc[i]
                    if isinstance(pr, str) and pr in grid.profiles.columns:
                        pmin = max(0.0, float(P_max_base[i]) * float(grid.profiles.loc[ts, pr]))
                inflow_h[(hi, i)] = float(infl)
                capacity_h[(hi, i)] = float(cap)
                pmin_h[(hi, i)] = float(pmin)

        # ── pre-compute demand per (hour, consumer) ───────────────────
        demand_h: dict = {}
        for hi, ts in enumerate(day_timesteps):
            for j in self.model.s_load:
                avg = float(grid.consumer.loc[j, "demand_avg"]) * (1 - float(grid.consumer.loc[j, "flex_fraction"]))
                prof = grid.consumer.loc[j, "demand_ref"]
                d_now = float(grid.profiles.loc[ts, prof]) * avg
                if self._foreign_cons_lock is not None:
                    key = (int(ts), int(j))
                    if key in self._foreign_cons_lock.index:
                        locked = float(self._foreign_cons_lock.loc[key])
                        if np.isfinite(locked):
                            d_now = locked
                demand_h[(hi, j)] = float(d_now)

        # ── flexible load: reservoir parameters and per-hour value ────
        # Valued like generator storage in this path: the filling level is frozen at
        # window start and only the time-of-year factor varies within the window, so
        # the price stays a constant coefficient in the LP. The single-period path re-prices
        # from the current level every timestep instead.
        flexload_cost_h: dict = {}
        flex_storage_cap: dict = {}  # j → reservoir size, MWh
        flex_storage_ini: dict = {}  # j → level at window start, MWh
        flex_drain_mw: dict = {}  # j → the constant flexible demand it serves, MW
        flex_has_floor: dict = {}  # j → whether the level can be held at or above zero
        for j in flex_set:
            drain = float(grid.consumer.loc[j, "flex_fraction"]) * float(grid.consumer.loc[j, "demand_avg"])
            cap = float(grid.consumer.loc[j, "flex_storage"]) * drain
            # The per-hour charge cap is demand_avg * flex_fraction / flex_on_off, so the
            # drain can be met hour by hour exactly when flex_on_off <= 1. Where it cannot,
            # no floor can be imposed without making the window infeasible.
            on_off = float(grid.consumer.loc[j, "flex_on_off"])
            has_floor = 0.0 < on_off <= 1.0
            flex_drain_mw[int(j)] = drain
            flex_storage_cap[int(j)] = cap
            flex_has_floor[int(j)] = has_floor
            # Seed inside the bounds the window will enforce, so hour 0 cannot be
            # infeasible on a level that came from input data (flex_storagelevel_init
            # outside [0, 1]) or from an earlier window solved without a floor.
            ini = float(self._storage_flexload[j])
            flex_storage_ini[int(j)] = min(max(ini, 0.0), cap) if has_floor else min(ini, cap)
            for hi, ts in enumerate(day_timesteps):
                flexload_cost_h[(hi, j)] = float(self._flexload_storage_value(j, self._storage_flexload[j], ts))

        # ── pre-compute per-hour branch flow bounds (including DA locks) ─
        ac_lb: dict = {}  # (hi, b) → lb or None
        ac_ub: dict = {}  # (hi, b) → ub or None
        dc_lb: dict = {}
        dc_ub: dict = {}
        for hi, ts in enumerate(day_timesteps):
            for b in self.model.s_branch_ac:
                key = (int(ts), int(b))
                if (
                    self._border_ac_flow_lb is not None
                    and key in self._border_ac_flow_lb.index
                    and key in self._border_ac_flow_ub.index
                ):
                    lb = float(self._border_ac_flow_lb.loc[key])
                    ub = float(self._border_ac_flow_ub.loc[key])
                    ac_lb[(hi, b)] = lb if np.isfinite(lb) else None
                    ac_ub[(hi, b)] = ub if np.isfinite(ub) else None
                else:
                    lo, hi_ = self._default_ac_flow_bounds.get(int(b), (None, None))
                    ac_lb[(hi, b)] = lo
                    ac_ub[(hi, b)] = hi_
            for b in self.model.s_branch_dc:
                key = (int(ts), int(b))
                if (
                    self._border_dc_flow_lb is not None
                    and key in self._border_dc_flow_lb.index
                    and key in self._border_dc_flow_ub.index
                ):
                    lb = float(self._border_dc_flow_lb.loc[key])
                    ub = float(self._border_dc_flow_ub.loc[key])
                    dc_lb[(hi, b)] = lb if np.isfinite(lb) else None
                    dc_ub[(hi, b)] = ub if np.isfinite(ub) else None
                else:
                    lo, hi_ = self._default_dc_flow_bounds.get(int(b), (None, None))
                    dc_lb[(hi, b)] = lo
                    dc_ub[(hi, b)] = hi_

        # ── foreign gen locks ─────────────────────────────────────────
        foreign_lock: dict = {}  # (hi, i) → fixed MW or None
        if self._foreign_gen_lock is not None:
            for hi, ts in enumerate(day_timesteps):
                for i in self.model.s_gen:
                    key = (int(ts), int(i))
                    if key in self._foreign_gen_lock.index:
                        v = float(self._foreign_gen_lock.loc[key])
                        if np.isfinite(v):
                            foreign_lock[(hi, i)] = max(0.0, v)

        # ── build Pyomo model ─────────────────────────────────────────
        m = pyo.ConcreteModel()
        m.s_gen = pyo.Set(ordered=True, initialize=list(self.model.s_gen))
        m.s_gen_pump = pyo.Set(ordered=True, initialize=list(self.model.s_gen_pump))
        m.s_gen_storage = pyo.Set(ordered=True, initialize=list(self._idx_generatorsWithStorage))
        m.s_load = pyo.Set(ordered=True, initialize=list(self.model.s_load))
        m.s_load_flex = pyo.Set(ordered=True, initialize=list(self._idx_consumersWithFlexLoad))
        m.s_branch_ac = pyo.Set(ordered=True, initialize=list(self.model.s_branch_ac))
        m.s_branch_dc = pyo.Set(ordered=True, initialize=list(self.model.s_branch_dc))
        m.s_node = pyo.Set(ordered=True, initialize=node_list)
        m.s_h = pyo.RangeSet(0, H - 1)

        # ── variables ─────────────────────────────────────────────────
        m.varGeneration = pyo.Var(m.s_gen, m.s_h, within=pyo.NonNegativeReals)
        m.varPump = pyo.Var(m.s_gen_pump, m.s_h, within=pyo.NonNegativeReals)
        m.varStorage = pyo.Var(m.s_gen_storage, m.s_h, within=pyo.NonNegativeReals)
        m.varSpill = pyo.Var(m.s_gen_storage, m.s_h, within=pyo.NonNegativeReals)
        m.varLoadShed = pyo.Var(m.s_load, m.s_h, within=pyo.NonNegativeReals)
        m.varDumpLoad = pyo.Var(m.s_load, m.s_h, within=pyo.NonNegativeReals)
        m.varFlexLoad = pyo.Var(m.s_load_flex, m.s_h, within=pyo.NonNegativeReals)
        # Reservoir level, in MWh. Reals, not NonNegativeReals: the floor is a
        # constraint rather than a domain because it has to be skipped for a consumer
        # whose per-hour charge limit cannot keep up with its drain -- see
        # cFlexStorFloor. The single-period path has no floor at all; it lets the level go
        # negative and prices that at flexload_outside_cost.
        m.varFlexStorage = pyo.Var(m.s_load_flex, m.s_h, within=pyo.Reals)
        m.varAcBranchFlow = pyo.Var(m.s_branch_ac, m.s_h, within=pyo.Reals)
        m.varDcBranchFlow = pyo.Var(m.s_branch_dc, m.s_h, within=pyo.Reals)
        m.varVoltageAngle = pyo.Var(m.s_node, m.s_h, within=pyo.Reals, initialize=0.0)

        # ── objective: sum over all hours + end-of-day terminal storage value ─
        def _obj_rule(m):
            cost = 0
            for h in range(H):
                cost += sum(m.varGeneration[i, h] * gen_cost_h.get((h, i), 0.0) for i in m.s_gen)
                cost -= sum(m.varPump[i, h] * pump_cost_h.get((h, i), 0.0) for i in m.s_gen_pump)
                cost -= sum(m.varFlexLoad[j, h] * flexload_cost_h.get((h, j), 0.0) for j in m.s_load_flex)
                cost += sum(m.varLoadShed[j, h] * const.loadshedcost for j in m.s_load)
                cost += sum(m.varDumpLoad[j, h] * const.loadshedcost for j in m.s_load)
            # Terminal storage value: reward conservation of stored energy at end of day.
            # Without this, the solver has no incentive to leave storage full at day-end,
            # which would drain it to zero every day and destroy inter-day continuity.
            # Subtracting here (minimisation) means higher end storage → lower effective cost.
            cost -= sum(terminal_storval.get(i, 0.0) * m.varStorage[i, H - 1] for i in m.s_gen_storage)
            return cost

        m.OBJ = pyo.Objective(rule=_obj_rule, sense=pyo.minimize)

        # ── generation bounds (non-storage) ───────────────────────────
        def _gen_ub_ns_rule(m, i, h):
            if i in gen_storage_set:
                return pyo.Constraint.Skip
            locked = foreign_lock.get((h, i))
            pmax = locked if locked is not None else inflow_h.get((h, i), 0.0)
            return m.varGeneration[i, h] <= pmax

        m.cGenUbNs = pyo.Constraint(m.s_gen, m.s_h, rule=_gen_ub_ns_rule)

        def _gen_lb_ns_rule(m, i, h):
            if i in gen_storage_set:
                return pyo.Constraint.Skip
            locked = foreign_lock.get((h, i))
            if locked is not None:
                return m.varGeneration[i, h] >= locked
            infl = inflow_h.get((h, i), 0.0)
            pmin = pmin_h.get((h, i), 0.0)
            return m.varGeneration[i, h] >= max(0.0, min(infl, pmin))

        m.cGenLbNs = pyo.Constraint(m.s_gen, m.s_h, rule=_gen_lb_ns_rule)

        # ── generation bounds (storage generators) ────────────────────
        # gen[i,h] <= inflow[h,i] + storage_prev / dt  (availability)
        # gen[i,h] <= capacity[h,i]                    (hardware cap)
        def _gen_ub_stor_avail_rule(m, i, h):
            locked = foreign_lock.get((h, i))
            if locked is not None:
                return m.varGeneration[i, h] <= locked
            prev_s = storage_ini[i] if h == 0 else m.varStorage[i, h - 1]
            return m.varGeneration[i, h] <= inflow_h.get((h, i), 0.0) + prev_s / dt

        m.cGenUbStorAvail = pyo.Constraint(m.s_gen_storage, m.s_h, rule=_gen_ub_stor_avail_rule)

        def _gen_ub_stor_cap_rule(m, i, h):
            locked = foreign_lock.get((h, i))
            if locked is not None:
                return m.varGeneration[i, h] <= locked
            return m.varGeneration[i, h] <= capacity_h.get((h, i), 0.0)

        m.cGenUbStorCap = pyo.Constraint(m.s_gen_storage, m.s_h, rule=_gen_ub_stor_cap_rule)

        def _gen_lb_stor_rule(m, i, h):
            locked = foreign_lock.get((h, i))
            if locked is not None:
                return m.varGeneration[i, h] >= locked
            infl = inflow_h.get((h, i), 0.0)
            pmin = pmin_h.get((h, i), 0.0)
            return m.varGeneration[i, h] >= max(0.0, min(infl, pmin))

        m.cGenLbStor = pyo.Constraint(m.s_gen_storage, m.s_h, rule=_gen_lb_stor_rule)

        # ── storage capacity upper bound ───────────────────────────────
        def _stor_cap_rule(m, i, h):
            return m.varStorage[i, h] <= storage_cap[i]

        m.cStorCap = pyo.Constraint(m.s_gen_storage, m.s_h, rule=_stor_cap_rule)

        # ── KEY: storage continuity (equality with explicit spill) ─────
        # storage[h] = storage_prev + (inflow - gen + pump*eff - spill) * dt
        def _stor_cont_rule(m, i, h):
            prev_s = storage_ini[i] if h == 0 else m.varStorage[i, h - 1]
            infl = inflow_h.get((h, i), 0.0)
            eff = pump_eff_g.get(i, 1.0) if i in gen_pump_set else 1.0
            pump_term = m.varPump[i, h] * eff if i in gen_pump_set else 0
            return m.varStorage[i, h] == (prev_s + (infl - m.varGeneration[i, h] + pump_term - m.varSpill[i, h]) * dt)

        m.cStorCont = pyo.Constraint(m.s_gen_storage, m.s_h, rule=_stor_cont_rule)

        # Limit spill power by per-generator fraction of installed generator capacity.
        def _spill_cap_rule(m, i, h):
            return m.varSpill[i, h] <= spill_cap_mw.get(i, 0.0)

        m.cSpillCap = pyo.Constraint(m.s_gen_storage, m.s_h, rule=_spill_cap_rule)

        # ── pump bounds ────────────────────────────────────────────────
        # pump[i,h]*eff*dt + storage_prev <= storage_cap  (no spurious pump credit)
        def _pump_room_rule(m, i, h):
            eff = pump_eff_g.get(i, 1.0)
            if eff <= 0:
                return m.varPump[i, h] == 0
            prev_s = storage_ini[i] if h == 0 else m.varStorage[i, h - 1]
            return m.varPump[i, h] * eff * dt <= storage_cap[i] - prev_s

        m.cPumpRoom = pyo.Constraint(m.s_gen_pump, m.s_h, rule=_pump_room_rule)

        def _pump_cap_rule(m, i, h):
            return m.varPump[i, h] <= pump_cap_raw.get(i, 0.0)

        m.cPumpCap = pyo.Constraint(m.s_gen_pump, m.s_h, rule=_pump_cap_rule)

        # ── ramp rates within the day ──────────────────────────────────
        has_ramp = (self._ramp_up_pu is not None) or (self._ramp_down_pu is not None)
        if has_ramp:

            def _ramp_up_rule(m, i, h):
                if self._disable_nuclear_ramp_profile_conflict[int(i)]:
                    return pyo.Constraint.Skip
                ramp_pu = self._ramp_up_pu[i] if self._ramp_up_pu is not None else np.nan
                if np.isnan(ramp_pu):
                    return pyo.Constraint.Skip
                # In RT, skip ramp for fully pinned generators (capacity ≈ pmin, e.g.
                # DA-locked gas and nuclear). Same reasoning as the single-period path.
                if self._is_rt and abs(capacity_h.get((h, i), 0.0) - pmin_h.get((h, i), 0.0)) < 1e-6:
                    return pyo.Constraint.Skip
                if h == 0:
                    prev = self._gen_prev[i]
                    if np.isnan(prev):
                        return pyo.Constraint.Skip
                    return m.varGeneration[i, 0] <= prev + float(ramp_pu) * float(self._ramp_cap_mw[i])
                return m.varGeneration[i, h] <= m.varGeneration[i, h - 1] + float(ramp_pu) * float(self._ramp_cap_mw[i])

            m.cRampUp = pyo.Constraint(m.s_gen, m.s_h, rule=_ramp_up_rule)

            def _ramp_dn_rule(m, i, h):
                if self._disable_nuclear_ramp_profile_conflict[int(i)]:
                    return pyo.Constraint.Skip
                ramp_pu = self._ramp_down_pu[i] if self._ramp_down_pu is not None else np.nan
                if np.isnan(ramp_pu):
                    return pyo.Constraint.Skip
                # In RT, skip ramp for fully pinned generators — same as ramp-up rule.
                if self._is_rt and abs(capacity_h.get((h, i), 0.0) - pmin_h.get((h, i), 0.0)) < 1e-6:
                    return pyo.Constraint.Skip
                if h == 0:
                    prev = self._gen_prev[i]
                    if np.isnan(prev):
                        return pyo.Constraint.Skip
                    return m.varGeneration[i, 0] >= prev - float(ramp_pu) * float(self._ramp_cap_mw[i])
                return m.varGeneration[i, h] >= m.varGeneration[i, h - 1] - float(ramp_pu) * float(self._ramp_cap_mw[i])

            m.cRampDn = pyo.Constraint(m.s_gen, m.s_h, rule=_ramp_dn_rule)

        # ── flex load ─────────────────────────────────────────────────
        if list(m.s_load_flex):

            def _flex_ub_rule(m, j, h):
                avg = float(grid.consumer.loc[j, "demand_avg"])
                frac = float(grid.consumer.loc[j, "flex_fraction"])
                on_off = float(grid.consumer.loc[j, "flex_on_off"])
                return m.varFlexLoad[j, h] <= avg * frac / on_off

            m.cFlexUb = pyo.Constraint(m.s_load_flex, m.s_h, rule=_flex_ub_rule)

            # Reservoir continuity, mirroring cStorCont for generator storage:
            #   level[h] = level[h-1] + (charge - constant drain) * dt
            def _flex_stor_cont_rule(m, j, h):
                prev = flex_storage_ini[j] if h == 0 else m.varFlexStorage[j, h - 1]
                return m.varFlexStorage[j, h] == prev + (m.varFlexLoad[j, h] - flex_drain_mw[j]) * dt

            m.cFlexStorCont = pyo.Constraint(m.s_load_flex, m.s_h, rule=_flex_stor_cont_rule)

            # The window price is a constant coefficient, so the solver would otherwise
            # run the level to one extreme for the whole window: up to the per-hour
            # charge limit while the price is positive, or down with no charging at all
            # while generation costs more than the price is worth. The single-period path needs
            # no bounds because it re-prices from the level every timestep, and the
            # curve turns at 0 and at capacity long before the level gets there.
            def _flex_stor_cap_rule(m, j, h):
                return m.varFlexStorage[j, h] <= flex_storage_cap[j]

            m.cFlexStorCap = pyo.Constraint(m.s_load_flex, m.s_h, rule=_flex_stor_cap_rule)

            def _flex_stor_floor_rule(m, j, h):
                if not flex_has_floor[j]:
                    return pyo.Constraint.Skip
                return m.varFlexStorage[j, h] >= 0.0

            m.cFlexStorFloor = pyo.Constraint(m.s_load_flex, m.s_h, rule=_flex_stor_floor_rule)

        # ── power balance (one per node per hour) ─────────────────────
        Bbus = self._Bbus

        def _power_balance_rule(m, n, h):
            ni = node_position[n]
            lhs = 0
            for g in self._generators_at_node.get(n, []):
                lhs += m.varGeneration[g, h]
                if g in gen_pump_set:
                    lhs -= m.varPump[g, h]
            for j in self._loads_at_node.get(n, []):
                lhs -= demand_h.get((h, j), 0.0)
                lhs += m.varLoadShed[j, h]
                lhs -= m.varDumpLoad[j, h]
                if j in flex_set:
                    lhs -= m.varFlexLoad[j, h]
            for b in self._dcbranch_to_node.get(n, []):
                lhs += m.varDcBranchFlow[b, h]
            for b in self._dcbranch_from_node.get(n, []):
                lhs -= m.varDcBranchFlow[b, h]
            lhs = lhs / const.baseMVA
            rhs = 0
            row = Bbus[ni]
            for k in range(len(row.indices)):
                ni2 = int(row.indices[k])
                n2 = node_list[ni2]
                rhs -= float(row.data[k]) * m.varVoltageAngle[n2, h] * const.baseAngle
            expr = lhs == rhs
            if isinstance(expr, bool) and expr is True:
                return pyo.Constraint.Skip
            return expr

        m.cPowerbalance = pyo.Constraint(m.s_node, m.s_h, rule=_power_balance_rule)

        # ── DC power flow (flow-angle relationship) ───────────────────
        DA = self._DA

        def _flow_angle_rule(m, b, h):
            idx_b = branch_ac_position[b]
            lhs = m.varAcBranchFlow[b, h] / const.baseMVA
            rhs = 0
            row = DA[idx_b]
            for k in range(len(row.indices)):
                ni2 = int(row.indices[k])
                n2 = node_list[ni2]
                rhs += float(row.data[k]) * m.varVoltageAngle[n2, h] * const.baseAngle
            return lhs == rhs

        m.cFlowAngle = pyo.Constraint(m.s_branch_ac, m.s_h, rule=_flow_angle_rule)

        # ── reference angle = 0 per synchronous area ──────────────────
        for refnode in self.refnodes:
            for h in range(H):
                m.varVoltageAngle[refnode, h].fix(0.0)

        # ── branch capacity limits ─────────────────────────────────────
        def _ac_lb_rule(m, b, h):
            lb = ac_lb.get((h, b))
            return m.varAcBranchFlow[b, h] >= lb if lb is not None else pyo.Constraint.Skip

        m.cAcFlowLb = pyo.Constraint(m.s_branch_ac, m.s_h, rule=_ac_lb_rule)

        def _ac_ub_rule(m, b, h):
            ub = ac_ub.get((h, b))
            return m.varAcBranchFlow[b, h] <= ub if ub is not None else pyo.Constraint.Skip

        m.cAcFlowUb = pyo.Constraint(m.s_branch_ac, m.s_h, rule=_ac_ub_rule)

        def _dc_lb_rule(m, b, h):
            lb = dc_lb.get((h, b))
            return m.varDcBranchFlow[b, h] >= lb if lb is not None else pyo.Constraint.Skip

        m.cDcFlowLb = pyo.Constraint(m.s_branch_dc, m.s_h, rule=_dc_lb_rule)

        def _dc_ub_rule(m, b, h):
            ub = dc_ub.get((h, b))
            return m.varDcBranchFlow[b, h] <= ub if ub is not None else pyo.Constraint.Skip

        m.cDcFlowUb = pyo.Constraint(m.s_branch_dc, m.s_h, rule=_dc_ub_rule)

        # ── inter-area NTC constraints ─────────────────────────────────
        if self._inter_area_ntc is not None and len(self._inter_area_ntc) > 0 and hasattr(self, "_inter_area_ntc_rows"):
            ntc_rows = self._inter_area_ntc_rows
            m.s_ntc = pyo.RangeSet(0, len(ntc_rows) - 1)

            def _ntc_fwd_rule(m, cidx, h):
                row = ntc_rows[int(cidx)]
                cap = row["ntc_forward"]
                if not np.isfinite(cap):
                    return pyo.Constraint.Skip
                if not (row["ac_pos"] or row["ac_neg"] or row["dc_pos"] or row["dc_neg"]):
                    return pyo.Constraint.Skip
                net_t = (
                    sum(m.varAcBranchFlow[b, h] for b in row["ac_pos"])
                    - sum(m.varAcBranchFlow[b, h] for b in row["ac_neg"])
                    + sum(m.varDcBranchFlow[b, h] for b in row["dc_pos"])
                    - sum(m.varDcBranchFlow[b, h] for b in row["dc_neg"])
                )
                return net_t <= cap

            m.cNtcFwd = pyo.Constraint(m.s_ntc, m.s_h, rule=_ntc_fwd_rule)

            def _ntc_bwd_rule(m, cidx, h):
                row = ntc_rows[int(cidx)]
                cap = row["ntc_backward"]
                if not np.isfinite(cap):
                    return pyo.Constraint.Skip
                if not (row["ac_pos"] or row["ac_neg"] or row["dc_pos"] or row["dc_neg"]):
                    return pyo.Constraint.Skip
                net_t = (
                    sum(m.varAcBranchFlow[b, h] for b in row["ac_pos"])
                    - sum(m.varAcBranchFlow[b, h] for b in row["ac_neg"])
                    + sum(m.varDcBranchFlow[b, h] for b in row["dc_pos"])
                    - sum(m.varDcBranchFlow[b, h] for b in row["dc_neg"])
                )
                return net_t >= -cap

            m.cNtcBwd = pyo.Constraint(m.s_ntc, m.s_h, rule=_ntc_bwd_rule)

        pre_data = {
            "H": H,
            "day_timesteps": list(day_timesteps),
            "inflow_h": inflow_h,
            "gen_cost_h": gen_cost_h,
            "pump_cost_h": pump_cost_h,
            "flexload_cost_h": flexload_cost_h,
            "storage_ini": storage_ini,
            "storage_cap": storage_cap,
            "pump_eff_g": pump_eff_g,
            "gen_storage_set": gen_storage_set,
            "gen_pump_set": gen_pump_set,
            "flex_set": flex_set,
            "node_list": node_list,
        }
        return m, pre_data

    def _extract_and_store_day_joint_results(self, m, pre_data, results, duals, commit_hours=None):
        """Extract results from a solved 24h joint model and store to results DB.

        Emits one TimestepResult per committed hour in the optimisation window --
        the same contract the single-period path uses -- then updates self._storage to
        end-of-commit levels.
        """
        H = pre_data["H"]
        if commit_hours is None:
            commit_hours = H
        commit_hours = int(max(1, min(int(commit_hours), int(H))))
        day_timesteps = pre_data["day_timesteps"]
        # inflow_h = pre_data["inflow_h"]
        # storage_ini = pre_data["storage_ini"]
        # storage_cap = pre_data["storage_cap"]
        # pump_eff_g = pre_data["pump_eff_g"]
        # gen_storage_set = pre_data["gen_storage_set"]
        # gen_pump_set = pre_data["gen_pump_set"]
        # flex_set = pre_data["flex_set"]
        node_list = pre_data["node_list"]
        gen_list = list(self.model.s_gen)
        gen_pump_list = list(self.model.s_gen_pump)
        storage_gen_list = list(self._idx_generatorsWithStorage)
        flex_list = list(self._idx_consumersWithFlexLoad)
        branch_ac_list = list(self.model.s_branch_ac)
        branch_dc_list = list(self.model.s_branch_dc)
        # Spill is reported against the full generator list, so storage generators need
        # their position in it. Mapped once instead of scanned inside the hour loop.
        gen_position = {i: gi for gi, i in enumerate(gen_list)}
        grid = self._grid
        dt = float(self.timeDelta)

        for hi, ts in enumerate(day_timesteps[:commit_hours]):
            Pgen = [float(pyo.value(m.varGeneration[i, hi]) or 0.0) for i in gen_list]
            Ppump = [float(pyo.value(m.varPump[i, hi]) or 0.0) for i in gen_pump_list]
            Pflexload = [float(pyo.value(m.varFlexLoad[j, hi]) or 0.0) for j in flex_list]
            Pb = [float(pyo.value(m.varAcBranchFlow[b, hi]) or 0.0) for b in branch_ac_list]
            Pdc = [float(pyo.value(m.varDcBranchFlow[b, hi]) or 0.0) for b in branch_dc_list]
            theta = [float(pyo.value(m.varVoltageAngle[n, hi]) or 0.0) * const.baseAngle for n in node_list]

            # Load shedding aggregated to nodes
            Ploadshed = aggregate_to_nodes(
                grid, self.model.s_load, lambda j: float(pyo.value(m.varLoadShed[j, hi]) or 0.0)
            )
            Pdumpload = aggregate_to_nodes(
                grid, self.model.s_load, lambda j: float(pyo.value(m.varDumpLoad[j, hi]) or 0.0)
            )

            # Per-hour objective (cost for this timestep only)
            obj_h = (
                sum(Pgen[gi] * pre_data["gen_cost_h"].get((hi, gen_list[gi]), 0.0) for gi in range(len(gen_list)))
                - sum(
                    Ppump[pi] * pre_data["pump_cost_h"].get((hi, gen_pump_list[pi]), 0.0)
                    for pi in range(len(gen_pump_list))
                )
                - sum(
                    Pflexload[fi] * pre_data["flexload_cost_h"].get((hi, flex_list[fi]), 0.0)
                    for fi in range(len(flex_list))
                )
                + sum(float(pyo.value(m.varLoadShed[j, hi]) or 0.0) * const.loadshedcost for j in self.model.s_load)
                + sum(float(pyo.value(m.varDumpLoad[j, hi]) or 0.0) * const.loadshedcost for j in self.model.s_load)
            )

            # Storage levels and spilled energy
            stor_levels = []
            for i in storage_gen_list:
                stor_levels.append(float(pyo.value(m.varStorage[i, hi]) or 0.0))

            # Spilled energy per generator (all generators, not just storage)
            energy_spilled = np.zeros(len(gen_list))
            for i in storage_gen_list:
                spill_val = float(pyo.value(m.varSpill[i, hi]) or 0.0)
                energy_spilled[gen_position[i]] = spill_val * dt

            # Branch capacity sensitivity (dual of capacity constraint).
            # NOTE: cAcFlowUb/cAcFlowLb/cDcFlowUb are declared as
            # Constraint(s_branch_ac, s_h), so the index order is (branch, hour).
            # Membership is tested explicitly rather than caught as KeyError: a
            # transposed (hour, branch) lookup is often a *valid* key for another
            # branch, so an exception-based guard cannot detect that mistake.
            senseB = []
            for j in self._idx_branchesWithConstraints:
                d = 0.0
                if duals:
                    if (j, hi) in m.cAcFlowUb:
                        d = float(duals.get(m.cAcFlowUb[j, hi], 0.0) or 0.0)
                    elif (j, hi) in m.cAcFlowLb:
                        d = float(duals.get(m.cAcFlowLb[j, hi], 0.0) or 0.0)
                senseB.append(-abs(d / const.baseMVA))

            senseDcB = []
            for j in branch_dc_list:
                d = 0.0
                if duals:
                    if (j, hi) in m.cDcFlowUb:
                        d = float(duals.get(m.cDcFlowUb[j, hi], 0.0) or 0.0)
                    elif (j, hi) in m.cDcFlowLb:
                        d = float(duals.get(m.cDcFlowLb[j, hi], 0.0) or 0.0)
                senseDcB.append(-abs(d / const.baseMVA))

            # Nodal prices (dual of power balance)
            senseN = []
            for n in node_list:
                d = 0.0
                if duals:
                    try:
                        c = m.cPowerbalance[n, hi]
                        d = float(duals.get(c, 0.0) or 0.0)
                    except (KeyError, AttributeError):
                        d = 0.0
                senseN.append(abs(d / const.baseMVA))

            storageprice = [pre_data["gen_cost_h"].get((hi, i), 0.0) for i in storage_gen_list]
            flexload_storagelevel = [float(pyo.value(m.varFlexStorage[j, hi]) or 0.0) for j in flex_list]
            flexload_marginalprice = [pre_data["flexload_cost_h"].get((hi, j), 0.0) for j in flex_list]

            TimestepResult(
                timestep=grid.timerange[0] + ts,
                objective_function=obj_h,
                generator_power=Pgen,
                generator_pumped=Ppump,
                branch_power=Pb,
                dcbranch_power=Pdc,
                node_angle=theta,
                sensitivity_branch_capacity=senseB,
                sensitivity_dcbranch_capacity=senseDcB,
                sensitivity_node_power=senseN,
                storage=stor_levels,
                inflow_spilled=energy_spilled.tolist(),
                loadshed_power=Ploadshed.tolist(),
                dumpload_power=Pdumpload.tolist(),
                marginalprice=storageprice,
                flexload_power=Pflexload,
                flexload_storage=flexload_storagelevel,
                flexload_storagevalue=flexload_marginalprice,
                # Losses are not modelled in this path; lossmethod != 0 is rejected in __init__.
                branch_ac_losses=[0.0] * len(branch_ac_list),
                branch_dc_losses=[0.0] * len(branch_dc_list),
                # Fault scenarios run the single-period path only -- see LpFaultProblem.__init__.
                fault_start=None,
            ).store(results)

            rt_diagnostics.append_multi_period_debug(self, m, hi, grid.timerange[0] + ts)

            # Update gen_prev for context at day boundary
            for gi, i in enumerate(gen_list):
                self._gen_prev[i] = Pgen[gi]

        # Update self._storage to end-of-commit levels
        for i in storage_gen_list:
            self._storage[i] = float(pyo.value(m.varStorage[i, commit_hours - 1]) or 0.0)

        # Same for the flexible-load reservoirs, so the next window is priced from
        # where this one left off -- the single-period path does this in
        # _storeResultsAndUpdateStorage.
        for j in flex_list:
            self._storage_flexload[j] = float(pyo.value(m.varFlexStorage[j, commit_hours - 1]) or 0.0)

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
        m, pre_data = self._build_day_joint_model(day_timesteps)
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
        self._extract_and_store_day_joint_results(
            m,
            pre_data,
            results,
            duals,
            commit_hours=commit_hours,
        )

    # ------------------------------------------------------------------

    def append_rt_solver_debug_event(self, payload):
        """Public wrapper for optional debug append used by workflow helpers."""
        rt_diagnostics.append_payload(self, payload)

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
        if self._is_rt and self._rt_solver_debug_path is not None:
            last_saved = None
            if continue_from_last and (results is not None):
                try:
                    last_saved = results.get_last_timestep_in_results()
                except Exception:
                    last_saved = None
            payload = {
                "event": "rt_debug_solve_start",
                "debug_session_id": self._rt_debug_session_id,
                "written_at_utc": str(pd.Timestamp.utcnow().isoformat()),
                "continue_from_last": bool(continue_from_last),
                "last_saved_timestep": (int(last_saved) if last_saved is not None else None),
                "first_timestep": (int(timesteps_to_solve[0]) if timesteps_to_solve else None),
                "last_timestep": (int(timesteps_to_solve[-1]) if timesteps_to_solve else None),
                "n_timesteps": int(len(timesteps_to_solve)),
            }
            rt_diagnostics.append_payload(self, payload)

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

            rt_diagnostics.write_objective_trace(self)
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

            # Optional deep RT solver diagnostics (JSONL stream)
            rt_diagnostics.append_hourly_debug(self, timestep)

            # store results and update storage levels
            self._storeResultsAndUpdateStorage(timestep, results)

        if self._objective_mode == "multi_period" and self._daily_objective_trace:
            print("Daily objective trace (single-period-path accumulation):")
            for d in sorted(self._daily_objective_trace):
                print(f"  day={d}: objective_sum={self._daily_objective_trace[d]:.6f}")

        rt_diagnostics.write_objective_trace(self)

        return results

    def _update_progress(self, n=None, maxn=None):
        return
