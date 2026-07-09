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
import os
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pyomo.opt
from pyomo.contrib import appsi
from tqdm import tqdm

from . import constants as const


class LpProblem(pyo.ConcreteModel):
    """LP problem formulation

    Parameters
    ==========
    grid : GridData
        grid data object
    lossmethod : int
        loss method; 0=no losses, 1=linearised losses, 2=added as load
    penalty_twoway_flow : float
        penalty factor to discourage flow in both directions on a branch with losses
    """

    def __init__(
        self,
        grid,
        lossmethod=0,
        penalty_twoway_flow=0,
        objective_mode="hourly",
        objective_day_horizon_hours=24,
        objective_day_commit_hours=24,
        storage_initial_fill_scale=1.0,
        rt_residual_penalty=0.0,
        rt_foreign_da_lock_penalty_coeff=0.0,
        is_rt=False,
    ):
        # 1.
        super().__init__()

        # 2. Compute matrices used in power flow equaions
        print("Computing B and DA matrices...")
        self._Bbus, self._DA = grid.compute_power_flow_matrices()

        print("Initialising LP problem...")

        # Helpers
        self._lossmethod = lossmethod
        self._grid = grid
        self.timeDelta = grid.timeDelta
        self._objective_mode = str(objective_mode).strip().lower()
        if self._objective_mode not in {"hourly", "daily_24h"}:
            raise ValueError("objective_mode must be one of: hourly, daily_24h")
        self._objective_day_horizon_hours = int(max(1, int(objective_day_horizon_hours)))
        self._objective_day_commit_hours = int(max(1, int(objective_day_commit_hours)))
        self._rt_residual_penalty = float(rt_residual_penalty)
        self._rt_foreign_da_lock_penalty_coeff = float(rt_foreign_da_lock_penalty_coeff)
        self._rt_target_tracking_active = (
            self._rt_residual_penalty > 0.0
            or self._rt_foreign_da_lock_penalty_coeff > 0.0
        )
        self._is_rt = bool(is_rt)
        self._timestep_day_hour: dict[int, tuple[int, int]] = {}
        self._daily_objective_trace: dict[int, float] = {}
        self._hourly_objective_trace: list[tuple[int, int, int, float]] = []
        self._solver_persistent = False
        self._generators_at_node = grid.generator.groupby("node").groups
        self._loads_at_node = grid.consumer.groupby("node").groups
        self._branch_from_node = grid.branch.groupby("node_from").groups
        self._branch_to_node = grid.branch.groupby("node_to").groups
        self._dcbranch_from_node = grid.dcbranch.groupby("node_from").groups
        self._dcbranch_to_node = grid.dcbranch.groupby("node_to").groups
        for n in grid.node["id"]:
            # fill in so dict is defined for all nodes:
            if n not in self._generators_at_node:
                self._generators_at_node[n] = []
            if n not in self._loads_at_node:
                self._loads_at_node[n] = []
            if n not in self._branch_from_node:
                self._branch_from_node[n] = []
            if n not in self._branch_to_node:
                self._branch_to_node[n] = []
            if n not in self._dcbranch_from_node:
                self._dcbranch_from_node[n] = []
            if n not in self._dcbranch_to_node:
                self._dcbranch_to_node[n] = []

        self._idx_generatorsWithPumping = grid.getIdxGeneratorsWithPumping()
        self._idx_generatorsWithStorage = grid.getIdxGeneratorsWithStorage()
        self._idx_consumersWithFlexLoad = grid.getIdxConsumersWithFlexibleLoad()
        # Optional profile references for time-varying dispatch bounds.
        # pmax_ref scales installed pmax (availability factor), pmin_ref sets a
        # profile-driven floor as a fraction of installed pmax.
        self._pmax_ref = grid.generator["pmax_ref"] if "pmax_ref" in grid.generator.columns else None
        self._pmin_ref = grid.generator["pmin_ref"] if "pmin_ref" in grid.generator.columns else None
        self._rt_target_ref = grid.generator["rt_target_ref"] if "rt_target_ref" in grid.generator.columns else None
        self._rt_target_gen_indices = {
            int(i)
            for i in grid.generator.index
            if self._rt_target_ref is not None
            and isinstance(self._rt_target_ref.loc[i], str)
            and str(self._rt_target_ref.loc[i]).strip()
        }
        # Ramp-rate limits (MW per timestep). NaN means unconstrained.
        # Optional pump (charging) DA target references for DA-target deviation penalties.
        # Only applies to generators in s_gen_pump that have a non-empty rt_pump_target_ref.
        self._rt_pump_target_ref = grid.generator["rt_pump_target_ref"] if "rt_pump_target_ref" in grid.generator.columns else None
        _pump_idx_set = {int(j) for j in self._idx_generatorsWithPumping}
        self._rt_pump_target_gen_indices = {
            int(i)
            for i in grid.generator.index
            if int(i) in _pump_idx_set
            and self._rt_pump_target_ref is not None
            and isinstance(self._rt_pump_target_ref.loc[i], str)
            and str(self._rt_pump_target_ref.loc[i]).strip()
        }
        # Optional consumer (flexible load) DA target references for DA-target deviation penalties.
        self._rt_consumer_target_ref = grid.consumer["rt_target_ref"] if "rt_target_ref" in grid.consumer.columns else None
        self._rt_consumer_target_indices = {
            int(i)
            for i in grid.consumer.index
            if self._rt_consumer_target_ref is not None
            and isinstance(self._rt_consumer_target_ref.loc[i], str)
            and str(self._rt_consumer_target_ref.loc[i]).strip()
            and int(i) in self._idx_consumersWithFlexLoad
        }
        # Optional storage (reservoir/battery filling) DA target references for DA-target deviation penalties.
        # Stores normalized filling fraction [0, 1] to ensure storage state continuity matches DA.
        self._rt_storage_target_ref = grid.generator["rt_storage_target_ref"] if "rt_storage_target_ref" in grid.generator.columns else None
        self._rt_storage_target_indices = {
            int(i)
            for i in grid.generator.index
            if self._rt_storage_target_ref is not None
            and isinstance(self._rt_storage_target_ref.loc[i], str)
            and str(self._rt_storage_target_ref.loc[i]).strip()
            and int(i) in self._idx_generatorsWithStorage
        }
        # Optional BE residual cap profiles (MW) for directional BE balancing constraints.
        self._rt_residual_pos_cap_ref = "rt_residual_pos_cap_mw"
        self._rt_residual_neg_cap_ref = "rt_residual_neg_cap_mw"
        self._rt_residual_cap_active = (
            self._rt_residual_penalty > 0
            and self._rt_residual_pos_cap_ref in grid.profiles.columns
            and self._rt_residual_neg_cap_ref in grid.profiles.columns
        )
        self._idx_be_generators = {
            int(i)
            for i in grid.generator.index
            if str(grid.generator.loc[i, "node"]).startswith("BE")
        }
        self._idx_be_target_generators = {
            int(i) for i in self._rt_target_gen_indices if int(i) in self._idx_be_generators
        }
        self._idx_be_target_pumps = {
            int(i) for i in self._rt_pump_target_gen_indices if int(i) in self._idx_be_generators
        }
        self._idx_be_loads = {
            int(i)
            for i in grid.consumer.index
            if str(grid.consumer.loc[i, "node"]).startswith("BE")
        }
        self._idx_be_flexloads = {
            int(i)
            for i in self._idx_consumersWithFlexLoad
            if int(i) in self._idx_be_loads
        }
        self._idx_be_target_flexloads = {
            int(i) for i in self._rt_consumer_target_indices if int(i) in self._idx_be_flexloads
        }
        self._idx_be_storage = {
            int(i)
            for i in grid.generator.index
            if str(grid.generator.loc[i, "node"]).startswith("BE")
            and int(i) in self._idx_generatorsWithStorage
        }
        self._idx_be_target_storage = {
            int(i) for i in self._rt_storage_target_indices if int(i) in self._idx_be_storage
        }
        # Ramp-rate limits (MW per timestep). NaN means unconstrained.
        self._ramp_up_mw = grid.generator["ramp_up_mw"].values.copy() if "ramp_up_mw" in grid.generator.columns else None
        self._ramp_down_mw = grid.generator["ramp_down_mw"].values.copy() if "ramp_down_mw" in grid.generator.columns else None
        # If True for a generator, the ramp constraint is released at the start of every 24-hour
        # block (timestep % 24 == 0).  This models plant types (e.g. nuclear) whose output level
        # is decided day-ahead but is held constant throughout the day.
        # NOTE: Assumes simulations start at hour 0 (midnight).  Timestep indices from
        # continue_from_last runs preserve the original numbering so midnight detection remains valid.
        if "ramp_daily_reset" in grid.generator.columns:
            self._ramp_daily_reset = grid.generator["ramp_daily_reset"].fillna(False).astype(bool).values
        else:
            self._ramp_daily_reset = None
        # Previous-timestep generation dispatch; NaN signals first timestep (no ramp constraint).
        self._gen_prev = np.full(len(grid.generator), np.nan)
        self._idx_branchesWithConstraints = grid.getIdxBranchesWithFlowConstraints()
        # Optional sparse foreign DA locks (parquet long format) injected by prepare-rt.
        self._foreign_gen_lock = None
        self._foreign_cons_lock = None
        self._be_border_ac_flow_lock = None
        self._be_border_dc_flow_lock = None
        self._be_border_ac_flow_lb = None
        self._be_border_ac_flow_ub = None
        self._be_border_dc_flow_lb = None
        self._be_border_dc_flow_ub = None
        self._idx_be_border_ac = {
            int(idx)
            for idx, row in grid.branch.iterrows()
            if (
                str(row.get("node_from", "")).startswith("BE")
                and not str(row.get("node_to", "")).startswith("BE")
            )
            or (
                str(row.get("node_to", "")).startswith("BE")
                and not str(row.get("node_from", "")).startswith("BE")
            )
        }
        self._idx_be_border_dc = {
            int(idx)
            for idx, row in grid.dcbranch.iterrows()
            if (
                str(row.get("node_from", "")).startswith("BE")
                and not str(row.get("node_to", "")).startswith("BE")
            )
            or (
                str(row.get("node_to", "")).startswith("BE")
                and not str(row.get("node_from", "")).startswith("BE")
            )
        }
        self._be_border_ac_sign = {
            int(idx): (
                1.0 if str(grid.branch.loc[idx, "node_from"]).startswith("BE") else -1.0
            )
            for idx in self._idx_be_border_ac
        }
        self._be_border_dc_sign = {
            int(idx): (
                1.0 if str(grid.dcbranch.loc[idx, "node_from"]).startswith("BE") else -1.0
            )
            for idx in self._idx_be_border_dc
        }
        self._default_ac_flow_bounds = {}
        self._default_dc_flow_bounds = {}
        for b in grid.branch.index:
            cap = pd.to_numeric(grid.branch.loc[b, "capacity"], errors="coerce")
            if np.isfinite(cap):
                self._default_ac_flow_bounds[int(b)] = (-float(cap), float(cap))
            else:
                self._default_ac_flow_bounds[int(b)] = (None, None)
        for b in grid.dcbranch.index:
            cap = pd.to_numeric(grid.dcbranch.loc[b, "capacity"], errors="coerce")
            if np.isfinite(cap):
                self._default_dc_flow_bounds[int(b)] = (-float(cap), float(cap))
            else:
                self._default_dc_flow_bounds[int(b)] = (None, None)
        # Optional inter-area NTC constraints
        self._inter_area_ntc = grid.inter_area_ntc
        self._load_optional_foreign_da_locks(grid)
        # self._fancy_progressbar = False

        # Initial values of marginal costs, storage and storage values
        _ini_fill_scale = float(max(0.0, min(1.0, storage_initial_fill_scale)))
        self._storage = (_ini_fill_scale * grid.generator["storage_ini"] * grid.generator["storage_cap"]).fillna(0)
        self._storage_flexload = (
            grid.consumer["flex_storagelevel_init"]
            * grid.consumer["flex_storage"]
            * grid.consumer["flex_fraction"]
            * grid.consumer["demand_avg"]
        ).fillna(0)
        self._energyspilled = grid.generator["storage_cap"].copy(deep=True)
        self._energyspilled[:] = 0

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

        # 3. Create pyomo model
        self._create_sets_and_parameters(grid)
        self._create_variables()
        self._create_objective(grid)
        self._powerbalance_rhs = self._get_powerbalance_rhs()
        # 3b. Constraints:
        self._create_constraint_powerflow_limit(grid)
        self._create_constraint_inter_area_ntc(grid)
        self._create_constraint_powerloss(grid, penalty_twoway_flow=penalty_twoway_flow)
        self._create_constraint_generator_output()
        self._create_constraint_rt_target_tracking()
        self._create_constraint_generator_pump(grid)
        self._create_constraint_load_flex(grid)
        self._create_constraint_rt_pump_target_tracking()
        self._create_constraint_rt_flexload_target_tracking()
        self._create_constraint_rt_storage_target_tracking()
        self._create_constraint_rt_io_target_tracking()
        self._create_constraint_rt_residual_caps()
        self._create_constraint_rt_source_directional_caps()
        self._create_constraint_rt_io_source_directional_caps()
        self._create_constraint_powerbalance(grid)
        self._create_constraint_powerflow_equation(grid)

    def _load_optional_foreign_da_locks(self, grid):
        """Load optional foreign DA lock tables from parquet artifacts.

        Expected long format:
        - generators: timestep, indx, output
        - consumers: timestep, indx, demand
        """
        gen_path = getattr(grid, "foreign_gen_lock_parquet", "")
        con_path = getattr(grid, "foreign_consumer_lock_parquet", "")
        ac_border_path = getattr(grid, "be_border_ac_flow_lock_parquet", "")
        dc_border_path = getattr(grid, "be_border_dc_flow_lock_parquet", "")
        ac_border_bounds_path = getattr(grid, "be_border_ac_flow_bounds_parquet", "")
        dc_border_bounds_path = getattr(grid, "be_border_dc_flow_bounds_parquet", "")

        if gen_path:
            p = Path(str(gen_path))
            if p.exists():
                try:
                    gdf = pd.read_parquet(p)
                    gdf["timestep"] = pd.to_numeric(gdf["timestep"], errors="coerce").fillna(-1).astype(int)
                    gdf["indx"] = pd.to_numeric(gdf["indx"], errors="coerce").fillna(-1).astype(int)
                    gdf["output"] = pd.to_numeric(gdf["output"], errors="coerce").fillna(0.0)
                    self._foreign_gen_lock = gdf.set_index(["timestep", "indx"])["output"]
                except Exception as exc:
                    warnings.warn(f"Failed reading foreign generator DA lock parquet '{p}': {exc}", UserWarning)

        if con_path:
            p = Path(str(con_path))
            if p.exists():
                try:
                    cdf = pd.read_parquet(p)
                    cdf["timestep"] = pd.to_numeric(cdf["timestep"], errors="coerce").fillna(-1).astype(int)
                    cdf["indx"] = pd.to_numeric(cdf["indx"], errors="coerce").fillna(-1).astype(int)
                    cdf["demand"] = pd.to_numeric(cdf["demand"], errors="coerce").fillna(0.0)
                    self._foreign_cons_lock = cdf.set_index(["timestep", "indx"])["demand"]
                except Exception as exc:
                    warnings.warn(f"Failed reading foreign consumer DA lock parquet '{p}': {exc}", UserWarning)

        if ac_border_path:
            p = Path(str(ac_border_path))
            if p.exists():
                try:
                    adf = pd.read_parquet(p)
                    adf["timestep"] = pd.to_numeric(adf["timestep"], errors="coerce").fillna(-1).astype(int)
                    adf["indx"] = pd.to_numeric(adf["indx"], errors="coerce").fillna(-1).astype(int)
                    adf["flow"] = pd.to_numeric(adf["flow"], errors="coerce").fillna(0.0)
                    self._be_border_ac_flow_lock = adf.set_index(["timestep", "indx"])["flow"]
                except Exception as exc:
                    warnings.warn(f"Failed reading BE-border AC flow lock parquet '{p}': {exc}", UserWarning)

        if dc_border_path:
            p = Path(str(dc_border_path))
            if p.exists():
                try:
                    ddf = pd.read_parquet(p)
                    ddf["timestep"] = pd.to_numeric(ddf["timestep"], errors="coerce").fillna(-1).astype(int)
                    ddf["indx"] = pd.to_numeric(ddf["indx"], errors="coerce").fillna(-1).astype(int)
                    ddf["flow"] = pd.to_numeric(ddf["flow"], errors="coerce").fillna(0.0)
                    self._be_border_dc_flow_lock = ddf.set_index(["timestep", "indx"])["flow"]
                except Exception as exc:
                    warnings.warn(f"Failed reading BE-border DC flow lock parquet '{p}': {exc}", UserWarning)

        if ac_border_bounds_path:
            p = Path(str(ac_border_bounds_path))
            if p.exists():
                try:
                    adf = pd.read_parquet(p)
                    adf["timestep"] = pd.to_numeric(adf["timestep"], errors="coerce").fillna(-1).astype(int)
                    adf["indx"] = pd.to_numeric(adf["indx"], errors="coerce").fillna(-1).astype(int)
                    adf["flow_lb"] = pd.to_numeric(adf["flow_lb"], errors="coerce")
                    adf["flow_ub"] = pd.to_numeric(adf["flow_ub"], errors="coerce")
                    adf_idx = adf.set_index(["timestep", "indx"])
                    self._be_border_ac_flow_lb = adf_idx["flow_lb"]
                    self._be_border_ac_flow_ub = adf_idx["flow_ub"]
                except Exception as exc:
                    warnings.warn(f"Failed reading BE-border AC flow bounds parquet '{p}': {exc}", UserWarning)

        if dc_border_bounds_path:
            p = Path(str(dc_border_bounds_path))
            if p.exists():
                try:
                    ddf = pd.read_parquet(p)
                    ddf["timestep"] = pd.to_numeric(ddf["timestep"], errors="coerce").fillna(-1).astype(int)
                    ddf["indx"] = pd.to_numeric(ddf["indx"], errors="coerce").fillna(-1).astype(int)
                    ddf["flow_lb"] = pd.to_numeric(ddf["flow_lb"], errors="coerce")
                    ddf["flow_ub"] = pd.to_numeric(ddf["flow_ub"], errors="coerce")
                    ddf_idx = ddf.set_index(["timestep", "indx"])
                    self._be_border_dc_flow_lb = ddf_idx["flow_lb"]
                    self._be_border_dc_flow_ub = ddf_idx["flow_ub"]
                except Exception as exc:
                    warnings.warn(f"Failed reading BE-border DC flow bounds parquet '{p}': {exc}", UserWarning)

    def _create_sets_and_parameters(self, grid_data):
        """Create pyomo model sets"""
        self.s_node = pyo.Set(ordered=True, initialize=grid_data.node["id"].tolist())
        self.s_branch_ac = pyo.Set(ordered=True, initialize=grid_data.branch.index.tolist())
        self.s_branch_dc = pyo.Set(ordered=True, initialize=grid_data.dcbranch.index.tolist())
        self.s_gen = pyo.Set(ordered=True, initialize=grid_data.generator.index.tolist())
        self.s_gen_pump = pyo.Set(ordered=True, initialize=grid_data.getIdxGeneratorsWithPumping())
        self.s_gen_storage = pyo.Set(ordered=True, initialize=grid_data.getIdxGeneratorsWithStorage())
        self.s_load = pyo.Set(ordered=True, initialize=grid_data.consumer.index.tolist())
        self.s_load_flex = pyo.Set(ordered=True, initialize=grid_data.getIdxConsumersWithFlexibleLoad())
        self.s_area = pyo.Set(ordered=True, initialize=grid_data.getAllAreas())

        # Mutable parameters
        # Quantities that change from timestep to the next:
        self.p_gen_pmin = pyo.Param(
            self.s_gen,
            within=pyo.Reals,
            default=0,
            mutable=True,
            initialize=grid_data.generator["pmin"].values,
        )
        self.p_gen_pmax = pyo.Param(
            self.s_gen,
            within=pyo.Reals,
            default=0,
            mutable=True,
            initialize=grid_data.generator["pmax"].values,
        )
        self.p_gen_cost = pyo.Param(
            self.s_gen,
            within=pyo.Reals,
            default=0,
            mutable=True,
            initialize=grid_data.generator["fuelcost"].values,
        )
        self.p_genpump_cost = pyo.Param(self.s_gen, within=pyo.Reals, default=0, mutable=True)
        # Optional per-generator curtailment cost. Defaults to zero = curtailment is free.
        self.p_curtail_cost = pyo.Param(self.s_gen, within=pyo.NonNegativeReals, default=0, mutable=True)
        self.p_pump_pmax = pyo.Param(
            self.s_gen_pump,
            within=pyo.NonNegativeReals,
            mutable=True,
            initialize={g: grid_data.generator.loc[g, "pump_cap"] for g in self.s_gen_pump},
        )
        self.p_rt_target = pyo.Param(self.s_gen, within=pyo.Reals, default=0, mutable=True)
        self.p_rt_target_active = pyo.Param(self.s_gen, within=pyo.Binary, default=0, mutable=True)
        self.p_rt_foreign_da_lock_penalty_gen = pyo.Param(self.s_gen, within=pyo.NonNegativeReals, default=0, mutable=True)
        self.p_demand = pyo.Param(self.s_load, within=pyo.Reals, default=0, mutable=True)
        self.p_rt_pump_target = pyo.Param(self.s_gen_pump, within=pyo.NonNegativeReals, default=0, mutable=True)
        self.p_rt_foreign_da_lock_penalty_pump = pyo.Param(self.s_gen_pump, within=pyo.NonNegativeReals, default=0, mutable=True)
        # Consumer (flexible load) RT DA target for DA-target deviation penalties
        self.p_rt_flexload_target = pyo.Param(self.s_load_flex, within=pyo.NonNegativeReals, default=0, mutable=True)
        self.p_rt_flexload_target_active = pyo.Param(self.s_load_flex, within=pyo.Binary, default=0, mutable=True)
        self.p_rt_foreign_da_lock_penalty_flex = pyo.Param(self.s_load_flex, within=pyo.NonNegativeReals, default=0, mutable=True)
        self.p_rt_storage_target = pyo.Param(self.s_gen_storage, within=pyo.NonNegativeReals, default=0, mutable=True)
        self.p_rt_storage_balance_rhs = pyo.Param(self.s_gen_storage, within=pyo.NonNegativeReals, default=0, mutable=True)
        self.p_rt_foreign_da_lock_penalty_storage = pyo.Param(self.s_gen_storage, within=pyo.NonNegativeReals, default=0, mutable=True)
        self.p_rt_residual_pos_cap = pyo.Param(within=pyo.NonNegativeReals, default=0, mutable=True)
        self.p_rt_residual_neg_cap = pyo.Param(within=pyo.NonNegativeReals, default=0, mutable=True)
        self.p_rt_io_target_ac = pyo.Param(self.s_branch_ac, within=pyo.Reals, default=0, mutable=True)
        self.p_rt_io_target_dc = pyo.Param(self.s_branch_dc, within=pyo.Reals, default=0, mutable=True)
        self.p_rt_io_target_active_ac = pyo.Param(self.s_branch_ac, within=pyo.Binary, default=0, mutable=True)
        self.p_rt_io_target_active_dc = pyo.Param(self.s_branch_dc, within=pyo.Binary, default=0, mutable=True)
        self.p_loadflex_cost = pyo.Param(
            self.s_load_flex,
            within=pyo.Reals,
            default=0,
            mutable=True,
            # initialize=grid_data.consumer.loc[self.s_load_flex, "flex_basevalue"].values,
        )
        if self._lossmethod == 2:
            # for storing power losses until next timestep
            self.p_branch_ac_power_loss = pyo.Param(self.s_branch_ac, within=pyo.Reals, default=0, mutable=True)
            self.p_branch_dc_power_loss = pyo.Param(self.s_branch_dc, within=pyo.Reals, default=0, mutable=True)
        elif self._lossmethod == 1:
            # for storing power flow from previous timestep
            self.p_branch_ac_powerflow12 = pyo.Param(self.s_branch_ac, within=pyo.Reals, initialize=0, mutable=True)
            self.p_branch_ac_powerflow21 = pyo.Param(self.s_branch_ac, within=pyo.Reals, initialize=0, mutable=True)
            self.p_branch_dc_powerflow12 = pyo.Param(self.s_branch_dc, within=pyo.Reals, initialize=0, mutable=True)
            self.p_branch_dc_powerflow21 = pyo.Param(self.s_branch_dc, within=pyo.Reals, initialize=0, mutable=True)

    def _create_variables(self):
        """Create pyomo model variables"""
        self.varAcBranchFlow = pyo.Var(self.s_branch_ac, within=pyo.Reals)
        self.varDcBranchFlow = pyo.Var(self.s_branch_dc, within=pyo.Reals)
        if self._lossmethod == 1:

            def maxflow(model, j):
                return (0, model._grid.branch.loc[j, "capacity"])

            def maxflow_dc(model, j):
                return (0, model._grid.dcbranch.loc[j, "capacity"])

            # Ref issue: https://github.com/powergama/powergama/issues/29
            # Adding bounds on 12 and 21 flows reduces the problem with simultaneously large 12 and 21 flows
            # that are not physical (but hard to avoid in circumstances when branch loss is actually beneficial
            # for the optimisation). However, it doesn't eliminate the problem.
            self.varAcBranchFlow12 = pyo.Var(self.s_branch_ac, within=pyo.NonNegativeReals, bounds=maxflow)
            self.varAcBranchFlow21 = pyo.Var(self.s_branch_ac, within=pyo.NonNegativeReals, bounds=maxflow)
            self.varDcBranchFlow12 = pyo.Var(self.s_branch_dc, within=pyo.NonNegativeReals, bounds=maxflow_dc)
            self.varDcBranchFlow21 = pyo.Var(self.s_branch_dc, within=pyo.NonNegativeReals, bounds=maxflow_dc)
            self.varLossAc12 = pyo.Var(self.s_branch_ac, within=pyo.NonNegativeReals)
            self.varLossAc21 = pyo.Var(self.s_branch_ac, within=pyo.NonNegativeReals)
            self.varLossDc12 = pyo.Var(self.s_branch_dc, within=pyo.NonNegativeReals)
            self.varLossDc21 = pyo.Var(self.s_branch_dc, within=pyo.NonNegativeReals)
        self.varGeneration = pyo.Var(self.s_gen, within=pyo.NonNegativeReals)
        self.varPump = pyo.Var(self.s_gen_pump, within=pyo.NonNegativeReals)
        self.varCurtailment = pyo.Var(self.s_gen, within=pyo.NonNegativeReals)
        self.varRtTargetDevPos = pyo.Var(self.s_gen, within=pyo.NonNegativeReals)
        self.varRtTargetDevNeg = pyo.Var(self.s_gen, within=pyo.NonNegativeReals)
        self.varRtSourceGenPosExceed = pyo.Var(self.s_gen, within=pyo.NonNegativeReals)
        self.varRtSourceGenNegExceed = pyo.Var(self.s_gen, within=pyo.NonNegativeReals)
        self.varFlexLoad = pyo.Var(self.s_load_flex, within=pyo.NonNegativeReals)
        self.varRtPumpTargetDevPos = pyo.Var(self.s_gen_pump, within=pyo.NonNegativeReals)
        self.varRtPumpTargetDevNeg = pyo.Var(self.s_gen_pump, within=pyo.NonNegativeReals)
        self.varRtSourcePumpPosExceed = pyo.Var(self.s_gen_pump, within=pyo.NonNegativeReals)
        self.varRtSourcePumpNegExceed = pyo.Var(self.s_gen_pump, within=pyo.NonNegativeReals)
        self.varRtResidualExceedPos = pyo.Var(within=pyo.NonNegativeReals)
        self.varRtResidualExceedNeg = pyo.Var(within=pyo.NonNegativeReals)
        self.varLoadShed = pyo.Var(self.s_load, within=pyo.NonNegativeReals)
        self.varRtSourceLoadShedExceed = pyo.Var(self.s_load, within=pyo.NonNegativeReals)
        self.varRtSourceFlexLoadExceed = pyo.Var(self.s_load_flex, within=pyo.NonNegativeReals)
        # Flexible load DA target tracking deviations (soft penalty)
        self.varRtFlexLoadTargetDevPos = pyo.Var(self.s_load_flex, within=pyo.NonNegativeReals)
        self.varRtFlexLoadTargetDevNeg = pyo.Var(self.s_load_flex, within=pyo.NonNegativeReals)
        self.varRtStorageTargetDevPos = pyo.Var(self.s_gen_storage, within=pyo.NonNegativeReals)
        self.varRtStorageTargetDevNeg = pyo.Var(self.s_gen_storage, within=pyo.NonNegativeReals)
        self.varRtIoTargetDevPosAc = pyo.Var(self.s_branch_ac, within=pyo.NonNegativeReals)
        self.varRtIoTargetDevNegAc = pyo.Var(self.s_branch_ac, within=pyo.NonNegativeReals)
        self.varRtIoTargetDevPosDc = pyo.Var(self.s_branch_dc, within=pyo.NonNegativeReals)
        self.varRtIoTargetDevNegDc = pyo.Var(self.s_branch_dc, within=pyo.NonNegativeReals)
        self.varRtSourceIoAcPosExceed = pyo.Var(self.s_branch_ac, within=pyo.NonNegativeReals)
        self.varRtSourceIoAcNegExceed = pyo.Var(self.s_branch_ac, within=pyo.NonNegativeReals)
        self.varRtSourceIoDcPosExceed = pyo.Var(self.s_branch_dc, within=pyo.NonNegativeReals)
        self.varRtSourceIoDcNegExceed = pyo.Var(self.s_branch_dc, within=pyo.NonNegativeReals)
        self.varVoltageAngle = pyo.Var(self.s_node, within=pyo.Reals, initialize=0.0)

    def _create_constraint_powerflow_limit(self, grid_data):
        """Constraint: Power flow limit"""

        def maxflowAc_rule(model, j):
            cap = grid_data.branch.loc[j, "capacity"]
            if not np.isinf(cap):
                expr = pyo.inequality(-cap, model.varAcBranchFlow[j], cap)
            else:
                expr = pyo.Constraint.Skip
            return expr

        def maxflowDc_rule(model, j):
            cap = grid_data.dcbranch.loc[j, "capacity"]
            if not np.isinf(cap):
                expr = pyo.inequality(-cap, model.varDcBranchFlow[j], cap)
            else:
                expr = pyo.Constraint.Skip
            return expr

        self.cMaxFlowAc = pyo.Constraint(self.s_branch_ac, rule=maxflowAc_rule)
        self.cMaxFlowDc = pyo.Constraint(self.s_branch_dc, rule=maxflowDc_rule)

    def _create_constraint_inter_area_ntc(self, grid_data):
        """Constraint: Directional NTC limits between areas.

        For each (area_from, area_to) pair, limit the net transfer
        from area_from -> area_to:
            -ntc_backward <= net_transfer <= ntc_forward
        """

        if self._inter_area_ntc is None or self._inter_area_ntc.shape[0] == 0:
            return

        self.s_inter_area_ntc = pyo.RangeSet(0, len(self._inter_area_ntc) - 1)
        self._inter_area_ntc_rows = {}

        for cidx, (_, row) in enumerate(self._inter_area_ntc.iterrows()):
            area_from = row["area_from"]
            area_to = row["area_to"]
            ntc_forward = float(row["ntc_forward"])
            ntc_backward = float(row["ntc_backward"])

            # Get branches between the two areas
            br_ac = grid_data.getInterAreaBranches(area_from=area_from, area_to=area_to, acdc="ac")
            br_dc = grid_data.getInterAreaBranches(area_from=area_from, area_to=area_to, acdc="dc")

            self._inter_area_ntc_rows[cidx] = {
                "ac_pos": list(br_ac["branches_pos"]),
                "ac_neg": list(br_ac["branches_neg"]),
                "dc_pos": list(br_dc["branches_pos"]),
                "dc_neg": list(br_dc["branches_neg"]),
                "ntc_forward": ntc_forward,
                "ntc_backward": ntc_backward,
            }

        def ntc_forward_rule(model, cidx):
            row = self._inter_area_ntc_rows[int(cidx)]
            cap = row["ntc_forward"]
            if not np.isfinite(cap):
                return pyo.Constraint.Skip
            if not (row["ac_pos"] or row["ac_neg"] or row["dc_pos"] or row["dc_neg"]):
                return pyo.Constraint.Skip
            net_transfer = (
                sum(model.varAcBranchFlow[b] for b in row["ac_pos"])
                - sum(model.varAcBranchFlow[b] for b in row["ac_neg"])
                + sum(model.varDcBranchFlow[b] for b in row["dc_pos"])
                - sum(model.varDcBranchFlow[b] for b in row["dc_neg"])
            )
            return net_transfer <= cap

        def ntc_backward_rule(model, cidx):
            row = self._inter_area_ntc_rows[int(cidx)]
            cap = row["ntc_backward"]
            if not np.isfinite(cap):
                return pyo.Constraint.Skip
            if not (row["ac_pos"] or row["ac_neg"] or row["dc_pos"] or row["dc_neg"]):
                return pyo.Constraint.Skip
            net_transfer = (
                sum(model.varAcBranchFlow[b] for b in row["ac_pos"])
                - sum(model.varAcBranchFlow[b] for b in row["ac_neg"])
                + sum(model.varDcBranchFlow[b] for b in row["dc_pos"])
                - sum(model.varDcBranchFlow[b] for b in row["dc_neg"])
            )
            return net_transfer >= -cap

        self.cInterAreaNtcForward = pyo.Constraint(self.s_inter_area_ntc, rule=ntc_forward_rule)
        self.cInterAreaNtcBackward = pyo.Constraint(self.s_inter_area_ntc, rule=ntc_backward_rule)

    def _powerloss_rules1(self, grid_data, penalty_twoway_flow=0):
        """Power loss proportional to flow, proportionality factor given by previous timestep

        P_loss = alpha P
        alpha = P_loss0/P0 (straight line from origo to operating point)
                r_pu = self._grid.dcbranch.loc[b, "resistance"]
                p_pu = self.varDcBranchFlow[b] / const.baseMVA
                loss_pu = r_pu * p_pu**2
                lossMVA = loss_pu * const.baseMVA * dclossmultiplier

        NOTE:
        With losses included, flow is split, flow=flow12-flow21. But there is no constraint
        saying flow can be only one direction. In some cases, the optimisation may find it
        beneficial to have increases losses by having large values for both flow12 and flow21
        (with flow still below capacity). High losses on a branch may be beneficial by
        allowing more flow on another line (under certain combinations of branch impedances
        and capacities.)
        The penalty_twoway_flow parameter may be used to discourange simultaneous (unphysical)
        twoway flow by adding a penalty in the objective function. This is not generally advices
        as it may have other unwanted effects (not sufficiently tested.)
        """

        def make_lossAc_rule12(br):
            def rule(model, j):
                flow_abs = model.p_branch_ac_powerflow12[j] + model.p_branch_ac_powerflow21[j]
                alpha = br.loc[j, "resistance"] * flow_abs / const.baseMVA
                expr = model.varLossAc12[j] == alpha * model.varAcBranchFlow12[j]
                return expr

            self.cLossAc12 = pyo.Constraint(self.s_branch_ac, rule=rule)

        def make_lossAc_rule21(br):
            def rule(model, j):
                flow_abs = model.p_branch_ac_powerflow12[j] + model.p_branch_ac_powerflow21[j]
                alpha = br.loc[j, "resistance"] * flow_abs / const.baseMVA
                expr = model.varLossAc21[j] == alpha * model.varAcBranchFlow21[j]
                return expr

            self.cLossAc21 = pyo.Constraint(self.s_branch_ac, rule=rule)

        def make_lossDc_rule12(br):
            def rule(model, j):
                flow_abs = model.p_branch_dc_powerflow12[j] + model.p_branch_dc_powerflow21[j]
                alpha = br.loc[j, "resistance"] * flow_abs / const.baseMVA
                expr = model.varLossDc12[j] == alpha * model.varDcBranchFlow12[j]
                return expr

            self.cLossDc12 = pyo.Constraint(self.s_branch_dc, rule=rule)

        def make_lossDc_rule21(br):
            def rule(model, j):
                flow_abs = model.p_branch_dc_powerflow12[j] + model.p_branch_dc_powerflow21[j]
                alpha = br.loc[j, "resistance"] * flow_abs / const.baseMVA
                expr = model.varLossDc21[j] == alpha * model.varDcBranchFlow21[j]
                return expr

            self.cLossDc21 = pyo.Constraint(self.s_branch_dc, rule=rule)

        make_lossAc_rule12(grid_data.branch)
        make_lossAc_rule21(grid_data.branch)
        make_lossDc_rule12(grid_data.dcbranch)
        make_lossDc_rule21(grid_data.dcbranch)

        # Add penalty in objective function to discourage simultaneous flow in both directions
        # This is not particularly elegant and should not be normally used
        # REF: Issue https://github.com/powergama/powergama/issues/29
        if penalty_twoway_flow > 0:
            print("Adding a cost to penalise simultaneous branch flow in both directions - ")
            self.OBJ.expr += penalty_twoway_flow * sum(
                self.varAcBranchFlow12[i] + self.varAcBranchFlow21[i] for i in self.s_branch_ac
            )

    def _create_constraint_powerloss(self, grid_data, penalty_twoway_flow=0):
        """Constraint: flow = flow12-flow21 & powerloss"""
        if self._lossmethod == 1:

            def flowAc_rule(model, j):
                expr = model.varAcBranchFlow[j] == model.varAcBranchFlow12[j] - model.varAcBranchFlow21[j]
                return expr

            def flowDc_rule(model, j):
                expr = model.varDcBranchFlow[j] == model.varDcBranchFlow12[j] - model.varDcBranchFlow21[j]
                return expr

            self.cFlowAc = pyo.Constraint(self.s_branch_ac, rule=flowAc_rule)
            self.cFlowDc = pyo.Constraint(self.s_branch_dc, rule=flowDc_rule)

        # 1b Losses vs flow
        if self._lossmethod == 1:
            self._powerloss_rules1(grid_data, penalty_twoway_flow)

    def _create_constraint_generator_output(self):
        """Constraint: Generator output limit"""

        # Generator output constraint is not necessary, as lower and upper
        # bounds are set for each timestep in _update_progress. Should not
        # be specified as constraint with with pmax as limit, since e.g.
        # PV may have higher production than generator rating.

        # HGS: Doing it anyway, cf Espen Bødal and Martin Kristiansen
        # TODO: Check that there are no problems with this.

        def genMaxLimit_rule(model, i):
            return model.varGeneration[i] <= self.p_gen_pmax[i]

        def genMinLimit_rule(model, i):
            return model.varGeneration[i] >= self.p_gen_pmin[i]

        self.cGenMaxLimit = pyo.Constraint(self.s_gen, rule=genMaxLimit_rule)
        self.cGenMinLimit = pyo.Constraint(self.s_gen, rule=genMinLimit_rule)

    def _create_constraint_rt_target_tracking(self):
        """Constraint: RT DA target dispatch deviation accounting."""

        if not self._rt_target_tracking_active:
            return

        def rt_target_rule(model, i):
            if int(i) not in self._rt_target_gen_indices:
                return pyo.Constraint.Skip
            return model.varGeneration[i] - self.p_rt_target[i] == model.varRtTargetDevPos[i] - model.varRtTargetDevNeg[i]

        self.cRtTargetTracking = pyo.Constraint(self.s_gen, rule=rt_target_rule)

    def _create_constraint_rt_pump_target_tracking(self):
        """Constraint: RT DA pump (charging) target deviation accounting.

        Mirrors _create_constraint_rt_target_tracking but for varPump (charging side).
        Without this, storage can freely charge in RT even when DA did not plan for it,
        causing independent energy accumulation and subsequent wind curtailment cascades.
        """
        if not self._rt_target_tracking_active:
            return

        def rt_pump_target_rule(model, i):
            if int(i) not in self._rt_pump_target_gen_indices:
                return pyo.Constraint.Skip
            return model.varPump[i] - self.p_rt_pump_target[i] == model.varRtPumpTargetDevPos[i] - model.varRtPumpTargetDevNeg[i]

        self.cRtPumpTargetTracking = pyo.Constraint(self.s_gen_pump, rule=rt_pump_target_rule)

    def _create_constraint_rt_flexload_target_tracking(self):
        """Constraint: RT DA flexible load (consumer) target deviation accounting.

        Flexible load can deviate from DA when other balancing channels are exhausted,
        but tracked deviations incur soft penalties to keep DA as the default solution.
        """
        if not self._rt_target_tracking_active:
            return

        def rt_flexload_target_rule(model, j):
            if int(j) not in self._rt_consumer_target_indices:
                return pyo.Constraint.Skip
            return model.varFlexLoad[j] - self.p_rt_flexload_target[j] == model.varRtFlexLoadTargetDevPos[j] - model.varRtFlexLoadTargetDevNeg[j]

        self.cRtFlexLoadTargetTracking = pyo.Constraint(self.s_load_flex, rule=rt_flexload_target_rule)

    def _create_constraint_rt_storage_target_tracking(self):
        """Constraint: RT DA storage filling trajectory deviation accounting."""

        if not self._rt_target_tracking_active:
            return

        def rt_storage_target_rule(model, i):
            if int(i) not in self._rt_storage_target_indices:
                return pyo.Constraint.Skip
            lhs = self.p_rt_storage_balance_rhs[i] - (self.timeDelta * model.varGeneration[i])
            if int(i) in self.s_gen_pump:
                pump_eff = float(self._grid.generator.loc[i, "pump_efficiency"])
                lhs += self.timeDelta * pump_eff * model.varPump[i]
            return lhs - self.p_rt_storage_target[i] == model.varRtStorageTargetDevPos[i] - model.varRtStorageTargetDevNeg[i]

        self.cRtStorageTargetTracking = pyo.Constraint(self.s_gen_storage, rule=rt_storage_target_rule)

    def _create_constraint_rt_io_target_tracking(self):
        """Constraint: DA BE border flow target deviation accounting for AC/DC branches."""

        if self._rt_residual_penalty <= 0:
            return

        def _ac_rule(model, b):
            if int(b) not in self._idx_be_border_ac:
                return pyo.Constraint.Skip
            return (
                self.p_rt_io_target_active_ac[b] * (model.varAcBranchFlow[b] - self.p_rt_io_target_ac[b])
                == model.varRtIoTargetDevPosAc[b] - model.varRtIoTargetDevNegAc[b]
            )

        def _dc_rule(model, b):
            if int(b) not in self._idx_be_border_dc:
                return pyo.Constraint.Skip
            return (
                self.p_rt_io_target_active_dc[b] * (model.varDcBranchFlow[b] - self.p_rt_io_target_dc[b])
                == model.varRtIoTargetDevPosDc[b] - model.varRtIoTargetDevNegDc[b]
            )

        self.cRtIoTargetTrackingAc = pyo.Constraint(self.s_branch_ac, rule=_ac_rule)
        self.cRtIoTargetTrackingDc = pyo.Constraint(self.s_branch_dc, rule=_dc_rule)

    def _create_constraint_rt_residual_caps(self):
        """Constraint: cap BE balancing action by directional residual need.

        Under BE residual-penalty mode, the solver may deviate from DA targets for balancing.
        This constraint limits aggregate BE balancing action to the residual magnitude
        per timestep (up/down), while allowing explicit overflow variables that are
        penalized in the objective.
        
        Directional semantics (residual sign determines allowed correction):
        - Positive residual (BE surplus): allow gen↑, pump↓ (help reduce), penalize others
        - Negative residual (BE deficit): allow gen↓, pump↑ (help inject), penalize others
        """

        if not self._rt_residual_cap_active:
            return

        def _be_balancing_deviation(model):
            dev_gen = sum(
                model.varGeneration[i] - self.p_rt_target[i]
                for i in self.s_gen
                if int(i) in self._idx_be_target_generators
            )
            dev_pump = sum(
                model.varPump[i] - self.p_rt_pump_target[i]
                for i in self.s_gen_pump
                if int(i) in self._idx_be_target_pumps
            )
            dev_loadshed = sum(
                model.varLoadShed[j]
                for j in self.s_load
                if int(j) in self._idx_be_loads
            )
            dev_flex = sum(
                model.varFlexLoad[j] - self.p_rt_flexload_target[j]
                for j in self.s_load_flex
                if int(j) in self._idx_be_target_flexloads
            )
            dev_io_ac = sum(
                self._be_border_ac_sign.get(int(b), 0.0)
                * self.p_rt_io_target_active_ac[b]
                * (model.varAcBranchFlow[b] - self.p_rt_io_target_ac[b])
                for b in self.s_branch_ac
                if int(b) in self._idx_be_border_ac
            )
            dev_io_dc = sum(
                self._be_border_dc_sign.get(int(b), 0.0)
                * self.p_rt_io_target_active_dc[b]
                * (model.varDcBranchFlow[b] - self.p_rt_io_target_dc[b])
                for b in self.s_branch_dc
                if int(b) in self._idx_be_border_dc
            )
            return dev_gen - dev_pump + dev_loadshed - dev_flex + dev_io_ac + dev_io_dc

        def _rt_residual_pos_cap_rule(model):
            return _be_balancing_deviation(model) <= self.p_rt_residual_pos_cap + model.varRtResidualExceedPos

        def _rt_residual_neg_cap_rule(model):
            return -_be_balancing_deviation(model) <= self.p_rt_residual_neg_cap + model.varRtResidualExceedNeg

        self.cRtResidualPosCap = pyo.Constraint(rule=_rt_residual_pos_cap_rule)
        self.cRtResidualNegCap = pyo.Constraint(rule=_rt_residual_neg_cap_rule)

    def _create_constraint_rt_source_directional_caps(self):
        """Constraint: cap each BE balancing source by the directional residual need.

        The residual sign determines which correction channels are meaningful. A zero
        directional cap soft-discourages the wrong-direction channel, while a positive
        cap allows that channel up to the residual magnitude before the penalty applies.
        """

        if not self._rt_residual_cap_active:
            return

        def _gen_pos_rule(model, i):
            if int(i) not in self._idx_be_target_generators:
                return pyo.Constraint.Skip
            return model.varRtTargetDevPos[i] <= self.p_rt_residual_pos_cap + model.varRtSourceGenPosExceed[i]

        def _gen_neg_rule(model, i):
            if int(i) not in self._idx_be_target_generators:
                return pyo.Constraint.Skip
            return model.varRtTargetDevNeg[i] <= self.p_rt_residual_neg_cap + model.varRtSourceGenNegExceed[i]

        def _pump_pos_rule(model, i):
            if int(i) not in self._idx_be_target_pumps:
                return pyo.Constraint.Skip
            return model.varRtPumpTargetDevNeg[i] <= self.p_rt_residual_pos_cap + model.varRtSourcePumpPosExceed[i]

        def _pump_neg_rule(model, i):
            if int(i) not in self._idx_be_target_pumps:
                return pyo.Constraint.Skip
            return model.varRtPumpTargetDevPos[i] <= self.p_rt_residual_neg_cap + model.varRtSourcePumpNegExceed[i]

        def _loadshed_rule(model, j):
            if int(j) not in self._idx_be_loads:
                return pyo.Constraint.Skip
            return model.varLoadShed[j] <= self.p_rt_residual_pos_cap + model.varRtSourceLoadShedExceed[j]

        def _flex_rule(model, j):
            if int(j) not in self._idx_be_flexloads:
                return pyo.Constraint.Skip
            return model.varFlexLoad[j] <= self.p_rt_residual_neg_cap + model.varRtSourceFlexLoadExceed[j]

        self.cRtSourceGenPosCap = pyo.Constraint(self.s_gen, rule=_gen_pos_rule)
        self.cRtSourceGenNegCap = pyo.Constraint(self.s_gen, rule=_gen_neg_rule)
        self.cRtSourcePumpPosCap = pyo.Constraint(self.s_gen_pump, rule=_pump_pos_rule)
        self.cRtSourcePumpNegCap = pyo.Constraint(self.s_gen_pump, rule=_pump_neg_rule)
        self.cRtSourceLoadShedCap = pyo.Constraint(self.s_load, rule=_loadshed_rule)
        self.cRtSourceFlexLoadCap = pyo.Constraint(self.s_load_flex, rule=_flex_rule)

    def _create_constraint_rt_io_source_directional_caps(self):
        """Constraint: cap DA I/O deviation channels by directional residual need."""

        if not self._rt_residual_cap_active:
            return

        def _ac_pos_rule(model, b):
            if int(b) not in self._idx_be_border_ac:
                return pyo.Constraint.Skip
            return (
                    self.p_rt_io_target_active_ac[b] * model.varRtIoTargetDevPosAc[b]
                <= self.p_rt_residual_pos_cap + model.varRtSourceIoAcPosExceed[b]
            )

        def _ac_neg_rule(model, b):
            if int(b) not in self._idx_be_border_ac:
                return pyo.Constraint.Skip
            return (
                    self.p_rt_io_target_active_ac[b] * model.varRtIoTargetDevNegAc[b]
                <= self.p_rt_residual_neg_cap + model.varRtSourceIoAcNegExceed[b]
            )

        def _dc_pos_rule(model, b):
            if int(b) not in self._idx_be_border_dc:
                return pyo.Constraint.Skip
            return (
                    self.p_rt_io_target_active_dc[b] * model.varRtIoTargetDevPosDc[b]
                <= self.p_rt_residual_pos_cap + model.varRtSourceIoDcPosExceed[b]
            )

        def _dc_neg_rule(model, b):
            if int(b) not in self._idx_be_border_dc:
                return pyo.Constraint.Skip
            return (
                    self.p_rt_io_target_active_dc[b] * model.varRtIoTargetDevNegDc[b]
                <= self.p_rt_residual_neg_cap + model.varRtSourceIoDcNegExceed[b]
            )

        self.cRtSourceIoAcPosCap = pyo.Constraint(self.s_branch_ac, rule=_ac_pos_rule)
        self.cRtSourceIoAcNegCap = pyo.Constraint(self.s_branch_ac, rule=_ac_neg_rule)
        self.cRtSourceIoDcPosCap = pyo.Constraint(self.s_branch_dc, rule=_dc_pos_rule)
        self.cRtSourceIoDcNegCap = pyo.Constraint(self.s_branch_dc, rule=_dc_neg_rule)

    def _create_constraint_generator_pump(self, grid_data):
        """Constraint: Pump output limit (respects both hardware cap and remaining reservoir space)."""

        def pump_rule(model, g):
            expr = model.varPump[g] <= model.p_pump_pmax[g]
            return expr

        self.cPump = pyo.Constraint(self.s_gen_pump, rule=pump_rule)

    def _create_constraint_load_flex(self, grid_data):
        """Constraint: Flexible load limit"""

        def flexload_rule(model, i):
            flexLoadMax = (
                grid_data.consumer.loc[i, "demand_avg"]
                * grid_data.consumer.loc[i, "flex_fraction"]
                / grid_data.consumer.loc[i, "flex_on_off"]
            )
            expr = model.varFlexLoad[i] <= flexLoadMax
            return expr

        self.cFlexload = pyo.Constraint(self.s_load_flex, rule=flexload_rule)

    def _create_constraint_powerbalance(self, grid_data):
        """ConstraintPower balance (power flow equation)  (Pnode = B theta)"""

        def powerbalance_rule(model, n):
            lhs = 0
            for g in self._generators_at_node[n]:
                # this is a generator connected to node n
                lhs += model.varGeneration[g]
                if g in model.s_gen_pump:
                    lhs -= model.varPump[g]
            for lod in self._loads_at_node[n]:
                lhs -= self.p_demand[lod]
                lhs += model.varLoadShed[lod]
                if lod in model.s_load_flex:
                    lhs -= model.varFlexLoad[lod]
            for b in self._dcbranch_to_node[n]:
                lhs += model.varDcBranchFlow[b]
                if model._lossmethod == 1:
                    lhs -= model.varLossDc12[b]
                elif model._lossmethod == 2:
                    lhs -= model.p_branch_dc_power_loss[b] / 2
            for b in self._dcbranch_from_node[n]:
                lhs += -model.varDcBranchFlow[b]
                if model._lossmethod == 1:
                    lhs -= model.varLossDc21[b]
                elif model._lossmethod == 2:
                    lhs -= model.p_branch_dc_power_loss[b] / 2
            if self._lossmethod == 1:
                # we define flow12 as flow leaving node 1, and flow21 as flow leaving node 2
                # so subtract loss at "receiving" node but not at "sending" node
                # i.e. loss12 at to-node and loss21 at from-node
                for b in self._branch_to_node[n]:
                    lhs += -model.varLossAc12[b]
                for b in self._branch_from_node[n]:
                    lhs += -model.varLossAc21[b]
            elif self._lossmethod == 2:
                # add ac branch losses as load, equally split between sending and receiving node
                for b in self._branch_to_node[n]:
                    # positive sign for flow into node
                    lhs -= model.p_branch_ac_power_loss[b] / 2
                for b in self._branch_from_node[n]:
                    lhs -= model.p_branch_ac_power_loss[b] / 2

            lhs = lhs / const.baseMVA

            # self._powerbalance_rhs = self._get_powerbalance_rhs()
            rhs = self._powerbalance_rhs[n]

            expr = lhs == rhs
            # Skip constraint if it is trivial (otherwise run-time error)
            # TODO: Check if this is safe
            if (type(expr) is bool) and (expr is True):
                expr = pyo.Constraint.Skip
            return expr

        self.cPowerbalance = pyo.Constraint(self.s_node, rule=powerbalance_rule)

    def _create_constraint_powerflow_equation(self, grid_data):
        """Constraint: Power balance (power flow vs voltage angle)"""

        # 1.
        def flowangle_rule(model, b):
            lhs = model.varAcBranchFlow[b]
            lhs = lhs / const.baseMVA
            rhs = 0
            # TODO: This can surely be simplified:
            # node id's are strings, but Bbus and DA matrices need matrix indices (int)
            idx_branch = list(self.s_branch_ac).index(b)
            # idx_branch = grid_data.branch.index.get_loc(b)  # Check if this works
            for i in range(len(self._DA[idx_branch].indices)):
                idx_node2 = self._DA[idx_branch].indices[i]
                DA_element = self._DA[idx_branch].data[i]
                n2 = list(model.s_node)[idx_node2]  # list since pyomo set is 1-based (could probably use +1 instead)
                rhs += DA_element * model.varVoltageAngle[n2] * const.baseAngle
            expr = lhs == rhs
            return expr

        self.cFlowAngle = pyo.Constraint(self.s_branch_ac, rule=flowangle_rule)

        # 2. Reference voltag angle)
        def referenceNode_rule(model, n):
            if n in self.refnodes:
                expr = model.varVoltageAngle[n] == 0
            else:
                expr = pyo.Constraint.Skip
            return expr

        self.cReferenceNode = pyo.Constraint(self.s_node, rule=referenceNode_rule)

    def _create_objective(self, grid_data):
        """Create pyomo model objective function"""

        def cost_rule(model):
            """Operational costs: cost of gen, load shed and curtailment"""

            # Operational costs phase 1 (if stage2DeltaTime>0)
            cost = sum(model.varGeneration[i] * self.p_gen_cost[i] for i in model.s_gen)
            cost -= sum(model.varPump[i] * self.p_genpump_cost[i] for i in model.s_gen_pump)
            cost -= sum(model.varFlexLoad[i] * self.p_loadflex_cost[i] for i in model.s_load_flex)
            cost += sum(model.varLoadShed[i] * const.loadshedcost for i in model.s_load)
            cost += sum(model.varCurtailment[i] * self.p_curtail_cost[i] for i in model.s_gen)
            if self._rt_foreign_da_lock_penalty_coeff > 0:
                cost += sum(
                    self.p_rt_foreign_da_lock_penalty_gen[i]
                    * (model.varRtTargetDevPos[i] + model.varRtTargetDevNeg[i])
                    for i in model.s_gen
                    if int(i) not in self._idx_be_target_generators
                )
                cost += sum(
                    self.p_rt_foreign_da_lock_penalty_pump[i]
                    * (model.varRtPumpTargetDevPos[i] + model.varRtPumpTargetDevNeg[i])
                    for i in model.s_gen_pump
                    if int(i) not in self._idx_be_target_pumps
                )
                cost += sum(
                    self.p_rt_foreign_da_lock_penalty_flex[j]
                    * (model.varRtFlexLoadTargetDevPos[j] + model.varRtFlexLoadTargetDevNeg[j])
                    for j in model.s_load_flex
                    if int(j) not in self._idx_be_target_flexloads
                )
                cost += sum(
                    self.p_rt_foreign_da_lock_penalty_storage[i]
                    * (model.varRtStorageTargetDevPos[i] + model.varRtStorageTargetDevNeg[i])
                    for i in model.s_gen_storage
                    if int(i) not in self._idx_be_target_storage
                )

            if self._rt_residual_penalty > 0:
                cost += self._rt_residual_penalty * sum(
                    model.varRtSourceGenPosExceed[i] + model.varRtSourceGenNegExceed[i]
                    for i in model.s_gen
                    if int(i) in self._idx_be_target_generators
                )
                cost += self._rt_residual_penalty * sum(
                    model.varRtSourcePumpPosExceed[i] + model.varRtSourcePumpNegExceed[i]
                    for i in model.s_gen_pump
                    if int(i) in self._idx_be_target_pumps
                )
                cost += self._rt_residual_penalty * sum(
                    model.varRtSourceLoadShedExceed[i]
                    for i in model.s_load
                    if int(i) in self._idx_be_loads
                )
                cost += self._rt_residual_penalty * sum(
                    model.varRtSourceFlexLoadExceed[i]
                    for i in model.s_load_flex
                    if int(i) in self._idx_be_target_flexloads
                )
                cost += self._rt_residual_penalty * sum(
                    model.varRtSourceIoAcPosExceed[b] + model.varRtSourceIoAcNegExceed[b]
                    for b in model.s_branch_ac
                    if int(b) in self._idx_be_border_ac
                )
                cost += self._rt_residual_penalty * sum(
                    model.varRtSourceIoDcPosExceed[b] + model.varRtSourceIoDcNegExceed[b]
                    for b in model.s_branch_dc
                    if int(b) in self._idx_be_border_dc
                )
                cost += self._rt_residual_penalty * (
                    model.varRtResidualExceedPos + model.varRtResidualExceedNeg
                )
            return cost
        self.OBJ = pyo.Objective(rule=cost_rule, sense=pyo.minimize)

    def _get_powerbalance_rhs(self):
        """Get rhs expression in powerbalance constraint"""
        # This is taken out of the constraint creation function to speed up constraint creation
        rhs = dict()
        for n in self.s_node:
            rhs[n] = 0
            # node id's are strings, but Bbus and DA matrices need matrix indices (int)
            idx_node = list(self.s_node).index(n)
            for i in range(len(self._Bbus[idx_node].indices)):
                idx_node2 = self._Bbus[idx_node].indices[i]
                B_element = self._Bbus[idx_node].data[i]
                n2 = list(self.s_node)[idx_node2]  # list since pyomo set is 1-based (could probably use +1 instead)
                rhs[n] -= B_element * self.varVoltageAngle[n2] * const.baseAngle
        return rhs

    def _get_timesteps_to_solve(self, continue_from_last=False, results=None):
        numTimesteps = len(self._grid.timerange)
        time_steps = range(numTimesteps)
        if continue_from_last:
            timestep_last = results.get_last_timestep_in_results()
            # first step is last plus one
            time_steps = range(timestep_last + 1, numTimesteps)
            print(f"Continue from timestep {timestep_last}.")
        return time_steps

    def _build_day_hour_index(self, timesteps_to_solve):
        """Build mapping timestep -> (day_index, hour_in_day) for diagnostics and horizon grouping."""
        self._timestep_day_hour = {}
        for offset, ts in enumerate(timesteps_to_solve):
            day = int(offset) // int(self._objective_day_horizon_hours)
            hour = int(offset) % int(self._objective_day_horizon_hours)
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
        rates, BE border locks) is the same physics as the hourly model,
        now indexed by (entity, hour) instead of (entity) alone.

        Returns (model, pre_data_dict).
        """
        H = len(day_timesteps)
        grid = self._grid
        dt = float(self.timeDelta)
        node_list = list(self.s_node)
        branch_ac_list = list(self.s_branch_ac)
        gen_storage_set = frozenset(self._idx_generatorsWithStorage)
        gen_pump_set = frozenset(self._idx_generatorsWithPumping)
        flex_set = frozenset(self._idx_consumersWithFlexLoad)

        # ── pre-compute storage capacity and pump parameters ──────────
        storage_ini = {int(i): float(self._storage[i]) for i in self._idx_generatorsWithStorage}
        storage_cap = {int(i): max(0.0, float(grid.generator.loc[i, "storage_cap"]))
                       for i in self._idx_generatorsWithStorage}
        pump_eff_g = {int(i): float(grid.generator.loc[i, "pump_efficiency"])
                      for i in self._idx_generatorsWithPumping}
        pump_cap_raw = {int(i): float(grid.generator.loc[i, "pump_cap"])
                        for i in self._idx_generatorsWithPumping}

        # ── pre-compute storval for each storage gen (BOD filling level) ─
        gen_cost_h: dict = {}   # (h, i) → total cost including storval
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
        for i in self.s_gen:
            if i not in gen_storage_set:
                base_cost = float(pyo.value(self.p_gen_cost[i]))
                for hi in range(H):
                    gen_cost_h[(hi, i)] = base_cost

        curtail_cost = {int(i): float(pyo.value(self.p_curtail_cost[i])) for i in self.s_gen}

        # ── pre-compute inflow / pmax / pmin per (hour, gen) ─────────
        P_max_base = grid.generator["pmax"]
        P_min_base = grid.generator["pmin"]
        inflow_h: dict = {}      # (hi, i) → inflow MW
        capacity_h: dict = {}   # (hi, i) → installed pmax × availability
        pmin_h: dict = {}        # (hi, i) → pmin lower bound

        for hi, ts in enumerate(day_timesteps):
            for i in self.s_gen:
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
            for j in self.s_load:
                avg = float(grid.consumer.loc[j, "demand_avg"]) * (
                    1 - float(grid.consumer.loc[j, "flex_fraction"])
                )
                prof = grid.consumer.loc[j, "demand_ref"]
                d_now = float(grid.profiles.loc[ts, prof]) * avg
                if self._foreign_cons_lock is not None:
                    key = (int(ts), int(j))
                    if key in self._foreign_cons_lock.index:
                        locked = float(self._foreign_cons_lock.loc[key])
                        if np.isfinite(locked):
                            d_now = locked
                demand_h[(hi, j)] = float(d_now)

        flexload_cost_h: dict = {}
        for hi in range(H):
            for j in flex_set:
                flexload_cost_h[(hi, j)] = float(pyo.value(self.p_loadflex_cost[j]))

        # ── pre-compute per-hour branch flow bounds (including DA locks) ─
        ac_lb: dict = {}   # (hi, b) → lb or None
        ac_ub: dict = {}   # (hi, b) → ub or None
        dc_lb: dict = {}
        dc_ub: dict = {}
        for hi, ts in enumerate(day_timesteps):
            for b in self.s_branch_ac:
                key = (int(ts), int(b))
                if (self._be_border_ac_flow_lb is not None
                        and key in self._be_border_ac_flow_lb.index
                        and key in self._be_border_ac_flow_ub.index):
                    lb = float(self._be_border_ac_flow_lb.loc[key])
                    ub = float(self._be_border_ac_flow_ub.loc[key])
                    ac_lb[(hi, b)] = lb if np.isfinite(lb) else None
                    ac_ub[(hi, b)] = ub if np.isfinite(ub) else None
                elif (self._be_border_ac_flow_lock is not None
                        and key in self._be_border_ac_flow_lock.index):
                    f = float(self._be_border_ac_flow_lock.loc[key])
                    if np.isfinite(f):
                        ac_lb[(hi, b)] = f; ac_ub[(hi, b)] = f
                    else:
                        lo, hi_ = self._default_ac_flow_bounds.get(int(b), (None, None))
                        ac_lb[(hi, b)] = lo; ac_ub[(hi, b)] = hi_
                else:
                    lo, hi_ = self._default_ac_flow_bounds.get(int(b), (None, None))
                    ac_lb[(hi, b)] = lo; ac_ub[(hi, b)] = hi_
            for b in self.s_branch_dc:
                key = (int(ts), int(b))
                if (self._be_border_dc_flow_lb is not None
                        and key in self._be_border_dc_flow_lb.index
                        and key in self._be_border_dc_flow_ub.index):
                    lb = float(self._be_border_dc_flow_lb.loc[key])
                    ub = float(self._be_border_dc_flow_ub.loc[key])
                    dc_lb[(hi, b)] = lb if np.isfinite(lb) else None
                    dc_ub[(hi, b)] = ub if np.isfinite(ub) else None
                elif (self._be_border_dc_flow_lock is not None
                        and key in self._be_border_dc_flow_lock.index):
                    f = float(self._be_border_dc_flow_lock.loc[key])
                    if np.isfinite(f):
                        dc_lb[(hi, b)] = f; dc_ub[(hi, b)] = f
                    else:
                        lo, hi_ = self._default_dc_flow_bounds.get(int(b), (None, None))
                        dc_lb[(hi, b)] = lo; dc_ub[(hi, b)] = hi_
                else:
                    lo, hi_ = self._default_dc_flow_bounds.get(int(b), (None, None))
                    dc_lb[(hi, b)] = lo; dc_ub[(hi, b)] = hi_

        # ── foreign gen locks ─────────────────────────────────────────
        foreign_lock: dict = {}  # (hi, i) → fixed MW or None
        if self._foreign_gen_lock is not None:
            for hi, ts in enumerate(day_timesteps):
                for i in self.s_gen:
                    key = (int(ts), int(i))
                    if key in self._foreign_gen_lock.index:
                        v = float(self._foreign_gen_lock.loc[key])
                        if np.isfinite(v):
                            foreign_lock[(hi, i)] = max(0.0, v)

        # ── build Pyomo model ─────────────────────────────────────────
        m = pyo.ConcreteModel()
        m.s_gen = pyo.Set(ordered=True, initialize=list(self.s_gen))
        m.s_gen_pump = pyo.Set(ordered=True, initialize=list(self.s_gen_pump))
        m.s_gen_storage = pyo.Set(ordered=True, initialize=list(self._idx_generatorsWithStorage))
        m.s_load = pyo.Set(ordered=True, initialize=list(self.s_load))
        m.s_load_flex = pyo.Set(ordered=True, initialize=list(self._idx_consumersWithFlexLoad))
        m.s_branch_ac = pyo.Set(ordered=True, initialize=list(self.s_branch_ac))
        m.s_branch_dc = pyo.Set(ordered=True, initialize=list(self.s_branch_dc))
        m.s_node = pyo.Set(ordered=True, initialize=node_list)
        m.s_h = pyo.RangeSet(0, H - 1)

        # ── variables ─────────────────────────────────────────────────
        m.varGeneration = pyo.Var(m.s_gen, m.s_h, within=pyo.NonNegativeReals)
        m.varPump = pyo.Var(m.s_gen_pump, m.s_h, within=pyo.NonNegativeReals)
        m.varStorage = pyo.Var(m.s_gen_storage, m.s_h, within=pyo.NonNegativeReals)
        m.varSpill = pyo.Var(m.s_gen_storage, m.s_h, within=pyo.NonNegativeReals)
        m.varLoadShed = pyo.Var(m.s_load, m.s_h, within=pyo.NonNegativeReals)
        m.varCurtailment = pyo.Var(m.s_gen, m.s_h, within=pyo.NonNegativeReals)
        m.varFlexLoad = pyo.Var(m.s_load_flex, m.s_h, within=pyo.NonNegativeReals)
        m.varAcBranchFlow = pyo.Var(m.s_branch_ac, m.s_h, within=pyo.Reals)
        m.varDcBranchFlow = pyo.Var(m.s_branch_dc, m.s_h, within=pyo.Reals)
        m.varVoltageAngle = pyo.Var(m.s_node, m.s_h, within=pyo.Reals, initialize=0.0)

        # ── objective: sum over all hours + end-of-day terminal storage value ─
        def _obj_rule(m):
            cost = 0
            for h in range(H):
                cost += sum(m.varGeneration[i, h] * gen_cost_h.get((h, i), 0.0)
                            for i in m.s_gen)
                cost -= sum(m.varPump[i, h] * pump_cost_h.get((h, i), 0.0)
                            for i in m.s_gen_pump)
                cost -= sum(m.varFlexLoad[j, h] * flexload_cost_h.get((h, j), 0.0)
                            for j in m.s_load_flex)
                cost += sum(m.varLoadShed[j, h] * const.loadshedcost
                            for j in m.s_load)
                cost += sum(m.varCurtailment[i, h] * curtail_cost.get(i, 0.0)
                            for i in m.s_gen)
            # Terminal storage value: reward conservation of stored energy at end of day.
            # Without this, the solver has no incentive to leave storage full at day-end,
            # which would drain it to zero every day and destroy inter-day continuity.
            # Subtracting here (minimisation) means higher end storage → lower effective cost.
            cost -= sum(terminal_storval.get(i, 0.0) * m.varStorage[i, H - 1]
                        for i in m.s_gen_storage)
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
            return m.varStorage[i, h] == (
                prev_s + (infl - m.varGeneration[i, h] + pump_term - m.varSpill[i, h]) * dt
            )
        m.cStorCont = pyo.Constraint(m.s_gen_storage, m.s_h, rule=_stor_cont_rule)

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
        has_ramp = (self._ramp_up_mw is not None) or (self._ramp_down_mw is not None)
        if has_ramp:
            def _ramp_up_rule(m, i, h):
                ramp = self._ramp_up_mw[i] if self._ramp_up_mw is not None else np.nan
                if np.isnan(ramp):
                    return pyo.Constraint.Skip
                # In RT, skip ramp for fully pinned generators (capacity ≈ pmin, e.g.
                # DA-locked gas and nuclear). Same reasoning as the hourly path.
                if self._is_rt and abs(capacity_h.get((h, i), 0.0) - pmin_h.get((h, i), 0.0)) < 1e-6:
                    return pyo.Constraint.Skip
                if h == 0:
                    prev = self._gen_prev[i]
                    if np.isnan(prev):
                        return pyo.Constraint.Skip
                    return m.varGeneration[i, 0] <= prev + ramp
                return m.varGeneration[i, h] <= m.varGeneration[i, h - 1] + ramp
            m.cRampUp = pyo.Constraint(m.s_gen, m.s_h, rule=_ramp_up_rule)

            def _ramp_dn_rule(m, i, h):
                ramp = self._ramp_down_mw[i] if self._ramp_down_mw is not None else np.nan
                if np.isnan(ramp):
                    return pyo.Constraint.Skip
                # In RT, skip ramp for fully pinned generators — same as ramp-up rule.
                if self._is_rt and abs(capacity_h.get((h, i), 0.0) - pmin_h.get((h, i), 0.0)) < 1e-6:
                    return pyo.Constraint.Skip
                if h == 0:
                    prev = self._gen_prev[i]
                    if np.isnan(prev):
                        return pyo.Constraint.Skip
                    return m.varGeneration[i, 0] >= prev - ramp
                return m.varGeneration[i, h] >= m.varGeneration[i, h - 1] - ramp
            m.cRampDn = pyo.Constraint(m.s_gen, m.s_h, rule=_ramp_dn_rule)

        # ── flex load ─────────────────────────────────────────────────
        if list(m.s_load_flex):
            def _flex_ub_rule(m, j, h):
                avg = float(grid.consumer.loc[j, "demand_avg"])
                frac = float(grid.consumer.loc[j, "flex_fraction"])
                on_off = float(grid.consumer.loc[j, "flex_on_off"])
                return m.varFlexLoad[j, h] <= avg * frac / on_off
            m.cFlexUb = pyo.Constraint(m.s_load_flex, m.s_h, rule=_flex_ub_rule)

        # ── power balance (one per node per hour) ─────────────────────
        Bbus = self._Bbus

        def _power_balance_rule(m, n, h):
            ni = node_list.index(n)
            lhs = 0
            for g in self._generators_at_node.get(n, []):
                lhs += m.varGeneration[g, h]
                if g in gen_pump_set:
                    lhs -= m.varPump[g, h]
            for j in self._loads_at_node.get(n, []):
                lhs -= demand_h.get((h, j), 0.0)
                lhs += m.varLoadShed[j, h]
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
            idx_b = branch_ac_list.index(b)
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
        if (self._inter_area_ntc is not None
                and len(self._inter_area_ntc) > 0
                and hasattr(self, "_inter_area_ntc_rows")):
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

        Replicates the data contract of _storeResultsAndUpdateStorage for each
        committed hour in the optimisation window, then updates self._storage
        to end-of-commit levels.
        """
        H = pre_data["H"]
        if commit_hours is None:
            commit_hours = H
        commit_hours = int(max(1, min(int(commit_hours), int(H))))
        day_timesteps = pre_data["day_timesteps"]
        inflow_h = pre_data["inflow_h"]
        storage_ini = pre_data["storage_ini"]
        storage_cap = pre_data["storage_cap"]
        pump_eff_g = pre_data["pump_eff_g"]
        gen_storage_set = pre_data["gen_storage_set"]
        gen_pump_set = pre_data["gen_pump_set"]
        flex_set = pre_data["flex_set"]
        node_list = pre_data["node_list"]
        gen_list = list(self.s_gen)
        gen_pump_list = list(self.s_gen_pump)
        storage_gen_list = list(self._idx_generatorsWithStorage)
        flex_list = list(self._idx_consumersWithFlexLoad)
        branch_ac_list = list(self.s_branch_ac)
        branch_dc_list = list(self.s_branch_dc)
        grid = self._grid
        dt = float(self.timeDelta)

        for hi, ts in enumerate(day_timesteps[:commit_hours]):
            Pgen = [float(pyo.value(m.varGeneration[i, hi]) or 0.0) for i in gen_list]
            Ppump = [float(pyo.value(m.varPump[i, hi]) or 0.0) for i in gen_pump_list]
            Pflexload = [float(pyo.value(m.varFlexLoad[j, hi]) or 0.0) for j in flex_list]
            Pb = [float(pyo.value(m.varAcBranchFlow[b, hi]) or 0.0) for b in branch_ac_list]
            Pdc = [float(pyo.value(m.varDcBranchFlow[b, hi]) or 0.0) for b in branch_dc_list]
            theta = [float(pyo.value(m.varVoltageAngle[n, hi]) or 0.0) * const.baseAngle
                     for n in node_list]

            # Load shedding aggregated to nodes
            Ploadshed = pd.Series(index=grid.node.id, data=0.0, dtype=float)
            for j in self.s_load:
                node_j = grid.consumer["node"][j]
                Ploadshed[node_j] += float(pyo.value(m.varLoadShed[j, hi]) or 0.0)

            # Per-hour objective (cost for this timestep only)
            obj_h = (
                sum(Pgen[gi] * pre_data["gen_cost_h"].get((hi, gen_list[gi]), 0.0)
                    for gi in range(len(gen_list)))
                - sum(Ppump[pi] * pre_data["pump_cost_h"].get((hi, gen_pump_list[pi]), 0.0)
                      for pi in range(len(gen_pump_list)))
                - sum(Pflexload[fi] * pre_data["flexload_cost_h"].get((hi, flex_list[fi]), 0.0)
                      for fi in range(len(flex_list)))
                + sum(float(pyo.value(m.varLoadShed[j, hi]) or 0.0) * const.loadshedcost
                      for j in self.s_load)
            )

            # Storage levels and spilled energy
            stor_levels = []
            for i in storage_gen_list:
                stor_levels.append(float(pyo.value(m.varStorage[i, hi]) or 0.0))

            # Spilled energy per generator (all generators, not just storage)
            energy_spilled = np.zeros(len(gen_list))
            for si, i in enumerate(storage_gen_list):
                spill_val = float(pyo.value(m.varSpill[i, hi]) or 0.0)
                gi = gen_list.index(i)
                energy_spilled[gi] = spill_val * dt

            # Branch capacity sensitivity (dual of capacity constraint)
            senseB = []
            for j in self._idx_branchesWithConstraints:
                d = 0.0
                if duals:
                    try:
                        c = m.cAcFlowUb[hi, j]
                        d = float(duals.get(c, 0.0) or 0.0)
                    except (KeyError, AttributeError):
                        try:
                            c = m.cAcFlowLb[hi, j]
                            d = float(duals.get(c, 0.0) or 0.0)
                        except (KeyError, AttributeError):
                            d = 0.0
                senseB.append(-abs(d / const.baseMVA))

            senseDcB = []
            for j in branch_dc_list:
                d = 0.0
                if duals:
                    try:
                        c = m.cDcFlowUb[hi, j]
                        d = float(duals.get(c, 0.0) or 0.0)
                    except (KeyError, AttributeError):
                        d = 0.0
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
            flexload_storagelevel = self._storage_flexload[self._idx_consumersWithFlexLoad]
            flexload_marginalprice = [pre_data["flexload_cost_h"].get((hi, j), 0.0)
                                      for j in flex_list]

            results.addResultsFromTimestep(
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
                marginalprice=storageprice,
                flexload_power=Pflexload,
                flexload_storage=flexload_storagelevel.tolist(),
                flexload_storagevalue=flexload_marginalprice,
                branch_ac_losses=[0.0] * len(branch_ac_list),
                branch_dc_losses=[0.0] * len(branch_dc_list),
                fault_start=None,
            )

            # Update gen_prev for context at day boundary
            for gi, i in enumerate(gen_list):
                self._gen_prev[i] = Pgen[gi]

        # Update self._storage to end-of-commit levels
        for i in storage_gen_list:
            self._storage[i] = float(pyo.value(m.varStorage[i, commit_hours - 1]) or 0.0)

    def _solve_and_store_day_joint(self, day_timesteps, results, solver_name, commit_hours=None):
        """Build, solve, and store results for one rolling-horizon LP window.

        Creates a fresh APPSI HiGHS instance for the day model, respecting
        the same environment variable seeds/threads as the hourly path.
        """
        ts0 = day_timesteps[0]
        if commit_hours is None:
            commit_hours = len(day_timesteps)
        commit_hours = int(max(1, min(int(commit_hours), len(day_timesteps))))
        target_horizon = int(max(1, int(self._objective_day_horizon_hours)))
        solve_h = int(len(day_timesteps))
        tail_note = ""
        if solve_h < target_horizon:
            tail_note = f", tail-window truncated from target {target_horizon}h"
        print(
            f"\n[24h joint] window starting at timestep {ts0} "
            f"({solve_h}h solve, {commit_hours}h commit{tail_note}) ..."
        )
        m, pre_data = self._build_day_joint_model(day_timesteps)
        print(f"  Model built: {len(list(m.s_gen))*len(day_timesteps)} gen×h vars, "
              f"{len(list(m.s_node))*len(day_timesteps)} node×h power-balance constraints")

        # Create solver (only APPSI HiGHS supported for dual extraction)
        if solver_name != "appsi_highs":
            warnings.warn(
                f"daily_24h mode with solver='{solver_name}': only 'appsi_highs' supports "
                "dual extraction for nodal prices. Proceeding but prices may be zero.",
                UserWarning,
            )
        day_opt = appsi.solvers.highs.Highs()
        seed_raw = str(os.environ.get("POWERGAMA_HIGHS_RANDOM_SEED", "")).strip()
        if seed_raw:
            try:
                day_opt.highs_options["random_seed"] = int(seed_raw)
            except Exception:
                pass
        threads_raw = str(os.environ.get("POWERGAMA_HIGHS_THREADS", "")).strip()
        if threads_raw:
            try:
                t = int(threads_raw)
                if t >= 1:
                    day_opt.highs_options["threads"] = t
            except Exception:
                pass

        res = day_opt.solve(m)
        if res.termination_condition != appsi.base.TerminationCondition.optimal:
            raise RuntimeError(
                f"[24h joint] non-optimal at timestep {ts0}: {res.termination_condition}"
            )
        day_opt.load_vars()
        duals = day_opt.get_duals()
        print(f"  Solved OK. Extracting results ...")
        self._extract_and_store_day_joint_results(
            m,
            pre_data,
            results,
            duals,
            commit_hours=commit_hours,
        )

    # ------------------------------------------------------------------

    def _write_objective_trace_if_requested(self):
        """Write optional hourly objective trace CSV for mode parity diagnostics."""
        trace_path = str(os.environ.get("POWERGAMA_OBJECTIVE_TRACE_CSV", "")).strip()
        if not trace_path or not self._hourly_objective_trace:
            return
        try:
            out_path = Path(trace_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            trace_df = pd.DataFrame(
                self._hourly_objective_trace,
                columns=["timestep", "day_index", "hour_in_day", "objective_value"],
            )
            trace_df["objective_mode"] = str(self._objective_mode)
            trace_df["objective_day_horizon_hours"] = int(self._objective_day_horizon_hours)
            trace_df.to_csv(out_path, index=False)
            print(f"Wrote objective trace CSV: {out_path}")
        except Exception as ex:
            warnings.warn(f"Failed to write POWERGAMA_OBJECTIVE_TRACE_CSV='{trace_path}': {ex}", UserWarning)

    def _relax_and_retry(self, opt, warmstart, count, solve_args):
        raise NotImplementedError

    # TODO: Update to allow persistent model solving
    # (remove+add constraint instead of mutable parameters)
    def _updateLpProblem(self, timestep):
        """
        Function that updates LP problem for a given timestep, due to changed
        power demand, power inflow and marginal generator costs
        """

        # 1. Generator output limits:
        #    -> power output constraints
        P_storage = self._storage / self.timeDelta
        P_max = self._grid.generator["pmax"]
        P_min = self._grid.generator["pmin"]
        for i in self.s_gen:
            inflow_factor = self._grid.generator.loc[i, "inflow_fac"]
            inflow_profile = self._grid.generator.loc[i, "inflow_ref"]

            # Time-varying pmax factor is applied to installed capacity.
            pmax_factor = 1.0
            if self._pmax_ref is not None:
                pmax_ref = self._pmax_ref.iloc[i]
                if isinstance(pmax_ref, str) and pmax_ref in self._grid.profiles.columns:
                    pmax_factor = self._grid.profiles.loc[timestep, pmax_ref]
            capacity_now = max(0, P_max[i] * pmax_factor)
            P_inflow = capacity_now * inflow_factor * self._grid.profiles.loc[timestep, inflow_profile]

            # pmin_ref profile is interpreted as fraction of installed pmax.
            pmin_now = P_min[i]
            if self._pmin_ref is not None:
                pmin_ref = self._pmin_ref.iloc[i]
                if isinstance(pmin_ref, str) and pmin_ref in self._grid.profiles.columns:
                    pmin_now = max(0, P_max[i] * self._grid.profiles.loc[timestep, pmin_ref])
            if i not in self._idx_generatorsWithStorage:
                """
                Don't let P_max limit the output (e.g. solar PV)
                This won't affect fuel based generators with zero storage,
                since these should have inflow=p_max in any case
                """
                self.p_gen_pmin[i] = max(min(P_inflow, pmin_now), 0)
                self.p_gen_pmax[i] = P_inflow
            else:
                # generator has storage
                self.p_gen_pmin[i] = max(min(max(0, P_inflow + P_storage[i]), pmin_now), 0)
                self.p_gen_pmax[i] = min(max(0, P_inflow + P_storage[i]), capacity_now)

            # Optional hard lock of foreign generators to DA output (sparse parquet lock table).
            if self._foreign_gen_lock is not None:
                key = (int(timestep), int(i))
                if key in self._foreign_gen_lock.index:
                    da_output = float(self._foreign_gen_lock.loc[key])
                    if not np.isfinite(da_output):
                        da_output = 0.0
                    da_output = max(0.0, da_output)
                    self.p_gen_pmin[i] = da_output
                    self.p_gen_pmax[i] = da_output

            if self._rt_target_ref is not None:
                rt_ref = self._rt_target_ref.iloc[i]
                if isinstance(rt_ref, str) and rt_ref in self._grid.profiles.columns:
                    self.p_rt_target_active[i] = 1
                    self.p_rt_target[i] = max(0.0, float(P_max[i]) * float(self._grid.profiles.loc[timestep, rt_ref]))
                else:
                    self.p_rt_target_active[i] = 0
                    self.p_rt_target[i] = 0.0
            else:
                self.p_rt_target_active[i] = 0
                self.p_rt_target[i] = 0.0

        # 1b. Apply ramp-rate limits based on previous-timestep dispatch.
        # Update pump (charging) DA targets for DA-target deviation penalties.
        if self._rt_pump_target_ref is not None and self._rt_pump_target_gen_indices:
            for i in self.s_gen_pump:
                rt_pump_ref = self._rt_pump_target_ref.iloc[int(i)]
                if isinstance(rt_pump_ref, str) and rt_pump_ref in self._grid.profiles.columns:
                    pump_cap = float(self._grid.generator.loc[i, "pump_cap"])
                    self.p_rt_pump_target[i] = max(0.0, pump_cap * float(self._grid.profiles.loc[timestep, rt_pump_ref]))
                else:
                    self.p_rt_pump_target[i] = 0.0

        # Update consumer (flexible load) DA targets for DA-target deviation penalties
        if self._rt_consumer_target_ref is not None and self._rt_consumer_target_indices:
            for j in self.s_load_flex:
                rt_cons_ref = self._rt_consumer_target_ref.iloc[int(j)]
                if isinstance(rt_cons_ref, str) and rt_cons_ref in self._grid.profiles.columns:
                    self.p_rt_flexload_target_active[j] = 1
                    demand_avg = float(self._grid.consumer.loc[j, "demand_avg"])
                    self.p_rt_flexload_target[j] = max(0.0, demand_avg * float(self._grid.profiles.loc[timestep, rt_cons_ref]))
                else:
                    self.p_rt_flexload_target_active[j] = 0
                    self.p_rt_flexload_target[j] = 0.0

        # Update storage DA filling targets and one-step storage balance RHS.
        if self._rt_storage_target_ref is not None and self._rt_storage_target_indices:
            for i in self.s_gen_storage:
                inflow_factor = self._grid.generator.loc[i, "inflow_fac"]
                inflow_profile = self._grid.generator.loc[i, "inflow_ref"]
                capacity = self._grid.generator.loc[i, "pmax"]
                gen_inflow = capacity * inflow_factor * self._grid.profiles[inflow_profile][timestep]
                self.p_rt_storage_balance_rhs[i] = max(0.0, float(self._storage[i] + gen_inflow * self.timeDelta))

                rt_storage_ref = self._rt_storage_target_ref.iloc[int(i)]
                if isinstance(rt_storage_ref, str) and rt_storage_ref in self._grid.profiles.columns:
                    storage_cap = float(self._grid.generator.loc[i, "storage_cap"])
                    target_frac = float(self._grid.profiles.loc[timestep, rt_storage_ref])
                    self.p_rt_storage_target[i] = max(0.0, storage_cap * max(0.0, min(1.0, target_frac)))
                else:
                    self.p_rt_storage_target[i] = 0.0
        
        # Update storage (filling state) DA targets for DA-target deviation penalties
        if self._rt_residual_cap_active:
            pos_val = pd.to_numeric(
                self._grid.profiles.loc[timestep, self._rt_residual_pos_cap_ref], errors="coerce"
            )
            neg_val = pd.to_numeric(
                self._grid.profiles.loc[timestep, self._rt_residual_neg_cap_ref], errors="coerce"
            )
            self.p_rt_residual_pos_cap = max(0.0, float(pos_val) if np.isfinite(pos_val) else 0.0)
            self.p_rt_residual_neg_cap = max(0.0, float(neg_val) if np.isfinite(neg_val) else 0.0)

        # 1b. Apply ramp-rate limits based on previous-timestep dispatch.
        #     Generators with NaN ramp values or NaN _gen_prev (first timestep) are unconstrained.
        # Daily-reset generators have their _gen_prev cleared at the start of each 24-hour window,
        # so they are free to choose a new level at midnight while staying fixed intra-day.
        if self._ramp_daily_reset is not None and timestep % 24 == 0:
            for i in self.s_gen:
                if self._ramp_daily_reset[i]:
                    self._gen_prev[i] = np.nan
        if self._ramp_up_mw is not None or self._ramp_down_mw is not None:
            for i in self.s_gen:
                prev = self._gen_prev[i]
                if np.isnan(prev):
                    continue  # first timestep: no ramp constraint
                ramp_up = self._ramp_up_mw[i] if self._ramp_up_mw is not None else np.nan
                ramp_dn = self._ramp_down_mw[i] if self._ramp_down_mw is not None else np.nan
                pmax_now = pyo.value(self.p_gen_pmax[i])
                pmin_now = pyo.value(self.p_gen_pmin[i])
                # In RT, skip ramp constraints for fully pinned generators (pmin ≈ pmax),
                # e.g. nuclear and DA-locked gas. The profile lock already uniquely defines
                # the feasible output; applying the ramp on top clips pmax below pmin and
                # the guard silently lowers pmin, causing gas backoff and spurious load shedding.
                # In DA, generators are never pinned this way so this branch is never taken.
                if self._is_rt and pmax_now - pmin_now < 1e-6:
                    continue
                if not np.isnan(ramp_up):
                    pmax_now = min(pmax_now, prev + ramp_up)
                if not np.isnan(ramp_dn):
                    pmin_now = max(pmin_now, prev - ramp_dn)
                # Guard: pmin must not exceed pmax after ramp clipping
                pmin_now = min(pmin_now, pmax_now)
                self.p_gen_pmax[i] = max(pmax_now, 0)
                self.p_gen_pmin[i] = max(pmin_now, 0)

        # TODO: re-create constraint - if persistent solver

        # 2. Update demand
        #    -> power balance constraint
        for i in self.s_load:
            average = self._grid.consumer.loc[i, "demand_avg"] * (1 - self._grid.consumer.loc[i, "flex_fraction"])
            profile_ref = self._grid.consumer.loc[i, "demand_ref"]
            demand_now = self._grid.profiles.loc[timestep, profile_ref] * average
            # Optional hard lock of foreign consumers to DA demand (sparse parquet lock table).
            if self._foreign_cons_lock is not None:
                key = (int(timestep), int(i))
                if key in self._foreign_cons_lock.index:
                    locked_demand = float(self._foreign_cons_lock.loc[key])
                    if np.isfinite(locked_demand):
                        demand_now = locked_demand
            self.p_demand[i] = demand_now

        # 3. Cost parameters
        #    -> update objective function

        # 1c. Update pump capacity based on remaining reservoir room.
        #     Prevents the LP from scheduling pump output beyond storage_cap,
        #     which would otherwise cause silent energy spill while still earning
        #     the full pump cost credit (degeneracy when reservoir is full).
        pump_eff = self._grid.generator["pump_efficiency"]
        storage_cap = self._grid.generator["storage_cap"]
        pump_cap_raw = self._grid.generator["pump_cap"]
        for i in self._idx_generatorsWithPumping:
            remaining_mwh = max(0.0, storage_cap[i] - self._storage[i])
            max_pump_by_room = remaining_mwh / (pump_eff[i] * self.timeDelta) if pump_eff[i] > 0 else 0.0
            self.p_pump_pmax[i] = min(pump_cap_raw[i], max_pump_by_room)

        # 3a. generators with storage (storage value)
        for i in self._idx_generatorsWithStorage:
            this_type_filling = self._grid.generator.loc[i, "storval_filling_ref"]
            this_type_time = self._grid.generator.loc[i, "storval_time_ref"]
            storagecapacity = self._grid.generator.loc[i, "storage_cap"]
            fillinglevel = self._storage[i] / storagecapacity
            filling_col = int(round(fillinglevel * 100))
            storagevalue = (
                self._grid.generator.loc[i, "storage_price"]
                * self._grid.storagevalue_filling.loc[filling_col, this_type_filling]
                * self._grid.storagevalue_time.loc[timestep, this_type_time]
            )
            self.p_gen_cost[i] = storagevalue
            if i in self._idx_generatorsWithPumping:
                deadband = self._grid.generator.pump_deadband[i]
                self.p_genpump_cost[i] = storagevalue - deadband

        # 3b. flexible load (storage value)
        for i in self._idx_consumersWithFlexLoad:
            this_type_filling = self._grid.consumer.loc[i, "flex_storval_filling"]
            this_type_time = self._grid.consumer.loc[i, "flex_storval_time"]
            # Compute storage capacity in Mwh (from value in hours)
            storagecapacity_flexload = (
                self._grid.consumer.loc[i, "flex_storage"]  # h
                * self._grid.consumer.loc[i, "flex_fraction"]
                * self._grid.consumer.loc[i, "demand_avg"]
            )  # MW
            fillinglevel = self._storage_flexload[i] / storagecapacity_flexload
            filling_col = int(round(fillinglevel * 100))
            if fillinglevel > 1:
                storagevalue_flex = -const.flexload_outside_cost
            elif fillinglevel < 0:
                storagevalue_flex = const.flexload_outside_cost
            else:
                storagevalue_flex = (
                    self._grid.consumer.flex_basevalue[i]
                    * self._grid.storagevalue_filling.loc[filling_col, this_type_filling]
                    * self._grid.storagevalue_time.loc[timestep, this_type_time]
                )
            self.p_loadflex_cost[i] = storagevalue_flex

        # 3c. Non-BE DA-locking penalty weights (dimensionless coeff * DA source marginal cost/storage value).
        _coeff = max(0.0, float(self._rt_foreign_da_lock_penalty_coeff))
        for i in self.s_gen:
            if int(i) in self._idx_be_target_generators or int(i) not in self._rt_target_gen_indices:
                self.p_rt_foreign_da_lock_penalty_gen[i] = 0.0
                continue
            _price = max(0.0, float(self.p_gen_cost[i]))
            self.p_rt_foreign_da_lock_penalty_gen[i] = _coeff * _price
        for i in self.s_gen_pump:
            if int(i) in self._idx_be_target_pumps or int(i) not in self._rt_pump_target_gen_indices:
                self.p_rt_foreign_da_lock_penalty_pump[i] = 0.0
                continue
            _price = max(0.0, float(self.p_genpump_cost[i]))
            self.p_rt_foreign_da_lock_penalty_pump[i] = _coeff * _price
        for i in self.s_load_flex:
            if int(i) in self._idx_be_target_flexloads or int(i) not in self._rt_consumer_target_indices:
                self.p_rt_foreign_da_lock_penalty_flex[i] = 0.0
                continue
            _price = max(0.0, float(self.p_loadflex_cost[i]))
            self.p_rt_foreign_da_lock_penalty_flex[i] = _coeff * _price
        for i in self.s_gen_storage:
            if int(i) in self._idx_be_target_storage or int(i) not in self._rt_storage_target_indices:
                self.p_rt_foreign_da_lock_penalty_storage[i] = 0.0
                continue
            _price = max(0.0, float(self.p_gen_cost[i]))
            self.p_rt_foreign_da_lock_penalty_storage[i] = _coeff * _price

        # 4. Optional hard lock of BE cross-border branch flows to DA values
        for b in self.s_branch_ac:
            key = (int(timestep), int(b))
            if self._be_border_ac_flow_lock is not None and key in self._be_border_ac_flow_lock.index:
                da_flow = float(self._be_border_ac_flow_lock.loc[key])
                if np.isfinite(da_flow):
                    self.p_rt_io_target_ac[b] = da_flow
                    self.p_rt_io_target_active_ac[b] = 1
                else:
                    self.p_rt_io_target_ac[b] = 0.0
                    self.p_rt_io_target_active_ac[b] = 0
            else:
                self.p_rt_io_target_ac[b] = 0.0
                self.p_rt_io_target_active_ac[b] = 0
            if (
                self._be_border_ac_flow_lb is not None
                and self._be_border_ac_flow_ub is not None
                and key in self._be_border_ac_flow_lb.index
                and key in self._be_border_ac_flow_ub.index
            ):
                lb = float(self._be_border_ac_flow_lb.loc[key])
                ub = float(self._be_border_ac_flow_ub.loc[key])
                if not np.isfinite(lb):
                    lb = None
                if not np.isfinite(ub):
                    ub = None
                self.varAcBranchFlow[b].setlb(lb)
                self.varAcBranchFlow[b].setub(ub)
            elif self._be_border_ac_flow_lock is not None and key in self._be_border_ac_flow_lock.index:
                da_flow = float(self._be_border_ac_flow_lock.loc[key])
                if not np.isfinite(da_flow):
                    da_flow = 0.0
                self.varAcBranchFlow[b].setlb(da_flow)
                self.varAcBranchFlow[b].setub(da_flow)
            else:
                lb, ub = self._default_ac_flow_bounds.get(int(b), (None, None))
                self.varAcBranchFlow[b].setlb(lb)
                self.varAcBranchFlow[b].setub(ub)

        for b in self.s_branch_dc:
            key = (int(timestep), int(b))
            if self._be_border_dc_flow_lock is not None and key in self._be_border_dc_flow_lock.index:
                da_flow = float(self._be_border_dc_flow_lock.loc[key])
                if np.isfinite(da_flow):
                    self.p_rt_io_target_dc[b] = da_flow
                    self.p_rt_io_target_active_dc[b] = 1
                else:
                    self.p_rt_io_target_dc[b] = 0.0
                    self.p_rt_io_target_active_dc[b] = 0
            else:
                self.p_rt_io_target_dc[b] = 0.0
                self.p_rt_io_target_active_dc[b] = 0
            if (
                self._be_border_dc_flow_lb is not None
                and self._be_border_dc_flow_ub is not None
                and key in self._be_border_dc_flow_lb.index
                and key in self._be_border_dc_flow_ub.index
            ):
                lb = float(self._be_border_dc_flow_lb.loc[key])
                ub = float(self._be_border_dc_flow_ub.loc[key])
                if not np.isfinite(lb):
                    lb = None
                if not np.isfinite(ub):
                    ub = None
                self.varDcBranchFlow[b].setlb(lb)
                self.varDcBranchFlow[b].setub(ub)
            elif self._be_border_dc_flow_lock is not None and key in self._be_border_dc_flow_lock.index:
                da_flow = float(self._be_border_dc_flow_lock.loc[key])
                if not np.isfinite(da_flow):
                    da_flow = 0.0
                self.varDcBranchFlow[b].setlb(da_flow)
                self.varDcBranchFlow[b].setub(da_flow)
            else:
                lb, ub = self._default_dc_flow_bounds.get(int(b), (None, None))
                self.varDcBranchFlow[b].setlb(lb)
                self.varDcBranchFlow[b].setub(ub)

        return

    def _update_persistent_model(self, opt):
        """Update objective function, constraints and bounds in persistent model"""
        # Mutable parameters is not enough
        #
        # TODO: Code for persistent solver
        # https://pyomo.readthedocs.io/en/latest/solvers/persistent_solvers.html
        # to use pwersistent solvers, probably have to set instance at the
        # start, and then modify it in each iteration rather than giving
        # it as an argument to opt.solve:
        #    opt.set_instance(self.concretemodel)
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
        for c in self.cGenMaxLimit.values():
            opt.remove_constraint(c)
        for c in self.cGenMinLimit.values():
            opt.remove_constraint(c)
        # del self.cGenMaxLimit
        # del self.cGenMinLimit
        # self._create_constraint_generator_output()
        for c in self.cGenMaxLimit.values():
            opt.add_constraint(c)
        for c in self.cGenMinLimit.values():
            opt.add_constraint(c)

        # 2.
        for c in self.cPowerbalance.values():
            opt.remove_constraint(c)
        # del self.cPowerbalance
        # self._create_constraint_powerbalance(self._grid)
        for c in self.cPowerbalance.values():
            opt.add_constraint(c)

        # 3.
        # no remove_object?
        # del self.OBJ
        # self._create_objective(self._grid)
        opt.set_objective(self.OBJ)

    # def _compute_transmission_loss(self, branch_flows, aclossmultiplier=1):
    #    """Compute power losses based on power flow"""
    #    for b in self.s_branch_ac:
    #        r = self._grid.branch.loc[b, "resistance"]
    #        branch_flows = self.p_branch_ac_powerflow[b]
    #        lossMVA = r * branch_flows[b] ** 2 / const.baseMVA
    #        # A multiplication factor to account for reactive current losses
    #        lossMVA = lossMVA * aclossmultiplier

    def _update_params_powerlosses(self, aclossmultiplier=1, dclossmultiplier=1):
        """Compute/update parameters used for transmission loss calculations"""
        if self._lossmethod == 1:
            # store power flows for next timestep (used for loss calculation)
            for b in self.s_branch_ac:
                # Github issue https://github.com/powergama/powergama/issues/29
                # write it in terms of varAcBranchFlow and not varBranchFlow12 because
                # under certain circumstances, varAcBranchFlow12 and varAcBranchFlow21
                # may both be large (if high loss is beneficial) since there is no constraint
                # forbidding simultaneous flow in both direction (only an indirect cost
                # via generation costs)
                if self.varAcBranchFlow[b].value > 0:
                    self.p_branch_ac_powerflow12[b] = self.varAcBranchFlow[b].value
                    self.p_branch_ac_powerflow21[b] = 0
                else:
                    self.p_branch_ac_powerflow12[b] = 0
                    self.p_branch_ac_powerflow21[b] = -self.varAcBranchFlow[b].value
            for b in self.s_branch_dc:
                # not the same issue for dc lines because there are not power flow equations
                self.p_branch_dc_powerflow12[b] = self.varDcBranchFlow12[b].value
                self.p_branch_dc_powerflow21[b] = self.varDcBranchFlow21[b].value
        elif self._lossmethod == 2:
            # compute power losses and store for next timestep
            for b in self.s_branch_ac:
                r = self._grid.branch.loc[b, "resistance"]
                lossMVA = r * self.varAcBranchFlow[b] ** 2 / const.baseMVA
                # A multiplication factor to account for reactive current losses
                lossMVA = lossMVA * aclossmultiplier
                self.p_branch_ac_power_loss[b] = lossMVA
            for b in self.s_branch_dc:
                r_pu = self._grid.dcbranch.loc[b, "resistance"]
                p_pu = self.varDcBranchFlow[b] / const.baseMVA
                loss_pu = r_pu * p_pu**2
                lossMVA = loss_pu * const.baseMVA * dclossmultiplier
                self.p_branch_dc_power_loss[b] = lossMVA

    def _get_fault_start(self, timestep):
        # Used by LpFaultProblem
        return None

    def _storeResultsAndUpdateStorage(self, timestep, results):
        """Store timestep results in local arrays, and update storage"""

        # 1. Update generator storage:
        inflow_profile_refs = self._grid.generator["inflow_ref"]
        inflow_factor = self._grid.generator["inflow_fac"]
        capacity = self._grid.generator["pmax"]
        pumpedIn = np.zeros(len(capacity))
        energyIn = np.zeros(len(capacity))
        energyOut = np.zeros(len(capacity))
        for i in self.s_gen:
            genInflow = capacity[i] * inflow_factor[i] * self._grid.profiles[inflow_profile_refs[i]][timestep]
            energyIn[i] = genInflow * self.timeDelta
            energyOut[i] = self.varGeneration[i].value * self.timeDelta

        for i in self._idx_generatorsWithPumping:
            Ppump = self.varPump[i].value
            pumpedIn[i] = Ppump * self._grid.generator["pump_efficiency"][i] * self.timeDelta
        energyStorable = self._storage + energyIn + pumpedIn - energyOut
        storagecapacity = self._grid.generator["storage_cap"]
        # self._storage[i] = min(storagecapacity,energyStorable)
        self._storage = np.vstack((storagecapacity, energyStorable)).min(axis=0)
        self._energyspilled = energyStorable - self._storage

        # 2. Update flexible load storage
        for i in self._idx_consumersWithFlexLoad:
            energyIn_flexload = self.varFlexLoad[i].value * self.timeDelta
            energyOut_flexload = (
                self._grid.consumer["flex_fraction"][i] * self._grid.consumer["demand_avg"][i] * self.timeDelta
            )
            self._storage_flexload[i] += energyIn_flexload - energyOut_flexload

        # 1c. Record this timestep's dispatch for ramp constraints in next timestep.
        for i in self.s_gen:
            v = self.varGeneration[i].value
            self._gen_prev[i] = v if v is not None else 0.0

        # 3. Collect variable values from optimisation result
        F = self.OBJ()
        Pgen = [self.varGeneration[i].value for i in self.s_gen]
        Ppump = [self.varPump[i].value for i in self.s_gen_pump]
        Pflexload = [self.varFlexLoad[i].value for i in self.s_load_flex]
        Pb = [self.varAcBranchFlow[i].value for i in self.s_branch_ac]
        Pdc = [self.varDcBranchFlow[i].value for i in self.s_branch_dc]
        theta = [self.varVoltageAngle[i].value * const.baseAngle for i in self.s_node]
        # load shedding is aggregated to nodes (due to old code)
        Ploadshed = pd.Series(index=self._grid.node.id, data=[0] * len(self._grid.node.id), dtype=float)
        for j in self.s_load:
            node = self._grid.consumer["node"][j]
            Ploadshed[node] += self.varLoadShed[j].value

        # 4 Collect dual values
        # 4a. branch capacity sensitivity (whether pos or neg flow)
        senseB = []
        for j in self._idx_branchesWithConstraints:
            # for j in self.concretemodel.BRANCH_AC:
            c = self.cMaxFlowAc[j]
            senseB.append(-abs(self.dual[c] / const.baseMVA))
        senseDcB = []
        for j in self.s_branch_dc:
            c = self.cMaxFlowDc[j]
            senseDcB.append(-abs(self.dual[c] / const.baseMVA))

        # 4b. node demand sensitivity (energy balance)
        # TODO: Without abs(...) the value jumps between pos and neg. Why?
        senseN = []
        for j in self.s_node:
            try:
                c = self.cPowerbalance[j]
                senseN.append(abs(self.dual[c] / const.baseMVA))
            except KeyError:
                # Constraint was skipped (trivially 0==0, e.g. isolated hub/virtual node
                # with no generators, loads, or branches). Nodal price = 0.
                senseN.append(0.0)

        # consider spilled energy only for generators with storage<infinity
        # energyspilled = zeros(energyStorable.shape)
        # indx = self._grid.getIdxGeneratorsWithNonzeroInflow()
        # energyspilled[indx] = energyStorable[indx]-self._storage[indx]
        energyspilled = self._energyspilled
        storagelevel = self._storage[self._idx_generatorsWithStorage]
        storageprice = [self.p_gen_cost[i].value for i in self._idx_generatorsWithStorage]
        flexload_storagelevel = self._storage_flexload[self._idx_consumersWithFlexLoad]
        flexload_marginalprice = [self.p_loadflex_cost[i].value for i in self._idx_consumersWithFlexLoad]

        # TODO: Only keep track of inflow spilled for generators with
        # nonzero inflow

        # Extract power losses
        if self._lossmethod == 0:
            acPowerLoss = [0] * len(self.s_branch_ac)
            dcPowerLoss = [0] * len(self.s_branch_dc)
        elif self._lossmethod == 1:
            acPowerLoss = [self.varLossAc12[b].value + self.varLossAc21[b].value for b in self.s_branch_ac]
            dcPowerLoss = [self.varLossDc12[b].value + self.varLossDc21[b].value for b in self.s_branch_dc]
        elif self._lossmethod == 2:
            acPowerLoss = list(self.p_branch_ac_power_loss.extract_values().values())
            dcPowerLoss = list(self.p_branch_dc_power_loss.extract_values().values())
        else:
            raise Exception("Lossmethod must be 0,1 or 2")

        results.addResultsFromTimestep(
            timestep=self._grid.timerange[0] + timestep,
            objective_function=F,
            generator_power=Pgen,
            generator_pumped=Ppump,
            branch_power=Pb,
            dcbranch_power=Pdc,
            node_angle=theta,
            sensitivity_branch_capacity=senseB,
            sensitivity_dcbranch_capacity=senseDcB,
            sensitivity_node_power=senseN,
            storage=storagelevel.tolist(),
            inflow_spilled=energyspilled.tolist(),
            loadshed_power=Ploadshed.tolist(),
            marginalprice=storageprice,
            flexload_power=Pflexload,
            flexload_storage=flexload_storagelevel.tolist(),
            flexload_storagevalue=flexload_marginalprice,
            branch_ac_losses=acPowerLoss,
            branch_dc_losses=dcPowerLoss,
            fault_start=self._get_fault_start(timestep),
        )

        return

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
        if solver == "gurobi":
            opt = pyo.SolverFactory("gurobi", solver_io="python")
            print(":) Using direct python interface to solver")
        elif solver == "gurobi_direct":  # think this is the same as gurobi above
            opt = pyo.SolverFactory(solver)
            print(":) Using gurobi_direct")
        elif solver == "gurobi_persistent":
            opt = pyo.SolverFactory("gurobi_persistent")
            print(":) Using persistent (in-memory) python interface to solver")
            print("-- Experimental --")
            symbolic_solver_labels = True
            if "symbolic_solver_labels" in solve_args:
                symbolic_solver_labels = solve_args["symbolic_solver_labels"]
                solve_args.pop("symbolic_solver_labels")
            opt.set_instance(self, symbolic_solver_labels=symbolic_solver_labels)
        elif solver == "appsi_highs":
            # opt = pyo.SolverFactory(solver)
            opt = appsi.solvers.Highs()
            # Keep solve() from raising on infeasible/unbounded runs; load values explicitly on optimal.
            opt.config.load_solution = False

            # Optional reproducibility controls for APPsi/HiGHS.
            # These keep workflow APIs unchanged while enabling seed/thread control per run.
            seed_raw = str(os.environ.get("POWERGAMA_HIGHS_RANDOM_SEED", "")).strip()
            if seed_raw:
                try:
                    opt.highs_options["random_seed"] = int(seed_raw)
                except Exception:
                    warnings.warn(
                        f"Ignoring invalid POWERGAMA_HIGHS_RANDOM_SEED='{seed_raw}' (expected integer).",
                        UserWarning,
                    )

            threads_raw = str(os.environ.get("POWERGAMA_HIGHS_THREADS", "")).strip()
            if threads_raw:
                try:
                    threads = int(threads_raw)
                    if threads >= 1:
                        opt.highs_options["threads"] = threads
                except Exception:
                    warnings.warn(
                        f"Ignoring invalid POWERGAMA_HIGHS_THREADS='{threads_raw}' (expected integer >= 1).",
                        UserWarning,
                    )

            parallel_raw = str(os.environ.get("POWERGAMA_HIGHS_PARALLEL", "")).strip().lower()
            if parallel_raw in {"off", "on", "choose"}:
                opt.highs_options["parallel"] = parallel_raw

            if opt.available():
                print(":) Found solver")
            else:
                print(":( Could not find solver {}. Returning.".format(solver))
                raise Exception("Could not find LP solver {}".format(solver))
        else:
            solver_io = None
            # Some solver plugins (e.g. pyomo.contrib.highs) do not accept
            # the executable kwarg in their ConfigDict.
            if solver_path:
                opt = pyo.SolverFactory(solver, executable=solver_path, solver_io=solver_io)
            else:
                opt = pyo.SolverFactory(solver, solver_io=solver_io)
            if opt.available():
                opt_exec = None
                if hasattr(opt, "executable"):
                    try:
                        opt_exec = opt.executable()
                    except Exception:
                        opt_exec = None
                if opt_exec:
                    print(":) Found solver here: {}".format(opt_exec))
                else:
                    print(":) Found solver")
            else:
                print(":( Could not find solver {}. Returning.".format(solver))
                raise Exception("Could not find LP solver {}".format(solver))

        # Enable access to dual values
        if not isinstance(opt, appsi.solvers.highs.Highs):
            self.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        timesteps_to_solve = self._get_timesteps_to_solve(continue_from_last=continue_from_last, results=results)
        self._build_day_hour_index(timesteps_to_solve)
        if self._objective_mode == "daily_24h":
            # ── true 24h joint LP path ────────────────────────────────
            if self._lossmethod != 0:
                warnings.warn(
                    "daily_24h mode currently ignores lossmethod != 0 (losses not modelled in 24h LP).",
                    UserWarning,
                )
            if continue_from_last:
                self._storage.loc[self._idx_generatorsWithStorage] = results.db.getResultStorageFillingAll(
                    timestep=timesteps_to_solve[0] - 1
                )
                self._storage_flexload.loc[self._idx_consumersWithFlexLoad] = results.db.getResultFlexloadStorageFillingAll(
                    timestep=timesteps_to_solve[0] - 1
                )
                if self._ramp_up_mw is not None or self._ramp_down_mw is not None:
                    prev_gen = results.db.getResultGeneratorPowerAll(timestep=timesteps_to_solve[0] - 1)
                    for i in self.s_gen:
                        self._gen_prev[i] = prev_gen.get(i, 0.0)

            # Rolling-horizon solve: optimise over objective horizon, commit only first 24h.
            # This enables next-day lookahead (e.g. 48h solve / 24h commit).
            solve_h = int(max(1, int(self._objective_day_horizon_hours)))
            commit_h = int(max(1, min(int(self._objective_day_commit_hours), solve_h)))
            windows = []
            for start in range(0, len(timesteps_to_solve), commit_h):
                win = list(timesteps_to_solve[start : start + solve_h])
                if win:
                    windows.append(win)

            print(
                f"Solving (24h joint LP) — {len(windows)} window(s): "
                f"{solve_h}h lookahead / {commit_h}h commit ..."
            )
            for win in windows:
                # For daily_reset generators, clear _gen_prev at commit boundary (same rule intent as hourly)
                if self._ramp_daily_reset is not None:
                    for i in self.s_gen:
                        if self._ramp_daily_reset[i]:
                            self._gen_prev[i] = np.nan
                self._solve_and_store_day_joint(win, results, solver, commit_hours=commit_h)

            self._write_objective_trace_if_requested()
            return results
            # Update internal variable for storage filling level (self._storage and self._storage_flexload)
            # this must be done before updateLpProblem below
            self._storage.loc[self._idx_generatorsWithStorage] = results.db.getResultStorageFillingAll(
                timestep=timesteps_to_solve[0] - 1
            )
            self._storage_flexload.loc[self._idx_consumersWithFlexLoad] = results.db.getResultFlexloadStorageFillingAll(
                timestep=timesteps_to_solve[0] - 1
            )
            # Restore previous-timestep generation for ramp constraints
            if self._ramp_up_mw is not None or self._ramp_down_mw is not None:
                prev_gen = results.db.getResultGeneratorPowerAll(timestep=timesteps_to_solve[0] - 1)
                for i in self.s_gen:
                    self._gen_prev[i] = prev_gen.get(i, 0.0)

        if self._lossmethod in [1, 2]:
            print("Computing losses in first timestep")
            self._updateLpProblem(timestep=timesteps_to_solve[0])
            res = opt.solve(self)
            if isinstance(res, appsi.solvers.highs.HighsResults):
                if res.termination_condition != appsi.base.TerminationCondition.optimal:
                    raise RuntimeError(
                        "APPSI HIGHS non-optimal termination before timestep loop: "
                        f"{res.termination_condition}"
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
                self.write(
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
                res = opt.solve(self, **solve_args)
            except Exception as ex:
                # do something ()
                print(f"SOLVE ERROR at timestep={timestep}. Tries to solve again")
                if solver == "appsi_highs":
                    # APPsi/HiGHS may emit cascading low-level row-bound update
                    # errors on immediate retry; preserve original failure signal.
                    raise
                try:
                    res = opt.solve(self, **solve_args)
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
                        f"APPSI HIGHS non-optimal termination at timestep={timestep}: "
                        f"{res.termination_condition}"
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
                obj_value = float(self.OBJ())
                self._daily_objective_trace[day_idx] = self._daily_objective_trace.get(day_idx, 0.0) + obj_value
                self._hourly_objective_trace.append((int(timestep), int(day_idx), int(hour_in_day), obj_value))

            # store results and update storage levels
            self._storeResultsAndUpdateStorage(timestep, results)

        if self._objective_mode == "daily_24h" and self._daily_objective_trace:
            print("Daily objective trace (hourly-path accumulation):")
            for d in sorted(self._daily_objective_trace):
                print(f"  day={d}: objective_sum={self._daily_objective_trace[d]:.6f}")

        self._write_objective_trace_if_requested()

        return results

    def _update_progress(self, n=None, maxn=None):
        return
