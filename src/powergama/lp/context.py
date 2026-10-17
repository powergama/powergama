"""Everything derived once from the grid data, shared by both formulations.

``GridContext`` is the second half of what lets the model builders be plain
functions: ``LpConfig`` carries what the caller asked for, this carries what the
grid implies. Both are settled before any Pyomo component exists and neither
changes afterwards, so a builder can take ``(model, grid, ctx, cfg)`` and reach
for nothing else.

Field names keep the spelling the rest of the code already uses, minus the
leading underscore -- including the inherited camelCase of
``idx_generatorsWithPumping`` and friends. That makes ``LpProblem.__getattr__``
a mechanical name-for-name redirect rather than a translation table, which is
worth more during a refactor than tidier names.

One thing is computed here that used to be computed elsewhere, because it is
a pure function of the grid:

``inter_area_ntc_rows``
    was a side effect of building the single-period NTC constraint, which the
    multi-period builder then read back behind a ``hasattr`` guard. That made
    multi-period silently depend on the single-period model having been built
    first, and silently drop every NTC constraint if it had not.

``_get_powerbalance_rhs`` deliberately does *not* move here despite also being
keyed on the B matrix: it builds Pyomo expressions over ``varVoltageAngle``, so
it needs a model and belongs with the single-period builder.

``T45_NUCLEAR_OPERATIONAL_LIMITS`` is read here. It is project-specific, and
the flag it produces is consumed by the ramp rules of both formulations.
"""

import os
from dataclasses import dataclass, field, fields

import numpy as np
import pandas as pd


@dataclass
class GridContext:
    """Grid-derived lookups, index sets and matrices. Built once, never mutated."""

    # --- power flow structure ---------------------------------------------
    Bbus: object = None
    DA: object = None
    refnodes: list = field(default_factory=list)

    # --- node -> entity maps ----------------------------------------------
    generators_at_node: dict = field(default_factory=dict)
    loads_at_node: dict = field(default_factory=dict)
    branch_from_node: dict = field(default_factory=dict)
    branch_to_node: dict = field(default_factory=dict)
    dcbranch_from_node: dict = field(default_factory=dict)
    dcbranch_to_node: dict = field(default_factory=dict)

    # --- entity id lists, in model index order ------------------------------
    # The multi-period builder needs these without a built single-period model;
    # they are the same lists its Pyomo Sets are initialised from.
    node_ids: list = field(default_factory=list)
    gen_ids: list = field(default_factory=list)
    load_ids: list = field(default_factory=list)
    branch_ac_ids: list = field(default_factory=list)
    branch_dc_ids: list = field(default_factory=list)

    # --- index sets --------------------------------------------------------
    idx_generatorsWithPumping: list = field(default_factory=list)
    idx_generatorsWithStorage: list = field(default_factory=list)
    idx_consumersWithFlexLoad: list = field(default_factory=list)
    idx_branchesWithConstraints: list = field(default_factory=list)

    # --- optional profile references for time-varying bounds ---------------
    pmax_ref: object = None
    pmin_ref: object = None

    # --- RT day-ahead target references ------------------------------------
    rt_target_ref: object = None
    rt_target_gen_indices: set = field(default_factory=set)
    rt_consumer_target_ref: object = None
    rt_consumer_target_indices: set = field(default_factory=set)
    rt_storage_target_ref: object = None
    rt_storage_target_indices: set = field(default_factory=set)

    # --- ramping ------------------------------------------------------------
    ramp_up_pu: object = None
    ramp_down_pu: object = None
    ramp_cap_mw: object = None
    ramp_daily_reset: object = None
    disable_nuclear_ramp_profile_conflict: object = None

    # --- RT generator classification (read only by rt_diagnostics, except
    #     wind/solar/hydro_ror which the deviation pricing uses) -------------
    idx_rt_peak_gen: set = field(default_factory=set)
    idx_rt_normal_gas: set = field(default_factory=set)
    idx_rt_wind: set = field(default_factory=set)
    idx_rt_solar: set = field(default_factory=set)
    idx_rt_nuclear: set = field(default_factory=set)
    idx_rt_biomass: set = field(default_factory=set)
    idx_rt_fossil_other: set = field(default_factory=set)
    idx_rt_storage_gens: set = field(default_factory=set)
    idx_rt_hydro_ror: set = field(default_factory=set)

    # --- branch flow bounds and NTC ----------------------------------------
    default_ac_flow_bounds: dict = field(default_factory=dict)
    default_dc_flow_bounds: dict = field(default_factory=dict)
    inter_area_ntc: object = None
    inter_area_ntc_rows: dict = field(default_factory=dict)

    # --- border branches, identified by the project-specific DA lock tables ---
    # Grid-shaped lookups even though their source is an injected artifact, so
    # the constraint rules that read them can take ctx like everything else.
    idx_border_ac: set = field(default_factory=set)
    idx_border_dc: set = field(default_factory=set)
    border_ac_sign: dict = field(default_factory=dict)
    border_dc_sign: dict = field(default_factory=dict)

    @classmethod
    def from_grid(cls, grid, Bbus, DA, refnodes, locks):
        """Derive everything the builders need from *grid*.

        Bbus/DA/refnodes are passed in rather than computed here: the matrices are
        expensive and the caller already prints progress around them. *locks* is a
        :class:`~powergama.lp.project_locks.DaLocks`, needed only for the border
        branch sets it identifies.
        """
        ctx = cls(
            Bbus=Bbus,
            DA=DA,
            refnodes=refnodes,
            idx_border_ac=locks.idx_border_ac,
            idx_border_dc=locks.idx_border_dc,
            border_ac_sign=locks.border_ac_sign,
            border_dc_sign=locks.border_dc_sign,
        )
        ctx._build_node_maps(grid)
        ctx._build_index_sets(grid)
        ctx._build_profile_refs(grid)
        ctx._build_ramping(grid)
        ctx._build_rt_classification(grid)
        ctx._build_flow_bounds(grid)
        ctx._build_inter_area_ntc(grid)
        return ctx

    # ------------------------------------------------------------------

    def _build_node_maps(self, grid):
        self.generators_at_node = grid.generator.groupby("node").groups
        self.loads_at_node = grid.consumer.groupby("node").groups
        self.branch_from_node = grid.branch.groupby("node_from").groups
        self.branch_to_node = grid.branch.groupby("node_to").groups
        self.dcbranch_from_node = grid.dcbranch.groupby("node_from").groups
        self.dcbranch_to_node = grid.dcbranch.groupby("node_to").groups
        for n in grid.node["id"]:
            # fill in so dict is defined for all nodes:
            if n not in self.generators_at_node:
                self.generators_at_node[n] = []
            if n not in self.loads_at_node:
                self.loads_at_node[n] = []
            if n not in self.branch_from_node:
                self.branch_from_node[n] = []
            if n not in self.branch_to_node:
                self.branch_to_node[n] = []
            if n not in self.dcbranch_from_node:
                self.dcbranch_from_node[n] = []
            if n not in self.dcbranch_to_node:
                self.dcbranch_to_node[n] = []

    def _build_index_sets(self, grid):
        self.node_ids = grid.node["id"].tolist()
        self.gen_ids = grid.generator.index.tolist()
        self.load_ids = grid.consumer.index.tolist()
        self.branch_ac_ids = grid.branch.index.tolist()
        self.branch_dc_ids = grid.dcbranch.index.tolist()
        self.idx_generatorsWithPumping = grid.getIdxGeneratorsWithPumping()
        self.idx_generatorsWithStorage = grid.getIdxGeneratorsWithStorage()
        self.idx_consumersWithFlexLoad = grid.getIdxConsumersWithFlexibleLoad()
        self.idx_branchesWithConstraints = grid.getIdxBranchesWithFlowConstraints()

    def _build_profile_refs(self, grid):
        # Optional profile references for time-varying dispatch bounds.
        # pmax_ref scales installed pmax (availability factor), pmin_ref sets a
        # profile-driven floor as a fraction of installed pmax.
        self.pmax_ref = grid.generator["pmax_ref"] if "pmax_ref" in grid.generator.columns else None
        self.pmin_ref = grid.generator["pmin_ref"] if "pmin_ref" in grid.generator.columns else None
        self.rt_target_ref = grid.generator["rt_target_ref"] if "rt_target_ref" in grid.generator.columns else None
        self.rt_target_gen_indices = {
            int(i)
            for i in grid.generator.index
            if self.rt_target_ref is not None
            and isinstance(self.rt_target_ref.loc[i], str)
            and str(self.rt_target_ref.loc[i]).strip()
        }
        # Optional consumer (flexible load) DA target references for DA-target deviation penalties.
        self.rt_consumer_target_ref = (
            grid.consumer["rt_target_ref"] if "rt_target_ref" in grid.consumer.columns else None
        )
        self.rt_consumer_target_indices = {
            int(i)
            for i in grid.consumer.index
            if self.rt_consumer_target_ref is not None
            and isinstance(self.rt_consumer_target_ref.loc[i], str)
            and str(self.rt_consumer_target_ref.loc[i]).strip()
            and int(i) in self.idx_consumersWithFlexLoad
        }
        # Optional storage (reservoir/battery filling) DA target references for DA-target deviation penalties.
        # Stores normalized filling fraction [0, 1] to ensure storage state continuity matches DA.
        self.rt_storage_target_ref = (
            grid.generator["rt_storage_target_ref"] if "rt_storage_target_ref" in grid.generator.columns else None
        )
        self.rt_storage_target_indices = {
            int(i)
            for i in grid.generator.index
            if self.rt_storage_target_ref is not None
            and isinstance(self.rt_storage_target_ref.loc[i], str)
            and str(self.rt_storage_target_ref.loc[i]).strip()
            and int(i) in self.idx_generatorsWithStorage
        }

    def _build_ramping(self, grid):
        # Ramp-rate limits in per-unit of installed generator capacity. NaN means unconstrained.
        self.ramp_up_pu = grid.generator["ramp_up_pu"].values.copy() if "ramp_up_pu" in grid.generator.columns else None
        self.ramp_down_pu = (
            grid.generator["ramp_down_pu"].values.copy() if "ramp_down_pu" in grid.generator.columns else None
        )
        self.ramp_cap_mw = pd.to_numeric(grid.generator["pmax"], errors="coerce").fillna(0.0).values
        # If True for a generator, the ramp constraint is released at the start of every 24-hour
        # block (timestep % 24 == 0).  This models plant types (e.g. nuclear) whose output level
        # is decided day-ahead but is held constant throughout the day.
        # NOTE: Assumes simulations start at hour 0 (midnight).  Timestep indices from
        # continue_from_last runs preserve the original numbering so midnight detection remains valid.
        if "ramp_daily_reset" in grid.generator.columns:
            self.ramp_daily_reset = grid.generator["ramp_daily_reset"].fillna(False).astype(bool).values
        else:
            self.ramp_daily_reset = None

        # Optional bypass for nuclear ramp/daily-reset when nuclear operational
        # profile limits are active (pmax_ref/pmin_ref runtime binding).
        # This avoids infeasible day-joint intersections where a constant
        # intra-day ramp policy conflicts with hour-varying profile bounds.
        _nuclear_limits_active = str(os.environ.get("T45_NUCLEAR_OPERATIONAL_LIMITS", "")).strip().lower() not in {
            "",
            "0",
            "false",
            "off",
            "no",
        }
        self.disable_nuclear_ramp_profile_conflict = np.zeros(len(grid.generator), dtype=bool)
        if _nuclear_limits_active and ("type" in grid.generator.columns):
            _gtype = grid.generator["type"].astype(str).str.lower().str.strip()
            _is_nuclear = _gtype.eq("nuclear")
            _has_pmax_ref = pd.Series(False, index=grid.generator.index)
            _has_pmin_ref = pd.Series(False, index=grid.generator.index)
            if self.pmax_ref is not None:
                _has_pmax_ref = self.pmax_ref.astype(str).str.strip().ne("")
            if self.pmin_ref is not None:
                _has_pmin_ref = self.pmin_ref.astype(str).str.strip().ne("")
            if "nuclear_fully_constrained" in grid.generator.columns:
                _fully_constrained = grid.generator["nuclear_fully_constrained"].fillna(False).astype(bool)
            else:
                # Backward-compatible fallback: matching non-empty pmax/pmin refs
                # typically indicate a fully pinned nuclear profile.
                _pmax_txt = (
                    self.pmax_ref.astype(str).str.strip()
                    if self.pmax_ref is not None
                    else pd.Series("", index=grid.generator.index)
                )
                _pmin_txt = (
                    self.pmin_ref.astype(str).str.strip()
                    if self.pmin_ref is not None
                    else pd.Series("", index=grid.generator.index)
                )
                _fully_constrained = _pmax_txt.ne("") & _pmax_txt.eq(_pmin_txt)
            _disable = _is_nuclear & _has_pmax_ref & _has_pmin_ref & _fully_constrained
            if _disable.any():
                self.disable_nuclear_ramp_profile_conflict[_disable.values] = True

    def _build_rt_classification(self, grid):
        # Classify BE generators by type/tag for differentiated deviation pricing.
        _gtype = (
            grid.generator["type"].astype(str).str.lower()
            if "type" in grid.generator.columns
            else pd.Series("", index=grid.generator.index)
        )
        _gdesc = (
            grid.generator["desc"].astype(str)
            if "desc" in grid.generator.columns
            else pd.Series("", index=grid.generator.index)
        )
        # Generator type classification for RT deviation cost attribution in debug output.
        # These classify by generator type only — no country filter.
        _storage_gens_set = self.idx_generatorsWithStorage
        self.idx_rt_peak_gen = {
            int(i) for i in grid.generator.index if _gtype.loc[i] == "fossil_gas" and "[RT]" in str(_gdesc.loc[i])
        }
        self.idx_rt_normal_gas = {
            int(i) for i in grid.generator.index if _gtype.loc[i] == "fossil_gas" and "[RT]" not in str(_gdesc.loc[i])
        }
        self.idx_rt_wind = {int(i) for i in grid.generator.index if _gtype.loc[i] in {"wind_off", "wind_on", "wind"}}
        self.idx_rt_solar = {int(i) for i in grid.generator.index if _gtype.loc[i] == "solar"}
        self.idx_rt_nuclear = {int(i) for i in grid.generator.index if _gtype.loc[i] == "nuclear"}
        self.idx_rt_biomass = {int(i) for i in grid.generator.index if _gtype.loc[i] == "biomass"}
        self.idx_rt_fossil_other = {int(i) for i in grid.generator.index if _gtype.loc[i] == "fossil_other"}
        self.idx_rt_storage_gens = set(_storage_gens_set)
        self.idx_rt_hydro_ror = {
            int(i)
            for i in grid.generator.index
            if _gtype.loc[i] == "hydro"
            and int(i) not in _storage_gens_set
            and float(pd.to_numeric(grid.generator.loc[i, "pump_cap"], errors="coerce") or 0.0) <= 0.0
            and float(pd.to_numeric(grid.generator.loc[i, "storage_cap"], errors="coerce") or 0.0) <= 0.0
        }

    def _build_flow_bounds(self, grid):
        for b in grid.branch.index:
            cap = pd.to_numeric(grid.branch.loc[b, "capacity"], errors="coerce")
            if np.isfinite(cap):
                self.default_ac_flow_bounds[int(b)] = (-float(cap), float(cap))
            else:
                self.default_ac_flow_bounds[int(b)] = (None, None)
        for b in grid.dcbranch.index:
            cap = pd.to_numeric(grid.dcbranch.loc[b, "capacity"], errors="coerce")
            if np.isfinite(cap):
                self.default_dc_flow_bounds[int(b)] = (-float(cap), float(cap))
            else:
                self.default_dc_flow_bounds[int(b)] = (None, None)

    def _build_inter_area_ntc(self, grid):
        """Resolve each NTC row to the branches crossing that area boundary.

        Both formulations need this. It used to be built as a side effect of the
        single-period constraint builder, so multi-period depended on construction
        order to see it at all.
        """
        self.inter_area_ntc = grid.inter_area_ntc
        if self.inter_area_ntc is None or self.inter_area_ntc.shape[0] == 0:
            return
        for cidx, (_, row) in enumerate(self.inter_area_ntc.iterrows()):
            area_from = row["area_from"]
            area_to = row["area_to"]
            # Get branches between the two areas
            br_ac = grid.getInterAreaBranches(area_from=area_from, area_to=area_to, acdc="ac")
            br_dc = grid.getInterAreaBranches(area_from=area_from, area_to=area_to, acdc="dc")
            self.inter_area_ntc_rows[cidx] = {
                "ac_pos": list(br_ac["branches_pos"]),
                "ac_neg": list(br_ac["branches_neg"]),
                "dc_pos": list(br_dc["branches_pos"]),
                "dc_neg": list(br_dc["branches_neg"]),
                "ntc_forward": float(row["ntc_forward"]),
                "ntc_backward": float(row["ntc_backward"]),
            }


# The names LpProblem.__getattr__ resolves against the context, spelled with the
# leading underscore the rest of the code uses. Explicit for the same reason as
# CONFIG_ATTRS: a new field must not silently shadow an LpProblem attribute.
CONTEXT_ATTRS = frozenset(f.name for f in fields(GridContext))
