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

``inter_area_rules_rows``
    was a side effect of building the single-period NTC constraint, which the
    multi-period builder then read back behind a ``hasattr`` guard. That made
    multi-period silently depend on the single-period model having been built
    first, and silently drop every NTC constraint if it had not.

``_get_powerbalance_rhs`` deliberately does *not* move here despite also being
keyed on the B matrix: it builds Pyomo expressions over ``varVoltageAngle``, so
it needs a model and belongs with the single-period builder.
"""

from dataclasses import dataclass, field, fields

import numpy as np
import pandas as pd

# Generator columns naming a profile that overrides a static value, and the value
# that means "no profile". Unlike inflow_ref, "const" need not exist as a column.
OVERRIDE_PROFILE_REFS = ("pmax_ref", "pmin_ref", "storage_ini_ref")
CONST_PROFILE_REF = "const"


def override_profile_refs(generator, name):
    """Column *name* of *generator* as profile names, "" where there is no profile.

    None if the column is absent (grid data read from an older results database).
    "const", empty strings and NaN (also the string "nan") all mean no profile.
    """
    if name not in generator.columns:
        return None
    col = generator[name]
    refs = col.astype(object).where(col.notna(), "").astype(str).str.strip()
    # "nan" is what a NaN cell becomes after a round trip through astype(str).
    return refs.mask(refs.eq(CONST_PROFILE_REF) | refs.str.lower().eq("nan"), "")


def _span(index) -> str:
    return "none" if len(index) == 0 else f"{index[0]}..{index[-1]}"


def _rows(rows) -> str:
    return ", ".join(str(r) for r in rows[:10]) + (f" ... (+{len(rows) - 10} more)" if len(rows) > 10 else "")


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
    # Position in grid.node of each consumer's node (-1 if not a node), and the node
    # ids as an Index: per-consumer results are summed onto nodes every timestep.
    consumer_node_pos: object = None
    node_index: object = None

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

    # --- flexible-load reservoirs ------------------------------------------
    # Per consumer with flexible load: the constant flexible demand the reservoir
    # serves (MW), its size (MWh), and whether its level can be held at or above
    # zero. The per-timestep charge limit is demand_avg * flex_fraction / flex_on_off,
    # so the drain can be met timestep by timestep exactly when flex_on_off <= 1;
    # where it cannot, no floor can be imposed without making the LP infeasible.
    flex_drain_mw: dict = field(default_factory=dict)
    flex_storage_cap: dict = field(default_factory=dict)
    flex_has_floor: dict = field(default_factory=dict)

    # --- generator profile references that override a static value ---------
    # Each is a Series of profile column names indexed like grid.generator, with
    # "" where the generator has no profile ("const", empty or NaN in the data),
    # or None when no generator has one. Validated against grid.profiles once.
    pmax_ref: object = None
    pmin_ref: object = None
    storage_ini_ref: object = None

    # --- storage values (lp/storage_value.py) -------------------------------
    # The time factor of a storage generator / flexible-load consumer is a column of
    # grid.profiles (position; -1 = no profile, a factor of 1), or, with the deprecated
    # separate table grid.storagevalue_time (storval_time_legacy), a column of that. The
    # filling-level factor is a column of grid.storagevalue_filling (position). Keyed by
    # generator / consumer id, only for the rows the LP values. Validated once.
    storval_time_legacy: bool = False
    storval_time_pos: dict = field(default_factory=dict)
    storval_filling_pos: dict = field(default_factory=dict)
    storval_flex_time_pos: dict = field(default_factory=dict)
    storval_flex_filling_pos: dict = field(default_factory=dict)

    # --- ramping ------------------------------------------------------------
    ramp_up_pu: object = None
    ramp_down_pu: object = None
    ramp_cap_mw: object = None

    # --- branch flow bounds and NTC ----------------------------------------
    default_ac_flow_bounds: dict = field(default_factory=dict)
    default_dc_flow_bounds: dict = field(default_factory=dict)
    inter_area_rules: object = None
    inter_area_rules_rows: dict = field(default_factory=dict)

    @classmethod
    def from_grid(cls, grid, Bbus, DA, refnodes):
        """Derive everything the builders need from *grid*.

        Bbus/DA/refnodes are passed in rather than computed here: the matrices are
        expensive and the caller already prints progress around them.
        """
        ctx = cls(Bbus=Bbus, DA=DA, refnodes=refnodes)
        ctx._build_node_maps(grid)
        ctx._build_index_sets(grid)
        ctx._build_flexload(grid)
        ctx._build_profile_refs(grid)
        ctx._build_storage_values(grid)
        ctx._build_ramping(grid)
        ctx._build_flow_bounds(grid)
        ctx._build_inter_area_rules(grid)
        return ctx

    # ------------------------------------------------------------------

    def _build_node_maps(self, grid):
        self.generators_at_node = grid.generator.groupby("node").groups
        self.loads_at_node = grid.consumer.groupby("node").groups
        self.branch_from_node = grid.branch.groupby("node_from").groups
        self.branch_to_node = grid.branch.groupby("node_to").groups
        self.dcbranch_from_node = grid.dcbranch.groupby("node_from").groups
        self.dcbranch_to_node = grid.dcbranch.groupby("node_to").groups
        self.node_index = pd.Index(grid.node["id"])
        self.consumer_node_pos = self.node_index.get_indexer(grid.consumer["node"])
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

    def _build_flexload(self, grid):
        for j in self.idx_consumersWithFlexLoad:
            drain = float(grid.consumer.loc[j, "flex_fraction"]) * float(grid.consumer.loc[j, "demand_avg"])
            on_off = float(grid.consumer.loc[j, "flex_on_off"])
            self.flex_drain_mw[j] = drain
            self.flex_storage_cap[j] = float(grid.consumer.loc[j, "flex_storage"]) * drain
            self.flex_has_floor[j] = 0.0 < on_off <= 1.0

    def flex_storage_seed(self, j, level):
        """The reservoir level an LP starts from: *level*, moved inside the bounds it enforces.

        Keeps the first timestep feasible on a level that came from input data
        (flex_storagelevel_init outside [0, 1]) or from a consumer without a floor.
        """
        cap = self.flex_storage_cap[j]
        return min(max(level, 0.0), cap) if self.flex_has_floor[j] else min(level, cap)

    def _build_profile_refs(self, grid):
        """Resolve the generator profile references that override a static value.

        pmax_ref scales installed pmax (availability factor), pmin_ref sets the floor
        as a fraction of installed pmax, storage_ini_ref sets the initial filling as a
        fraction of storage_cap. "const" (the default), an empty cell or NaN means no
        profile. Anything else must name a column of grid.profiles, checked here once
        rather than silently ignored every timestep.
        """
        profile_cols = set(grid.profiles.columns) if grid.profiles is not None else set()
        problems = []
        for name in OVERRIDE_PROFILE_REFS:
            refs = override_profile_refs(grid.generator, name)
            if refs is None:
                setattr(self, name, None)
                continue
            unknown = refs[refs.ne("") & ~refs.isin(profile_cols)]
            if len(unknown) > 0:
                if "desc" in grid.generator.columns:
                    who = [f"{i} ({grid.generator.at[i, 'desc']})" for i in unknown.index]
                else:
                    who = [str(i) for i in unknown.index]
                shown = ", ".join(who[:10]) + (f" ... (+{len(who) - 10} more)" if len(who) > 10 else "")
                problems.append(f"{name}: no profile column(s) {sorted(unknown.unique())} for generator(s) {shown}")
            setattr(self, name, refs if refs.ne("").any() else None)
        if problems:
            raise ValueError(
                "Generator profile references name missing profile columns "
                f'(use "{CONST_PROFILE_REF}" for no profile):\n  ' + "\n  ".join(problems)
            )

    def _build_storage_values(self, grid):
        """Resolve the storage-value curves of the generators with storage and the consumers
        with flexible load (the rows the LP values), and check them once.

        - ``storagevalue_filling`` must have the rows 0..100 (one per percent of filling) and
          a column for every ``storval_filling_ref`` / ``flex_storval_filling`` of those rows.
        - Without ``storagevalue_time``, ``storval_time_ref`` / ``flex_storval_time`` name a
          column of ``grid.profiles``; ``"const"``, an empty cell or NaN is a factor of 1.
        - With the deprecated ``storagevalue_time``, they name a column of that table, for
          every row (the two are not mixed).

        Rows the LP does not value are not checked. Anything else raises ``ValueError``.
        """
        gens = list(self.idx_generatorsWithStorage)
        flex = list(self.idx_consumersWithFlexLoad)
        if not gens and not flex:
            return
        filling = grid.storagevalue_filling
        if filling is None:
            raise ValueError(
                "Storage values: the grid has generators with storage or consumers with flexible load, but no "
                "storage-value filling curves (readProfileData(..., storagevalue_filling=...))"
            )
        if not filling.index.equals(pd.RangeIndex(101)):
            raise ValueError(
                "Storage values: the filling curves (storagevalue_filling) must have the rows 0..100, one per "
                f"percent of filling; they have {len(filling.index)} rows ({_span(filling.index)})"
            )
        legacy = grid.storagevalue_time is not None
        self.storval_time_legacy = legacy
        time_table = grid.storagevalue_time if legacy else grid.profiles
        time_where = "storagevalue_time" if legacy else "the profiles"
        problems = []

        def resolve(table, what, ids, column, columns, where, const_is_one):
            if column not in table.columns:
                problems.append(f"the {what} table has no '{column}' column")
                return {}
            if const_is_one:
                names = override_profile_refs(table, column)
            else:
                names = table[column].astype(object).where(table[column].notna(), "").astype(str)
            out = {}
            unknown = {}
            for i in ids:
                name = names.loc[i]
                if const_is_one and name == "":
                    out[i] = -1
                    continue
                pos = int(columns.get_indexer([name])[0])
                if pos < 0:
                    unknown.setdefault(name, []).append(i)
                out[i] = pos
            for name, rows in unknown.items():
                problems.append(f"{what} {column} '{name}' is not a column of {where}, for {what}(s) {_rows(rows)}")
            return out

        fill_cols = filling.columns
        time_cols = time_table.columns
        gen, cons = grid.generator, grid.consumer
        self.storval_filling_pos = resolve(
            gen, "generator", gens, "storval_filling_ref", fill_cols, "storagevalue_filling", False
        )
        self.storval_time_pos = resolve(gen, "generator", gens, "storval_time_ref", time_cols, time_where, not legacy)
        self.storval_flex_filling_pos = resolve(
            cons, "consumer", flex, "flex_storval_filling", fill_cols, "storagevalue_filling", False
        )
        self.storval_flex_time_pos = resolve(
            cons, "consumer", flex, "flex_storval_time", time_cols, time_where, not legacy
        )
        if problems:
            raise ValueError("Storage values:\n  " + "\n  ".join(problems))

    def _build_ramping(self, grid):
        # Ramp-rate limits in per-unit of installed generator capacity. NaN means unconstrained.
        self.ramp_up_pu = grid.generator["ramp_up_pu"].values.copy() if "ramp_up_pu" in grid.generator.columns else None
        self.ramp_down_pu = (
            grid.generator["ramp_down_pu"].values.copy() if "ramp_down_pu" in grid.generator.columns else None
        )
        self.ramp_cap_mw = pd.to_numeric(grid.generator["pmax"], errors="coerce").fillna(0.0).values

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

    def _build_inter_area_rules(self, grid):
        """Resolve each NTC row to the branches crossing that area boundary.

        Both formulations need this. It used to be built as a side effect of the
        single-period constraint builder, so multi-period depended on construction
        order to see it at all.
        """
        self.inter_area_rules = grid.inter_area_rules
        if self.inter_area_rules is None or self.inter_area_rules.shape[0] == 0:
            return
        for cidx, (_, row) in enumerate(self.inter_area_rules.iterrows()):
            area_from = row["area_from"]
            area_to = row["area_to"]
            # Get branches between the two areas
            br_ac = grid.getInterAreaBranches(area_from=area_from, area_to=area_to, acdc="ac")
            br_dc = grid.getInterAreaBranches(area_from=area_from, area_to=area_to, acdc="dc")
            self.inter_area_rules_rows[cidx] = {
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
