"""Everything the real-time extension derives once from the grid and the DA locks.

The RT counterpart of :class:`~powergama.lp.context.GridContext`, held by the
extension rather than the core, so a normal run never computes any of it.
"""

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class RtContext:
    """RT target references, generator classification and border branches. Built once."""

    # --- day-ahead target references and the entities that carry one --------
    target_ref: object = None
    target_gen_indices: set = field(default_factory=set)
    consumer_target_ref: object = None
    consumer_target_indices: set = field(default_factory=set)
    storage_target_ref: object = None
    storage_target_indices: set = field(default_factory=set)
    pump_target_ref: object = None
    pump_target_indices: set = field(default_factory=set)

    # --- generator classification (read only by the diagnostics, except
    #     wind/solar/hydro_ror which the deviation pricing uses) -------------
    idx_peak_gen: set = field(default_factory=set)
    idx_normal_gas: set = field(default_factory=set)
    idx_wind: set = field(default_factory=set)
    idx_solar: set = field(default_factory=set)
    idx_nuclear: set = field(default_factory=set)
    idx_biomass: set = field(default_factory=set)
    idx_fossil_other: set = field(default_factory=set)
    idx_storage_gens: set = field(default_factory=set)
    idx_hydro_ror: set = field(default_factory=set)

    # --- border branches, identified by the DA lock tables --------------------
    idx_border_ac: set = field(default_factory=set)
    idx_border_dc: set = field(default_factory=set)
    border_ac_sign: dict = field(default_factory=dict)
    border_dc_sign: dict = field(default_factory=dict)

    @classmethod
    def from_grid(cls, grid, ctx, locks):
        """Derive the RT lookups from *grid*, the core *ctx* and the loaded *locks*."""
        rt_ctx = cls(
            idx_border_ac=locks.idx_border_ac,
            idx_border_dc=locks.idx_border_dc,
            border_ac_sign=locks.border_ac_sign,
            border_dc_sign=locks.border_dc_sign,
        )
        rt_ctx._build_target_refs(grid, ctx)
        rt_ctx._build_classification(grid, ctx)
        return rt_ctx

    def _build_target_refs(self, grid, ctx):
        # Generator dispatch targets.
        self.target_ref = grid.generator["rt_target_ref"] if "rt_target_ref" in grid.generator.columns else None
        self.target_gen_indices = {
            int(i)
            for i in grid.generator.index
            if self.target_ref is not None
            and isinstance(self.target_ref.loc[i], str)
            and str(self.target_ref.loc[i]).strip()
        }
        # Consumer (flexible load) demand targets.
        self.consumer_target_ref = grid.consumer["rt_target_ref"] if "rt_target_ref" in grid.consumer.columns else None
        self.consumer_target_indices = {
            int(i)
            for i in grid.consumer.index
            if self.consumer_target_ref is not None
            and isinstance(self.consumer_target_ref.loc[i], str)
            and str(self.consumer_target_ref.loc[i]).strip()
            and int(i) in ctx.idx_consumersWithFlexLoad
        }
        # Pumping targets of pumped storage, as a fraction of pump_cap.
        self.pump_target_ref = (
            grid.generator["rt_pump_target_ref"] if "rt_pump_target_ref" in grid.generator.columns else None
        )
        self.pump_target_indices = {
            int(i)
            for i in grid.generator.index
            if self.pump_target_ref is not None
            and isinstance(self.pump_target_ref.loc[i], str)
            and str(self.pump_target_ref.loc[i]).strip()
            and int(i) in ctx.idx_generatorsWithPumping
        }
        # Storage (reservoir/battery) filling targets, as a fraction [0, 1] of capacity, so
        # that storage state continuity matches DA.
        self.storage_target_ref = (
            grid.generator["rt_storage_target_ref"] if "rt_storage_target_ref" in grid.generator.columns else None
        )
        self.storage_target_indices = {
            int(i)
            for i in grid.generator.index
            if self.storage_target_ref is not None
            and isinstance(self.storage_target_ref.loc[i], str)
            and str(self.storage_target_ref.loc[i]).strip()
            and int(i) in ctx.idx_generatorsWithStorage
        }

    def _build_classification(self, grid, ctx):
        # Classify generators by type/tag for differentiated deviation pricing and for
        # RT deviation cost attribution in debug output. By type only, no country filter.
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
        _storage_gens_set = ctx.idx_generatorsWithStorage
        self.idx_peak_gen = {
            int(i) for i in grid.generator.index if _gtype.loc[i] == "fossil_gas" and "[RT]" in str(_gdesc.loc[i])
        }
        self.idx_normal_gas = {
            int(i) for i in grid.generator.index if _gtype.loc[i] == "fossil_gas" and "[RT]" not in str(_gdesc.loc[i])
        }
        self.idx_wind = {int(i) for i in grid.generator.index if _gtype.loc[i] in {"wind_off", "wind_on", "wind"}}
        self.idx_solar = {int(i) for i in grid.generator.index if _gtype.loc[i] == "solar"}
        self.idx_nuclear = {int(i) for i in grid.generator.index if _gtype.loc[i] == "nuclear"}
        self.idx_biomass = {int(i) for i in grid.generator.index if _gtype.loc[i] == "biomass"}
        self.idx_fossil_other = {int(i) for i in grid.generator.index if _gtype.loc[i] == "fossil_other"}
        self.idx_storage_gens = set(_storage_gens_set)
        self.idx_hydro_ror = {
            int(i)
            for i in grid.generator.index
            if _gtype.loc[i] == "hydro"
            and int(i) not in _storage_gens_set
            and float(pd.to_numeric(grid.generator.loc[i, "pump_cap"], errors="coerce") or 0.0) <= 0.0
            and float(pd.to_numeric(grid.generator.loc[i, "storage_cap"], errors="coerce") or 0.0) <= 0.0
        }
