"""Everything the real-time extension derives once from the grid and the DA locks.

The RT counterpart of :class:`~powergama.lp.context.GridContext`, held by the
extension rather than the core, so a normal run never computes any of it.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..context import CONST_PROFILE_REF, override_profile_refs

# The deviation prices of each kind of tracked row: (table, price up, price down, the target
# column that makes a row tracked). "Up" is more than DA: more output, pumping or demand.
# Each price has an optional profile multiplier, the same name with "_ref".
PRICE_COLUMNS = {
    "gen": ("generator", "rt_price_up", "rt_price_down", "rt_target_ref"),
    "pump": ("generator", "rt_pump_price_up", "rt_pump_price_down", "rt_pump_target_ref"),
    "cons": ("consumer", "rt_price_up", "rt_price_down", "rt_target_ref"),
}


@dataclass
class RtContext:
    """RT target references, deviation prices and border branches. Built once."""

    # --- day-ahead target references and the entities that carry one --------
    target_ref: object = None
    target_gen_indices: set = field(default_factory=set)
    consumer_target_ref: object = None
    consumer_target_indices: set = field(default_factory=set)
    pump_target_ref: object = None
    pump_target_indices: set = field(default_factory=set)

    # --- deviation prices (EUR/MWh), per tracked row: price(t) = constant x profile(t) ----
    # Each is a PriceRefs (or None when no row of that kind is tracked).
    gen_price_up: object = None
    gen_price_down: object = None
    pump_price_up: object = None
    pump_price_down: object = None
    cons_price_up: object = None
    cons_price_down: object = None

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
        rt_ctx._build_prices(grid)
        return rt_ctx

    def _build_target_refs(self, grid, ctx):
        # Storage is settled on its power, generation and pumping, not on its filling level.
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

    def _build_prices(self, grid):
        """Resolve the deviation prices of the tracked rows, and check them once.

        Every row with a target needs both prices, as finite numbers; their ``*_ref``
        columns are checked for every row, as the core checks ``pmax_ref``.
        """
        problems = []
        tracked = {
            "gen": self.target_gen_indices,
            "pump": self.pump_target_indices,
            "cons": self.consumer_target_indices,
        }
        for kind, (table_name, up, down, target) in PRICE_COLUMNS.items():
            table = getattr(grid, table_name)
            rows = sorted(tracked[kind])
            for direction, column in (("up", up), ("down", down)):
                refs = _price_refs(table, table_name, column, rows, target, grid.profiles, problems)
                setattr(self, f"{kind}_price_{direction}", refs)
        if problems:
            raise ValueError("RT deviation prices:\n  " + "\n  ".join(problems))


@dataclass
class PriceRefs:
    """One deviation price of the tracked rows of one kind: ``constant x profile(t)``.

    ``position`` is the column of ``grid.profiles`` (-1: no profile, a factor of 1).
    """

    rows: np.ndarray
    constant: np.ndarray
    position: np.ndarray

    def at(self, profiles, timestep) -> np.ndarray:
        """The prices at *timestep*, aligned with ``rows``."""
        values = self.constant.copy()
        used = self.position >= 0
        if used.any():
            factors = profiles.iloc[profiles.index.get_loc(timestep), self.position[used]].to_numpy(dtype=float)
            values[used] = self.constant[used] * factors
        return values


def _who(table, rows) -> str:
    if "desc" in table.columns:
        who = [f"{i} ({table.at[i, 'desc']})" for i in rows]
    else:
        who = [str(i) for i in rows]
    return ", ".join(who[:10]) + (f" ... (+{len(who) - 10} more)" if len(who) > 10 else "")


def _price_refs(table, table_name, column, rows, target, profiles, problems):
    """The :class:`PriceRefs` of *column* for the tracked *rows*; problems are appended."""
    profile_cols = profiles.columns if profiles is not None else pd.Index([])
    refs = override_profile_refs(table, f"{column}_ref")
    if refs is not None:
        unknown = refs[refs.ne("") & ~refs.isin(profile_cols)]
        if len(unknown):
            problems.append(
                f"{table_name} {column}_ref: no profile column(s) {sorted(unknown.unique())} for row(s) "
                f'{_who(table, list(unknown.index))} (use "{CONST_PROFILE_REF}" for no profile)'
            )
            return None
    if not rows:
        return None
    if column in table.columns:
        constant = pd.to_numeric(table.loc[rows, column], errors="coerce").to_numpy(dtype=float)
    else:
        constant = np.full(len(rows), np.nan)
    missing = [r for r, c in zip(rows, constant) if not np.isfinite(c)]
    if missing:
        problems.append(
            f"{table_name} {column} is missing for row(s) {_who(table, missing)}, which have a target "
            f"({target}): every tracked row needs rt_price_up and rt_price_down (EUR/MWh; "
            "powergama.lp.rt.price_columns writes them)"
        )
        return None
    names = refs.loc[rows] if refs is not None else pd.Series("", index=rows)
    position = np.array([-1 if n == "" else int(profile_cols.get_loc(n)) for n in names], dtype=int)
    for n in sorted(set(names) - {""}):
        if profiles[n].isna().any():
            problems.append(
                f"{table_name} {column}_ref profile '{n}' has NaN, for row(s) {_who(table, list(names.index[names.eq(n)]))}"
            )
    return PriceRefs(rows=np.asarray(rows, dtype=int), constant=constant, position=position)
