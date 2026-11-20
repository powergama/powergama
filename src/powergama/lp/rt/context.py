"""Everything the real-time extension derives once from the grid.

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
    """RT target references, deviation prices and inter-area rule targets. Built once."""

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

    # --- inter-area rules held to their DA summed flow, by position in the table ----
    # Each is a RuleTarget: a rule with a da_flow_ref and a balancing penalty or a lock.
    rules: dict = field(default_factory=dict)

    @classmethod
    def from_grid(cls, grid, ctx):
        """Derive the RT lookups from *grid* and the core *ctx*."""
        rt_ctx = cls()
        rt_ctx._build_target_refs(grid, ctx)
        rt_ctx._build_prices(grid)
        rt_ctx.rules = build_rule_targets(grid, ctx)
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


@dataclass
class RuleTarget:
    """One inter-area rule RT holds to its DA summed flow.

    ``position`` is the rule's position in ``grid.inter_area_rules`` (the core's
    ``inter_area_rules_rows`` key); ``column`` the profile column of the DA summed flow
    (``da_flow_ref``) and ``profile_position`` its position in ``grid.profiles``.
    ``balancing_penalty`` is EUR/MWh (0: none); ``lock_share`` is lambda (None: no lock);
    ``width_forward`` / ``width_backward`` are the lock window either side of DA, MW
    (None: that side open).
    """

    position: int
    area_from: str
    area_to: str
    column: str
    profile_position: int
    balancing_penalty: float
    lock_share: float | None
    width_forward: float | None
    width_backward: float | None


def _lock_width(share: float, ntc: float) -> float | None:
    """(1 - share) x ntc; lambda = 1 is a width of 0 whatever the NTC, an infinite NTC no bound."""
    if share >= 1.0:
        return 0.0
    if not np.isfinite(ntc):
        return None
    return (1.0 - share) * max(0.0, ntc)


def build_rule_targets(grid, ctx) -> dict[int, RuleTarget]:
    """The inter-area rules with a ``da_flow_ref`` and a balancing penalty or a lock, by position.

    A rule without ``da_flow_ref`` gets neither, whatever its ``balancing_penalty`` and
    ``lock_share``. Raises ``ValueError`` on a ``da_flow_ref``
    that is not a profile column, NaN in a referenced column, a negative balancing penalty, a lock share outside [0, 1], or a rule with a
    ``da_flow_ref`` and no branch across its boundary.
    """
    rules = grid.inter_area_rules
    if rules is None or len(rules) == 0:
        return {}
    refs = override_profile_refs(rules, "da_flow_ref")
    if refs is None:
        return {}
    profiles = grid.profiles
    profile_cols = profiles.columns if profiles is not None else pd.Index([])
    nan = pd.Series(np.nan, index=rules.index)
    penalty = pd.to_numeric(
        rules["balancing_penalty"] if "balancing_penalty" in rules.columns else nan, errors="coerce"
    )
    share = pd.to_numeric(rules["lock_share"] if "lock_share" in rules.columns else nan, errors="coerce")
    problems = []
    out = {}
    for position, k in enumerate(rules.index):
        name = refs.loc[k]
        if name == "":
            continue
        who = f"inter_area_rules row {k} ({rules.at[k, 'area_from']} -> {rules.at[k, 'area_to']})"
        p = 0.0 if np.isnan(penalty.loc[k]) else float(penalty.loc[k])
        x = None if np.isnan(share.loc[k]) else float(share.loc[k])
        if name not in profile_cols:
            problems.append(f"{who}: da_flow_ref '{name}' is not a profile column")
            continue
        if profiles[name].isna().any():
            problems.append(f"{who}: profile '{name}' (da_flow_ref) has NaN")
        if not np.isfinite(p) or p < 0.0:
            problems.append(f"{who}: balancing_penalty must be >= 0, not {p}")
        if x is not None and not 0.0 <= x <= 1.0:
            problems.append(f"{who}: lock_share must be within [0, 1], not {x}")
        row = ctx.inter_area_rules_rows[position]
        if not (row["ac_pos"] or row["ac_neg"] or row["dc_pos"] or row["dc_neg"]):
            problems.append(f"{who}: has a da_flow_ref but no branch across its boundary")
            continue
        if p == 0.0 and x is None:
            continue
        out[position] = RuleTarget(
            position=position,
            area_from=str(rules.at[k, "area_from"]),
            area_to=str(rules.at[k, "area_to"]),
            column=name,
            profile_position=int(profile_cols.get_loc(name)),
            balancing_penalty=p,
            lock_share=x,
            width_forward=None if x is None else _lock_width(x, row["ntc_forward"]),
            width_backward=None if x is None else _lock_width(x, row["ntc_backward"]),
        )
    if problems:
        raise ValueError("RT inter-area rules:\n  " + "\n  ".join(problems))
    return out
