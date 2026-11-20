"""A readable summary of the DA-derived ("channel") columns of a balancing run's inputs.

A *channel* is a profile column of a :class:`~powergama.lp.rt.prepare.RefColumns`
(``da_gen_<d>``, ``da_pump_<d>``, ``da_cons_<d>``, ``da_carry_<d>``, ``da_storage_<d>``,
``da_headroom_<d>``, the ``da_price`` family, ``da_flow_<k>``, ``rt_target_zero``).
:func:`summarize_channels` lists, for each channel, the RT rows (generator, consumer and
inter-area rule) that reference it and how; :func:`write_channel_summary` writes that as ``channels.csv`` and a
short overview, ``prepare_rt_summary.md``. :func:`~powergama.lp.rt.prepare.prepare_rt`
writes both next to ``da_solution.parquet``.

Nothing here knows about energy types or areas: rows are grouped by the ``area`` (from the
node table) and ``type`` columns as they are.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .prepare import DA_INDEX, ZERO_TARGET, RefColumns, _capacity, _is_flexible, _ref_frame, override_profile_refs

SUMMARY_FILE = "prepare_rt_summary.md"
CHANNELS_FILE = "channels.csv"

FAMILIES = (
    "da_gen",
    "da_pump",
    "da_cons",
    "da_carry",
    "da_storage",
    "da_headroom",
    "da_price",
    "da_flow",
    "zero",
    "other",
)
"""The channel families, in the order of the summary. ``other`` is any column not named by
:mod:`~powergama.lp.rt.prepare`."""

FAMILY_TITLE = {
    "da_gen": "DA generator output",
    "da_pump": "DA pumping",
    "da_cons": "DA flexible demand",
    "da_carry": "RT availability less the DA curtailment",
    "da_storage": "DA storage filling",
    "da_headroom": "peak-tranche headroom",
    "da_price": "deviation price of a storage unit (EUR/MWh)",
    "da_flow": "DA summed flow of an inter-area rule (MW)",
    "zero": "zero",
    "other": "other columns",
}

FAMILY_FORMULA = {
    "da_gen": "DA output of d / pmax of d, clipped to [0, 1]",
    "da_pump": "DA pumping of d / pump_cap of d, clipped to [0, 1]",
    "da_cons": "DA flexible demand of d / demand_avg of d, clipped to [0, 1]",
    "da_carry": "(RT availability + DA output - DA availability), within [0, pmax], / pmax of d",
    "da_storage": "DA filling of d at the start of the timestep / storage_cap of d, clipped to [0, 1] "
    "(at the DA run's first timestep: the filling DA started from)",
    "da_headroom": "(pmax of the unit - DA output of the unit) / pmax of the unit, clipped to [0, 1]; "
    "the unit is the block_group of DA row d, d its first row",
    "da_price": "c0 +/- the marginal cost of d, its storage value v at the DA filling level (storage_price x "
    "F(DA filling) x T(t), or the flexible-load value): generation c0 + (fuelcost + v) up, c0 - (fuelcost + v) "
    "down; pumping c0 -/+ (v x pump_efficiency - pump_deadband); flexible demand c0 +/- v",
    "da_flow": "the DA flows of the AC and DC branches across the boundary of rule k, summed area_from -> "
    "area_to (the signs of the NTC constraint), MW, not clipped",
    "zero": "0",
    "other": "set by the caller",
}

FAMILY_PATTERN = {
    "da_gen": "da_gen_<d>",
    "da_pump": "da_pump_<d>",
    "da_cons": "da_cons_<d>",
    "da_carry": "da_carry_<d>",
    "da_storage": "da_storage_<d>",
    "da_headroom": "da_headroom_<d>",
    "da_price": "da_[pump_|cons_]price_<up|down>_<d>",
    "da_flow": "da_flow_<k>",
    "zero": ZERO_TARGET,
    "other": "<name>",
}

USAGES = (
    "target",
    "pump_target",
    "cons_target",
    "fix",
    "cap",
    "pmin",
    "storage_ini",
    "price_up",
    "price_down",
    "pump_price_up",
    "pump_price_down",
    "cons_price_up",
    "cons_price_down",
    "flow_ref",
)
"""How an RT row uses a channel, in the order of the summary:

- ``target``: generator ``rt_target_ref``;
- ``pump_target``: generator ``rt_pump_target_ref``;
- ``cons_target``: consumer ``rt_target_ref``;
- ``fix``: generator ``pmin_ref`` and ``pmax_ref``, both to the channel;
- ``cap``: generator ``pmax_ref`` only;
- ``pmin``: generator ``pmin_ref`` only;
- ``storage_ini``: generator ``storage_ini_ref``;
- ``price_up``, ``price_down``: generator ``rt_price_up_ref``, ``rt_price_down_ref``;
- ``pump_price_up``, ``pump_price_down``: generator ``rt_pump_price_up_ref``, ``rt_pump_price_down_ref``;
- ``cons_price_up``, ``cons_price_down``: consumer ``rt_price_up_ref``, ``rt_price_down_ref``;
- ``flow_ref``: inter-area rule ``da_flow_ref`` (its balancing penalty and lock act on that flow).
"""

USAGE_TEXT = {
    "target": "tracking target (`rt_target_ref`)",
    "pump_target": "pumping target (`rt_pump_target_ref`)",
    "cons_target": "flexible-demand target (consumer `rt_target_ref`)",
    "fix": "fixed (`pmin_ref` = `pmax_ref`)",
    "cap": "upper bound (`pmax_ref` only)",
    "pmin": "lower bound (`pmin_ref` only)",
    "storage_ini": "initial filling (`storage_ini_ref`)",
    "price_up": "deviation price up (`rt_price_up_ref`)",
    "price_down": "deviation price down (`rt_price_down_ref`)",
    "pump_price_up": "pumping deviation price up (`rt_pump_price_up_ref`)",
    "pump_price_down": "pumping deviation price down (`rt_pump_price_down_ref`)",
    "cons_price_up": "flexible-demand deviation price up (consumer `rt_price_up_ref`)",
    "cons_price_down": "flexible-demand deviation price down (consumer `rt_price_down_ref`)",
    "flow_ref": "DA summed flow (`da_flow_ref`)",
}

USAGE_TABLE = {
    "target": "generator",
    "pump_target": "generator",
    "cons_target": "consumer",
    "fix": "generator",
    "cap": "generator",
    "pmin": "generator",
    "storage_ini": "generator",
    "price_up": "generator",
    "price_down": "generator",
    "pump_price_up": "generator",
    "pump_price_down": "generator",
    "cons_price_up": "consumer",
    "cons_price_down": "consumer",
    "flow_ref": "inter_area_rules",
}

CHANNEL_COLUMNS = [
    "name",
    "family",
    "formula",
    "da_row",
    "n_rows",
    "usages",
    "rt_rows",
    "usage",
    "desc",
    "node",
    "area",
    "type",
]
"""The columns of ``channels.csv``. ``rt_rows`` (``<table>:<row>``), ``usage`` (the usages
of that row, joined by ``+``), ``desc``, ``node``, ``area`` and ``type`` are lists joined
by ``;``, one entry per referencing row in the same order; ``usages`` is the set of usages
over all rows, joined by ``;``. ``da_row`` is the DA row the channel is keyed on (for
``da_flow_<k>``, the rule row k)."""

_FAMILY_RE = re.compile(r"^(da_gen|da_pump|da_cons|da_carry|da_storage|da_headroom|da_flow)_(\d+)$")

_PRICE_RE = re.compile(r"^da_(?:pump_|cons_)?price_(?:up|down)_(\d+)$")

ADDED_ROW_FAMILIES = frozenset({"da_headroom"})
"""Families keyed on a DA row but referenced by rows added for RT (no DA row of their own)."""


def channel_family(name: str) -> tuple[str, int | None]:
    """The family of a channel name, and its DA row (None for ``zero`` and ``other``)."""
    if name == ZERO_TARGET:
        return "zero", None
    match = _FAMILY_RE.match(str(name))
    if match is None:
        price = _PRICE_RE.match(str(name))
        if price is not None:
            return "da_price", int(price.group(1))
        return "other", None
    return match.group(1), int(match.group(2))


def _effective(table: pd.DataFrame, part: pd.DataFrame, col: str, names: set) -> pd.Series:
    """The reference in *col* of each table row once *part* is applied; non-channels are NaN."""
    if col in table.columns:
        values = table[col].astype(object).copy()
    else:
        values = pd.Series(np.nan, index=table.index, dtype=object)
    if col in part.columns:
        cells = part[col].dropna()
        values.loc[cells.index] = cells.astype(object)
    return values.where(values.map(lambda v: isinstance(v, str) and v in names))


def _references(
    refs: RefColumns, generator: pd.DataFrame, consumer: pd.DataFrame, rules: pd.DataFrame | None = None
) -> pd.DataFrame:
    """One row per (channel, table, RT row, usage)."""
    names = set(refs.profiles.columns)
    out: list[tuple] = []
    gen_part = _ref_frame(refs.generator)
    cons_part = _ref_frame(refs.consumer)
    rule_part = _ref_frame(refs.inter_area_rules)
    if rules is None and not rule_part.empty:
        raise ValueError("RefColumns sets inter-area rules, but no inter-area table was given")
    checked = [("generator", generator, gen_part), ("consumer", consumer, cons_part)]
    if rules is not None:
        checked.append(("inter_area_rules", rules, rule_part))
    for what, table, part in checked:
        missing = part.index.difference(table.index)
        if len(missing):
            raise ValueError(f"RefColumns sets RT {what} row(s) {list(missing[:5])}, which the {what} table lacks")

    eff = {col: _effective(generator, gen_part, col, names) for col in GEN_REF_COLUMNS + tuple(GEN_PRICE_USAGES)}
    for r in generator.index:
        tgt, pump, lo, hi, ini = (eff[c].at[r] for c in GEN_REF_COLUMNS)
        if isinstance(tgt, str):
            out.append((tgt, "generator", r, "target"))
        if isinstance(pump, str):
            out.append((pump, "generator", r, "pump_target"))
        if isinstance(lo, str) and lo == hi:
            out.append((lo, "generator", r, "fix"))
        else:
            if isinstance(hi, str):
                out.append((hi, "generator", r, "cap"))
            if isinstance(lo, str):
                out.append((lo, "generator", r, "pmin"))
        if isinstance(ini, str):
            out.append((ini, "generator", r, "storage_ini"))
        for col, usage in GEN_PRICE_USAGES.items():
            if isinstance(eff[col].at[r], str):
                out.append((eff[col].at[r], "generator", r, usage))
    cons_tgt = _effective(consumer, cons_part, "rt_target_ref", names)
    cons_eff = {col: _effective(consumer, cons_part, col, names) for col in CONS_PRICE_USAGES}
    for r in consumer.index:
        if isinstance(cons_tgt.at[r], str):
            out.append((cons_tgt.at[r], "consumer", r, "cons_target"))
        for col, usage in CONS_PRICE_USAGES.items():
            if isinstance(cons_eff[col].at[r], str):
                out.append((cons_eff[col].at[r], "consumer", r, usage))
    if rules is not None:
        flow = _effective(rules, rule_part, "da_flow_ref", names)
        for r in rules.index:
            if isinstance(flow.at[r], str):
                out.append((flow.at[r], "inter_area_rules", r, "flow_ref"))
    return pd.DataFrame(out, columns=["name", "table", "row", "usage"])


GEN_REF_COLUMNS = ("rt_target_ref", "rt_pump_target_ref", "pmin_ref", "pmax_ref", "storage_ini_ref")
GEN_PRICE_USAGES = {
    "rt_price_up_ref": "price_up",
    "rt_price_down_ref": "price_down",
    "rt_pump_price_up_ref": "pump_price_up",
    "rt_pump_price_down_ref": "pump_price_down",
}
CONS_PRICE_USAGES = {"rt_price_up_ref": "cons_price_up", "rt_price_down_ref": "cons_price_down"}


def _rule_info(rules: pd.DataFrame) -> pd.DataFrame:
    """``desc`` (balancing penalty and lock share), ``node`` (""), ``area`` (``from->to``), ``type`` ("") of each rule."""
    info = pd.DataFrame(index=rules.index)
    nan = pd.Series(np.nan, index=rules.index)
    penalty = pd.to_numeric(
        rules["balancing_penalty"] if "balancing_penalty" in rules.columns else nan, errors="coerce"
    )
    share = pd.to_numeric(rules["lock_share"] if "lock_share" in rules.columns else nan, errors="coerce")
    info["desc"] = [
        "balancing_penalty " + (f"{p:g}" if np.isfinite(p) else "-") + ", lock_share " + (f"{x:g}" if np.isfinite(x) else "-")
        for p, x in zip(penalty, share)
    ]
    info["node"] = ""
    info["area"] = rules["area_from"].astype(str) + "->" + rules["area_to"].astype(str)
    info["type"] = ""
    return info


def _row_info(table: pd.DataFrame, node: pd.DataFrame | None) -> pd.DataFrame:
    """``desc``, ``node``, ``area``, ``type`` of each row of an RT table ("" where unknown)."""
    info = pd.DataFrame(index=table.index)
    for col in ("desc", "node", "type"):
        info[col] = table[col].astype(object) if col in table.columns else ""
    if "area" in table.columns:
        info["area"] = table["area"].astype(object)
    elif node is not None and {"id", "area"} <= set(node.columns) and "node" in table.columns:
        areas = node.drop_duplicates("id").set_index("id")["area"]
        info["area"] = table["node"].map(areas).astype(object)
    else:
        info["area"] = ""
    return info.where(info.notna(), "").map(str)


def _join(values) -> str:
    return ";".join(str(v) for v in values)


def summarize_channels(
    refs: RefColumns,
    generator: pd.DataFrame,
    consumer: pd.DataFrame,
    *,
    da_index: str = DA_INDEX,
    node: pd.DataFrame | None = None,
    inter_area_rules: pd.DataFrame | None = None,
    sources: dict[str, str] | None = None,
    header: dict[str, str] | None = None,
) -> tuple[str, pd.DataFrame]:
    """The channels of *refs*, as a short Markdown overview and a table.

    Parameters
    ----------
    refs:
        The combined :class:`~powergama.lp.rt.prepare.RefColumns` (targets, pins, carried
        curtailment, initial storage, ...). Every column of ``refs.profiles`` is a channel.
    generator, consumer:
        The RT tables, before or after :func:`~powergama.lp.rt.prepare.apply_refs`: a cell
        *refs* sets wins over the table's, a cell it leaves unset (NA) is read from the table.
        ``desc``, ``node``, ``type`` (and ``area``) are reported as they are.
    da_index:
        The column with the DA row of each RT row; only reported (``da_row`` comes from the
        channel name, which is keyed on the DA row).
    node:
        The node table (``id``, ``area``), for the ``area`` of each row through its ``node``.
        Not needed if the tables have an ``area`` column; without either, no area grouping.
    inter_area_rules:
        The RT inter-area table, if *refs* sets ``da_flow_ref`` on it (or it has one); its
        rows are reported as ``inter_area_rules:<row>``, their ``area`` as ``from->to`` and
        their ``desc`` as the balancing penalty and lock share. *refs* setting rules without it raises.
    sources:
        Where each family comes from, by family name (``da_gen``, ``da_carry``, ...), for
        the summary; a family without one is reported without a source.
    header:
        Lines for the top of the summary, ``{label: text}`` (e.g. the DA run, its time
        range, the files written, the option counts), in the given order.

    Returns
    -------
    markdown:
        The overview: the header, then per family its formula, source, and the rows using it
        grouped by usage, area and type (counts only), then the rows without any channel.
    channels:
        One row per channel, in the order of ``refs.profiles``; see :data:`CHANNEL_COLUMNS`.
        A channel no row references has ``n_rows`` 0.
    """
    sources = dict(sources or {})
    rules = inter_area_rules
    if rules is not None:
        rules = _with_refs(rules, _ref_frame(refs.inter_area_rules))
    long = _references(refs, generator, consumer, rules)
    _check_da_rows(long, generator, consumer, da_index)
    info = {"generator": _row_info(generator, node), "consumer": _row_info(consumer, node)}
    if rules is not None:
        info["inter_area_rules"] = _rule_info(rules)

    rows = []
    for name in refs.profiles.columns:
        family, d = channel_family(name)
        used = long[long["name"] == name]
        entries: list[tuple[str, object, str]] = []
        for (what, r), group in used.groupby(["table", "row"], sort=False):
            entries.append((what, r, "+".join(u for u in USAGES if u in set(group["usage"]))))
        entries.sort(key=lambda e: (TABLE_ORDER.index(e[0]), e[1]))
        details = [info[what].loc[r] for what, r, _ in entries]
        rows.append(
            {
                "name": name,
                "family": family,
                "formula": FAMILY_FORMULA[family],
                "da_row": pd.NA if d is None else d,
                "n_rows": len(entries),
                "usages": _join(u for u in USAGES if u in set(used["usage"])),
                "rt_rows": _join(f"{what}:{r}" for what, r, _ in entries),
                "usage": _join(u for _, _, u in entries),
                "desc": _join(x["desc"] for x in details),
                "node": _join(x["node"] for x in details),
                "area": _join(x["area"] for x in details),
                "type": _join(x["type"] for x in details),
            }
        )
    channels = pd.DataFrame(rows, columns=CHANNEL_COLUMNS)
    channels["da_row"] = channels["da_row"].astype("Int64")

    lines = ["# prepare_rt channels", ""]
    for label, text in (header or {}).items():
        lines.append(f"- **{label}**: {text}")
    if header:
        lines.append("")
    n_gen = long.loc[long["table"] == "generator", "row"].nunique()
    n_cons = long.loc[long["table"] == "consumer", "row"].nunique()
    users = f"{_n(n_gen, 'generator')} and {_n(n_cons, 'consumer')}"
    if rules is not None:
        n_rules = long.loc[long["table"] == "inter_area_rules", "row"].nunique()
        users = f"{_n(n_gen, 'generator')}, {_n(n_cons, 'consumer')} and {_n(n_rules, 'inter-area rule')}"
    lines.append(
        f"{_n(len(channels), 'channel')} (DA-derived profile columns), referenced by {users}. "
        f"One row per channel, with the rows using it: `{CHANNELS_FILE}`."
    )
    for family in FAMILIES:
        names = channels.loc[channels["family"] == family, "name"]
        if names.empty:
            continue
        lines += ["", f"## `{FAMILY_PATTERN[family]}`: {FAMILY_TITLE[family]} ({_n(len(names), 'channel')})", ""]
        lines.append(f"- Formula: {FAMILY_FORMULA[family]}.")
        if family in sources:
            lines.append(f"- Source: {sources[family]}.")
        used = long[long["name"].isin(set(names))]
        if used.empty:
            lines.append("- Not referenced by any row.")
        for usage in USAGES:
            sub = used[used["usage"] == usage]
            if sub.empty:
                continue
            what = USAGE_TABLE[usage]
            details = info[what].loc[sorted(set(sub["row"]))]
            groups = _groups(details)
            lines.append(
                f"- Used as {USAGE_TEXT[usage]} by {_n(len(details), TABLE_NOUN[what])}"
                + (f": {groups}" if groups else "")
            )
            if what == "inter_area_rules":
                for r, row in details.iterrows():
                    lines.append(f"  - rule {r} ({row['area']}): {row['desc']}")
        unused = int((channels.loc[names.index, "n_rows"] == 0).sum())
        if unused and not used.empty:
            lines.append(f"- {unused} of them not referenced by any row.")

    lines += ["", "## Not covered", ""]
    covered = {what: set(long.loc[long["table"] == what, "row"]) for what in TABLE_ORDER}
    any_left = False
    checked = [("generator", generator), ("consumer", consumer)]
    if rules is not None:
        checked.append(("inter_area_rules", rules))
    for what, table in checked:
        left = [r for r in table.index if r not in covered[what]]
        if not left:
            continue
        any_left = True
        reasons = pd.Series([_uncovered_reason(what, table, r) for r in left], index=left)
        lines.append(f"- {_n(len(left), TABLE_NOUN[what])} without any channel:")
        for reason, idx in reasons.groupby(reasons, sort=False).groups.items():
            groups = _groups(info[what].loc[list(idx)])
            lines.append(f"  - {reason} ({len(idx)})" + (f": {groups}" if groups else ""))
    if not any_left:
        lines.append(
            "Every generator and consumer row uses a channel."
            if rules is None
            else "Every generator, consumer and inter-area rule row uses a channel."
        )
    return "\n".join(lines) + "\n", channels


def _check_da_rows(long: pd.DataFrame, generator: pd.DataFrame, consumer: pd.DataFrame, da_index: str):
    """A channel keyed on DA row d is only referenced by RT rows of DA row d (where *da_index* is known).

    A headroom channel (:data:`ADDED_ROW_FAMILIES`) is only referenced by rows without a DA row.
    """
    tables = {"generator": generator, "consumer": consumer}
    for (what, name), group in long.groupby(["table", "name"]):
        family, d = channel_family(name)
        if what == "inter_area_rules":
            # keyed on the rule's own row, not on a DA row
            bad = [r for r in sorted(set(group["row"])) if d is not None and r != d]
            if bad:
                raise ValueError(f"Inter-area rule(s) {bad[:5]} reference '{name}', the DA flow of rule {d}")
            continue
        table = tables[what]
        if d is None or da_index not in table.columns:
            continue
        mapped = pd.to_numeric(table.loc[sorted(set(group["row"])), da_index], errors="coerce")
        if family in ADDED_ROW_FAMILIES:
            bad = mapped.index[mapped.notna()]
            if len(bad):
                raise ValueError(f"RT {what} row(s) {list(bad[:5])} reference '{name}' but have a {da_index}")
            continue
        bad = mapped.index[mapped.ne(d)]
        if len(bad):
            raise ValueError(f"RT {what} row(s) {list(bad[:5])} reference '{name}' but their {da_index} is not {d}")


def _uncovered_reason(what: str, table: pd.DataFrame, r) -> str:
    if what == "inter_area_rules":
        return "no `da_flow_ref` (no balancing penalty or lock in RT)"
    if what == "generator":
        if "pmax" in table.columns and not _capacity(table.at[r, "pmax"]) > 0.0:
            return "no capacity (`pmax <= 0`)"
        return "`pmax > 0`, no channel"
    if not _is_flexible(table, r):
        return "no flexible load"
    return "flexible load, no channel"


TABLE_ORDER = ("generator", "consumer", "inter_area_rules")
TABLE_NOUN = {"generator": "generator", "consumer": "consumer", "inter_area_rules": "inter-area rule"}


def _with_refs(table: pd.DataFrame, part: pd.DataFrame) -> pd.DataFrame:
    """*table* with the cells *part* sets (NA in *part* leaves the table's value); no-profile refs NaN."""
    table = table.copy()
    for col in part.columns:
        cells = part[col].dropna()
        if col not in table.columns:
            table[col] = pd.Series(np.nan, index=table.index, dtype=object)
        elif cells.dtype == object and table[col].dtype != object:
            table[col] = table[col].astype(object)
        table.loc[cells.index, col] = cells
    if "da_flow_ref" in table.columns:
        refs = override_profile_refs(table, "da_flow_ref")
        table["da_flow_ref"] = refs.where(refs.ne(""), np.nan)
    return table


def _n(count: int, noun: str) -> str:
    return f"{count} {noun}" + ("" if count == 1 else "s")


def _counts(values: pd.Series) -> list[tuple[str, int]]:
    counts = values.value_counts(sort=False)
    return sorted(((str(k), int(v)) for k, v in counts.items()), key=lambda kv: (-kv[1], kv[0]))


def _groups(details: pd.DataFrame) -> str:
    """Counts by area, then type: ``area A 6 (gas 4, wind 2); area B 1 (gas 1)``."""
    has_area = details["area"].ne("").any()
    has_type = details["type"].ne("").any()

    def types(frame):
        return ", ".join(f"{t or '-'} {n}" for t, n in _counts(frame["type"]))

    if has_area:
        parts = []
        for area, n in _counts(details["area"]):
            text = f"area {area or '-'} {n}"
            if has_type:
                text += f" ({types(details[details['area'] == area])})"
            parts.append(text)
        return "; ".join(parts)
    if has_type:
        return types(details)
    return ""


def write_channel_summary(
    refs: RefColumns,
    generator: pd.DataFrame,
    consumer: pd.DataFrame,
    outdir,
    **kwargs,
) -> dict[str, Path]:
    """Write :func:`summarize_channels` into *outdir*: ``prepare_rt_summary.md`` and ``channels.csv``.

    The keyword arguments are those of :func:`summarize_channels`. Returns the paths as
    ``{"summary": ..., "channels": ...}``.
    """
    markdown, channels = summarize_channels(refs, generator, consumer, **kwargs)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = outdir / SUMMARY_FILE
    summary.write_text(markdown, encoding="utf-8")
    table = outdir / CHANNELS_FILE
    channels.to_csv(table, index=False)
    return {"summary": summary, "channels": table}
