"""Turn a day-ahead run into the inputs of a balancing (real-time) run.

DA and RT are separate runs. :func:`prepare_rt` is the step between them: it reads the
DA grid tables and the DA dispatch (only through :class:`powergama.Results`), and writes
the RT grid tables and one profile file, ``da_solution.parquet``, with the DA-derived
columns those tables reference. The RT run then reads them like any other input::

    prepared = prepare_rt(res_da, "rt_inputs")
    data = powergama.GridData()
    data.readGridData(**prepared.grid_files())  # node, branch, ... CSV files
    data.readProfileData(
        filename=["profiles_rt.csv", prepared.files["da_solution"]],
        timerange=timerange,
    )
    lp = powergama.LpProblem(data, extensions=[RtExtension(...)])

The DA ``Results`` may come from the DA run itself or from its database
(``Results.from_existing(..., timerange=...)``).

Every RT generator and consumer row carries ``da_index``, the DA row it comes from (NaN
for a row added for RT only). The DA-derived columns are named after the DA row, so RT
rows that come from the same DA row share a column:

==================  ==============================================  ==================================
column              value (clipped to [0, 1])                       referenced by
==================  ==============================================  ==================================
``da_gen_<d>``      DA output of d / ``pmax`` of d                  generator ``rt_target_ref``
``da_pump_<d>``     DA pumping of d / ``pump_cap`` of d             generator ``rt_pump_target_ref``
``da_cons_<d>``     DA flexible demand of d / ``demand_avg`` of d   consumer ``rt_target_ref``
``rt_target_zero``  0                                               rows with no DA row
``da_carry_<d>``    (RT avail. + DA output - DA avail.) / ``pmax``  generator ``pmax_ref`` (and
                                                                    ``pmin_ref``), *carry_curtailment*
``da_storage_<d>``  DA filling of d at the start of the timestep    generator ``storage_ini_ref``,
                    / ``storage_cap`` of d                          *initial_storage*
``da_headroom_<d>`` (cap - DA output of the unit) / cap, cap = the  ``pmax_ref`` of the unit's peak
                    unit's ``pmax`` (``block_group``); d = the DA   tranche, *peak_tranches*
                    row of its first row
``da_price_up_<d>`` c0 + (fuelcost + v), v the storage value of d   generator ``rt_price_up_ref``
``da_price_down_<d>`` at the DA filling level (and the pumping and  (``rt_price_down_ref``, ...),
``da_pump_price_*`` flexible-demand equivalents), EUR/MWh, not      *prices*, storage units only
``da_cons_price_*`` clipped; see :func:`price_columns`
==================  ==============================================  ==================================

The pin options also reference ``da_gen_<d>`` (``pmax_ref``, and ``pmin_ref`` to fix) and
``rt_target_zero`` (``pmin_ref``); see :func:`pin_columns`.

Nothing here knows about energy types: which units are tracked, pinned or split is
decided by the caller, as data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import storage_value as storval
from ..context import GridContext, override_profile_refs

ZERO_TARGET = "rt_target_zero"
"""The one all-zero target column, referenced by RT rows that have no DA row."""

DA_INDEX = "da_index"

GRID_TABLES = ("node", "branch", "dcbranch", "generator", "consumer")
# The inter-area table; its name follows the GridData attribute.
INTER_AREA_TABLES = ("inter_area_rules",)


@dataclass
class RefColumns:
    """DA-derived profile columns, and the references to them in the RT tables.

    The result of :func:`target_columns` and the other ``*_columns`` functions; several
    are joined by :func:`combine_columns`.

    profiles:
        Index = DA timesteps, float64; only the columns some row references.
    generator:
        Index = RT generator rows; one column per generator table column to set. A missing
        cell (NA) leaves the table's value as it is; ``""`` is a value (no profile, no
        target). Rows not in the index are left as they are.
    consumer:
        The same for the RT consumer rows. A Series sets the one column it is named after.
    """

    profiles: pd.DataFrame
    generator: pd.DataFrame
    consumer: pd.DataFrame | pd.Series


TargetColumns = RefColumns
"""The result of :func:`target_columns`; the first name of :class:`RefColumns`."""


def _ref_frame(refs) -> pd.DataFrame:
    if refs is None:
        return pd.DataFrame()
    if isinstance(refs, pd.Series):
        return refs.to_frame()
    return refs


def _merge_refs(frames: list[pd.DataFrame], what: str) -> pd.DataFrame:
    """Join reference frames cell by cell; a cell set to two different values raises."""
    out = pd.DataFrame()
    for frame in frames:
        for col in frame.columns:
            new = frame[col]
            if col not in out.columns:
                out = out.reindex(out.index.union(new.index, sort=False))
                out[col] = new
                continue
            old = out[col].reindex(new.index)
            clash = old.notna() & new.notna() & old.ne(new)
            if clash.any():
                rows = list(new.index[clash])
                raise ValueError(
                    f"RT {what} column '{col}' is set to different values for row(s) {rows[:5]}: "
                    f"{list(old[clash][:5])} and {list(new[clash][:5])}"
                )
            out = out.reindex(out.index.union(new.index, sort=False))
            out[col] = out[col].where(out[col].notna(), new.reindex(out.index))
    return out


def combine_columns(*parts: RefColumns) -> RefColumns:
    """Join the results of several ``*_columns`` functions into one.

    The profile columns are joined by name: a name in several parts must have the same
    timesteps and bit-identical values (two rules that reference the same DA column),
    otherwise ``ValueError``. The references are joined cell by cell: a cell that two
    parts set to different values raises ``ValueError``.
    """
    if not parts:
        raise ValueError("combine_columns needs at least one part")
    index = parts[0].profiles.index
    columns: dict[str, pd.Series] = {}
    for part in parts:
        if not part.profiles.index.equals(index):
            raise ValueError(
                f"The parts are not on the same timesteps: {_span(part.profiles.index)} and {_span(index)}"
            )
        for name in part.profiles.columns:
            values = part.profiles[name]
            if name in columns:
                old = columns[name]
                if old.dtype != values.dtype or not np.array_equal(old.to_numpy(), values.to_numpy()):
                    raise ValueError(f"Profile column '{name}' has different values in two parts")
                continue
            columns[name] = values
    profiles = pd.DataFrame(columns, index=index, dtype=np.float64)
    generator = _merge_refs([_ref_frame(p.generator) for p in parts], "generator")
    consumer = _merge_refs([_ref_frame(p.consumer) for p in parts], "consumer")
    return RefColumns(profiles=profiles, generator=generator, consumer=consumer)


def clear_refs(table: pd.DataFrame, mask, columns, *, timesteps, which: str = "generator") -> RefColumns:
    """A :class:`RefColumns` that clears *columns* (sets ``""``, no profile) on the *mask* rows.

    For joining with :func:`combine_columns`, e.g. to drop a unit's ``pmin_ref``/``pmax_ref``
    from the DA grid. *table* is the RT table the mask is on, *which* names it
    (``"generator"`` or ``"consumer"``), *timesteps* are those of the other parts. The
    profile frame has no column; a cleared cell that another part sets to a profile raises
    in :func:`combine_columns`.
    """
    if which not in ("generator", "consumer"):
        raise ValueError(f"which must be 'generator' or 'consumer', not {which!r}")
    if isinstance(columns, str):
        columns = [columns]
    rows = table.index[_mask(mask, table, "clear_refs")]
    refs = pd.DataFrame({col: pd.Series("", index=rows, dtype=object) for col in columns}, index=rows)
    profiles = pd.DataFrame(index=pd.Index(list(timesteps)), dtype=np.float64)
    empty = pd.DataFrame()
    return RefColumns(
        profiles=profiles,
        generator=refs if which == "generator" else empty,
        consumer=refs if which == "consumer" else empty,
    )


def apply_refs(table: pd.DataFrame, refs) -> pd.DataFrame:
    """A copy of the RT *table* with the references of *refs* (a :class:`RefColumns` part) set.

    A column the table does not have is added, ``""`` (or NaN for a numeric column) where
    *refs* sets nothing.
    """
    table = table.copy()
    refs = _ref_frame(refs)
    for col in refs.columns:
        values = refs[col].dropna()
        text = refs[col].dtype == object
        if col not in table.columns:
            table[col] = pd.Series("" if text else np.nan, index=table.index, dtype=object if text else np.float64)
        elif text and table[col].dtype != object:
            table[col] = table[col].astype(object)
        if len(values):
            table.loc[values.index, col] = values
    return table


def _mask(mask, table: pd.DataFrame, name: str) -> pd.Series:
    """*mask* as a boolean Series on the rows of *table*; None is no row."""
    if mask is None:
        return pd.Series(False, index=table.index, dtype=bool)
    if not isinstance(mask, pd.Series):
        raise ValueError(f"'{name}' must be a boolean pandas Series on the RT table's index")
    if not mask.index.equals(table.index):
        raise ValueError(f"'{name}' is not on the RT table's index")
    if mask.isna().any() or not (pd.api.types.is_bool_dtype(mask.dtype)):
        raise ValueError(f"'{name}' must be boolean, without missing values")
    return mask.astype(bool)


@dataclass
class PreparedRt:
    """What :func:`prepare_rt` wrote.

    tables:
        The RT grid tables, keyed like the ``GridData`` attributes: node, branch, dcbranch,
        generator, consumer, and the inter-area table if the DA grid has one.
    da_solution:
        The DA-derived profile columns (index = stored DA timesteps).
    files:
        The file of each table and ``"da_solution"`` (the parquet file).
    """

    tables: dict[str, pd.DataFrame]
    da_solution: pd.DataFrame
    files: dict[str, Path] = field(default_factory=dict)

    def grid_files(self) -> dict[str, Path]:
        """The table files as keyword arguments of ``GridData.readGridData``."""
        keywords = {
            "node": "nodes",
            "branch": "ac_branches",
            "dcbranch": "dc_branches",
            "generator": "generators",
            "consumer": "consumers",
        }
        out = {kw: self.files[name] for name, kw in keywords.items()}
        for name in INTER_AREA_TABLES:
            if name in self.files:
                out[name] = self.files[name]
        return out


def _mapping(table: pd.DataFrame, da_index: str, what: str) -> pd.Series:
    """The DA row of each RT row as a nullable integer Series."""
    if da_index not in table.columns:
        raise ValueError(f"RT {what} table has no '{da_index}' column (the DA row of each RT row)")
    raw = table[da_index]
    mapped = pd.to_numeric(raw, errors="coerce")
    bad = raw.notna() & mapped.isna()
    if bad.any():
        raise ValueError(f"RT {what} '{da_index}' is not a row number for row(s) {list(table.index[bad])}")
    whole = mapped.dropna()
    if not (whole == np.floor(whole)).all():
        raise ValueError(f"RT {what} '{da_index}' must hold whole row numbers")
    return mapped.astype("Int64")


def _capacity(value) -> float:
    try:
        cap = float(value)
    except (TypeError, ValueError):
        return math.nan
    return cap


def _check_index(name: str, frame: pd.DataFrame, expected: pd.Index):
    if not frame.index.equals(expected):
        raise ValueError(
            f"DA frame '{name}' is not on the expected timesteps: it has {len(frame.index)} rows "
            f"({_span(frame.index)}), expected {len(expected)} ({_span(expected)})"
        )


def _span(index) -> str:
    if len(index) == 0:
        return "none"
    return f"{index[0]}..{index[-1]}"


def _da_series(frame: pd.DataFrame, name: str, d: int, *, missing_is_zero: bool) -> pd.Series:
    """Column *d* of a DA frame; NaN raises. A missing column is zeros if allowed."""
    if d not in frame.columns:
        if missing_is_zero:
            return pd.Series(0.0, index=frame.index, dtype=np.float64)
        raise ValueError(f"DA frame '{name}' has no column for DA row {d}")
    s = frame[d]
    if s.isna().any():
        steps = list(s.index[s.isna()])
        shown = ", ".join(str(t) for t in steps[:5]) + (" ..." if len(steps) > 5 else "")
        raise ValueError(f"DA frame '{name}' has NaN for DA row {d} at timestep(s) {shown}")
    return s


def _da_capacity(table: pd.DataFrame | None, rt_table: pd.DataFrame, mapping: pd.Series, column: str, d: int, what):
    """The capacity of DA row *d*: from the DA table if given, else from its RT rows."""
    if table is not None:
        if d not in table.index:
            raise ValueError(f"RT {what} rows map to DA row {d}, which the DA {what} table does not have")
        return _capacity(table.at[d, column])
    caps = {_capacity(v) for v in rt_table.loc[mapping.eq(d).fillna(False).to_numpy(), column]}
    if len(caps) != 1:
        raise ValueError(
            f"RT {what} rows mapped to DA row {d} have different '{column}' {sorted(caps)}; "
            f"pass the DA {what} table to take it from there"
        )
    return caps.pop()


def _fraction(values: pd.Series, capacity: float, column: str, d: int, what: str) -> pd.Series:
    if not np.isfinite(capacity) or capacity <= 0.0:
        raise ValueError(f"DA {what} row {d} has '{column}' {capacity}: its DA values can't be a fraction of it")
    # Keep exactly this arithmetic (Series / float, then clip): results are compared bit for bit.
    return (values / float(capacity)).clip(0.0, 1.0)


def _da_gen_fraction(gen_frame, generator, gen_map, da_generator, d: int) -> pd.Series:
    """Column ``da_gen_<d>``: DA output of d / ``pmax`` of d. Targets and pins share it."""
    return _fraction(
        _da_series(gen_frame, "generator_output", d, missing_is_zero=False),
        _da_capacity(da_generator, generator, gen_map, "pmax", d, "generator"),
        "pmax",
        d,
        "generator",
    )


def target_columns(
    frames: dict[str, pd.DataFrame],
    generator: pd.DataFrame,
    consumer: pd.DataFrame,
    *,
    da_index: str = DA_INDEX,
    da_generator: pd.DataFrame | None = None,
    da_consumer: pd.DataFrame | None = None,
    timesteps=None,
) -> RefColumns:
    """DA tracking targets for the RT rows, and the profile columns they reference.

    Parameters
    ----------
    frames:
        DA dispatch as returned by :meth:`powergama.Results.get_dispatch_frames`. Uses
        ``generator_output``, ``pumping`` and ``flexload_demand``.
    generator, consumer:
        The RT tables. Each has a *da_index* column with the DA row of each RT row, or
        NaN for a row added for RT.
    da_index:
        Name of that column.
    da_generator, da_consumer:
        The DA tables, for the capacity of each DA row (``pmax``, ``pump_cap``,
        ``demand_avg``). If not given, it is taken from the RT rows mapped to that DA row,
        which must then agree.
    timesteps:
        The DA timesteps the frames must be on. If not given, every frame used must be on
        the timesteps of ``generator_output``.

    Rules
    -----
    - A generator with ``pmax > 0`` tracks ``da_gen_<d>``; with ``pump_cap > 0`` it also
      tracks ``da_pump_<d>`` (zeros if DA row d has no pumping). A generator with
      ``pmax <= 0`` (``pump_cap <= 0``) has no target ("").
    - A consumer with flexible load tracks ``da_cons_<d>`` (zeros if DA row d had no
      flexible load). Other consumers have no target.
    - A row with no DA row (NaN *da_index*) that would have a target tracks
      :data:`ZERO_TARGET`.
    - NaN in a DA value that is needed, a frame that is not on the expected timesteps, or
      a DA row that does not exist raises ``ValueError``.
    """
    gen_frame = frames["generator_output"]
    expected = pd.Index(list(timesteps)) if timesteps is not None else gen_frame.index
    _check_index("generator_output", gen_frame, expected)

    gen_map = _mapping(generator, da_index, "generator")
    cons_map = _mapping(consumer, da_index, "consumer")

    columns: dict[str, pd.Series] = {}
    zero_used = False

    def column(name, make):
        if name not in columns:
            columns[name] = make()
        return name

    # --- generator output -----------------------------------------------------
    gen_ref = pd.Series("", index=generator.index, dtype=object)
    for r in generator.index:
        if not _capacity(generator.at[r, "pmax"]) > 0.0:
            continue
        d = gen_map.at[r]
        if pd.isna(d):
            gen_ref.at[r] = ZERO_TARGET
            zero_used = True
            continue
        d = int(d)
        gen_ref.at[r] = column(
            f"da_gen_{d}", lambda d=d: _da_gen_fraction(gen_frame, generator, gen_map, da_generator, d)
        )

    # --- pumping --------------------------------------------------------------
    pump_ref = pd.Series("", index=generator.index, dtype=object)
    if "pump_cap" in generator.columns:
        pump_frame = None
        for r in generator.index:
            if not _capacity(generator.at[r, "pump_cap"]) > 0.0:
                continue
            d = gen_map.at[r]
            if pd.isna(d):
                pump_ref.at[r] = ZERO_TARGET
                zero_used = True
                continue
            if pump_frame is None:
                pump_frame = frames["pumping"]
                _check_index("pumping", pump_frame, expected)
            d = int(d)
            pump_ref.at[r] = column(
                f"da_pump_{d}",
                lambda d=d: _fraction(
                    _da_series(pump_frame, "pumping", d, missing_is_zero=True),
                    _da_capacity(da_generator, generator, gen_map, "pump_cap", d, "generator"),
                    "pump_cap",
                    d,
                    "generator",
                ),
            )

    # --- flexible load ----------------------------------------------------------
    cons_ref = pd.Series("", index=consumer.index, dtype=object)
    flex_frame = None
    for r in consumer.index:
        if not _is_flexible(consumer, r):
            continue
        d = cons_map.at[r]
        if pd.isna(d):
            cons_ref.at[r] = ZERO_TARGET
            zero_used = True
            continue
        if flex_frame is None:
            flex_frame = frames["flexload_demand"]
            _check_index("flexload_demand", flex_frame, expected)
        d = int(d)
        cons_ref.at[r] = column(
            f"da_cons_{d}",
            lambda d=d: _fraction(
                _da_series(flex_frame, "flexload_demand", d, missing_is_zero=True),
                _da_capacity(da_consumer, consumer, cons_map, "demand_avg", d, "consumer"),
                "demand_avg",
                d,
                "consumer",
            ),
        )

    if zero_used:
        columns[ZERO_TARGET] = pd.Series(0.0, index=expected, dtype=np.float64)

    profiles = pd.DataFrame(columns, index=expected, dtype=np.float64)
    gen_refs = pd.DataFrame({"rt_target_ref": gen_ref, "rt_pump_target_ref": pump_ref})
    return RefColumns(profiles=profiles, generator=gen_refs, consumer=cons_ref.rename("rt_target_ref"))


def _is_flexible(consumer: pd.DataFrame, r) -> bool:
    """Same test as ``GridData.getIdxConsumersWithFlexibleLoad``."""
    if "flex_fraction" not in consumer.columns:
        return False
    flex = _capacity(consumer.at[r, "flex_fraction"])
    return bool(0.0 < flex < np.inf and _capacity(consumer.at[r, "demand_avg"]) > 0.0)


def _generator_frame(frames: dict[str, pd.DataFrame], timesteps) -> tuple[pd.DataFrame, pd.Index]:
    gen_frame = frames["generator_output"]
    expected = pd.Index(list(timesteps)) if timesteps is not None else gen_frame.index
    _check_index("generator_output", gen_frame, expected)
    return gen_frame, expected


def _pinned_da_row(generator: pd.DataFrame, gen_map: pd.Series, r, option: str) -> int:
    """The DA row of RT generator row *r*, which *option* pins; raises if it can't be pinned."""
    if not _capacity(generator.at[r, "pmax"]) > 0.0:
        raise ValueError(f"'{option}' selects RT generator row {r}, which has pmax {generator.at[r, 'pmax']}")
    d = gen_map.at[r]
    if pd.isna(d):
        raise ValueError(f"'{option}' selects RT generator row {r}, which has no DA row")
    return int(d)


def _pin_refs(rows, pmin: dict, pmax: dict) -> pd.DataFrame:
    """``pmin_ref``/``pmax_ref`` of the pinned rows, with ``inflow_ref = "const"``, ``inflow_fac = 1``."""
    index = pd.Index(list(rows))
    return pd.DataFrame(
        {
            "pmin_ref": pd.Series([pmin.get(r) for r in index], index=index, dtype=object),
            "pmax_ref": pd.Series([pmax[r] for r in index], index=index, dtype=object),
            "inflow_ref": pd.Series("const", index=index, dtype=object),
            "inflow_fac": pd.Series(1.0, index=index, dtype=np.float64),
        },
        index=index,
    )


def pin_columns(
    frames: dict[str, pd.DataFrame],
    generator: pd.DataFrame,
    *,
    fix=None,
    cap=None,
    zero_pmin=None,
    da_index: str = DA_INDEX,
    da_generator: pd.DataFrame | None = None,
    timesteps=None,
) -> RefColumns:
    """Pin RT generators to their DA output through ``pmin_ref``/``pmax_ref``.

    Parameters
    ----------
    frames:
        DA dispatch as returned by :meth:`powergama.Results.get_dispatch_frames`. Uses
        ``generator_output``.
    generator:
        The RT generator table, with a *da_index* column (see :func:`target_columns`).
    fix, cap, zero_pmin:
        Boolean Series on the RT generator index, or None (no row).

        - *fix*: ``pmin_ref = pmax_ref = da_gen_<d>``, the unit produces its DA output;
        - *cap*: ``pmax_ref = da_gen_<d>``, the unit produces at most its DA output;
          ``pmin_ref`` is left as it is;
        - *zero_pmin*: only on *cap* rows, also ``pmin_ref =`` :data:`ZERO_TARGET`, so the
          unit can go down to zero.

        Every pinned row also gets ``inflow_ref = "const"`` and ``inflow_fac = 1``: the DA
        output already has the availability in it, and ``pmax_ref`` would apply it twice.
    da_index, da_generator, timesteps:
        As in :func:`target_columns`. ``da_gen_<d>`` is the column the target of the same
        unit references, with the same values.

    A row in both *fix* and *cap*, a *zero_pmin* row outside *cap*, a pinned row with
    ``pmax <= 0`` or without a DA row, and NaN in a DA output that is needed raise
    ``ValueError``.
    """
    fix = _mask(fix, generator, "fix")
    cap = _mask(cap, generator, "cap")
    zero_pmin = _mask(zero_pmin, generator, "zero_pmin")
    both = fix & cap
    if both.any():
        raise ValueError(f"RT generator row(s) {list(generator.index[both])} are in both 'fix' and 'cap'")
    outside = zero_pmin & ~cap
    if outside.any():
        raise ValueError(f"'zero_pmin' row(s) {list(generator.index[outside])} are not in 'cap'")

    gen_frame, expected = _generator_frame(frames, timesteps)
    gen_map = _mapping(generator, da_index, "generator")
    columns: dict[str, pd.Series] = {}
    pmin: dict = {}
    pmax: dict = {}
    rows = generator.index[fix | cap]
    for r in rows:
        d = _pinned_da_row(generator, gen_map, r, "fix" if fix.at[r] else "cap")
        name = f"da_gen_{d}"
        if name not in columns:
            columns[name] = _da_gen_fraction(gen_frame, generator, gen_map, da_generator, d)
        pmax[r] = name
        if fix.at[r]:
            pmin[r] = name
        elif zero_pmin.at[r]:
            pmin[r] = ZERO_TARGET
            columns[ZERO_TARGET] = pd.Series(0.0, index=expected, dtype=np.float64)

    profiles = pd.DataFrame(columns, index=expected, dtype=np.float64)
    return RefColumns(profiles=profiles, generator=_pin_refs(rows, pmin, pmax), consumer=pd.DataFrame())


def carry_curtailment_columns(
    frames: dict[str, pd.DataFrame],
    generator: pd.DataFrame,
    mask,
    *,
    da_available: pd.DataFrame,
    rt_available: pd.DataFrame,
    pin: str = "cap",
    da_index: str = DA_INDEX,
    da_generator: pd.DataFrame | None = None,
    timesteps=None,
) -> RefColumns:
    """Carry the DA curtailment of RT generators into RT, through ``pmax_ref``.

    The DA run curtailed ``da_available - da_output`` MW of the unit; RT may produce what
    is left of its own availability after the same curtailment. For DA row d, column
    ``da_carry_<d>`` is, as a fraction of ``pmax`` of d::

        ((rt_available + da_output - da_available).clip(0, pmax) / pmax).clip(0, 1)

    Parameters
    ----------
    frames:
        DA dispatch as returned by :meth:`powergama.Results.get_dispatch_frames`. Uses
        ``generator_output``.
    generator:
        The RT generator table, with a *da_index* column.
    mask:
        Boolean Series on the RT generator index: the rows to limit.
    da_available, rt_available:
        The availability of the unit in the DA and in the RT run, in MW (for example
        ``pmax * inflow_fac * inflow profile``, as the caller computes it). Index = the DA
        timesteps (those of *frames*), columns = **DA rows**: RT rows from the same DA row
        share one column. NaN raises; fill it before.
    pin:
        ``"cap"`` sets ``pmax_ref`` only; ``"fix"`` sets ``pmin_ref`` too, so the unit
        produces exactly that. Either way the rows get ``inflow_ref = "const"`` and
        ``inflow_fac = 1``.
    da_index, da_generator, timesteps:
        As in :func:`target_columns`.

    A masked row with ``pmax <= 0`` or without a DA row, a missing column, frames on other
    timesteps and NaN raise ``ValueError``.
    """
    if pin not in ("cap", "fix"):
        raise ValueError(f"pin must be 'cap' or 'fix', not {pin!r}")
    mask = _mask(mask, generator, "carry_curtailment")
    gen_frame, expected = _generator_frame(frames, timesteps)
    for name, frame in (("da_available", da_available), ("rt_available", rt_available)):
        _check_index(name, frame, expected)
    gen_map = _mapping(generator, da_index, "generator")
    columns: dict[str, pd.Series] = {}
    pmin: dict = {}
    pmax: dict = {}
    rows = generator.index[mask]
    for r in rows:
        d = _pinned_da_row(generator, gen_map, r, "carry_curtailment")
        name = f"da_carry_{d}"
        if name not in columns:
            capacity = _da_capacity(da_generator, generator, gen_map, "pmax", d, "generator")
            da_out = _da_series(gen_frame, "generator_output", d, missing_is_zero=False)
            da_avail = _da_series(da_available, "da_available", d, missing_is_zero=False)
            rt_avail = _da_series(rt_available, "rt_available", d, missing_is_zero=False)
            # The curtailment-carry limit, then the fraction of pmax, in exactly this
            # order: results are compared bit for bit.
            limit = (rt_avail + da_out - da_avail).clip(lower=0.0, upper=float(capacity))
            columns[name] = _fraction(limit, capacity, "pmax", d, "generator")
        pmax[r] = name
        if pin == "fix":
            pmin[r] = name

    profiles = pd.DataFrame(columns, index=expected, dtype=np.float64)
    return RefColumns(profiles=profiles, generator=_pin_refs(rows, pmin, pmax), consumer=pd.DataFrame())


@dataclass
class CarryCurtailment:
    """The *carry_curtailment* option of :func:`prepare_rt`; see :func:`carry_curtailment_columns`.

    *source* is a short text on where the availabilities come from (e.g. the RT profile
    file), reported in ``prepare_rt_summary.md``.
    """

    mask: pd.Series
    da_available: pd.DataFrame
    rt_available: pd.DataFrame
    pin: str = "cap"
    source: str | None = None


def _block_groups(generator: pd.DataFrame, mask: pd.Series, block_group: str) -> list[list]:
    """The masked rows grouped by *block_group*, each group's rows sorted, groups by their first row.

    An empty (or missing) *block_group* is a group of one. A group that is only partly in
    the mask raises: its tranche would be sized on part of the unit.
    """
    rows = sorted(generator.index[mask])
    if block_group in generator.columns:
        col = generator[block_group]
        labels = col.astype(object).where(col.notna(), "").astype(str).str.strip()
    else:
        labels = pd.Series("", index=generator.index, dtype=object)
    groups: dict = {}
    for r in rows:
        key = ("row", r) if labels.at[r] == "" else ("group", labels.at[r])
        groups.setdefault(key, []).append(r)
    for (kind, label), members in groups.items():
        if kind != "group":
            continue
        outside = [r for r in generator.index[labels.eq(label)] if not mask.at[r]]
        if outside:
            raise ValueError(
                f"'peak_tranches' selects only part of {block_group} '{label}': row(s) {outside[:5]} are not "
                f"selected, row(s) {members[:5]} are"
            )
    return sorted(groups.values(), key=lambda members: members[0])


def peak_tranche_columns(
    frames: dict[str, pd.DataFrame],
    generator: pd.DataFrame,
    mask,
    *,
    price: float,
    da_generator: pd.DataFrame | None = None,
    block_group: str = "block_group",
    ramp_pu: float | None = None,
    da_index: str = DA_INDEX,
    timesteps=None,
) -> tuple[pd.DataFrame, RefColumns]:
    """Add one RT peak tranche per unit: an extra row at *price* on the unit's DA headroom.

    The masked rows are grouped by *block_group* (the rows of one physical unit, e.g. price
    blocks; an empty value is a group of one). For each group, in the order of the group's
    first row (rows sorted by label), the tranche is a copy of that first row with

    - ``pmax`` = cap, the sum of the group's ``pmax`` (from *da_generator* at the group's DA
      rows if given, else from the RT rows);
    - ``fuelcost`` = *price*;
    - *da_index* NA: the row has no DA row of its own, so :func:`target_columns` gives it
      :data:`ZERO_TARGET`;
    - ``pmax_ref`` = ``da_headroom_<d>``, d the DA row of the group's first row, with values
      ``((cap - DA output of the group) / cap).clip(0, 1)``; the group's DA outputs are
      added to 0.0 one by one, in row order;
    - ``pmin_ref = ""``, ``inflow_ref = "const"``, ``inflow_fac = 1``;
    - ``ramp_up_pu`` and ``ramp_down_pu`` = *ramp_pu*, NaN (no ramp limit) when it is None.

    Everything else (``desc``, ``node``, ``type``, ``pmin``, ``block_group``, ...) is the
    first row's. A group whose cap is not positive gets no tranche. The tranches are
    appended at the labels ``len(generator)``, ``len(generator) + 1``, ...

    Parameters
    ----------
    frames:
        DA dispatch as returned by :meth:`powergama.Results.get_dispatch_frames`. Uses
        ``generator_output``.
    generator:
        The RT generator table, with a *da_index* column.
    mask:
        Boolean Series on the RT generator index: the rows of the units that get a tranche.
        A ``block_group`` must be selected whole.
    price:
        The tranches' ``fuelcost``, EUR/MWh.
    da_generator, da_index, timesteps:
        As in :func:`target_columns`.
    block_group:
        The column grouping the rows of one unit; without it every row is its own group.
    ramp_pu:
        The tranches' ramp limit up and down (``ramp_up_pu``, ``ramp_down_pu``, a fraction
        of their ``pmax`` per timestep, >= 0). None (the default): no ramp limit; the unit's
        own ramp limits are never inherited.

    Returns
    -------
    generator:
        A copy of *generator* with the tranche rows appended (*da_index* as ``Int64``).
    refs:
        The ``da_headroom_<d>`` columns and the tranche rows' ``pmax_ref``, ``pmin_ref``,
        ``inflow_ref`` and ``inflow_fac`` (already set in the returned table), for
        :func:`combine_columns` with the targets and pins computed on the returned table.

    A selected row without a DA row or with storage or pumping, a ``block_group`` selected
    in part, NaN in a DA output that is needed, frames on other timesteps and a tranche
    label already in use, and a *ramp_pu* that is negative or not finite raise ``ValueError``.
    """
    if ramp_pu is not None and not (np.isfinite(ramp_pu) and ramp_pu >= 0.0):
        raise ValueError(f"'peak_tranches' ramp_pu must be a finite number >= 0 or None, got {ramp_pu!r}")
    mask = _mask(mask, generator, "peak_tranches")
    gen_frame, expected = _generator_frame(frames, timesteps)
    gen_map = _mapping(generator, da_index, "generator")
    for col in ("storage_cap", "pump_cap"):
        if col in generator.columns:
            has = generator.index[mask & pd.to_numeric(generator[col], errors="coerce").fillna(0.0).gt(0.0)]
            if len(has):
                raise ValueError(f"'peak_tranches' selects RT generator row(s) {list(has[:5])} with {col} > 0")

    columns: dict[str, pd.Series] = {}
    firsts: list = []
    caps: list[float] = []
    names: list[str] = []
    for members in _block_groups(generator, mask, block_group):
        da_rows = []
        for r in members:
            d = gen_map.at[r]
            if pd.isna(d):
                raise ValueError(f"'peak_tranches' selects RT generator row {r}, which has no DA row")
            da_rows.append(int(d))
        if da_generator is not None:
            missing = [d for d in da_rows if d not in da_generator.index]
            if missing:
                raise ValueError(f"RT generator rows map to DA row(s) {missing}, which the DA generator table lacks")
            pmax = pd.to_numeric(da_generator.loc[da_rows, "pmax"], errors="coerce")
        else:
            pmax = pd.to_numeric(generator.loc[members, "pmax"], errors="coerce")
        cap = float(pmax.sum())
        if not cap > 0.0:
            continue
        # The tranche headroom, in exactly this order: results are compared bit for bit.
        da_unit = pd.Series(0.0, index=expected, dtype=np.float64)
        for d in da_rows:
            da_unit = da_unit + _da_series(gen_frame, "generator_output", d, missing_is_zero=False)
        name = f"da_headroom_{da_rows[0]}"
        columns[name] = ((cap - da_unit) / cap).clip(0.0, 1.0)
        firsts.append(members[0])
        caps.append(cap)
        names.append(name)

    labels = pd.RangeIndex(len(generator), len(generator) + len(firsts))
    taken = labels.intersection(generator.index)
    if len(taken):
        raise ValueError(f"The tranche rows would take RT generator label(s) {list(taken[:5])}, already in use")
    refs = pd.DataFrame(
        {
            "pmax_ref": pd.Series(names, index=labels, dtype=object),
            "pmin_ref": pd.Series("", index=labels, dtype=object),
            "inflow_ref": pd.Series("const", index=labels, dtype=object),
            "inflow_fac": pd.Series(1.0, index=labels, dtype=np.float64),
        },
        index=labels,
    )
    out = generator.copy()
    if len(firsts):
        new = generator.loc[firsts].copy()
        new.index = labels
        new["pmax"] = np.asarray(caps, dtype=np.float64)
        new["fuelcost"] = float(price)
        new[da_index] = pd.NA
        for col in refs.columns:
            # keep the table's dtype (e.g. pandas' str), so the RT rows' columns are unchanged
            new[col] = refs[col].astype(generator[col].dtype) if col in generator.columns else refs[col]
        for col in ("ramp_up_pu", "ramp_down_pu"):
            if ramp_pu is not None:
                new[col] = float(ramp_pu)
            elif col in new.columns:
                new[col] = np.nan
        out = pd.concat([out, new])
    out[da_index] = pd.to_numeric(out[da_index], errors="coerce").astype("Int64")
    profiles = pd.DataFrame(columns, index=expected, dtype=np.float64)
    return out, RefColumns(profiles=profiles, generator=refs, consumer=pd.DataFrame())


@dataclass
class PeakTranches:
    """The *peak_tranches* option of :func:`prepare_rt`; see :func:`peak_tranche_columns`.

    *mask* is on the DA generator index, *price* the tranches' fuel cost (EUR/MWh),
    *ramp_pu* their ramp limit up and down (None: no ramp limit). *source* is a short text
    on where the price and the selection come from, reported in ``prepare_rt_summary.md``.
    """

    mask: pd.Series
    price: float
    ramp_pu: float | None = None
    source: str | None = None


def _on_rt_rows(mask, index: pd.Index):
    """A mask on the DA generator rows, extended with False to the RT rows added after them."""
    if not isinstance(mask, pd.Series) or mask.index.equals(index) or not mask.index.isin(index).all():
        return mask  # None, already on the RT rows, or wrong: _mask decides
    return mask.reindex(index, fill_value=False)


LEGACY_DCBRANCH_COLUMNS = ("area_from", "area_to")
"""Columns ``GridData.getDcBranchAreas`` used to add to ``dcbranch`` in place, so DA
databases written before it was side-effect free store them in ``data_dcbranch``."""


def rt_tables_from_da(res_da) -> dict[str, pd.DataFrame]:
    """The DA grid tables as the start of the RT tables, each RT row its DA row.

    Copies of ``res_da.grid``'s node, branch, dcbranch, generator and consumer tables, and
    of ``inter_area_rules`` if the DA grid has one, keyed like the ``GridData`` attributes.
    Every generator and consumer row gets ``da_index`` (Int64), the DA row it comes from;
    rows a caller adds for RT only should have it NA. This is what :func:`prepare_rt`
    starts from; a caller that adds or changes rows starts from it too and then calls
    :func:`target_columns` and the other ``*_columns`` functions itself.

    *res_da* may be the DA run's own :class:`powergama.Results` or one read back with
    ``Results.from_existing``: the tables are the same either way. ``dcbranch`` drops
    :data:`LEGACY_DCBRANCH_COLUMNS`, which DA databases written by older versions carry.
    A DA table that already has a ``da_index`` column raises ``ValueError``.
    """
    grid = res_da.grid
    tables = {name: getattr(grid, name).copy() for name in GRID_TABLES}
    tables["dcbranch"] = tables["dcbranch"].drop(columns=list(LEGACY_DCBRANCH_COLUMNS), errors="ignore")
    for name in INTER_AREA_TABLES:
        extra = getattr(grid, name, None)
        if extra is not None:
            tables[name] = extra.copy()
            break
    for name in ("generator", "consumer"):
        if DA_INDEX in tables[name].columns:
            raise ValueError(f"The DA {name} table already has a '{DA_INDEX}' column")
        tables[name][DA_INDEX] = pd.array(tables[name].index, dtype="Int64")
    return tables


def prepare_rt(
    res_da,
    outdir,
    *,
    timeMaxMin=None,
    text_output: bool = False,
    fix=None,
    cap=None,
    zero_pmin=None,
    carry_curtailment: CarryCurtailment | None = None,
    initial_storage=None,
    peak_tranches: PeakTranches | None = None,
    prices: DeviationPrices | None = None,
    summary: bool = True,
    sources: dict[str, str] | None = None,
    summary_header: dict[str, str] | None = None,
) -> PreparedRt:
    """Write the inputs of an RT run from a complete DA run.

    Reads only ``res_da.grid`` and ``res_da.get_dispatch_frames()``. Copies the DA grid
    tables with :func:`rt_tables_from_da` (here every RT row is its DA row), sets the target
    references (see :func:`target_columns`) and those of the options below, and writes
    into *outdir*:

    - one CSV per grid table (``node.csv``, ``branch.csv``, ``dcbranch.csv``,
      ``generator.csv``, ``consumer.csv`` and the inter-area table if any);
    - ``da_solution.parquet``, the referenced DA columns. Its index is the stored DA
      timesteps, i.e. the rows of the profile files the DA run read, so
      ``readProfileData(filename=[..., "da_solution.parquet"], timerange=...)`` cuts it at
      the same rows as the other profile files;
    - with *text_output*, also ``da_solution.csv`` (for reading, not for input);
    - with *summary* (default), ``prepare_rt_summary.md`` and ``channels.csv``: the
      DA-derived columns, what they are and which RT rows use them how (see
      :func:`~powergama.lp.rt.summary.summarize_channels`).

    Parameters
    ----------
    res_da:
        The DA :class:`powergama.Results`. It must be complete
        (:meth:`~powergama.Results.is_complete`), otherwise ``ValueError``.
    outdir:
        Directory to write to; created if missing.
    timeMaxMin:
        ``[start, end]``: the DA timesteps ``start <= t < end`` to write; default the DA
        run's time range. The RT run reads ``da_solution.parquet`` at the rows of its own
        ``timerange``, which must be among them.
    text_output:
        Also write ``da_solution.csv``.
    fix, cap, zero_pmin:
        Boolean Series on the DA generator index (which is the RT one here): pin these
        units to their DA output, see :func:`pin_columns`.
    carry_curtailment:
        A :class:`CarryCurtailment`: limit these units to their RT availability less the
        DA curtailment, see :func:`carry_curtailment_columns`. Its frames are on the DA
        timesteps written (*timeMaxMin*).
    initial_storage:
        Boolean Series on the DA generator index (which is the RT one here): start these
        storage units from the DA filling at the RT run's first timestep, see
        :func:`initial_storage_columns`.
    peak_tranches:
        A :class:`PeakTranches`: add one row per unit (``block_group``) of these generators,
        at its price, on the unit's DA headroom, see :func:`peak_tranche_columns`. The rows
        are added to the copied tables first; the targets then give them
        :data:`ZERO_TARGET` like any RT row without a DA row. The masks of the other
        options stay on the DA rows (the tranches are never in them).
    prices:
        A :class:`DeviationPrices`: write the deviation prices of every RT row (the
        tranches included) with :func:`price_columns`, the storage values by default at
        the DA filling level (:func:`da_storage_values`). Without it no price is written,
        and an RT run raises on the first tracked row without one.
    summary:
        Write ``prepare_rt_summary.md`` and ``channels.csv``.
    sources:
        Where each channel family comes from, for the summary, by family (``da_gen``,
        ``da_pump``, ``da_cons``, ``da_carry``, ``da_storage``, ``da_headroom``,
        ``da_price``). The DA families default to the DA database, ``da_carry`` to
        ``carry_curtailment.source``, ``da_headroom`` to the DA database and
        ``peak_tranches.source``, ``da_price`` to the DA database and ``prices.source``.
    summary_header:
        Extra ``{label: text}`` lines for the top of the summary, after (or replacing) the
        DA run, its timesteps, the options and the files.

    The options and the targets share their columns (a pinned unit's ``da_gen_<d>`` is its
    target column too). A unit both pinned and in *carry_curtailment* raises.

    Planned keyword-only options (not implemented yet; they will be added here without
    changing the arguments above):

    - DA border flows for ``inter_area_rules.da_flow_ref``.
    """
    grid = res_da.grid
    if not res_da.is_complete():
        raise ValueError(
            "The DA run is not complete: its stored timesteps are not its time range "
            f"{_span(list(res_da.timerange))}. Re-run it before preparing RT."
        )
    if timeMaxMin is None:
        timeMaxMin = [res_da.timerange[0], res_da.timerange[-1] + 1]
    start, end = int(timeMaxMin[0]), int(timeMaxMin[-1])
    timesteps = [t for t in res_da.timerange if start <= t < end]
    if not timesteps:
        raise ValueError(f"No DA timestep in [{start}, {end})")
    frames = res_da.get_dispatch_frames(timeMaxMin=[start, end])

    tables = rt_tables_from_da(res_da)
    parts = []
    if peak_tranches is not None:
        tables["generator"], tranches = peak_tranche_columns(
            frames,
            tables["generator"],
            _mask(peak_tranches.mask, grid.generator, "peak_tranches").reindex(
                tables["generator"].index, fill_value=False
            ),
            price=peak_tranches.price,
            ramp_pu=peak_tranches.ramp_pu,
            da_generator=grid.generator,
            timesteps=timesteps,
        )
        parts.append(tranches)
    rt_index = tables["generator"].index
    fix, cap, zero_pmin, initial_storage = (_on_rt_rows(m, rt_index) for m in (fix, cap, zero_pmin, initial_storage))
    if carry_curtailment is not None:
        carry_curtailment = CarryCurtailment(
            _on_rt_rows(carry_curtailment.mask, rt_index),
            carry_curtailment.da_available,
            carry_curtailment.rt_available,
            pin=carry_curtailment.pin,
            source=carry_curtailment.source,
        )

    targets = target_columns(
        frames,
        tables["generator"],
        tables["consumer"],
        da_generator=grid.generator,
        da_consumer=grid.consumer,
        timesteps=timesteps,
    )
    parts.insert(0, targets)
    if fix is not None or cap is not None or zero_pmin is not None:
        parts.append(
            pin_columns(
                frames,
                tables["generator"],
                fix=fix,
                cap=cap,
                zero_pmin=zero_pmin,
                da_generator=grid.generator,
                timesteps=timesteps,
            )
        )
    if carry_curtailment is not None:
        parts.append(
            carry_curtailment_columns(
                frames,
                tables["generator"],
                carry_curtailment.mask,
                da_available=carry_curtailment.da_available,
                rt_available=carry_curtailment.rt_available,
                pin=carry_curtailment.pin,
                da_generator=grid.generator,
                timesteps=timesteps,
            )
        )
    if initial_storage is not None:
        # The level at the start of a timestep is the one stored for the timestep before,
        # back to the DA run's start.
        da_start = res_da.timerange[0]
        storage_frames = frames if start <= da_start else res_da.get_dispatch_frames(timeMaxMin=[da_start, end])
        parts.append(
            initial_storage_columns(
                storage_frames,
                tables["generator"],
                initial_storage,
                da_generator=grid.generator,
                da_profiles=grid.profiles,
                timesteps=timesteps,
            )
        )
    if prices is not None:
        renewable = _on_rt_rows(prices.renewable, rt_index)
        values = prices.storage_value
        if values is None:
            values = da_storage_values(res_da, timeMaxMin=[start, end])
        parts.append(
            price_columns(
                tables["generator"],
                tables["consumer"],
                fee=prices.fee,
                renewable=renewable,
                curtailment_factor=prices.curtailment_factor,
                storage_value=values.get("generator"),
                flex_value=values.get("consumer"),
                timesteps=timesteps,
            )
        )
    refs = combine_columns(*parts)
    tables["generator"] = apply_refs(tables["generator"], refs.generator)
    tables["consumer"] = apply_refs(tables["consumer"], refs.consumer)
    da_solution = refs.profiles

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    for name, table in tables.items():
        path = outdir / f"{name}.csv"
        table.to_csv(path, index=False)
        files[name] = path
    path = outdir / "da_solution.parquet"
    da_solution.to_parquet(path)
    files["da_solution"] = path
    if text_output:
        da_solution.to_csv(outdir / "da_solution.csv")
    if summary:
        from .summary import write_channel_summary

        da_file = getattr(getattr(res_da, "db", None), "filename", None)
        da_run = f"DA run `{Path(da_file).name}`" if da_file else "DA run"
        family_sources = {family: da_run for family in ("da_gen", "da_pump", "da_cons", "da_storage")}
        if carry_curtailment is not None:
            family_sources["da_carry"] = (
                f"DA output from {da_run}; availabilities: {carry_curtailment.source}"
                if carry_curtailment.source
                else f"DA output from {da_run}; availabilities given by the caller"
            )
        if peak_tranches is not None:
            family_sources["da_headroom"] = f"DA output from {da_run}" + (
                f"; {peak_tranches.source}" if peak_tranches.source else ""
            )
        if prices is not None:
            family_sources["da_price"] = (
                f"storage values at the DA filling level from {da_run}"
                if prices.storage_value is None
                else "storage values given by the caller"
            ) + (f"; {prices.source}" if prices.source else "")
        family_sources.update(sources or {})
        option_counts = {
            "fix": fix,
            "cap": cap,
            "zero_pmin": zero_pmin,
            "carry_curtailment": None if carry_curtailment is None else carry_curtailment.mask,
            "initial_storage": initial_storage,
        }
        options = ", ".join(
            f"{name} {int(_mask(m, tables['generator'], name).sum())}" for name, m in option_counts.items()
        )
        if carry_curtailment is not None:
            options = options.replace("carry_curtailment ", f"carry_curtailment (pin={carry_curtailment.pin}) ")
        if peak_tranches is not None:
            selected = int(_mask(peak_tranches.mask, grid.generator, "peak_tranches").sum())
            added = len(tables["generator"]) - len(grid.generator)
            ramp = (
                ", free ramp" if peak_tranches.ramp_pu is None else f", ramp {peak_tranches.ramp_pu:g} pu"
            )
            options += f", peak_tranches {selected} (+{added} tranche rows at {peak_tranches.price:g} EUR/MWh{ramp})"
        if prices is not None:
            n_res = int(_mask(_on_rt_rows(prices.renewable, rt_index), tables["generator"], "renewable").sum())
            options += (
                f", prices (c0 {prices.fee:g} EUR/MWh, {n_res} renewable at {prices.curtailment_factor:g} x c0 down)"
            )
        header = {
            "DA run": f"`{da_file}`" if da_file else "(in memory)",
            "DA timesteps": f"{timesteps[0]}..{timesteps[-1]} ({len(timesteps)})",
            "Options (generator rows)": options,
            "Files": ", ".join(f"`{p.name}`" for p in files.values()),
        }
        header.update(summary_header or {})
        files.update(
            write_channel_summary(
                refs,
                tables["generator"],
                tables["consumer"],
                outdir,
                node=tables["node"],
                sources=family_sources,
                header=header,
            )
        )
    return PreparedRt(tables=tables, da_solution=da_solution, files=files)


def _da_initial_fraction(da_generator: pd.DataFrame, da_profiles: pd.DataFrame | None, d: int) -> float:
    """The filling DA row *d* started the DA run with, as a fraction of ``storage_cap``.

    The DA ``storage_ini_ref`` profile at its first row if DA had one (as ``LpProblem``
    reads it), else ``storage_ini`` clipped to [0, 1] (NaN is 0, as in ``LpProblem``).
    """
    refs = override_profile_refs(da_generator, "storage_ini_ref")
    ref = refs.at[d] if refs is not None else ""
    if ref:
        if da_profiles is None or ref not in da_profiles.columns:
            raise ValueError(
                f"DA generator row {d} started from the profile '{ref}' (storage_ini_ref): pass the DA "
                "profiles as da_profiles"
            )
        fraction = float(pd.to_numeric(da_profiles[ref].iloc[[0]], errors="coerce").to_numpy(dtype=float)[0])
        if np.isnan(fraction):
            raise ValueError(f"DA storage_ini_ref profile '{ref}' of generator row {d} is NaN at its first row")
        return fraction
    ini = pd.to_numeric(pd.Series([da_generator.at[d, "storage_ini"]]), errors="coerce").fillna(0.0)
    return float(ini.clip(0.0, 1.0).iloc[0])


def initial_storage_columns(
    frames: dict[str, pd.DataFrame],
    generator: pd.DataFrame,
    mask,
    *,
    da_generator: pd.DataFrame,
    da_profiles: pd.DataFrame | None = None,
    da_index: str = DA_INDEX,
    timesteps=None,
) -> RefColumns:
    """Start RT storage from the DA storage level, through ``storage_ini_ref``.

    For DA row d, column ``da_storage_<d>`` is the DA filling at the **start** of each DA
    timestep, as a fraction of ``storage_cap`` of d:

    - at the DA run's first timestep, the filling DA started from: its ``storage_ini``
      clipped to [0, 1], or, if DA had a ``storage_ini_ref``, that profile's value at its
      first row (from *da_profiles*);
    - at a later timestep t, ``(Res_Storage[t - 1] / storage_cap).clip(0, 1)``:
      ``Res_Storage`` at t - 1 is the filling at the end of t - 1.

    An RT run whose time range starts at t then starts from the filling DA had at t
    (``LpProblem`` reads ``storage_ini_ref`` at the run's first timestep).

    Parameters
    ----------
    frames:
        DA dispatch as returned by :meth:`powergama.Results.get_dispatch_frames`. Uses
        ``storage``, which must start at the DA run's first timestep and have consecutive
        timesteps (``get_dispatch_frames()`` without *timeMaxMin* does).
    generator:
        The RT generator table, with a *da_index* column.
    mask:
        Boolean Series on the RT generator index: the rows to start from DA.
    da_generator:
        The DA generator table (``storage_cap``, ``storage_ini``, ``storage_ini_ref``).
    da_profiles:
        The DA profiles (``res_da.grid.profiles``), needed if a masked unit's DA row has a
        ``storage_ini_ref``.
    da_index:
        As in :func:`target_columns`.
    timesteps:
        The DA timesteps to write; default those of ``storage``. Each must be in it.

    A masked row without storage (``storage_cap`` not finite and positive, in RT or DA),
    without a DA row or whose DA row has no stored filling, and NaN raise ``ValueError``.
    """
    mask = _mask(mask, generator, "initial_storage")
    storage = frames["storage"]
    if len(storage.index) == 0:
        raise ValueError("DA frame 'storage' has no timestep")
    first = int(storage.index[0])
    if not np.array_equal(storage.index.to_numpy(), np.arange(first, first + len(storage.index))):
        raise ValueError("DA frame 'storage' must have consecutive timesteps from the DA run's first")
    expected = pd.Index(list(timesteps)) if timesteps is not None else storage.index
    outside = expected.difference(storage.index)
    if len(outside):
        raise ValueError(f"DA frame 'storage' has no timestep(s) {list(outside[:5])}")
    gen_map = _mapping(generator, da_index, "generator")
    columns: dict[str, pd.Series] = {}
    refs: dict = {}
    rows = generator.index[mask]
    for r in rows:
        if not 0.0 < _capacity(generator.at[r, "storage_cap"]) < np.inf:
            raise ValueError(f"'initial_storage' selects RT generator row {r}, which has no storage")
        d = gen_map.at[r]
        if pd.isna(d):
            raise ValueError(f"'initial_storage' selects RT generator row {r}, which has no DA row")
        d = int(d)
        name = f"da_storage_{d}"
        if name not in columns:
            if d not in da_generator.index:
                raise ValueError(f"RT generator rows map to DA row {d}, which the DA generator table does not have")
            cap = _capacity(da_generator.at[d, "storage_cap"])
            if not 0.0 < cap < np.inf:
                raise ValueError(f"DA generator row {d} has no storage ('storage_cap' {cap})")
            level = _da_series(storage, "storage", d, missing_is_zero=False)
            fraction = (level.shift(1) / float(cap)).clip(0.0, 1.0)
            fraction.iloc[0] = _da_initial_fraction(da_generator, da_profiles, d)
            columns[name] = fraction.reindex(expected)
        refs[r] = name

    profiles = pd.DataFrame(columns, index=expected, dtype=np.float64)
    index = pd.Index(list(rows))
    generator_refs = pd.DataFrame({"storage_ini_ref": pd.Series([refs[r] for r in index], index=index, dtype=object)})
    return RefColumns(profiles=profiles, generator=generator_refs, consumer=pd.DataFrame())


# --- deviation prices -------------------------------------------------------------

PRICE_REF_CONST = "const"

GENERATOR_PRICE_COLUMNS = ("rt_price_up", "rt_price_down")
PUMP_PRICE_COLUMNS = ("rt_pump_price_up", "rt_pump_price_down")
CONSUMER_PRICE_COLUMNS = ("rt_price_up", "rt_price_down")


def _da_grid_context(grid):
    """A :class:`~powergama.lp.context.GridContext` with just the storage-value curves resolved."""
    ctx = GridContext()
    ctx.idx_generatorsWithStorage = grid.getIdxGeneratorsWithStorage()
    ctx.idx_consumersWithFlexLoad = grid.getIdxConsumersWithFlexibleLoad()
    ctx._build_storage_values(grid)
    return ctx


def da_storage_values(res_da, timeMaxMin=None) -> dict[str, pd.DataFrame]:
    """The value of stored energy of every DA storage unit at its DA filling level, per timestep.

    For DA timestep t, the value at the filling the unit had at the **start** of t (the
    filling DA started from at the DA run's first timestep, ``Res_Storage`` / the flexible
    reservoir level of t - 1 afterwards), evaluated with PowerGAMA's own storage-value
    functions (:mod:`powergama.lp.storage_value`) on the DA grid:

    - ``"generator"``: ``storage_price x F(filling) x T(t)`` of every generator with storage
      (columns = DA generator rows);
    - ``"consumer"``: ``flex_basevalue x F(filling) x T(t)`` of every consumer with flexible
      load (columns = DA consumer rows).

    That is the storage value the single-period DA run itself used at t (``Res_Storage``'s
    ``marginalprice``); a multi-period DA run fixed it per window instead, and this
    re-evaluates it at every timestep. Index = the DA timesteps ``start <= t < end`` of
    *timeMaxMin* (default the DA run's time range), in EUR/MWh. The DA run must be
    complete; a stored NaN raises ``ValueError``.
    """
    grid = res_da.grid
    if not res_da.is_complete():
        raise ValueError("The DA run is not complete: its storage values can't be evaluated")
    da_start = int(res_da.timerange[0])
    if timeMaxMin is None:
        timeMaxMin = [da_start, res_da.timerange[-1] + 1]
    start, end = int(timeMaxMin[0]), int(timeMaxMin[-1])
    if start < da_start:
        raise ValueError(f"timeMaxMin starts at {start}, before the DA run ({da_start})")
    gens = list(grid.getIdxGeneratorsWithStorage())
    flex = list(grid.getIdxConsumersWithFlexibleLoad())
    steps = pd.Index([t for t in res_da.timerange if start <= t < end], dtype="int64")
    out = {
        "generator": pd.DataFrame(index=steps, columns=pd.Index(gens, dtype="int64"), dtype=np.float64),
        "consumer": pd.DataFrame(index=steps, columns=pd.Index(flex, dtype="int64"), dtype=np.float64),
    }
    if not gens and not flex:
        return out
    ctx = _da_grid_context(grid)
    every = pd.Index([t for t in res_da.timerange if da_start <= t < end], dtype="int64")

    def start_levels(stored: pd.DataFrame, initial: np.ndarray, what: str) -> pd.DataFrame:
        stored = stored.reindex(index=every)
        if stored.iloc[:-1].isna().any().any():
            raise ValueError(f"DA {what} levels have NaN")
        levels = stored.shift(1)
        levels.iloc[0] = initial
        return levels

    if gens:
        cap = grid.generator.loc[gens, "storage_cap"].to_numpy(dtype=float)
        storage = res_da.get_dispatch_frames(timeMaxMin=[da_start, end])["storage"][gens]
        # The level LpProblem starts from: storage_ini (or storage_ini_ref at its first row) x cap.
        initial = np.array([_da_initial_fraction(grid.generator, grid.profiles, d) for d in gens]) * cap
        levels = start_levels(storage, np.clip(initial, 0.0, cap), "storage")
        for t in steps:
            out["generator"].loc[t] = storval.generator_storage_values(
                grid, ctx, gens, levels.loc[t].to_numpy(dtype=float), int(t) - da_start
            )
    if flex:
        cons = grid.consumer
        stored = res_da.db.getResultFrame("Res_FlexibleLoad", "storage", [da_start, end])
        stored = stored.reindex(columns=pd.Index(flex, dtype="int64")).astype("float64")
        initial = (
            (cons["flex_storagelevel_init"] * cons["flex_storage"] * cons["flex_fraction"] * cons["demand_avg"])
            .fillna(0)
            .loc[flex]
            .to_numpy(dtype=float)
        )
        levels = start_levels(stored, initial, "flexible-load reservoir")
        for t in steps:
            out["consumer"].loc[t] = [
                storval.flexload_storage_value(grid, ctx, j, float(levels.at[t, j]), int(t) - da_start) for j in flex
            ]
    return out


def _value_series(frame, name: str, d: int, expected: pd.Index, what: str) -> pd.Series:
    if frame is None or d not in frame.columns:
        raise ValueError(f"RT {what} rows of DA row {d} have storage: '{name}' has no column for it")
    missing = expected.difference(frame.index)
    if len(missing):
        raise ValueError(f"'{name}' has no timestep(s) {list(missing[:5])}")
    values = frame[d].reindex(expected).astype(np.float64)
    if values.isna().any():
        raise ValueError(f"'{name}' has NaN for DA row {d}")
    return values


def price_columns(
    generator: pd.DataFrame,
    consumer: pd.DataFrame,
    *,
    fee: float,
    renewable=None,
    curtailment_factor: float = 2.0,
    storage_value: pd.DataFrame | None = None,
    flex_value: pd.DataFrame | None = None,
    da_index: str = DA_INDEX,
    timesteps=None,
) -> RefColumns:
    """The RT deviation prices of the RT rows (EUR/MWh), as unit data.

    "Up" is more than DA (more output, pumping or flexible demand), "down" less. *fee* is
    the balancing fee ``c0``, paid on every MWh of deviation; ``c`` is the unit's marginal
    cost: ``fuelcost``, plus the storage value at the DA filling for a unit with storage.

    - Generator rows in *renewable*: ``rt_price_up = -c0``,
      ``rt_price_down = curtailment_factor x c0``.
    - Other generator rows: ``rt_price_up = c0 + c``, ``rt_price_down = c0 - c``.
    - Generators with pumping: ``rt_pump_price_up = c0 - (v x pump_efficiency -
      pump_deadband)``, ``rt_pump_price_down = c0 + (v x pump_efficiency - pump_deadband)``,
      v the storage value at the DA filling.
    - Consumers with flexible load: ``rt_price_up = c0 + v``, ``rt_price_down = c0 - v``,
      v the flexible-load value at the DA filling.

    A price that does not vary in time is a constant in ``rt_price_up``/``rt_price_down``
    (``*_ref = "const"``). Where it varies (storage units: generators with storage, pumping,
    flexible load), the constant is 1 and ``*_ref`` names a profile column holding the price,
    named after the DA row d: ``da_price_up_<d>``, ``da_price_down_<d>``,
    ``da_pump_price_up_<d>``, ``da_pump_price_down_<d>``, ``da_cons_price_up_<d>``,
    ``da_cons_price_down_<d>``. RT computes ``constant x profile(t)``.

    Parameters
    ----------
    generator, consumer:
        The RT tables (``fuelcost``, ``storage_cap``, ``pump_cap``, ``pump_efficiency``,
        ``pump_deadband``; the flexible-load columns), with a *da_index* column.
    fee:
        ``c0``, EUR/MWh.
    renewable:
        Boolean Series on the RT generator index: the rows priced as renewables.
    curtailment_factor:
        The renewables' down price as a multiple of ``c0``.
    storage_value:
        The storage value of each DA generator row with storage, EUR/MWh: index = DA
        timesteps, columns = DA generator rows. ``da_storage_values(res_da)["generator"]``
        evaluates it at the DA filling; any other values may be given.
    flex_value:
        The same for the DA consumer rows with flexible load
        (``da_storage_values(res_da)["consumer"]``).
    da_index:
        As in :func:`target_columns`.
    timesteps:
        The timesteps of the profile columns; default those of *storage_value*, else of
        *flex_value*, else none.

    Every generator row, every row with pumping and every flexible consumer gets a price
    (rows without a target ignore it). A storage row without a DA row or without its value,
    a NaN, or two RT rows of one DA row that need different values in one column raise
    ``ValueError``.
    """
    fee = float(fee)
    factor = float(curtailment_factor)
    res_mask = _mask(renewable, generator, "renewable")
    if timesteps is not None:
        expected = pd.Index(list(timesteps))
    elif storage_value is not None:
        expected = storage_value.index
    elif flex_value is not None:
        expected = flex_value.index
    else:
        expected = pd.Index([], dtype="int64")
    gen_map = _mapping(generator, da_index, "generator")
    columns: dict[str, pd.Series] = {}

    def column(name: str, values: pd.Series) -> str:
        if name in columns:
            if not np.array_equal(columns[name].to_numpy(), values.to_numpy()):
                raise ValueError(f"Price column '{name}' would get different values from two RT rows")
        else:
            columns[name] = values
        return name

    def da_row(table, mapping, r, what) -> int:
        d = mapping.at[r]
        if pd.isna(d):
            raise ValueError(f"RT {what} row {r} has storage but no DA row: its storage value is unknown")
        return int(d)

    def number(table, r, col, what) -> float:
        value = _capacity(table.at[r, col]) if col in table.columns else math.nan
        if not np.isfinite(value):
            raise ValueError(f"RT {what} row {r} has '{col}' {value}: its deviation price needs a number")
        return value

    up = pd.Series(np.nan, index=generator.index, dtype=np.float64)
    down = pd.Series(np.nan, index=generator.index, dtype=np.float64)
    up_ref = pd.Series(PRICE_REF_CONST, index=generator.index, dtype=object)
    down_ref = pd.Series(PRICE_REF_CONST, index=generator.index, dtype=object)
    pump_up = pd.Series(np.nan, index=generator.index, dtype=np.float64)
    pump_down = pd.Series(np.nan, index=generator.index, dtype=np.float64)
    pump_up_ref = pd.Series(None, index=generator.index, dtype=object)
    pump_down_ref = pd.Series(None, index=generator.index, dtype=object)
    for r in generator.index:
        has_storage = 0.0 < _capacity(generator.at[r, "storage_cap"]) < np.inf if "storage_cap" in generator else False
        if res_mask.at[r]:
            up.at[r] = -fee
            down.at[r] = factor * fee
        elif has_storage:
            d = da_row(generator, gen_map, r, "generator")
            v = _value_series(storage_value, "storage_value", d, expected, "generator")
            cost = number(generator, r, "fuelcost", "generator") + v
            up.at[r] = down.at[r] = 1.0
            up_ref.at[r] = column(f"da_price_up_{d}", fee + cost)
            down_ref.at[r] = column(f"da_price_down_{d}", fee - cost)
        else:
            cost = number(generator, r, "fuelcost", "generator")
            up.at[r] = fee + cost
            down.at[r] = fee - cost
        if "pump_cap" in generator.columns and 0.0 < _capacity(generator.at[r, "pump_cap"]) < np.inf:
            if not has_storage:
                raise ValueError(f"RT generator row {r} has pumping but no storage")
            d = da_row(generator, gen_map, r, "generator")
            v = _value_series(storage_value, "storage_value", d, expected, "generator")
            credit = v * number(generator, r, "pump_efficiency", "generator") - number(
                generator, r, "pump_deadband", "generator"
            )
            pump_up.at[r] = pump_down.at[r] = 1.0
            pump_up_ref.at[r] = column(f"da_pump_price_up_{d}", fee - credit)
            pump_down_ref.at[r] = column(f"da_pump_price_down_{d}", fee + credit)

    cons_map = _mapping(consumer, da_index, "consumer")
    flex_rows = [r for r in consumer.index if _is_flexible(consumer, r)]
    c_up = pd.Series(np.nan, index=flex_rows, dtype=np.float64)
    c_down = pd.Series(np.nan, index=flex_rows, dtype=np.float64)
    c_up_ref = pd.Series(None, index=flex_rows, dtype=object)
    c_down_ref = pd.Series(None, index=flex_rows, dtype=object)
    for r in flex_rows:
        d = da_row(consumer, cons_map, r, "consumer")
        v = _value_series(flex_value, "flex_value", d, expected, "consumer")
        c_up.at[r] = c_down.at[r] = 1.0
        c_up_ref.at[r] = column(f"da_cons_price_up_{d}", fee + v)
        c_down_ref.at[r] = column(f"da_cons_price_down_{d}", fee - v)

    profiles = pd.DataFrame(columns, index=expected, dtype=np.float64)
    gen_refs = pd.DataFrame(
        {
            "rt_price_up": up,
            "rt_price_down": down,
            "rt_price_up_ref": up_ref,
            "rt_price_down_ref": down_ref,
            "rt_pump_price_up": pump_up,
            "rt_pump_price_down": pump_down,
            "rt_pump_price_up_ref": pump_up_ref,
            "rt_pump_price_down_ref": pump_down_ref,
        }
    )
    cons_refs = pd.DataFrame(
        {
            "rt_price_up": c_up,
            "rt_price_down": c_down,
            "rt_price_up_ref": c_up_ref,
            "rt_price_down_ref": c_down_ref,
        },
        index=pd.Index(flex_rows, dtype=consumer.index.dtype),
    )
    return RefColumns(profiles=profiles, generator=gen_refs, consumer=cons_refs)


@dataclass
class DeviationPrices:
    """The deviation-price rules :func:`prepare_rt` applies, see :func:`price_columns`.

    fee:
        ``c0``, EUR/MWh.
    renewable:
        Boolean Series on the DA generator index: the rows priced as renewables.
    curtailment_factor:
        The renewables' down price as a multiple of ``c0``.
    storage_value:
        ``{"generator": ..., "consumer": ...}`` as returned by :func:`da_storage_values`;
        default: that, from the DA run (the storage value at the DA filling level).
    source:
        Where the rules come from, for the summary.
    """

    fee: float
    renewable: pd.Series | None = None
    curtailment_factor: float = 2.0
    storage_value: dict[str, pd.DataFrame] | None = None
    source: str | None = None
