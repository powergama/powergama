"""The balancing ("real-time") extension: :class:`RtExtension`.

RT is not a third formulation. It extends the single-period one: the same
physics, plus deviation tracking against a day-ahead schedule, and an objective
that minimises deviation instead of cost. Switch it on by passing it to the
problem::

    from powergama.lp.rt import RtExtension

    lp = powergama.LpProblem(data, extensions=[RtExtension()])

RT is the second step of a two-step workflow. A day-ahead run produces a
schedule; a prepare-rt step (:func:`prepare_rt` for the tables and the soft
targets) turns it into RT inputs -- hard border-flow windows (parquet
files), pins and fixed demand (``pmin_ref``/``pmax_ref`` and ``demand_ref``
profile columns, ordinary grid data) and soft targets (``rt_target_ref`` / ``rt_pump_target_ref`` profile
columns, border-flow targets) whose deviations are penalised at the units' deviation
prices (``rt_price_up``/``rt_price_down`` and the pumping equivalents, unit data written
by :func:`price_columns`).
Everything else is optimised freely.

The core never names RT. Everything here is reached through the hooks of
:class:`~powergama.lp.extension.ModelExtension`, so without the extension the
lock files are not read, no RT context is derived and no RT parameter, variable
or constraint is declared. ``tests/test_rt_isolation.py`` keeps it that way.

=========================  ==================================================
module                     holds
=========================  ==================================================
:mod:`.extension`          :class:`RtExtension`, wiring the pieces to the hooks
:mod:`.context`            :class:`RtContext`, derived once from the grid and locks
:mod:`.build`              RT parameters, deviation variables, tracking
                           constraints and the objective edit
:mod:`.data`               per-timestep locks, targets and deviation prices
:mod:`.diagnostics`        the optional JSONL debug stream
:mod:`.prepare`            :func:`prepare_rt`, from a day-ahead run to the RT
                           inputs (tables and the DA-solution profile file);
                           :func:`rt_tables_from_da`, the table copy it starts from
:mod:`.summary`            :func:`summarize_channels`, the overview of the
                           DA-derived columns that :func:`prepare_rt` writes
=========================  ==================================================
"""

from .extension import RtExtension
from .prepare import (
    ZERO_TARGET,
    CarryCurtailment,
    DeviationPrices,
    PeakTranches,
    PreparedRt,
    RefColumns,
    TargetColumns,
    apply_refs,
    carry_curtailment_columns,
    clear_refs,
    combine_columns,
    da_storage_values,
    initial_storage_columns,
    peak_tranche_columns,
    pin_columns,
    prepare_rt,
    price_columns,
    rt_tables_from_da,
    target_columns,
)
from .summary import channel_family, summarize_channels, write_channel_summary

__all__ = [
    "RtExtension",
    "ZERO_TARGET",
    "CarryCurtailment",
    "DeviationPrices",
    "PeakTranches",
    "PreparedRt",
    "RefColumns",
    "TargetColumns",
    "apply_refs",
    "carry_curtailment_columns",
    "channel_family",
    "clear_refs",
    "combine_columns",
    "da_storage_values",
    "initial_storage_columns",
    "peak_tranche_columns",
    "pin_columns",
    "prepare_rt",
    "price_columns",
    "rt_tables_from_da",
    "summarize_channels",
    "target_columns",
    "write_channel_summary",
]
