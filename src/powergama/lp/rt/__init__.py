"""The balancing ("real-time") extension: :class:`RtExtension`.

RT is not a third formulation. It extends the single-period one: the same
physics, plus deviation tracking against a day-ahead schedule, and an objective
that minimises deviation instead of cost. Switch it on by passing it to the
problem::

    from powergama.lp.rt import RtExtension

    lp = powergama.LpProblem(data, extensions=[RtExtension(balancing_fee_eur_per_mwh=100)])

RT is the second step of a two-step workflow. A day-ahead run produces a
schedule; an external prepare-rt step turns it into RT inputs -- hard locks
(parquet files: pinned foreign generators, fixed foreign demand, border-flow
windows) and soft targets (``rt_target_ref`` / ``rt_pump_target_ref`` profile
columns, border-flow targets) whose deviations are penalised.
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
=========================  ==================================================
"""

from .extension import RtExtension

__all__ = ["RtExtension"]
