"""The contract between a solved LP and the results database.

:class:`TimestepResult` names, once, the values the results database expects for a
single timestep. Both solve paths in :mod:`powergama.LpProblemPyomo` -- the sequential
hourly loop and the multi-period rolling horizon -- build one of these and call
:meth:`TimestepResult.store`, so there is a single site that knows the shape of
``Results.addResultsFromTimestep``.

The two paths read from genuinely different objects (hourly from ``self.var*[i]`` and
``self.dual``, multi-period from a separate ``(entity, hour)``-indexed model and a dual
dictionary), so what is shared here is the *contract*, not the extraction. Nothing in
this module touches a Pyomo model or an ``LpProblem``; ``results`` is used only for the
one call, and no ``Results`` type is imported.

Ordering is part of the contract. Every list is positional, and the position means
something different per field -- generator order, node order, "generators with storage"
order, and so on. See ``Results.addResultsFromTimestep`` for which index each one
follows; getting it wrong writes plausible numbers against the wrong entity.
"""

import dataclasses

import pandas as pd


@dataclasses.dataclass(slots=True)
class TimestepResult:
    """One timestep's solved values, in MW / EUR-per-MWh, ready for the database."""

    timestep: int
    objective_function: float
    generator_power: list
    generator_pumped: list
    branch_power: list
    dcbranch_power: list
    node_angle: list
    sensitivity_branch_capacity: list
    sensitivity_dcbranch_capacity: list
    sensitivity_node_power: list
    storage: list
    inflow_spilled: list
    loadshed_power: list
    dumpload_power: list
    marginalprice: list
    flexload_power: list
    flexload_storage: list
    flexload_storagevalue: list
    branch_ac_losses: list
    branch_dc_losses: list
    fault_start: int | None = None

    def store(self, results) -> None:
        """Write this timestep to the results database.

        The keyword names are taken from the field names rather than spelled out, so a
        field added here without a matching parameter on ``addResultsFromTimestep``
        fails loudly instead of being dropped.
        """
        results.addResultsFromTimestep(**{f.name: getattr(self, f.name) for f in dataclasses.fields(self)})


def aggregate_to_nodes(grid, loads, value_of) -> pd.Series:
    """Sum a per-consumer quantity onto the nodes the consumers sit at.

    Load shedding and dump load are stored per node, not per consumer, so both solve
    paths fold their consumer-indexed variables up this way.

    Parameters
    ----------
    grid : GridData
        Supplies ``node.id`` (the output index) and ``consumer["node"]`` (the mapping).
    loads : iterable
        Consumer indices to include.
    value_of : callable
        ``value_of(j)`` returns the MW value for consumer ``j``.
    """
    out = pd.Series(index=grid.node.id, data=0.0, dtype=float)
    for j in loads:
        out[grid.consumer["node"][j]] += value_of(j)
    return out
