"""The contract between a solved LP and the results database.

:class:`TimestepResult` names, once, the values the results database expects for a
single timestep. The solve loop in :mod:`powergama.lp` builds one for each committed
timestep of a window (``lp.formulation.extract``) and calls
:meth:`TimestepResult.store`, so there is a single site that knows the shape of
``Results.addResultsFromTimestep``. Nothing in this module touches a Pyomo model or an
``LpProblem``; ``results`` is used only for the one call, and no ``Results`` type is
imported.

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
    loads = list(loads)
    values = pd.Series([value_of(j) for j in loads], index=grid.consumer["node"].loc[loads].to_numpy(), dtype=float)
    return values.groupby(level=0, sort=False).sum().reindex(grid.node.id, fill_value=0.0)
