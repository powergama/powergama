"""Results of a solve kept in memory, for short solves that need no database.

:class:`MemoryResults` can be passed to :meth:`powergama.LpProblem.solve` in place of
:class:`powergama.Results`. It keeps every stored timestep in lists and gives them
back as DataFrames, with no sqlite file. It suits many short solves, such as a few
timesteps re-solved on a modified grid from a state taken from a full run::

    state = SimulationState.from_results(res, timestep=99)
    out = MemoryResults(grid)
    lp.solve(out, timesteps=range(100, 124), state=state)
    out.loadshed()                     # timestep x node, MW

It has none of the plotting and analysis methods of :class:`~powergama.Results`.
"""

import numpy as np
import pandas as pd


def _all(n):
    return lambda grid: list(range(n(grid)))


# Field of TimestepResult -> the element index each position of its list refers to,
# as in Results.addResultsFromTimestep.
_COLUMNS = {
    "generator_power": _all(lambda g: len(g.generator)),
    "inflow_spilled": _all(lambda g: len(g.generator)),
    "generator_pumped": lambda g: g.getIdxGeneratorsWithPumping(),
    "branch_power": _all(lambda g: len(g.branch)),
    "branch_ac_losses": _all(lambda g: len(g.branch)),
    "sensitivity_branch_capacity": lambda g: g.getIdxBranchesWithFlowConstraints(),
    "dcbranch_power": _all(lambda g: 0 if g.dcbranch is None else len(g.dcbranch)),
    "branch_dc_losses": _all(lambda g: 0 if g.dcbranch is None else len(g.dcbranch)),
    "sensitivity_dcbranch_capacity": _all(lambda g: 0 if g.dcbranch is None else len(g.dcbranch)),
    "node_angle": _all(lambda g: len(g.node)),
    "sensitivity_node_power": _all(lambda g: len(g.node)),
    "loadshed_power": _all(lambda g: len(g.node)),
    "dumpload_power": _all(lambda g: len(g.node)),
    "storage": lambda g: g.getIdxGeneratorsWithStorage(),
    "marginalprice": lambda g: g.getIdxGeneratorsWithStorage(),
    "flexload_power": lambda g: g.getIdxConsumersWithFlexibleLoad(),
    "flexload_storage": lambda g: g.getIdxConsumersWithFlexibleLoad(),
    "flexload_storagevalue": lambda g: g.getIdxConsumersWithFlexibleLoad(),
}


class MemoryResults:
    """Results of a solve, kept in memory.

    Parameters
    ----------
    grid : GridData
        The grid the solve is for; gives the element indices of the columns.
    """

    def __init__(self, grid):
        self.grid = grid
        self.timerange = grid.timerange
        self._timesteps = []
        self._objective = []
        self._values = {name: [] for name in _COLUMNS}

    def addResultsFromTimestep(self, timestep, objective_function, **values):
        """Store one timestep; called by the solve loop (see TimestepResult.store)."""
        unknown = set(values) - set(_COLUMNS)
        if unknown:
            raise TypeError(f"Unknown result fields: {sorted(unknown)}")
        self._timesteps.append(int(timestep))
        self._objective.append(float(objective_function))
        for name in _COLUMNS:
            v = values.get(name)
            self._values[name].append(None if v is None else np.asarray(v, dtype=float))

    def __len__(self):
        return len(self._timesteps)

    @property
    def timesteps(self) -> list:
        """The stored timesteps, as stored by Results (``grid.timerange[0] + t``)."""
        return list(self._timesteps)

    def objective(self) -> pd.Series:
        """Objective function value per timestep."""
        return pd.Series(self._objective, index=pd.Index(self._timesteps, name="timestep"), name="objective")

    def frame(self, name) -> pd.DataFrame:
        """One stored quantity as a DataFrame: timestep x element index.

        `name` is a field of :class:`powergama.timestep_result.TimestepResult`, e.g.
        "generator_power", "loadshed_power", "storage" or "flexload_storage". Columns
        are the element indices the quantity is given for (all nodes, generators with
        storage, and so on).
        """
        if name not in _COLUMNS:
            raise KeyError(f"{name!r}; one of {sorted(_COLUMNS)}")
        columns = _COLUMNS[name](self.grid)
        rows = [np.full(len(columns), np.nan) if v is None else v for v in self._values[name]]
        data = np.vstack(rows) if rows else np.empty((0, len(columns)))
        return pd.DataFrame(data, index=pd.Index(self._timesteps, name="timestep"), columns=columns)

    def loadshed(self) -> pd.DataFrame:
        """Load shed per node (MW): timestep x node index."""
        return self.frame("loadshed_power")
