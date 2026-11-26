"""aggregate_to_nodes: per-consumer values summed onto nodes, with and without a GridContext."""

import pandas as pd

import powergama
from powergama.timestep_result import aggregate_to_nodes


def _reference(grid, loads, value_of):
    """The earlier pandas groupby version."""
    values = pd.Series([value_of(j) for j in loads], index=grid.consumer["node"].loc[loads].to_numpy(), dtype=float)
    return values.groupby(level=0, sort=False).sum().reindex(grid.node.id, fill_value=0.0)


def test_aggregate_to_nodes_matches_groupby(testcase_9bus_data):
    grid = testcase_9bus_data
    # two consumers on one node, and a missing value (None counts as 0)
    grid.consumer.loc[1, "node"] = grid.consumer.loc[0, "node"]
    ctx = powergama.LpProblem(grid)._ctx
    loads = list(grid.consumer.index)
    values = {j: (None if j == 2 else 10.0 * (j + 1)) for j in loads}
    want = _reference(grid, loads, values.get)
    for got in (aggregate_to_nodes(grid, loads, values.get), aggregate_to_nodes(grid, loads, values.get, ctx)):
        pd.testing.assert_series_equal(got, want, check_names=False)
