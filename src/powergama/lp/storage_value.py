"""Marginal value of stored energy, in generator storage and flexible-load reservoirs.

Shared by both formulations, which is why it sits here rather than under either
of them. Both evaluate it from the level at the start of the LP: the single-period
loop every timestep, the multi-period path once per window.
"""

import numpy as np

from .. import constants as const


def _positions(index, labels):
    """Positions of *labels* in *index*; a missing label raises, as ``.loc`` would."""
    pos = index.get_indexer(labels)
    if (pos < 0).any():
        raise KeyError(f"Not found in storage value table: {sorted(set(np.asarray(labels)[pos < 0].tolist()))}")
    return pos


def generator_storage_values(grid, gens, storage_levels, timestep):
    """Marginal value of energy in the storage of each generator in *gens*, in EUR/MWh.

    storage_price × filling-level factor × time factor. Both formulations value the
    energy left in storage at the end of the LP with this (the terminal storage
    value); the filling level is that at the start of the LP and *timestep* is its
    last timestep. Returns an array aligned with *gens*.
    """
    gens = list(gens)
    if not gens:
        return np.zeros(0)
    gen = grid.generator.loc[gens]
    fillinglevel = np.clip(np.asarray(storage_levels, dtype=float) / gen["storage_cap"].to_numpy(float), 0.0, 1.0)
    filling_row = _positions(grid.storagevalue_filling.index, np.rint(fillinglevel * 100).astype(int))
    filling_col = _positions(grid.storagevalue_filling.columns, gen["storval_filling_ref"])
    time_col = _positions(grid.storagevalue_time.columns, gen["storval_time_ref"])
    filling = grid.storagevalue_filling.to_numpy(float)[filling_row, filling_col]
    time = grid.storagevalue_time.to_numpy(float)[grid.storagevalue_time.index.get_loc(timestep), time_col]
    return gen["storage_price"].to_numpy(float) * filling * time


def flexload_storage_value(grid, i, storage_level, timestep):
    """Marginal value of energy in consumer *i*'s flexible-load reservoir, in EUR/MWh.

    The value follows the same filling-level × time-of-year curves used for
    generator storage, so it falls as the reservoir fills. Outside [0, 1] the
    curves do not apply and a flat penalty pushes the level back into range.

    Both solve paths price flexible load with this. The single-period loop re-evaluates
    it every timestep from the level carried in the caller's reservoir state; the
    multi-period path evaluates it once per window from the level at window start,
    the same approximation it already makes for generator storage.
    """
    # Reservoir size in MWh: flex_storage is given in hours of flexible demand.
    storagecapacity_flexload = (
        grid.consumer.loc[i, "flex_storage"]  # h
        * grid.consumer.loc[i, "flex_fraction"]
        * grid.consumer.loc[i, "demand_avg"]  # MW
    )
    fillinglevel = storage_level / storagecapacity_flexload
    if fillinglevel > 1:
        return -const.flexload_outside_cost
    if fillinglevel < 0:
        return const.flexload_outside_cost
    filling_col = int(round(fillinglevel * 100))
    return (
        grid.consumer.flex_basevalue[i]
        * grid.storagevalue_filling.loc[filling_col, grid.consumer.loc[i, "flex_storval_filling"]]
        * grid.storagevalue_time.loc[timestep, grid.consumer.loc[i, "flex_storval_time"]]
    )
