"""Marginal value of stored energy, in generator storage and flexible-load reservoirs.

Shared by both formulations, which is why it sits here rather than under either
of them. Both evaluate it from the level at the start of the LP: the single-period
loop every timestep, the multi-period path once per window.

The value is a base value × a filling-level factor × a time factor. The filling-level
curves are ``grid.storagevalue_filling`` (101 rows, one per percent). The time factor
is an ordinary profile column (``storval_time_ref`` / ``flex_storval_time``), or, with
the deprecated separate table ``grid.storagevalue_time``, a column of that. Which
columns, and where they are, is resolved and checked once, in
:class:`~powergama.lp.context.GridContext` (``storval_*`` fields).
"""

import numpy as np

from .. import constants as const


def _profile_time_factors(profiles, positions, timestep):
    """Row *timestep* of the profile columns at *positions*; -1 (no profile) is a factor of 1."""
    factors = np.ones(len(positions))
    used = positions >= 0
    if used.any():
        factors[used] = profiles.iloc[profiles.index.get_loc(timestep), positions[used]].to_numpy(dtype=float)
    return factors


def generator_storage_values(grid, ctx, gens, storage_levels, timestep):
    """Marginal value of energy in the storage of each generator in *gens*, in EUR/MWh.

    storage_price × filling-level factor × time factor. Both formulations value the
    energy left in storage at the end of the LP with this (the terminal storage
    value); the filling level is that at the start of the LP and *timestep* is its
    last timestep. Returns an array aligned with *gens*, which must have storage.
    """
    gens = list(gens)
    if not gens:
        return np.zeros(0)
    gen = grid.generator.loc[gens]
    fillinglevel = np.clip(np.asarray(storage_levels, dtype=float) / gen["storage_cap"].to_numpy(float), 0.0, 1.0)
    filling_row = np.rint(fillinglevel * 100).astype(int)
    filling_col = np.array([ctx.storval_filling_pos[g] for g in gens], dtype=int)
    filling = grid.storagevalue_filling.to_numpy(float)[filling_row, filling_col]
    time_col = np.array([ctx.storval_time_pos[g] for g in gens], dtype=int)
    if ctx.storval_time_legacy:
        table = grid.storagevalue_time
        time = table.to_numpy(float)[table.index.get_loc(timestep), time_col]
    else:
        time = _profile_time_factors(grid.profiles, time_col, timestep)
    return gen["storage_price"].to_numpy(float) * filling * time


def flexload_storage_value(grid, ctx, i, storage_level, timestep):
    """Marginal value of energy in consumer *i*'s flexible-load reservoir, in EUR/MWh.

    The value follows the same filling-level × time curves used for generator
    storage, so it falls as the reservoir fills. Outside [0, 1] the curves do not
    apply and a flat penalty pushes the level back into range.

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
    filling = grid.storagevalue_filling.iat[filling_col, ctx.storval_flex_filling_pos[i]]
    if ctx.storval_time_legacy:
        time = grid.storagevalue_time.loc[timestep, grid.consumer.loc[i, "flex_storval_time"]]
    else:
        pos = ctx.storval_flex_time_pos[i]
        time = 1.0 if pos < 0 else grid.profiles.iat[grid.profiles.index.get_loc(timestep), pos]
    return grid.consumer.flex_basevalue[i] * filling * time
