"""Marginal value of stored energy in a flexible-load reservoir.

Shared by both formulations, which is why it sits here rather than under either
of them. The single-period loop re-evaluates it every timestep from the level it
carries in ``_storage_flexload``; the multi-period path evaluates it once per
window from the level at window start -- the same approximation it already makes
for generator storage.
"""

from .. import constants as const


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
