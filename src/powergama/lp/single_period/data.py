"""Per-timestep data for the single-period formulation.

The counterpart of :mod:`..multi_period.data`, and the one place the two
formulations are shaped differently. Here the model was built once and this
rewrites its mutable Params and variable bounds *in place*; there, the
equivalent module returns a snapshot that is baked into a model rebuilt for each
window. Collapsing that difference is what merging the formulations would mean.

These take the LpProblem rather than explicit arguments. Unlike the builders,
which read nothing mutable and so get real signatures, the updater reads and
writes simulation state -- storage levels, previous dispatch, the loss
linearisation point -- alongside grid data and config. An eight-argument
signature would be a longer way of spelling `self`, not an interface.
"""

import numpy as np
import pandas as pd

from ... import constants as const
from .. import storage_value


def _profile_values(profiles, timestep, refs, required=True):
    """Value at *timestep* of the profile each entry of *refs* names, as an array.

    With required=False, an entry that names no profile (not a string, or not a
    column) gives NaN instead of raising.
    """
    pos = profiles.columns.get_indexer(pd.Index(refs, dtype=object))
    missing = pos < 0
    if required and missing.any():
        raise KeyError(f"No such profile: {sorted(set(np.asarray(refs, dtype=object)[missing]))}")
    values = profiles.loc[timestep].to_numpy()[pos].astype(float)
    values[missing] = np.nan
    return values


def _store(param, index, values):
    """Write *values* into the mutable Param *param* at *index*, in one go.

    Bypasses Pyomo's per-element validation, which dominated the update time: the
    values are computed here from validated grid data. APPSI reads the new values
    from the Param when it next solves.
    """
    param.store_values(dict(zip(index, np.asarray(values, dtype=float).tolist())), check=False)


def update_timestep(lp, timestep):
    """
    Function that updates LP problem for a given timestep, due to changed
    power demand, power inflow and marginal generator costs

    Computed column-wise with numpy rather than generator by generator: this runs
    every timestep, and on a large grid the per-element pandas lookups used to cost
    more than solving the LP.
    """
    grid = lp._grid
    gen = grid.generator
    storage = np.asarray(lp._storage, dtype=float)
    is_storage = np.zeros(len(gen), dtype=bool)
    is_storage[lp._idx_generatorsWithStorage] = True

    # 1. Generator output limits:
    #    -> power output constraints
    P_max = gen["pmax"].to_numpy(float)
    P_min = gen["pmin"].to_numpy(float)

    # Time-varying pmax factor is applied to installed capacity.
    pmax_factor = np.ones(len(gen))
    if lp._pmax_ref is not None:
        factor = _profile_values(grid.profiles, timestep, lp._pmax_ref, required=False)
        pmax_factor = np.where(np.isnan(factor), 1.0, factor)
    capacity_now = np.maximum(0, P_max * pmax_factor)
    P_inflow = (
        capacity_now * gen["inflow_fac"].to_numpy(float) * _profile_values(grid.profiles, timestep, gen["inflow_ref"])
    )

    # pmin_ref profile is interpreted as fraction of installed pmax.
    pmin_now = P_min
    if lp._pmin_ref is not None:
        fraction = _profile_values(grid.profiles, timestep, lp._pmin_ref, required=False)
        pmin_now = np.where(np.isnan(fraction), P_min, np.maximum(0, P_max * fraction))

    # Don't let P_max limit the output of generators without storage (e.g. solar PV).
    # This won't affect fuel based generators with zero storage, since these should
    # have inflow=p_max in any case.
    available = np.maximum(0, P_inflow + storage / lp.timeDelta)
    pmin = np.where(
        is_storage, np.maximum(np.minimum(available, pmin_now), 0), np.maximum(np.minimum(P_inflow, pmin_now), 0)
    )
    pmax = np.where(is_storage, np.minimum(available, capacity_now), P_inflow)

    # 1b. Apply ramp-rate limits around previous-timestep dispatch,
    #     with delta caps scaled by installed capacity (ramp_pu * pmax_installed).
    #     Generators with NaN ramp values or NaN _gen_prev (first timestep) are unconstrained.
    # Daily-reset generators have their _gen_prev cleared at the start of each 24-hour window,
    # so they are free to choose a new level at midnight while staying fixed intra-day.
    if lp._ramp_daily_reset is not None and timestep % 24 == 0:
        lp._gen_prev[lp._ramp_daily_reset & ~lp._disable_nuclear_ramp_profile_conflict] = np.nan
    if lp._ramp_up_pu is not None or lp._ramp_down_pu is not None:
        prev = lp._gen_prev
        ramped = ~np.isnan(prev) & ~lp._disable_nuclear_ramp_profile_conflict
        # An extension may exempt fully pinned generators (pmin ≈ pmax) from the
        # ramp: clipping pmax below pmin makes the guard below lower pmin, moving the
        # generator off its pinned output. See ModelExtension.exempt_pinned_from_ramp.
        if lp._exempt_pinned_from_ramp:
            ramped &= ~(pmax - pmin < 1e-6)
        cap = lp._ramp_cap_mw.astype(float)
        pmax_ramped, pmin_ramped = pmax, pmin
        if lp._ramp_up_pu is not None:
            ramp_up = lp._ramp_up_pu.astype(float)
            pmax_ramped = np.where(~np.isnan(ramp_up), np.minimum(pmax, prev + ramp_up * cap), pmax)
        if lp._ramp_down_pu is not None:
            ramp_dn = lp._ramp_down_pu.astype(float)
            pmin_ramped = np.where(~np.isnan(ramp_dn), np.maximum(pmin, prev - ramp_dn * cap), pmin)
        # Guard: pmin must not exceed pmax after ramp clipping
        pmin_ramped = np.minimum(pmin_ramped, pmax_ramped)
        pmax = np.where(ramped, np.maximum(pmax_ramped, 0), pmax)
        pmin = np.where(ramped, np.maximum(pmin_ramped, 0), pmin)

    _store(lp.model.p_gen_pmax, lp.model.s_gen, pmax)
    _store(lp.model.p_gen_pmin, lp.model.s_gen, pmin)

    # 2. Update demand
    #    -> power balance constraint
    cons = grid.consumer
    average = cons["demand_avg"].to_numpy(float) * (1 - cons["flex_fraction"].to_numpy(float))
    _store(lp.model.p_demand, lp.model.s_load, _profile_values(grid.profiles, timestep, cons["demand_ref"]) * average)

    # 3. Cost parameters
    #    -> update objective function

    # 1c. Update pump capacity based on remaining reservoir room.
    #     Prevents the LP from scheduling pump output beyond storage_cap,
    #     which would otherwise cause silent energy spill while still earning
    #     the full pump cost credit (degeneracy when reservoir is full).
    pumps = lp._idx_generatorsWithPumping
    if pumps:
        pump_eff = gen["pump_efficiency"].to_numpy(float)[pumps]
        remaining_mwh = np.maximum(0.0, gen["storage_cap"].to_numpy(float)[pumps] - storage[pumps])
        with np.errstate(divide="ignore", invalid="ignore"):
            max_pump_by_room = np.where(pump_eff > 0, remaining_mwh / (pump_eff * lp.timeDelta), 0.0)
        _store(lp.model.p_pump_pmax, pumps, np.minimum(gen["pump_cap"].to_numpy(float)[pumps], max_pump_by_room))

    # 3a. generators with storage: level at the start of the timestep, inflow during
    #     it, and the value of the energy left at its end. The per-MWh costs of
    #     generating and pumping (p_gen_cost, p_genpump_cost) do not depend on the
    #     level and are set once, when the model is built.
    stores = lp._idx_generatorsWithStorage
    if stores:
        _store(lp.model.p_storage_prev, stores, storage[stores])
        _store(lp.model.p_storage_inflow, stores, P_inflow[stores])
        _store(
            lp.model.p_storage_value,
            stores,
            storage_value.generator_storage_values(grid, stores, storage[stores], timestep),
        )

    # 3b. flexible load (storage value)
    for i in lp._idx_consumersWithFlexLoad:
        lp.model.p_loadflex_cost[i] = storage_value.flexload_storage_value(
            lp._grid, i, lp._storage_flexload[i], timestep
        )

    # 4. Extensions, which may override what is set above. The core sets no
    #    branch-flow variable bounds: the rating is enforced by cMaxFlowAc/cMaxFlowDc.
    for ext in lp._extensions:
        ext.update_timestep(lp, timestep)

    return


def update_loss_parameters(lp, aclossmultiplier=1, dclossmultiplier=1):
    """Compute/update parameters used for transmission loss calculations.

    The losses for the coming timestep are linearised around the flows of the one
    just solved, read from varAcBranchFlow / varDcBranchFlow. Both methods use the
    net flow only, which is also what the results database stores, so a resumed run
    can seed the same operating point (seed_loss_operating_point).
    """
    if lp._lossmethod == 1:
        # store power flows for next timestep (used for loss calculation)
        # Github issue https://github.com/powergama/powergama/issues/29
        # write it in terms of the net flow and not the directional flow12/flow21 because
        # under certain circumstances, flow12 and flow21 may both be large (if high loss
        # is beneficial) since there is no constraint forbidding simultaneous flow in
        # both direction (only an indirect cost via generation costs). That holds for DC
        # branches as much as AC: nothing physical prevents it there either (issue #41).
        for b in lp.model.s_branch_ac:
            flow = lp.model.varAcBranchFlow[b].value
            lp.model.p_branch_ac_powerflow12[b] = max(flow, 0)
            lp.model.p_branch_ac_powerflow21[b] = max(-flow, 0)
        for b in lp.model.s_branch_dc:
            flow = lp.model.varDcBranchFlow[b].value
            lp.model.p_branch_dc_powerflow12[b] = max(flow, 0)
            lp.model.p_branch_dc_powerflow21[b] = max(-flow, 0)
    elif lp._lossmethod == 2:
        # compute power losses and store for next timestep
        for b in lp.model.s_branch_ac:
            r = lp._grid.branch.loc[b, "resistance"]
            lossMVA = r * lp.model.varAcBranchFlow[b] ** 2 / const.baseMVA
            # A multiplication factor to account for reactive current losses
            lossMVA = lossMVA * aclossmultiplier
            lp.model.p_branch_ac_power_loss[b] = lossMVA
        for b in lp.model.s_branch_dc:
            r_pu = lp._grid.dcbranch.loc[b, "resistance"]
            p_pu = lp.model.varDcBranchFlow[b] / const.baseMVA
            loss_pu = r_pu * p_pu**2
            lossMVA = loss_pu * const.baseMVA * dclossmultiplier
            lp.model.p_branch_dc_power_loss[b] = lossMVA


def seed_loss_operating_point(lp, ac_flows, dc_flows):
    """Set the flows the next loss update linearises around, from stored results.

    *ac_flows* and *dc_flows* map branch index to net flow (MW), as stored in
    Res_Branches / Res_DcBranches. Used when resuming a run: the preceding
    timestep's solution then exists only in the database, not in Var.value.
    """
    for b in lp.model.s_branch_ac:
        lp.model.varAcBranchFlow[b].value = ac_flows[b]
    for b in lp.model.s_branch_dc:
        lp.model.varDcBranchFlow[b].value = dc_flows[b]
