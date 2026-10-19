"""Per-window data: rewrite the model's mutable Params for the window about to be solved.

The model was built once for its window length (:mod:`.build`); this writes the
time-varying data into it *in place* before every solve, rather than rebuilding.

The window's first hour is computed from the simulation state -- storage levels,
the flexible-load reservoirs, the previous dispatch the ramp limits use -- exactly
as the single-period model always did, so a window of one timestep is that model.
Later hours only need the input data; their coupling to the state is through the
LP's own variables.

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


def _by_hour(index, hours):
    """Keys (i, h) for every *i* in *index* and every *h* in *hours*, hour by hour."""
    return [(i, h) for h in hours for i in index]


def _generator_limits(lp, timestep):
    """Installed capacity available, inflow and minimum output at *timestep*, in MW."""
    grid = lp._grid
    gen = grid.generator
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
    return capacity_now, P_inflow, pmin_now


def update_window(lp, timesteps):
    """
    Update the LP problem for the window of *timesteps*, due to changed
    power demand, power inflow and marginal generator costs

    Computed column-wise with numpy rather than generator by generator: this runs
    every window, and on a large grid the per-element pandas lookups used to cost
    more than solving the LP.
    """
    model = lp.model
    grid = lp._grid
    gen = grid.generator
    H = len(timesteps)
    timestep = timesteps[0]
    storage = np.asarray(lp._storage, dtype=float)
    is_storage = np.zeros(len(gen), dtype=bool)
    is_storage[lp._idx_generatorsWithStorage] = True

    # 1. Generator output limits:
    #    -> power output constraints
    # The first hour's availability from storage is known: what is stored now.
    capacity_now, P_inflow, pmin_now = _generator_limits(lp, timestep)
    inflow = [P_inflow]

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
    pmax_all, pmin_all = [pmax], [pmin]

    # Later hours: availability from storage is a constraint on the hour before
    # (cGenUbStorAvail), and ramp limits are constraints between the hours (cRampUp,
    # cRampDn), so the bounds are installed capacity and inflow only.
    for ts in timesteps[1:]:
        capacity_now, P_inflow, pmin_now = _generator_limits(lp, ts)
        inflow.append(P_inflow)
        pmax_all.append(np.where(is_storage, capacity_now, P_inflow))
        pmin_all.append(np.maximum(np.minimum(P_inflow, pmin_now), 0))

    hours = range(H)
    _store(model.p_gen_pmax, _by_hour(model.s_gen, hours), np.concatenate(pmax_all))
    _store(model.p_gen_pmin, _by_hour(model.s_gen, hours), np.concatenate(pmin_all))
    _update_ramp_limits(lp, timesteps)

    # 2. Update demand
    #    -> power balance constraint
    cons = grid.consumer
    average = cons["demand_avg"].to_numpy(float) * (1 - cons["flex_fraction"].to_numpy(float))
    demand = [_profile_values(grid.profiles, ts, cons["demand_ref"]) * average for ts in timesteps]
    _store(model.p_demand, _by_hour(model.s_load, hours), np.concatenate(demand))

    # 3. Cost parameters
    #    -> update objective function

    # 1c. Update pump capacity based on remaining reservoir room.
    #     Prevents the LP from scheduling pump output beyond storage_cap,
    #     which would otherwise cause silent energy spill while still earning
    #     the full pump cost credit (degeneracy when reservoir is full).
    #     Later hours are limited by the room the hour before leaves (cPumpRoom).
    pumps = lp._idx_generatorsWithPumping
    if pumps:
        pump_eff = gen["pump_efficiency"].to_numpy(float)[pumps]
        pump_cap = gen["pump_cap"].to_numpy(float)[pumps]
        remaining_mwh = np.maximum(0.0, gen["storage_cap"].to_numpy(float)[pumps] - storage[pumps])
        with np.errstate(divide="ignore", invalid="ignore"):
            max_pump_by_room = np.where(pump_eff > 0, remaining_mwh / (pump_eff * lp.timeDelta), 0.0)
        pump_pmax = [np.minimum(pump_cap, max_pump_by_room)] + [pump_cap] * (H - 1)
        _store(model.p_pump_pmax, _by_hour(pumps, hours), np.concatenate(pump_pmax))

    # 3a. generators with storage: level at the start of the window, inflow during
    #     each hour, and the value of the energy left at the window's end. The
    #     per-MWh costs of generating and pumping (p_gen_cost, p_genpump_cost) do not
    #     depend on the level and are set once, when the model is built.
    stores = lp._idx_generatorsWithStorage
    if stores:
        _store(model.p_storage_prev, stores, storage[stores])
        _store(model.p_storage_inflow, _by_hour(stores, hours), np.concatenate([q[stores] for q in inflow]))
        _store(
            model.p_storage_value,
            stores,
            storage_value.generator_storage_values(grid, stores, storage[stores], timesteps[-1]),
        )

    # 3b. flexible load: reservoir level at the start of the window, and the value of
    #     the energy delivered to it, from that level, for each hour
    for i in lp._idx_consumersWithFlexLoad:
        model.p_flex_storage_prev[i] = lp._ctx.flex_storage_seed(i, float(lp._storage_flexload[i]))
        for h, ts in enumerate(timesteps):
            model.p_loadflex_cost[i, h] = storage_value.flexload_storage_value(lp._grid, i, lp._storage_flexload[i], ts)

    # 4. Extensions, which may override what is set above. They only run in windows
    #    of one timestep. The core sets no branch-flow variable bounds: the rating is
    #    enforced by cMaxFlowAc/cMaxFlowDc.
    for ext in lp._extensions:
        ext.update_timestep(lp, timestep)

    return


def _update_ramp_limits(lp, timesteps):
    """Write the ramp limits between consecutive hours of the window, in MW.

    A generator with a daily reset is free to change its output at midnight, so its
    limit into a midnight hour is lifted to twice its installed capacity -- more than
    it could ever change by.
    """
    model = lp.model
    cap = lp._ramp_cap_mw.astype(float)
    reset_hours = [h for h, ts in enumerate(timesteps) if ts % 24 == 0]
    for param, keys, ramp in (
        (model.p_ramp_up_mw, list(model.s_ramp_up), lp._ramp_up_pu),
        (model.p_ramp_dn_mw, list(model.s_ramp_dn), lp._ramp_down_pu),
    ):
        if not keys:
            continue
        gens = np.array([i for i, _ in keys])
        hours = np.array([h for _, h in keys])
        limit = ramp.astype(float)[gens] * cap[gens]
        if lp._ramp_daily_reset is not None and reset_hours:
            released = lp._ramp_daily_reset[gens] & np.isin(hours, reset_hours)
            limit = np.where(released, 2 * cap[gens], limit)
        _store(param, keys, limit)


def update_loss_parameters(lp, aclossmultiplier=1, dclossmultiplier=1):
    """Compute/update parameters used for transmission loss calculations.

    The losses for the coming timestep are linearised around the flows of the one
    just solved, read from varAcBranchFlow / varDcBranchFlow. Both methods use the
    net flow only, which is also what the results database stores, so a resumed run
    can seed the same operating point (seed_loss_operating_point). Losses are only
    modelled in windows of one timestep, hence hour 0 throughout.
    """
    m = lp.model
    if lp._lossmethod == 1:
        # store power flows for next timestep (used for loss calculation)
        # Github issue https://github.com/powergama/powergama/issues/29
        # write it in terms of the net flow and not the directional flow12/flow21 because
        # under certain circumstances, flow12 and flow21 may both be large (if high loss
        # is beneficial) since there is no constraint forbidding simultaneous flow in
        # both direction (only an indirect cost via generation costs). That holds for DC
        # branches as much as AC: nothing physical prevents it there either (issue #41).
        for b in m.s_branch_ac:
            flow = m.varAcBranchFlow[b, 0].value
            m.p_branch_ac_powerflow12[b, 0] = max(flow, 0)
            m.p_branch_ac_powerflow21[b, 0] = max(-flow, 0)
        for b in m.s_branch_dc:
            flow = m.varDcBranchFlow[b, 0].value
            m.p_branch_dc_powerflow12[b, 0] = max(flow, 0)
            m.p_branch_dc_powerflow21[b, 0] = max(-flow, 0)
    elif lp._lossmethod == 2:
        # compute power losses and store for next timestep
        for b in m.s_branch_ac:
            r = lp._grid.branch.loc[b, "resistance"]
            lossMVA = r * m.varAcBranchFlow[b, 0] ** 2 / const.baseMVA
            # A multiplication factor to account for reactive current losses
            lossMVA = lossMVA * aclossmultiplier
            m.p_branch_ac_power_loss[b, 0] = lossMVA
        for b in m.s_branch_dc:
            r_pu = lp._grid.dcbranch.loc[b, "resistance"]
            p_pu = m.varDcBranchFlow[b, 0] / const.baseMVA
            loss_pu = r_pu * p_pu**2
            lossMVA = loss_pu * const.baseMVA * dclossmultiplier
            m.p_branch_dc_power_loss[b, 0] = lossMVA


def seed_loss_operating_point(lp, ac_flows, dc_flows):
    """Set the flows the next loss update linearises around, from stored results.

    *ac_flows* and *dc_flows* map branch index to net flow (MW), as stored in
    Res_Branches / Res_DcBranches. Used when resuming a run: the preceding
    timestep's solution then exists only in the database, not in Var.value.
    """
    for b in lp.model.s_branch_ac:
        lp.model.varAcBranchFlow[b, 0].value = ac_flows[b]
    for b in lp.model.s_branch_dc:
        lp.model.varDcBranchFlow[b, 0].value = dc_flows[b]
