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
import pyomo.environ as pyo

from ... import constants as const
from .. import storage_value
from ..rt import data as rt_data


def update_timestep(lp, timestep):
    """
    Function that updates LP problem for a given timestep, due to changed
    power demand, power inflow and marginal generator costs
    """

    # 1. Generator output limits:
    #    -> power output constraints
    P_storage = lp._storage / lp.timeDelta
    P_max = lp._grid.generator["pmax"]
    P_min = lp._grid.generator["pmin"]
    for i in lp.model.s_gen:
        inflow_factor = lp._grid.generator.loc[i, "inflow_fac"]
        inflow_profile = lp._grid.generator.loc[i, "inflow_ref"]

        # Time-varying pmax factor is applied to installed capacity.
        pmax_factor = 1.0
        if lp._pmax_ref is not None:
            pmax_ref = lp._pmax_ref.iloc[i]
            if isinstance(pmax_ref, str) and pmax_ref in lp._grid.profiles.columns:
                pmax_factor = lp._grid.profiles.loc[timestep, pmax_ref]
        capacity_now = max(0, P_max[i] * pmax_factor)
        P_inflow = capacity_now * inflow_factor * lp._grid.profiles.loc[timestep, inflow_profile]

        # pmin_ref profile is interpreted as fraction of installed pmax.
        pmin_now = P_min[i]
        if lp._pmin_ref is not None:
            pmin_ref = lp._pmin_ref.iloc[i]
            if isinstance(pmin_ref, str) and pmin_ref in lp._grid.profiles.columns:
                pmin_now = max(0, P_max[i] * lp._grid.profiles.loc[timestep, pmin_ref])
        if i not in lp._idx_generatorsWithStorage:
            """
            Don't let P_max limit the output (e.g. solar PV)
            This won't affect fuel based generators with zero storage,
            since these should have inflow=p_max in any case
            """
            lp.model.p_gen_pmin[i] = max(min(P_inflow, pmin_now), 0)
            lp.model.p_gen_pmax[i] = P_inflow
        else:
            # generator has storage
            lp.model.p_gen_pmin[i] = max(min(max(0, P_inflow + P_storage[i]), pmin_now), 0)
            lp.model.p_gen_pmax[i] = min(max(0, P_inflow + P_storage[i]), capacity_now)

        # Optional hard lock of foreign generators to DA output (sparse parquet lock table).
        if lp._foreign_gen_lock is not None:
            key = (int(timestep), int(i))
            if key in lp._foreign_gen_lock.index:
                da_output = float(lp._foreign_gen_lock.loc[key])
                if not np.isfinite(da_output):
                    da_output = 0.0
                da_output = max(0.0, da_output)
                lp.model.p_gen_pmin[i] = da_output
                lp.model.p_gen_pmax[i] = da_output

    # 1b. Apply ramp-rate limits (PU of installed capacity) around previous dispatch.
    # 1b. Apply ramp-rate limits around previous-timestep dispatch,
    #     with delta caps scaled by installed capacity (ramp_pu * pmax_installed).
    #     Generators with NaN ramp values or NaN _gen_prev (first timestep) are unconstrained.
    # Daily-reset generators have their _gen_prev cleared at the start of each 24-hour window,
    # so they are free to choose a new level at midnight while staying fixed intra-day.
    if lp._ramp_daily_reset is not None and timestep % 24 == 0:
        for i in lp.model.s_gen:
            if lp._disable_nuclear_ramp_profile_conflict[int(i)]:
                continue
            if lp._ramp_daily_reset[i]:
                lp._gen_prev[i] = np.nan
    if lp._ramp_up_pu is not None or lp._ramp_down_pu is not None:
        for i in lp.model.s_gen:
            if lp._disable_nuclear_ramp_profile_conflict[int(i)]:
                continue
            prev = lp._gen_prev[i]
            if np.isnan(prev):
                continue  # first timestep: no ramp constraint
            ramp_up = lp._ramp_up_pu[i] if lp._ramp_up_pu is not None else np.nan
            ramp_dn = lp._ramp_down_pu[i] if lp._ramp_down_pu is not None else np.nan
            pmax_now = pyo.value(lp.model.p_gen_pmax[i])
            pmin_now = pyo.value(lp.model.p_gen_pmin[i])
            # In RT, skip ramp constraints for fully pinned generators (pmin ≈ pmax),
            # e.g. nuclear and DA-locked gas. The profile lock already uniquely defines
            # the feasible output; applying the ramp on top clips pmax below pmin and
            # the guard silently lowers pmin, causing gas backoff and spurious load shedding.
            # In DA, generators are never pinned this way so this branch is never taken.
            if lp._is_rt and pmax_now - pmin_now < 1e-6:
                continue
            if not np.isnan(ramp_up):
                pmax_now = min(pmax_now, prev + float(ramp_up) * float(lp._ramp_cap_mw[i]))
            if not np.isnan(ramp_dn):
                pmin_now = max(pmin_now, prev - float(ramp_dn) * float(lp._ramp_cap_mw[i]))
            # Guard: pmin must not exceed pmax after ramp clipping
            pmin_now = min(pmin_now, pmax_now)
            lp.model.p_gen_pmax[i] = max(pmax_now, 0)
            lp.model.p_gen_pmin[i] = max(pmin_now, 0)

    # TODO: re-create constraint - if persistent solver

    # 2. Update demand
    #    -> power balance constraint
    for i in lp.model.s_load:
        average = lp._grid.consumer.loc[i, "demand_avg"] * (1 - lp._grid.consumer.loc[i, "flex_fraction"])
        profile_ref = lp._grid.consumer.loc[i, "demand_ref"]
        demand_now = lp._grid.profiles.loc[timestep, profile_ref] * average
        # Optional hard lock of foreign consumers to DA demand (sparse parquet lock table).
        if lp._foreign_cons_lock is not None:
            key = (int(timestep), int(i))
            if key in lp._foreign_cons_lock.index:
                locked_demand = float(lp._foreign_cons_lock.loc[key])
                if np.isfinite(locked_demand):
                    demand_now = locked_demand
        lp.model.p_demand[i] = demand_now

    # 3. Cost parameters
    #    -> update objective function

    # 1c. Update pump capacity based on remaining reservoir room.
    #     Prevents the LP from scheduling pump output beyond storage_cap,
    #     which would otherwise cause silent energy spill while still earning
    #     the full pump cost credit (degeneracy when reservoir is full).
    pump_eff = lp._grid.generator["pump_efficiency"]
    storage_cap = lp._grid.generator["storage_cap"]
    pump_cap_raw = lp._grid.generator["pump_cap"]
    for i in lp._idx_generatorsWithPumping:
        remaining_mwh = max(0.0, storage_cap[i] - lp._storage[i])
        max_pump_by_room = remaining_mwh / (pump_eff[i] * lp.timeDelta) if pump_eff[i] > 0 else 0.0
        lp.model.p_pump_pmax[i] = min(pump_cap_raw[i], max_pump_by_room)

    # 3a. generators with storage (storage value)
    for i in lp._idx_generatorsWithStorage:
        this_type_filling = lp._grid.generator.loc[i, "storval_filling_ref"]
        this_type_time = lp._grid.generator.loc[i, "storval_time_ref"]
        storagecapacity = lp._grid.generator.loc[i, "storage_cap"]
        fillinglevel = lp._storage[i] / storagecapacity
        filling_col = int(round(fillinglevel * 100))
        storagevalue = (
            lp._grid.generator.loc[i, "storage_price"]
            * lp._grid.storagevalue_filling.loc[filling_col, this_type_filling]
            * lp._grid.storagevalue_time.loc[timestep, this_type_time]
        )
        lp.model.p_gen_cost[i] = storagevalue
        if i in lp._idx_generatorsWithPumping:
            deadband = lp._grid.generator.pump_deadband[i]
            lp.model.p_genpump_cost[i] = storagevalue - deadband

    # 3b. flexible load (storage value)
    for i in lp._idx_consumersWithFlexLoad:
        lp.model.p_loadflex_cost[i] = storage_value.flexload_storage_value(
            lp._grid, i, lp._storage_flexload[i], timestep
        )

    # 4. Reset AC/DC branch flow bounds: to the DA-locked window where the
    #    project-specific bounds artifact provides one, otherwise clear them.
    #    The rating is already enforced by cMaxFlowAc/cMaxFlowDc; repeating it
    #    as a variable bound makes the solver price congestion on the bound
    #    (reduced cost) and leave the constraint dual at zero (issue #39).
    #    Runs in every mode. The RT border-flow *targets* that used to share
    #    this loop are written by rt_data.update_border_flow_targets.
    for b in lp.model.s_branch_ac:
        key = (int(timestep), int(b))
        if (
            lp._border_ac_flow_lb is not None
            and lp._border_ac_flow_ub is not None
            and key in lp._border_ac_flow_lb.index
            and key in lp._border_ac_flow_ub.index
        ):
            lb = float(lp._border_ac_flow_lb.loc[key])
            ub = float(lp._border_ac_flow_ub.loc[key])
            if not np.isfinite(lb):
                lb = None
            if not np.isfinite(ub):
                ub = None
            lp.model.varAcBranchFlow[b].setlb(lb)
            lp.model.varAcBranchFlow[b].setub(ub)
        else:
            lp.model.varAcBranchFlow[b].setlb(None)
            lp.model.varAcBranchFlow[b].setub(None)

    for b in lp.model.s_branch_dc:
        key = (int(timestep), int(b))
        if (
            lp._border_dc_flow_lb is not None
            and lp._border_dc_flow_ub is not None
            and key in lp._border_dc_flow_lb.index
            and key in lp._border_dc_flow_ub.index
        ):
            lb = float(lp._border_dc_flow_lb.loc[key])
            ub = float(lp._border_dc_flow_ub.loc[key])
            if not np.isfinite(lb):
                lb = None
            if not np.isfinite(ub):
                ub = None
            lp.model.varDcBranchFlow[b].setlb(lb)
            lp.model.varDcBranchFlow[b].setub(ub)
        else:
            lp.model.varDcBranchFlow[b].setlb(None)
            lp.model.varDcBranchFlow[b].setub(None)

    if lp._is_rt:
        rt_data.update_generator_targets(lp, timestep, P_max)
        rt_data.update_flexload_and_storage_targets(lp, timestep)
        rt_data.update_deviation_prices(lp, timestep)
        rt_data.update_border_flow_targets(lp, timestep)

    return


def update_loss_parameters(lp, aclossmultiplier=1, dclossmultiplier=1):
    """Compute/update parameters used for transmission loss calculations"""
    if lp._lossmethod == 1:
        # store power flows for next timestep (used for loss calculation)
        for b in lp.model.s_branch_ac:
            # Github issue https://github.com/powergama/powergama/issues/29
            # write it in terms of varAcBranchFlow and not varBranchFlow12 because
            # under certain circumstances, varAcBranchFlow12 and varAcBranchFlow21
            # may both be large (if high loss is beneficial) since there is no constraint
            # forbidding simultaneous flow in both direction (only an indirect cost
            # via generation costs)
            if lp.model.varAcBranchFlow[b].value > 0:
                lp.model.p_branch_ac_powerflow12[b] = lp.model.varAcBranchFlow[b].value
                lp.model.p_branch_ac_powerflow21[b] = 0
            else:
                lp.model.p_branch_ac_powerflow12[b] = 0
                lp.model.p_branch_ac_powerflow21[b] = -lp.model.varAcBranchFlow[b].value
        for b in lp.model.s_branch_dc:
            # not the same issue for dc lines because there are not power flow equations
            lp.model.p_branch_dc_powerflow12[b] = lp.model.varDcBranchFlow12[b].value
            lp.model.p_branch_dc_powerflow21[b] = lp.model.varDcBranchFlow21[b].value
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
