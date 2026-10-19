"""Read a solved multi-period window into one TimestepResult per committed step.

Shares the 21-field contract with the single-period extractor through
TimestepResult, but not the code: this reads an (entity, hour)-indexed model and
a dual dict, where the other reads scalar vars and a Pyomo Suffix. It also
recomputes the per-hour objective from the prepared cost data, because the
model's OBJ is the total over the whole window.
"""

import numpy as np
import pyomo.environ as pyo

from ... import constants as const
from ...timestep_result import TimestepResult, aggregate_to_nodes


def extract_and_store(lp, m, wd, results, duals, commit_hours=None):
    """Extract results from a solved 24h joint model and store to results DB.

    Emits one TimestepResult per committed hour in the optimisation window --
    the same contract the single-period path uses -- then updates lp._storage to
    end-of-commit levels.
    """
    ctx, grid = lp._ctx, lp._grid
    H = wd.H
    if commit_hours is None:
        commit_hours = H
    commit_hours = int(max(1, min(int(commit_hours), int(H))))
    day_timesteps = wd.day_timesteps
    # inflow_h = wd.inflow_h
    # storage_ini = wd.storage_ini
    # storage_cap = wd.storage_cap
    # pump_eff_g = wd.pump_eff_g
    # gen_storage_set = wd.gen_storage_set
    # gen_pump_set = wd.gen_pump_set
    # flex_set = wd.flex_set
    node_list = wd.node_list
    gen_list = list(ctx.gen_ids)
    gen_pump_list = list(ctx.idx_generatorsWithPumping)
    storage_gen_list = list(lp._idx_generatorsWithStorage)
    flex_list = list(lp._idx_consumersWithFlexLoad)
    branch_ac_list = list(ctx.branch_ac_ids)
    branch_dc_list = list(ctx.branch_dc_ids)
    # Spill is reported against the full generator list, so storage generators need
    # their position in it. Mapped once instead of scanned inside the hour loop.
    gen_position = {i: gi for gi, i in enumerate(gen_list)}
    grid = grid
    dt = float(lp.timeDelta)

    for hi, ts in enumerate(day_timesteps[:commit_hours]):
        Pgen = [float(pyo.value(m.varGeneration[i, hi]) or 0.0) for i in gen_list]
        Ppump = [float(pyo.value(m.varPump[i, hi]) or 0.0) for i in gen_pump_list]
        Pflexload = [float(pyo.value(m.varFlexLoad[j, hi]) or 0.0) for j in flex_list]
        Pb = [float(pyo.value(m.varAcBranchFlow[b, hi]) or 0.0) for b in branch_ac_list]
        Pdc = [float(pyo.value(m.varDcBranchFlow[b, hi]) or 0.0) for b in branch_dc_list]
        theta = [float(pyo.value(m.varVoltageAngle[n, hi]) or 0.0) * const.baseAngle for n in node_list]

        # Load shedding aggregated to nodes
        Ploadshed = aggregate_to_nodes(grid, ctx.load_ids, lambda j: float(pyo.value(m.varLoadShed[j, hi]) or 0.0))
        Pdumpload = aggregate_to_nodes(grid, ctx.load_ids, lambda j: float(pyo.value(m.varDumpLoad[j, hi]) or 0.0))

        # Storage levels, at the start and end of this hour
        stor_levels = [float(pyo.value(m.varStorage[i, hi]) or 0.0) for i in storage_gen_list]
        if hi == 0:
            stor_prev = [wd.storage_ini[i] for i in storage_gen_list]
        else:
            stor_prev = [float(pyo.value(m.varStorage[i, hi - 1]) or 0.0) for i in storage_gen_list]
        spill = [float(pyo.value(m.varSpill[i, hi]) or 0.0) for i in storage_gen_list]
        overflow = [float(pyo.value(m.varStorageOverflow[i, hi]) or 0.0) for i in storage_gen_list]

        # Per-hour objective (cost rate for this timestep only), defined as the
        # single-period extractor does: the value of the energy taken from storage
        # this hour stands in for the terminal value of the energy left in it.
        obj_h = (
            sum(Pgen[gi] * wd.gen_cost_h.get((hi, gen_list[gi]), 0.0) for gi in range(len(gen_list)))
            + sum(Ppump[pi] * wd.pump_cost_h.get((hi, gen_pump_list[pi]), 0.0) for pi in range(len(gen_pump_list)))
            - sum(Pflexload[fi] * wd.flexload_cost_h.get((hi, flex_list[fi]), 0.0) for fi in range(len(flex_list)))
            + sum(float(pyo.value(m.varLoadShed[j, hi]) or 0.0) * const.loadshedcost for j in ctx.load_ids)
            + sum(float(pyo.value(m.varDumpLoad[j, hi]) or 0.0) * const.loadshedcost for j in ctx.load_ids)
            + sum(overflow) * const.loadshedcost
            + sum(spill) * const.spillcost
            + sum(
                wd.terminal_storval.get(i, 0.0) * (stor_prev[k] + wd.inflow_h.get((hi, i), 0.0) * dt - stor_levels[k])
                for k, i in enumerate(storage_gen_list)
            )
            / dt
        )

        # Spilled energy per generator. As in the single-period path, a generator
        # with zero storage capacity spills whatever inflow it does not use (e.g.
        # curtailed wind), and one with infinite capacity spills nothing.
        energy_spilled = np.zeros(len(gen_list))
        for gi, i in enumerate(gen_list):
            if i not in wd.gen_storage_set and grid.generator.loc[i, "storage_cap"] == 0:
                energy_spilled[gi] = (wd.inflow_h.get((hi, i), 0.0) - Pgen[gi]) * dt
        for k, i in enumerate(storage_gen_list):
            energy_spilled[gen_position[i]] = (spill[k] + overflow[k]) * dt

        # Branch capacity sensitivity (dual of capacity constraint).
        # NOTE: cAcFlowUb/cAcFlowLb/cDcFlowUb are declared as
        # Constraint(s_branch_ac, s_h), so the index order is (branch, hour).
        # Membership is tested explicitly rather than caught as KeyError: a
        # transposed (hour, branch) lookup is often a *valid* key for another
        # branch, so an exception-based guard cannot detect that mistake.
        senseB = []
        for j in lp._idx_branchesWithConstraints:
            d = 0.0
            if duals:
                if (j, hi) in m.cAcFlowUb:
                    d = float(duals.get(m.cAcFlowUb[j, hi], 0.0) or 0.0)
                elif (j, hi) in m.cAcFlowLb:
                    d = float(duals.get(m.cAcFlowLb[j, hi], 0.0) or 0.0)
            senseB.append(-abs(d / const.baseMVA))

        senseDcB = []
        for j in branch_dc_list:
            d = 0.0
            if duals:
                if (j, hi) in m.cDcFlowUb:
                    d = float(duals.get(m.cDcFlowUb[j, hi], 0.0) or 0.0)
                elif (j, hi) in m.cDcFlowLb:
                    d = float(duals.get(m.cDcFlowLb[j, hi], 0.0) or 0.0)
            senseDcB.append(-abs(d / const.baseMVA))

        # Nodal prices (dual of power balance)
        senseN = []
        for n in node_list:
            d = 0.0
            if duals:
                try:
                    c = m.cPowerbalance[n, hi]
                    d = float(duals.get(c, 0.0) or 0.0)
                except (KeyError, AttributeError):
                    d = 0.0
            senseN.append(abs(d / const.baseMVA))

        storageprice = [wd.terminal_storval.get(i, 0.0) for i in storage_gen_list]
        flexload_storagelevel = [float(pyo.value(m.varFlexStorage[j, hi]) or 0.0) for j in flex_list]
        flexload_marginalprice = [wd.flexload_cost_h.get((hi, j), 0.0) for j in flex_list]

        TimestepResult(
            timestep=grid.timerange[0] + ts,
            objective_function=obj_h,
            generator_power=Pgen,
            generator_pumped=Ppump,
            branch_power=Pb,
            dcbranch_power=Pdc,
            node_angle=theta,
            sensitivity_branch_capacity=senseB,
            sensitivity_dcbranch_capacity=senseDcB,
            sensitivity_node_power=senseN,
            storage=stor_levels,
            inflow_spilled=energy_spilled.tolist(),
            loadshed_power=Ploadshed.tolist(),
            dumpload_power=Pdumpload.tolist(),
            marginalprice=storageprice,
            flexload_power=Pflexload,
            flexload_storage=flexload_storagelevel,
            flexload_storagevalue=flexload_marginalprice,
            # Losses are not modelled in this path; lossmethod != 0 is rejected in __init__.
            branch_ac_losses=[0.0] * len(branch_ac_list),
            branch_dc_losses=[0.0] * len(branch_dc_list),
            # Fault scenarios run the single-period path only -- see LpFaultProblem.__init__.
            fault_start=None,
        ).store(results)

        # Update gen_prev for context at day boundary
        for gi, i in enumerate(gen_list):
            lp._gen_prev[i] = Pgen[gi]

    # Update lp._storage to end-of-commit levels
    for i in storage_gen_list:
        lp._storage[i] = float(pyo.value(m.varStorage[i, commit_hours - 1]) or 0.0)

    # Same for the flexible-load reservoirs, so the next window is priced from
    # where this one left off -- the single-period path does this in
    # _storeResultsAndUpdateStorage.
    for j in flex_list:
        lp._storage_flexload[j] = float(pyo.value(m.varFlexStorage[j, commit_hours - 1]) or 0.0)
