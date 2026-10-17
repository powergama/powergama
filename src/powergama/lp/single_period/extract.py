"""Read one solved single-period timestep into a TimestepResult.

Also advances the simulation state that makes the problem sequential: storage
levels, the flexible-load reservoir and the previous dispatch the ramp bounds
use. The two jobs sit together because the storage update is a mass balance over
the same extracted quantities.

The multi-period counterpart reads the same 21-field contract out of a different
object -- an (entity, hour)-indexed model and a dual dict, rather than scalar
vars and a Suffix -- which is why TimestepResult shares the contract and not the
extraction code.
"""

import numpy as np

from ... import constants as const
from ...timestep_result import TimestepResult, aggregate_to_nodes


def store_results_and_update_storage(lp, timestep, results):
    """Store timestep results in local arrays, and update storage"""

    # 1. Update generator storage:
    inflow_profile_refs = lp._grid.generator["inflow_ref"]
    inflow_factor = lp._grid.generator["inflow_fac"]
    capacity = lp._grid.generator["pmax"]
    pumpedIn = np.zeros(len(capacity))
    energyIn = np.zeros(len(capacity))
    energyOut = np.zeros(len(capacity))
    for i in lp.model.s_gen:
        genInflow = capacity[i] * inflow_factor[i] * lp._grid.profiles[inflow_profile_refs[i]][timestep]
        energyIn[i] = genInflow * lp.timeDelta
        energyOut[i] = lp.model.varGeneration[i].value * lp.timeDelta

    for i in lp._idx_generatorsWithPumping:
        Ppump = lp.model.varPump[i].value
        pumpedIn[i] = Ppump * lp._grid.generator["pump_efficiency"][i] * lp.timeDelta
    energyStorable = lp._storage + energyIn + pumpedIn - energyOut
    storagecapacity = lp._grid.generator["storage_cap"]
    # Keep storage state inside physical bounds for the next timestep.
    lp._storage = np.clip(energyStorable, 0.0, storagecapacity)
    lp._energyspilled = energyStorable - lp._storage

    # 2. Update flexible load storage
    for i in lp._idx_consumersWithFlexLoad:
        energyIn_flexload = lp.model.varFlexLoad[i].value * lp.timeDelta
        energyOut_flexload = lp._grid.consumer["flex_fraction"][i] * lp._grid.consumer["demand_avg"][i] * lp.timeDelta
        lp._storage_flexload[i] += energyIn_flexload - energyOut_flexload

    # 1c. Record this timestep's dispatch for ramp constraints in next timestep.
    for i in lp.model.s_gen:
        v = lp.model.varGeneration[i].value
        lp._gen_prev[i] = v if v is not None else 0.0

    # 3. Collect variable values from optimisation result
    F = lp.model.OBJ()
    Pgen = [lp.model.varGeneration[i].value for i in lp.model.s_gen]
    Ppump = [lp.model.varPump[i].value for i in lp.model.s_gen_pump]
    Pflexload = [lp.model.varFlexLoad[i].value for i in lp.model.s_load_flex]
    Pb = [lp.model.varAcBranchFlow[i].value for i in lp.model.s_branch_ac]
    Pdc = [lp.model.varDcBranchFlow[i].value for i in lp.model.s_branch_dc]
    theta = [lp.model.varVoltageAngle[i].value * const.baseAngle for i in lp.model.s_node]
    # load shedding is aggregated to nodes (due to old code)
    Ploadshed = aggregate_to_nodes(lp._grid, lp.model.s_load, lambda j: lp.model.varLoadShed[j].value)
    Pdumpload = aggregate_to_nodes(lp._grid, lp.model.s_load, lambda j: lp.model.varDumpLoad[j].value)

    # 4 Collect dual values
    # 4a. branch capacity sensitivity (whether pos or neg flow)
    senseB = []
    for j in lp._idx_branchesWithConstraints:
        # for j in lp.concretemodel.BRANCH_AC:
        c = lp.model.cMaxFlowAc[j]
        senseB.append(-abs(lp.dual[c] / const.baseMVA))
    senseDcB = []
    for j in lp.model.s_branch_dc:
        c = lp.model.cMaxFlowDc[j]
        senseDcB.append(-abs(lp.dual[c] / const.baseMVA))

    # 4b. node demand sensitivity (energy balance)
    # TODO: Without abs(...) the value jumps between pos and neg. Why?
    senseN = []
    for j in lp.model.s_node:
        try:
            c = lp.model.cPowerbalance[j]
            senseN.append(abs(lp.dual[c] / const.baseMVA))
        except KeyError:
            # Constraint was skipped (trivially 0==0, e.g. isolated hub/virtual node
            # with no generators, loads, or branches). Nodal price = 0.
            senseN.append(0.0)

    # consider spilled energy only for generators with storage<infinity
    # energyspilled = zeros(energyStorable.shape)
    # indx = lp._grid.getIdxGeneratorsWithNonzeroInflow()
    # energyspilled[indx] = energyStorable[indx]-lp._storage[indx]
    energyspilled = lp._energyspilled
    storagelevel = lp._storage[lp._idx_generatorsWithStorage]
    storageprice = [lp.model.p_gen_cost[i].value for i in lp._idx_generatorsWithStorage]
    flexload_storagelevel = lp._storage_flexload[lp._idx_consumersWithFlexLoad]
    flexload_marginalprice = [lp.model.p_loadflex_cost[i].value for i in lp._idx_consumersWithFlexLoad]

    # TODO: Only keep track of inflow spilled for generators with
    # nonzero inflow

    # Extract power losses
    if lp._lossmethod == 0:
        acPowerLoss = [0] * len(lp.model.s_branch_ac)
        dcPowerLoss = [0] * len(lp.model.s_branch_dc)
    elif lp._lossmethod == 1:
        acPowerLoss = [lp.model.varLossAc12[b].value + lp.model.varLossAc21[b].value for b in lp.model.s_branch_ac]
        dcPowerLoss = [lp.model.varLossDc12[b].value + lp.model.varLossDc21[b].value for b in lp.model.s_branch_dc]
    elif lp._lossmethod == 2:
        acPowerLoss = list(lp.model.p_branch_ac_power_loss.extract_values().values())
        dcPowerLoss = list(lp.model.p_branch_dc_power_loss.extract_values().values())
    else:
        raise Exception("Lossmethod must be 0,1 or 2")

    TimestepResult(
        timestep=lp._grid.timerange[0] + timestep,
        objective_function=F,
        generator_power=Pgen,
        generator_pumped=Ppump,
        branch_power=Pb,
        dcbranch_power=Pdc,
        node_angle=theta,
        sensitivity_branch_capacity=senseB,
        sensitivity_dcbranch_capacity=senseDcB,
        sensitivity_node_power=senseN,
        storage=storagelevel.tolist(),
        inflow_spilled=energyspilled.tolist(),
        loadshed_power=Ploadshed.tolist(),
        dumpload_power=Pdumpload.tolist(),
        marginalprice=storageprice,
        flexload_power=Pflexload,
        flexload_storage=flexload_storagelevel.tolist(),
        flexload_storagevalue=flexload_marginalprice,
        branch_ac_losses=acPowerLoss,
        branch_dc_losses=dcPowerLoss,
        fault_start=lp._get_fault_start(timestep),
    ).store(results)

    return
