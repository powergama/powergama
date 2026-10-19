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
from pyomo.environ import value as pyo_value

from ... import constants as const
from ...timestep_result import TimestepResult, aggregate_to_nodes
from . import data as sp_data


def store_results_and_update_storage(lp, timestep, results):
    """Store timestep results in local arrays, and update storage"""

    # 1. Update generator storage. Its level is an LP variable, so the next
    #    timestep starts from the solved level, and spilled energy is the solved
    #    spill (plus overflow). Generators without storage report the inflow they
    #    did not use, e.g. curtailed wind, as spilled.
    gen = lp._grid.generator
    Pgen = [lp.model.varGeneration[i].value for i in lp.model.s_gen]
    genInflow = (
        gen["pmax"].to_numpy(float)
        * gen["inflow_fac"].to_numpy(float)
        * sp_data._profile_values(lp._grid.profiles, timestep, gen["inflow_ref"])
    )
    energyIn = genInflow * lp.timeDelta
    energyOut = np.asarray(Pgen, dtype=float) * lp.timeDelta
    storage_prev = lp._storage.copy()
    # Without storage (zero capacity) everything not generated is spilled; with
    # infinite capacity, nothing is.
    energyStorable = lp._storage + energyIn - energyOut
    lp._storage = np.clip(energyStorable, 0.0, lp._grid.generator["storage_cap"])
    lp._energyspilled = energyStorable - lp._storage
    for i in lp._idx_generatorsWithStorage:
        lp._storage[i] = lp.model.varStorage[i].value
        lp._energyspilled[i] = (lp.model.varSpill[i].value + lp.model.varStorageOverflow[i].value) * lp.timeDelta

    # 2. Update flexible load storage
    for i in lp._idx_consumersWithFlexLoad:
        energyIn_flexload = lp.model.varFlexLoad[i].value * lp.timeDelta
        energyOut_flexload = lp._grid.consumer["flex_fraction"][i] * lp._grid.consumer["demand_avg"][i] * lp.timeDelta
        lp._storage_flexload[i] += energyIn_flexload - energyOut_flexload

    # 1c. Record this timestep's dispatch for ramp constraints in next timestep.
    lp._gen_prev[:] = [v if v is not None else 0.0 for v in Pgen]

    # 3. Collect variable values from optimisation result
    F = lp._objective_value
    if "storage_value" in lp._objective_term_names:
        # Report the value of the energy taken from storage in place of the value
        # of the energy left in it, so that the objective is an operating cost
        # comparable across timesteps rather than dominated by a large negative
        # constant. Both are rates, EUR/h: see single_period.build.objective_terms.
        F += (
            sum(
                pyo_value(lp.model.p_storage_value[i])
                * (storage_prev[i] + pyo_value(lp.model.p_storage_inflow[i]) * lp.timeDelta)
                for i in lp._idx_generatorsWithStorage
            )
            / lp.timeDelta
        )
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
    storageprice = [lp.model.p_storage_value[i].value for i in lp._idx_generatorsWithStorage]
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
