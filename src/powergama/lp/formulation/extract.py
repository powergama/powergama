"""Read the committed hours of a solved window into TimestepResults.

Also advances the simulation state that makes the problem sequential: storage
levels, the flexible-load reservoir and the previous dispatch the ramp bounds
use, to where the last committed hour leaves them. The two jobs sit together
because the storage update is a mass balance over the same extracted quantities.
"""

import numpy as np
from pyomo.environ import value as pyo_value

from ... import constants as const
from ...timestep_result import TimestepResult, aggregate_to_nodes
from . import data as window_data


def store_window(lp, timesteps, commit, objective_values, results):
    """Store the first *commit* hours of the window *timesteps*, and update the state.

    *objective_values* are the values of the objective's terms in each committed hour
    (the sum of objective_terms for that hour), as evaluated by the solve loop.
    """
    m = lp.model
    gen = lp._grid.generator
    dt = lp.timeDelta
    H = len(timesteps)
    stores = lp._idx_generatorsWithStorage
    reports_storage_value = "storage_value" in lp._objective_term_names
    pmax_inflow = gen["pmax"].to_numpy(float) * gen["inflow_fac"].to_numpy(float)
    storage_cap = gen["storage_cap"]

    for h, timestep in enumerate(timesteps[:commit]):
        # 1. Update generator storage. Its level is an LP variable, so the next hour
        #    starts from the solved level, and spilled energy is the solved spill.
        #    Generators without storage report the inflow they did not use,
        #    e.g. curtailed wind, as spilled.
        Pgen = [m.varGeneration[i, h].value for i in m.s_gen]
        genInflow = pmax_inflow * window_data._profile_values(lp._grid.profiles, timestep, gen["inflow_ref"])
        energyIn = genInflow * dt
        energyOut = np.asarray(Pgen, dtype=float) * dt
        storage_prev = lp._storage.copy()
        # Without storage (zero capacity) everything not generated is spilled; with
        # infinite capacity, nothing is.
        energyStorable = lp._storage + energyIn - energyOut
        lp._storage = np.clip(energyStorable, 0.0, storage_cap)
        lp._energyspilled = energyStorable - lp._storage
        for i in stores:
            lp._storage[i] = m.varStorage[i, h].value
            lp._energyspilled[i] = m.varSpill[i, h].value * dt

        # 2. Update flexible load storage, an LP variable like generator storage
        for i in lp._idx_consumersWithFlexLoad:
            lp._storage_flexload[i] = m.varFlexStorage[i, h].value

        # 1c. Record this hour's dispatch for ramp constraints in the next window.
        lp._gen_prev[:] = [v if v is not None else 0.0 for v in Pgen]

        # 3. Collect variable values from optimisation result
        F = objective_values[h]
        if reports_storage_value:
            # Report the value of the energy taken from storage in place of the value
            # of the energy left in it, so that the objective is an operating cost
            # comparable across timesteps rather than dominated by a large negative
            # constant. Both are rates, EUR/h: see build.objective_terms. Only the
            # window's last hour has the terminal term; the others leave it out.
            last = h == H - 1
            F += (
                sum(
                    pyo_value(m.p_storage_value[i])
                    * (
                        storage_prev[i]
                        + pyo_value(m.p_storage_inflow[i, h]) * dt
                        - (0.0 if last else m.varStorage[i, h].value)
                    )
                    for i in stores
                )
                / dt
            )
        Ppump = [m.varPump[i, h].value for i in m.s_gen_pump]
        Pflexload = [m.varFlexLoad[i, h].value for i in m.s_load_flex]
        Pb = [m.varAcBranchFlow[i, h].value for i in m.s_branch_ac]
        Pdc = [m.varDcBranchFlow[i, h].value for i in m.s_branch_dc]
        theta = [m.varVoltageAngle[i, h].value * const.baseAngle for i in m.s_node]
        # load shedding is aggregated to nodes (due to old code)
        Ploadshed = aggregate_to_nodes(lp._grid, m.s_load, lambda j: m.varLoadShed[j, h].value)
        Pdumpload = aggregate_to_nodes(lp._grid, m.s_load, lambda j: m.varDumpLoad[j, h].value)

        # 4 Collect dual values
        # 4a. branch capacity sensitivity (whether pos or neg flow)
        senseB = []
        for j in lp._idx_branchesWithConstraints:
            c = m.cMaxFlowAc[j, h]
            senseB.append(-abs(lp.dual[c] / const.baseMVA))
        senseDcB = []
        for j in m.s_branch_dc:
            c = m.cMaxFlowDc[j, h]
            senseDcB.append(-abs(lp.dual[c] / const.baseMVA))

        # 4b. node demand sensitivity (energy balance)
        # TODO: Without abs(...) the value jumps between pos and neg. Why?
        senseN = []
        for j in m.s_node:
            try:
                c = m.cPowerbalance[j, h]
                senseN.append(abs(lp.dual[c] / const.baseMVA))
            except KeyError:
                # Constraint was skipped (trivially 0==0, e.g. isolated hub/virtual node
                # with no generators, loads, or branches). Nodal price = 0.
                senseN.append(0.0)

        energyspilled = lp._energyspilled
        storagelevel = lp._storage[stores]
        storageprice = [m.p_storage_value[i].value for i in stores]
        flexload_storagelevel = lp._storage_flexload[lp._idx_consumersWithFlexLoad]
        flexload_marginalprice = [m.p_loadflex_cost[i, h].value for i in lp._idx_consumersWithFlexLoad]

        # Extract power losses
        if lp._lossmethod == 0:
            acPowerLoss = [0] * len(m.s_branch_ac)
            dcPowerLoss = [0] * len(m.s_branch_dc)
        elif lp._lossmethod == 1:
            acPowerLoss = [m.varLossAc12[b, h].value + m.varLossAc21[b, h].value for b in m.s_branch_ac]
            dcPowerLoss = [m.varLossDc12[b, h].value + m.varLossDc21[b, h].value for b in m.s_branch_dc]
        elif lp._lossmethod == 2:
            acPowerLoss = [m.p_branch_ac_power_loss[b, h].value for b in m.s_branch_ac]
            dcPowerLoss = [m.p_branch_dc_power_loss[b, h].value for b in m.s_branch_dc]
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
