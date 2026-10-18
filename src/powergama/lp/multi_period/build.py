"""The multi-period formulation: a joint LP over a window of several periods.

Every coefficient is a constant computed by :mod:`.data` before the model is
built, so there are no mutable Params here at all -- the opposite of the
single-period model, which is built once and mutated each timestep.

A window of length 1 ought to be the single-period problem. It is not, today:
storage draw is priced twice here (per MWh generated *and* through a terminal
storage value that the continuity equality links), ramp limits bind differently
for energy-limited generators, and flexible load is hard-bounded here but
unbounded there. See userguide/model_formulations.md.
"""

import numpy as np
import pyomo.environ as pyo

from ... import constants as const


def build_window_model(lp, wd):
    """Build the joint LP over one window from its prepared data."""
    ctx, grid = lp._ctx, lp._grid
    # ── build Pyomo model ─────────────────────────────────────────
    m = pyo.ConcreteModel()
    m.s_gen = pyo.Set(ordered=True, initialize=list(ctx.gen_ids))
    m.s_gen_pump = pyo.Set(ordered=True, initialize=list(ctx.idx_generatorsWithPumping))
    m.s_gen_storage = pyo.Set(ordered=True, initialize=list(lp._idx_generatorsWithStorage))
    m.s_load = pyo.Set(ordered=True, initialize=list(ctx.load_ids))
    m.s_load_flex = pyo.Set(ordered=True, initialize=list(lp._idx_consumersWithFlexLoad))
    m.s_branch_ac = pyo.Set(ordered=True, initialize=list(ctx.branch_ac_ids))
    m.s_branch_dc = pyo.Set(ordered=True, initialize=list(ctx.branch_dc_ids))
    m.s_node = pyo.Set(ordered=True, initialize=wd.node_list)
    m.s_h = pyo.RangeSet(0, wd.H - 1)

    # ── variables ─────────────────────────────────────────────────
    m.varGeneration = pyo.Var(m.s_gen, m.s_h, within=pyo.NonNegativeReals)
    m.varPump = pyo.Var(m.s_gen_pump, m.s_h, within=pyo.NonNegativeReals)
    m.varStorage = pyo.Var(m.s_gen_storage, m.s_h, within=pyo.NonNegativeReals)
    m.varSpill = pyo.Var(m.s_gen_storage, m.s_h, within=pyo.NonNegativeReals)
    m.varLoadShed = pyo.Var(m.s_load, m.s_h, within=pyo.NonNegativeReals)
    m.varDumpLoad = pyo.Var(m.s_load, m.s_h, within=pyo.NonNegativeReals)
    m.varFlexLoad = pyo.Var(m.s_load_flex, m.s_h, within=pyo.NonNegativeReals)
    # Reservoir level, in MWh. Reals, not NonNegativeReals: the floor is a
    # constraint rather than a domain because it has to be skipped for a consumer
    # whose per-hour charge limit cannot keep up with its drain -- see
    # cFlexStorFloor. The single-period path has no floor at all; it lets the level go
    # negative and prices that at flexload_outside_cost.
    m.varFlexStorage = pyo.Var(m.s_load_flex, m.s_h, within=pyo.Reals)
    m.varAcBranchFlow = pyo.Var(m.s_branch_ac, m.s_h, within=pyo.Reals)
    m.varDcBranchFlow = pyo.Var(m.s_branch_dc, m.s_h, within=pyo.Reals)
    m.varVoltageAngle = pyo.Var(m.s_node, m.s_h, within=pyo.Reals, initialize=0.0)

    # ── objective: sum over all hours + end-of-day terminal storage value ─
    def _obj_rule(m):
        cost = 0
        for h in range(wd.H):
            cost += sum(m.varGeneration[i, h] * wd.gen_cost_h.get((h, i), 0.0) for i in m.s_gen)
            cost -= sum(m.varPump[i, h] * wd.pump_cost_h.get((h, i), 0.0) for i in m.s_gen_pump)
            cost -= sum(m.varFlexLoad[j, h] * wd.flexload_cost_h.get((h, j), 0.0) for j in m.s_load_flex)
            cost += sum(m.varLoadShed[j, h] * const.loadshedcost for j in m.s_load)
            cost += sum(m.varDumpLoad[j, h] * const.loadshedcost for j in m.s_load)
        # Terminal storage value: reward conservation of stored energy at end of day.
        # Without this, the solver has no incentive to leave storage full at day-end,
        # which would drain it to zero every day and destroy inter-day continuity.
        # Subtracting here (minimisation) means higher end storage → lower effective cost.
        cost -= sum(wd.terminal_storval.get(i, 0.0) * m.varStorage[i, wd.H - 1] for i in m.s_gen_storage)
        return cost

    m.OBJ = pyo.Objective(rule=_obj_rule, sense=pyo.minimize)

    # ── generation bounds (non-storage) ───────────────────────────
    def _gen_ub_ns_rule(m, i, h):
        if i in wd.gen_storage_set:
            return pyo.Constraint.Skip
        return m.varGeneration[i, h] <= wd.inflow_h.get((h, i), 0.0)

    m.cGenUbNs = pyo.Constraint(m.s_gen, m.s_h, rule=_gen_ub_ns_rule)

    def _gen_lb_ns_rule(m, i, h):
        if i in wd.gen_storage_set:
            return pyo.Constraint.Skip
        infl = wd.inflow_h.get((h, i), 0.0)
        pmin = wd.pmin_h.get((h, i), 0.0)
        return m.varGeneration[i, h] >= max(0.0, min(infl, pmin))

    m.cGenLbNs = pyo.Constraint(m.s_gen, m.s_h, rule=_gen_lb_ns_rule)

    # ── generation bounds (storage generators) ────────────────────
    # gen[i,h] <= inflow[h,i] + storage_prev / wd.dt  (availability)
    # gen[i,h] <= capacity[h,i]                    (hardware cap)
    def _gen_ub_stor_avail_rule(m, i, h):
        prev_s = wd.storage_ini[i] if h == 0 else m.varStorage[i, h - 1]
        return m.varGeneration[i, h] <= wd.inflow_h.get((h, i), 0.0) + prev_s / wd.dt

    m.cGenUbStorAvail = pyo.Constraint(m.s_gen_storage, m.s_h, rule=_gen_ub_stor_avail_rule)

    def _gen_ub_stor_cap_rule(m, i, h):
        return m.varGeneration[i, h] <= wd.capacity_h.get((h, i), 0.0)

    m.cGenUbStorCap = pyo.Constraint(m.s_gen_storage, m.s_h, rule=_gen_ub_stor_cap_rule)

    def _gen_lb_stor_rule(m, i, h):
        infl = wd.inflow_h.get((h, i), 0.0)
        pmin = wd.pmin_h.get((h, i), 0.0)
        return m.varGeneration[i, h] >= max(0.0, min(infl, pmin))

    m.cGenLbStor = pyo.Constraint(m.s_gen_storage, m.s_h, rule=_gen_lb_stor_rule)

    # ── storage capacity upper bound ───────────────────────────────
    def _stor_cap_rule(m, i, h):
        return m.varStorage[i, h] <= wd.storage_cap[i]

    m.cStorCap = pyo.Constraint(m.s_gen_storage, m.s_h, rule=_stor_cap_rule)

    # ── KEY: storage continuity (equality with explicit spill) ─────
    # storage[h] = storage_prev + (inflow - gen + pump*eff - spill) * wd.dt
    def _stor_cont_rule(m, i, h):
        prev_s = wd.storage_ini[i] if h == 0 else m.varStorage[i, h - 1]
        infl = wd.inflow_h.get((h, i), 0.0)
        eff = wd.pump_eff_g.get(i, 1.0) if i in wd.gen_pump_set else 1.0
        pump_term = m.varPump[i, h] * eff if i in wd.gen_pump_set else 0
        return m.varStorage[i, h] == (prev_s + (infl - m.varGeneration[i, h] + pump_term - m.varSpill[i, h]) * wd.dt)

    m.cStorCont = pyo.Constraint(m.s_gen_storage, m.s_h, rule=_stor_cont_rule)

    # Limit spill power by per-generator fraction of installed generator capacity.
    def _spill_cap_rule(m, i, h):
        return m.varSpill[i, h] <= wd.spill_cap_mw.get(i, 0.0)

    m.cSpillCap = pyo.Constraint(m.s_gen_storage, m.s_h, rule=_spill_cap_rule)

    # ── pump bounds ────────────────────────────────────────────────
    # pump[i,h]*eff*wd.dt + storage_prev <= wd.storage_cap  (no spurious pump credit)
    def _pump_room_rule(m, i, h):
        eff = wd.pump_eff_g.get(i, 1.0)
        if eff <= 0:
            return m.varPump[i, h] == 0
        prev_s = wd.storage_ini[i] if h == 0 else m.varStorage[i, h - 1]
        return m.varPump[i, h] * eff * wd.dt <= wd.storage_cap[i] - prev_s

    m.cPumpRoom = pyo.Constraint(m.s_gen_pump, m.s_h, rule=_pump_room_rule)

    def _pump_cap_rule(m, i, h):
        return m.varPump[i, h] <= wd.pump_cap_raw.get(i, 0.0)

    m.cPumpCap = pyo.Constraint(m.s_gen_pump, m.s_h, rule=_pump_cap_rule)

    # ── ramp rates within the day ──────────────────────────────────
    has_ramp = (lp._ramp_up_pu is not None) or (lp._ramp_down_pu is not None)
    if has_ramp:

        def _ramp_up_rule(m, i, h):
            if lp._disable_nuclear_ramp_profile_conflict[int(i)]:
                return pyo.Constraint.Skip
            ramp_pu = lp._ramp_up_pu[i] if lp._ramp_up_pu is not None else np.nan
            if np.isnan(ramp_pu):
                return pyo.Constraint.Skip
            if h == 0:
                prev = lp._gen_prev[i]
                if np.isnan(prev):
                    return pyo.Constraint.Skip
                return m.varGeneration[i, 0] <= prev + float(ramp_pu) * float(lp._ramp_cap_mw[i])
            return m.varGeneration[i, h] <= m.varGeneration[i, h - 1] + float(ramp_pu) * float(lp._ramp_cap_mw[i])

        m.cRampUp = pyo.Constraint(m.s_gen, m.s_h, rule=_ramp_up_rule)

        def _ramp_dn_rule(m, i, h):
            if lp._disable_nuclear_ramp_profile_conflict[int(i)]:
                return pyo.Constraint.Skip
            ramp_pu = lp._ramp_down_pu[i] if lp._ramp_down_pu is not None else np.nan
            if np.isnan(ramp_pu):
                return pyo.Constraint.Skip
            if h == 0:
                prev = lp._gen_prev[i]
                if np.isnan(prev):
                    return pyo.Constraint.Skip
                return m.varGeneration[i, 0] >= prev - float(ramp_pu) * float(lp._ramp_cap_mw[i])
            return m.varGeneration[i, h] >= m.varGeneration[i, h - 1] - float(ramp_pu) * float(lp._ramp_cap_mw[i])

        m.cRampDn = pyo.Constraint(m.s_gen, m.s_h, rule=_ramp_dn_rule)

    # ── flex load ─────────────────────────────────────────────────
    if list(m.s_load_flex):

        def _flex_ub_rule(m, j, h):
            avg = float(grid.consumer.loc[j, "demand_avg"])
            frac = float(grid.consumer.loc[j, "flex_fraction"])
            on_off = float(grid.consumer.loc[j, "flex_on_off"])
            return m.varFlexLoad[j, h] <= avg * frac / on_off

        m.cFlexUb = pyo.Constraint(m.s_load_flex, m.s_h, rule=_flex_ub_rule)

        # Reservoir continuity, mirroring cStorCont for generator storage:
        #   level[h] = level[h-1] + (charge - constant drain) * wd.dt
        def _flex_stor_cont_rule(m, j, h):
            prev = wd.flex_storage_ini[j] if h == 0 else m.varFlexStorage[j, h - 1]
            return m.varFlexStorage[j, h] == prev + (m.varFlexLoad[j, h] - wd.flex_drain_mw[j]) * wd.dt

        m.cFlexStorCont = pyo.Constraint(m.s_load_flex, m.s_h, rule=_flex_stor_cont_rule)

        # The window price is a constant coefficient, so the solver would otherwise
        # run the level to one extreme for the whole window: up to the per-hour
        # charge limit while the price is positive, or down with no charging at all
        # while generation costs more than the price is worth. The single-period path needs
        # no bounds because it re-prices from the level every timestep, and the
        # curve turns at 0 and at capacity long before the level gets there.
        def _flex_stor_cap_rule(m, j, h):
            return m.varFlexStorage[j, h] <= wd.flex_storage_cap[j]

        m.cFlexStorCap = pyo.Constraint(m.s_load_flex, m.s_h, rule=_flex_stor_cap_rule)

        def _flex_stor_floor_rule(m, j, h):
            if not wd.flex_has_floor[j]:
                return pyo.Constraint.Skip
            return m.varFlexStorage[j, h] >= 0.0

        m.cFlexStorFloor = pyo.Constraint(m.s_load_flex, m.s_h, rule=_flex_stor_floor_rule)

    # ── power balance (one per node per hour) ─────────────────────
    Bbus = lp._Bbus

    def _power_balance_rule(m, n, h):
        ni = wd.node_position[n]
        lhs = 0
        for g in lp._generators_at_node.get(n, []):
            lhs += m.varGeneration[g, h]
            if g in wd.gen_pump_set:
                lhs -= m.varPump[g, h]
        for j in lp._loads_at_node.get(n, []):
            lhs -= wd.demand_h.get((h, j), 0.0)
            lhs += m.varLoadShed[j, h]
            lhs -= m.varDumpLoad[j, h]
            if j in wd.flex_set:
                lhs -= m.varFlexLoad[j, h]
        for b in lp._dcbranch_to_node.get(n, []):
            lhs += m.varDcBranchFlow[b, h]
        for b in lp._dcbranch_from_node.get(n, []):
            lhs -= m.varDcBranchFlow[b, h]
        lhs = lhs / const.baseMVA
        rhs = 0
        row = Bbus[ni]
        for k in range(len(row.indices)):
            ni2 = int(row.indices[k])
            n2 = wd.node_list[ni2]
            rhs -= float(row.data[k]) * m.varVoltageAngle[n2, h] * const.baseAngle
        expr = lhs == rhs
        if isinstance(expr, bool) and expr is True:
            return pyo.Constraint.Skip
        return expr

    m.cPowerbalance = pyo.Constraint(m.s_node, m.s_h, rule=_power_balance_rule)

    # ── DC power flow (flow-angle relationship) ───────────────────
    DA = lp._DA

    def _flow_angle_rule(m, b, h):
        idx_b = wd.branch_ac_position[b]
        lhs = m.varAcBranchFlow[b, h] / const.baseMVA
        rhs = 0
        row = DA[idx_b]
        for k in range(len(row.indices)):
            ni2 = int(row.indices[k])
            n2 = wd.node_list[ni2]
            rhs += float(row.data[k]) * m.varVoltageAngle[n2, h] * const.baseAngle
        return lhs == rhs

    m.cFlowAngle = pyo.Constraint(m.s_branch_ac, m.s_h, rule=_flow_angle_rule)

    # ── reference angle = 0 per synchronous area ──────────────────
    for refnode in lp.refnodes:
        for h in range(wd.H):
            m.varVoltageAngle[refnode, h].fix(0.0)

    # ── branch capacity limits ─────────────────────────────────────
    def _ac_lb_rule(m, b, h):
        lb = wd.ac_lb.get((h, b))
        return m.varAcBranchFlow[b, h] >= lb if lb is not None else pyo.Constraint.Skip

    m.cAcFlowLb = pyo.Constraint(m.s_branch_ac, m.s_h, rule=_ac_lb_rule)

    def _ac_ub_rule(m, b, h):
        ub = wd.ac_ub.get((h, b))
        return m.varAcBranchFlow[b, h] <= ub if ub is not None else pyo.Constraint.Skip

    m.cAcFlowUb = pyo.Constraint(m.s_branch_ac, m.s_h, rule=_ac_ub_rule)

    def _dc_lb_rule(m, b, h):
        lb = wd.dc_lb.get((h, b))
        return m.varDcBranchFlow[b, h] >= lb if lb is not None else pyo.Constraint.Skip

    m.cDcFlowLb = pyo.Constraint(m.s_branch_dc, m.s_h, rule=_dc_lb_rule)

    def _dc_ub_rule(m, b, h):
        ub = wd.dc_ub.get((h, b))
        return m.varDcBranchFlow[b, h] <= ub if ub is not None else pyo.Constraint.Skip

    m.cDcFlowUb = pyo.Constraint(m.s_branch_dc, m.s_h, rule=_dc_ub_rule)

    # ── inter-area NTC constraints ─────────────────────────────────
    # The hasattr guard this replaces existed because inter_area_ntc_rows was a
    # side effect of building the single-period model; it now comes from the context.
    if ctx.inter_area_ntc is not None and len(ctx.inter_area_ntc) > 0 and ctx.inter_area_ntc_rows:
        ntc_rows = lp._inter_area_ntc_rows
        m.s_ntc = pyo.RangeSet(0, len(ntc_rows) - 1)

        def _ntc_fwd_rule(m, cidx, h):
            row = ntc_rows[int(cidx)]
            cap = row["ntc_forward"]
            if not np.isfinite(cap):
                return pyo.Constraint.Skip
            if not (row["ac_pos"] or row["ac_neg"] or row["dc_pos"] or row["dc_neg"]):
                return pyo.Constraint.Skip
            net_t = (
                sum(m.varAcBranchFlow[b, h] for b in row["ac_pos"])
                - sum(m.varAcBranchFlow[b, h] for b in row["ac_neg"])
                + sum(m.varDcBranchFlow[b, h] for b in row["dc_pos"])
                - sum(m.varDcBranchFlow[b, h] for b in row["dc_neg"])
            )
            return net_t <= cap

        m.cNtcFwd = pyo.Constraint(m.s_ntc, m.s_h, rule=_ntc_fwd_rule)

        def _ntc_bwd_rule(m, cidx, h):
            row = ntc_rows[int(cidx)]
            cap = row["ntc_backward"]
            if not np.isfinite(cap):
                return pyo.Constraint.Skip
            if not (row["ac_pos"] or row["ac_neg"] or row["dc_pos"] or row["dc_neg"]):
                return pyo.Constraint.Skip
            net_t = (
                sum(m.varAcBranchFlow[b, h] for b in row["ac_pos"])
                - sum(m.varAcBranchFlow[b, h] for b in row["ac_neg"])
                + sum(m.varDcBranchFlow[b, h] for b in row["dc_pos"])
                - sum(m.varDcBranchFlow[b, h] for b in row["dc_neg"])
            )
            return net_t >= -cap

        m.cNtcBwd = pyo.Constraint(m.s_ntc, m.s_h, rule=_ntc_bwd_rule)

    return m
