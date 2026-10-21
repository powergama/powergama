"""Per-timestep real-time data: day-ahead locks, targets and deviation prices.

Called once per timestep by :meth:`RtExtension.update_timestep`, after the core
has written this timestep's generator limits and demand.

`apply_da_locks` overrides those core values: it pins locked generators,
replaces locked demand and sets the border-flow windows. The rest write only
``p_rt_*`` parameters, so they are independent of the single-period parameter
writes and of each other.

The locks used to be applied inside the core loops, in every mode. They are RT
inputs only -- produced from a day-ahead run by the external prepare-rt step --
so they now live here, and a normal run never reads them. Applying a generator
lock after the ramp clipping rather than before gives the same limits: the
clipping used to skip pinned generators in RT, and now the lock simply
overwrites whatever it computed. tests/test_sim_realtime.py covers both.

These take the LpProblem, for the same reason the single-period updater does:
they read mutable simulation state (_storage) alongside grid data, and an
eight-argument signature would be a longer way of spelling `self`. The RT
extension comes in as ``rt``, for its locks, context and settings.
"""

import numpy as np
import pyomo.environ as pyo


def apply_da_locks(lp, rt, timestep):
    """Apply the day-ahead hard locks: pinned generators, fixed demand, border-flow windows."""
    # Foreign generators pinned to their DA output.
    if rt.locks.foreign_gen_lock is not None:
        for i in lp.model.s_gen:
            key = (int(timestep), int(i))
            if key in rt.locks.foreign_gen_lock.index:
                da_output = float(rt.locks.foreign_gen_lock.loc[key])
                if not np.isfinite(da_output):
                    da_output = 0.0
                da_output = max(0.0, da_output)
                lp.model.p_gen_pmin[i, 0] = da_output
                lp.model.p_gen_pmax[i, 0] = da_output

    # Foreign consumers fixed to their DA demand.
    if rt.locks.foreign_cons_lock is not None:
        for i in lp.model.s_load:
            key = (int(timestep), int(i))
            if key in rt.locks.foreign_cons_lock.index:
                locked_demand = float(rt.locks.foreign_cons_lock.loc[key])
                if np.isfinite(locked_demand):
                    lp.model.p_demand[i, 0] = locked_demand

    # Border-flow windows as variable bounds, reset to none where this timestep has
    # no window. The branch rating stays with cMaxFlowAc/cMaxFlowDc (issue #39).
    _set_flow_bounds(lp.model.varAcBranchFlow, rt.locks.border_ac_flow_lb, rt.locks.border_ac_flow_ub, timestep)
    _set_flow_bounds(lp.model.varDcBranchFlow, rt.locks.border_dc_flow_lb, rt.locks.border_dc_flow_ub, timestep)


def _set_flow_bounds(var, lb_series, ub_series, timestep):
    # RT runs in windows of one timestep: hour 0 of the window is the timestep.
    for b, h in var:
        key = (int(timestep), int(b))
        lb = ub = None
        if lb_series is not None and ub_series is not None and key in lb_series.index and key in ub_series.index:
            lb = float(lb_series.loc[key])
            ub = float(ub_series.loc[key])
            if not np.isfinite(lb):
                lb = None
            if not np.isfinite(ub):
                ub = None
        var[b, h].setlb(lb)
        var[b, h].setub(ub)


def update_generator_targets(lp, rt, timestep):
    """Write each generator's DA dispatch target for this timestep."""
    P_max = lp._grid.generator["pmax"]
    for i in lp.model.s_gen:
        if rt.ctx.target_ref is not None:
            rt_ref = rt.ctx.target_ref.iloc[i]
            if isinstance(rt_ref, str) and rt_ref in lp._grid.profiles.columns:
                lp.model.p_rt_target[i] = max(0.0, float(P_max[i]) * float(lp._grid.profiles.loc[timestep, rt_ref]))
            else:
                lp.model.p_rt_target[i] = 0.0
        else:
            lp.model.p_rt_target[i] = 0.0


def update_pump_targets(lp, rt, timestep):
    """Write each pumped-storage unit's DA pumping target for this timestep."""
    if rt.ctx.pump_target_ref is None or not rt.ctx.pump_target_indices:
        return
    for i in lp.model.s_gen_pump:
        rt_ref = rt.ctx.pump_target_ref.iloc[int(i)]
        if isinstance(rt_ref, str) and rt_ref in lp._grid.profiles.columns:
            pump_cap = float(lp._grid.generator.loc[i, "pump_cap"])
            lp.model.p_rt_pump_target[i] = max(0.0, pump_cap * float(lp._grid.profiles.loc[timestep, rt_ref]))
        else:
            lp.model.p_rt_pump_target[i] = 0.0


def update_flexload_targets(lp, rt, timestep):
    """Write flexible-load DA targets."""
    # Update consumer (flexible load) DA targets for DA-target deviation penalties
    if rt.ctx.consumer_target_ref is not None and rt.ctx.consumer_target_indices:
        for j in lp.model.s_load_flex:
            rt_cons_ref = rt.ctx.consumer_target_ref.iloc[int(j)]
            if isinstance(rt_cons_ref, str) and rt_cons_ref in lp._grid.profiles.columns:
                demand_avg = float(lp._grid.consumer.loc[j, "demand_avg"])
                lp.model.p_rt_flexload_target[j] = max(
                    0.0, demand_avg * float(lp._grid.profiles.loc[timestep, rt_cons_ref])
                )
            else:
                lp.model.p_rt_flexload_target[j] = 0.0


def marginal_cost(lp, i):
    """Marginal cost of generator *i*, in EUR/MWh: its operating cost, plus the
    storage value for a generator that draws from storage."""
    cost = float(pyo.value(lp.model.p_gen_cost[i]))
    if i in lp.model.s_gen_storage:
        cost += float(pyo.value(lp.model.p_storage_value[i]))
    return cost


def update_deviation_prices(lp, rt, timestep):
    """Price this timestep's deviation variables from current marginal costs."""
    # Every tracked quantity is priced around its current marginal cost c:
    #   c+ = c0 + c (more than DA), c- = c0 - c (less than DA),
    # with c0 the balancing fee, so c0 is paid on every MWh of deviation either way.
    # Variable renewables are the exception (below). Ramp limits still govern which
    # redispatch is feasible.
    _fee = float(rt.balancing_fee_eur_per_mwh)
    _split_eps = 1e-6

    def _enforce_split_sum_guard(p_pos: float, p_neg: float, eps: float = _split_eps) -> tuple[float, float]:
        """Ensure split-variable coefficients satisfy p_pos + p_neg >= eps.

        This prevents degenerate/unbounded rays where both split variables can
        increase together without improving the physical deviation variable.
        The adjustment is applied to the smaller coefficient, matching the
        intended semantics of preserving the larger directional signal.
        """
        s = float(p_pos) + float(p_neg)
        if s >= float(eps):
            return float(p_pos), float(p_neg)
        add = float(eps) - s
        if float(p_pos) <= float(p_neg):
            return float(p_pos) + add, float(p_neg)
        return float(p_pos), float(p_neg) + add

    for i in lp.model.s_gen:
        if int(i) in rt.ctx.target_gen_indices:
            _gen_cost = marginal_cost(lp, i)
            _is_res = int(i) in rt.ctx.idx_wind or int(i) in rt.ctx.idx_solar or int(i) in rt.ctx.idx_hydro_ror
            if _is_res:
                # RES: reward upward and penalize curtailment with a tunable factor.
                # c+ = -c0
                # c- = rt_p_res_curtailment_factor * c0
                _dev_price_pos = -_fee
                _dev_price_neg = rt.res_curtailment_factor * _fee
            else:
                # Non-RES: c+ = c0 + cDA, c- = c0 - cDA
                # c- may be negative when cDA > c0 (incentive to reduce generation).
                _dev_price_pos = _fee + _gen_cost
                _dev_price_neg = _fee - _gen_cost
            _dev_price_pos, _dev_price_neg = _enforce_split_sum_guard(_dev_price_pos, _dev_price_neg)
            lp.model.p_rt_deviation_price_gen_pos[i] = _dev_price_pos
            lp.model.p_rt_deviation_price_gen_neg[i] = _dev_price_neg
        else:
            lp.model.p_rt_deviation_price_gen_pos[i] = 0.0
            lp.model.p_rt_deviation_price_gen_neg[i] = 0.0
    for i in lp.model.s_gen_pump:
        if int(i) in rt.ctx.pump_target_indices:
            # Pumping: c+ = c0 + c_pump, c- = c0 - c_pump, where c_pump is the cost of
            # pumping 1 MWh: the deadband, less the storage value of what reaches storage.
            # Pumping x more and generating eff*x more then costs c0*(1 + eff) + deadband,
            # whatever the storage value, so cycling never pays.
            _pump_credit = float(pyo.value(lp.model.p_storage_value[i])) * float(
                lp._grid.generator.loc[i, "pump_efficiency"]
            ) - float(pyo.value(lp.model.p_genpump_cost[i]))
            _c_pump_pos, _c_pump_neg = _enforce_split_sum_guard(_fee - _pump_credit, _fee + _pump_credit)
            lp.model.p_rt_deviation_price_pump_pos[i] = _c_pump_pos
            lp.model.p_rt_deviation_price_pump_neg[i] = _c_pump_neg
        else:
            lp.model.p_rt_deviation_price_pump_pos[i] = 0.0
            lp.model.p_rt_deviation_price_pump_neg[i] = 0.0
    for i in lp.model.s_load_flex:
        if int(i) in rt.ctx.consumer_target_indices:
            # Flex load / load-shedding: c+ = c0 + c_flex, c- = c0 - c_flex
            _flex_cost = float(pyo.value(lp.model.p_loadflex_cost[i, 0]))
            _c_flex_pos = _fee + _flex_cost
            _c_flex_neg = _fee - _flex_cost
            _c_flex_pos, _c_flex_neg = _enforce_split_sum_guard(_c_flex_pos, _c_flex_neg)
            lp.model.p_rt_deviation_price_flex_pos[i] = _c_flex_pos
            lp.model.p_rt_deviation_price_flex_neg[i] = _c_flex_neg
        else:
            lp.model.p_rt_deviation_price_flex_pos[i] = 0.0
            lp.model.p_rt_deviation_price_flex_neg[i] = 0.0


def update_border_flow_targets(lp, rt, timestep):
    """Write the DA border-flow targets for AC and DC branches."""
    for b in lp.model.s_branch_ac:
        key = (int(timestep), int(b))
        if rt.locks.border_ac_flow_lock is not None and key in rt.locks.border_ac_flow_lock.index:
            da_flow = float(rt.locks.border_ac_flow_lock.loc[key])
            if np.isfinite(da_flow):
                lp.model.p_rt_io_target_ac[b] = da_flow
                lp.model.p_rt_io_target_active_ac[b] = 1
            else:
                lp.model.p_rt_io_target_ac[b] = 0.0
                lp.model.p_rt_io_target_active_ac[b] = 0
        else:
            lp.model.p_rt_io_target_ac[b] = 0.0
            lp.model.p_rt_io_target_active_ac[b] = 0

    for b in lp.model.s_branch_dc:
        key = (int(timestep), int(b))
        if rt.locks.border_dc_flow_lock is not None and key in rt.locks.border_dc_flow_lock.index:
            da_flow = float(rt.locks.border_dc_flow_lock.loc[key])
            if np.isfinite(da_flow):
                lp.model.p_rt_io_target_dc[b] = da_flow
                lp.model.p_rt_io_target_active_dc[b] = 1
            else:
                lp.model.p_rt_io_target_dc[b] = 0.0
                lp.model.p_rt_io_target_active_dc[b] = 0
        else:
            lp.model.p_rt_io_target_dc[b] = 0.0
            lp.model.p_rt_io_target_active_dc[b] = 0
