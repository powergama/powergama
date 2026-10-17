"""Per-timestep real-time data: day-ahead targets and deviation prices.

Called from the single-period update once per timestep, under ``cfg.is_rt``.
These write only ``p_rt_*`` parameters, so they are independent of the
single-period parameter writes and of each other.

Two of them used to be interleaved rather than separate blocks, and are now
their own passes over the same index sets:

- the generator DA target was written inside the main generator loop;
- the border flow targets were written inside the loop that also resets AC/DC
  branch flow bounds, which is not RT-specific and has to run in every mode.

Both reorderings are safe because nothing between the old and new write points
reads the parameters being written -- the tracking constraints read them at
solve time -- and both are pinned by the LP-text oracle.

These take the LpProblem rather than explicit arguments, for the same reason
the single-period updater does: they read mutable simulation state (_storage,
_storage_flexload) alongside grid data and config, and an eight-argument
signature would be a longer way of spelling `self` rather than an interface.
"""

import numpy as np
import pyomo.environ as pyo


def update_generator_targets(lp, timestep, P_max):
    """Write each generator's DA dispatch target for this timestep."""
    for i in lp.model.s_gen:
        if lp._rt_target_ref is not None:
            rt_ref = lp._rt_target_ref.iloc[i]
            if isinstance(rt_ref, str) and rt_ref in lp._grid.profiles.columns:
                lp.model.p_rt_target_active[i] = 1
                lp.model.p_rt_target[i] = max(0.0, float(P_max[i]) * float(lp._grid.profiles.loc[timestep, rt_ref]))
            else:
                lp.model.p_rt_target_active[i] = 0
                lp.model.p_rt_target[i] = 0.0
        else:
            lp.model.p_rt_target_active[i] = 0
            lp.model.p_rt_target[i] = 0.0


def update_flexload_and_storage_targets(lp, timestep):
    """Write flexible-load DA targets, storage DA targets and the storage balance RHS."""
    # Update consumer (flexible load) DA targets for DA-target deviation penalties
    if lp._rt_consumer_target_ref is not None and lp._rt_consumer_target_indices:
        for j in lp.model.s_load_flex:
            rt_cons_ref = lp._rt_consumer_target_ref.iloc[int(j)]
            if isinstance(rt_cons_ref, str) and rt_cons_ref in lp._grid.profiles.columns:
                lp.model.p_rt_flexload_target_active[j] = 1
                demand_avg = float(lp._grid.consumer.loc[j, "demand_avg"])
                lp.model.p_rt_flexload_target[j] = max(
                    0.0, demand_avg * float(lp._grid.profiles.loc[timestep, rt_cons_ref])
                )
            else:
                lp.model.p_rt_flexload_target_active[j] = 0
                lp.model.p_rt_flexload_target[j] = 0.0

    # Update storage DA filling targets and one-step storage balance RHS.
    if lp._rt_storage_target_ref is not None and lp._rt_storage_target_indices:
        for i in lp.model.s_gen_storage:
            inflow_factor = lp._grid.generator.loc[i, "inflow_fac"]
            inflow_profile = lp._grid.generator.loc[i, "inflow_ref"]
            capacity = lp._grid.generator.loc[i, "pmax"]
            gen_inflow = capacity * inflow_factor * lp._grid.profiles[inflow_profile][timestep]
            lp.model.p_rt_storage_balance_rhs[i] = max(0.0, float(lp._storage[i] + gen_inflow * lp.timeDelta))

            rt_storage_ref = lp._rt_storage_target_ref.iloc[int(i)]
            if isinstance(rt_storage_ref, str) and rt_storage_ref in lp._grid.profiles.columns:
                storage_cap = float(lp._grid.generator.loc[i, "storage_cap"])
                target_frac = float(lp._grid.profiles.loc[timestep, rt_storage_ref])
                lp.model.p_rt_storage_target[i] = max(0.0, storage_cap * max(0.0, min(1.0, target_frac)))
            else:
                lp.model.p_rt_storage_target[i] = 0.0


def update_deviation_prices(lp, timestep):
    """Price this timestep's deviation variables from current marginal costs."""
    # In deviation objective mode, all generation technologies use the same
    # RT-local deviation pricing scheme (no DA-price coupling):
    #   deviation_price = max(0, current_marginal_cost + balancing_fee)
    # applied symmetrically to upward and downward deviation variables.
    # This ensures every technology follows a pure deviation-based RT scheme,
    # while ramp-rate constraints continue to govern feasible redispatch.
    _fee = float(lp._rt_balancing_fee_eur_per_mwh)
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
        if int(i) in lp._rt_target_gen_indices:
            _gen_cost = float(pyo.value(lp.model.p_gen_cost[i]))
            _is_res = int(i) in lp._idx_rt_wind or int(i) in lp._idx_rt_solar or int(i) in lp._idx_rt_hydro_ror
            if _is_res:
                # RES: reward upward and penalize curtailment with a tunable factor.
                # c+ = -c0
                # c- = rt_p_res_curtailment_factor * c0
                _dev_price_pos = -_fee
                _dev_price_neg = lp._rt_p_res_curtailment_factor * _fee
            else:
                # Non-RES: c+ = c0 + cDA, c- = c0 - cDA
                # c- may be negative when cDA > c0 (incentive to reduce generation).
                _dev_price_pos = _fee + _gen_cost
                _dev_price_neg = _fee - _gen_cost
            _dev_price_pos, _dev_price_neg = _enforce_split_sum_guard(_dev_price_pos, _dev_price_neg)
            lp.model.p_rt_deviation_price_gen_pos[i] = _dev_price_pos
            lp.model.p_rt_deviation_price_gen_neg[i] = _dev_price_neg
            lp.model.p_rt_deviation_price_gen[i] = max(_dev_price_pos, _dev_price_neg)
        else:
            lp.model.p_rt_deviation_price_gen[i] = 0.0
            lp.model.p_rt_deviation_price_gen_pos[i] = 0.0
            lp.model.p_rt_deviation_price_gen_neg[i] = 0.0
    for i in lp.model.s_load_flex:
        if int(i) in lp._rt_consumer_target_indices:
            # Flex load / load-shedding: c+ = c0 + c_flex, c- = c0 - c_flex
            _flex_cost = float(pyo.value(lp.model.p_loadflex_cost[i]))
            _c_flex_pos = _fee + _flex_cost
            _c_flex_neg = _fee - _flex_cost
            _c_flex_pos, _c_flex_neg = _enforce_split_sum_guard(_c_flex_pos, _c_flex_neg)
            lp.model.p_rt_deviation_price_flex_pos[i] = _c_flex_pos
            lp.model.p_rt_deviation_price_flex_neg[i] = _c_flex_neg
            lp.model.p_rt_deviation_price_flex[i] = 0.5 * (_c_flex_pos + _c_flex_neg)
        else:
            lp.model.p_rt_deviation_price_flex[i] = 0.0
            lp.model.p_rt_deviation_price_flex_pos[i] = 0.0
            lp.model.p_rt_deviation_price_flex_neg[i] = 0.0
    for i in lp.model.s_gen_storage:
        if int(i) in lp._rt_storage_target_indices:
            # Storage DA-marginal pricing: c+ = c0 + c_DA, c- = c0 - c_DA
            # c_DA is the DA marginal storage value (from da_storage_marginalprice parquet),
            # falling back to current storagevalue (p_gen_cost[i]) if not available.
            _c_da_storage = lp._da_storage_marginalprice.get((int(timestep), int(i)), None)
            if _c_da_storage is None:
                _c_da_storage = float(pyo.value(lp.model.p_gen_cost[i]))
            else:
                _c_da_storage = float(_c_da_storage)
            c_pos = _fee + _c_da_storage
            c_neg = _fee - _c_da_storage
            c_pos, c_neg = _enforce_split_sum_guard(c_pos, c_neg)
            lp.model.p_rt_deviation_price_storage_pos[i] = c_pos
            lp.model.p_rt_deviation_price_storage_neg[i] = c_neg
            lp.model.p_rt_deviation_price_storage[i] = 0.5 * (c_pos + c_neg)
        else:
            lp.model.p_rt_deviation_price_storage[i] = 0.0
            lp.model.p_rt_deviation_price_storage_pos[i] = 0.0
            lp.model.p_rt_deviation_price_storage_neg[i] = 0.0


def update_border_flow_targets(lp, timestep):
    """Write the DA border-flow targets for AC and DC branches.

    Split out of the loop that also resets branch flow bounds: that part runs in
    every mode and stays in the single-period updater.
    """
    for b in lp.model.s_branch_ac:
        key = (int(timestep), int(b))
        if lp._border_ac_flow_lock is not None and key in lp._border_ac_flow_lock.index:
            da_flow = float(lp._border_ac_flow_lock.loc[key])
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
        if lp._border_dc_flow_lock is not None and key in lp._border_dc_flow_lock.index:
            da_flow = float(lp._border_dc_flow_lock.loc[key])
            if np.isfinite(da_flow):
                lp.model.p_rt_io_target_dc[b] = da_flow
                lp.model.p_rt_io_target_active_dc[b] = 1
            else:
                lp.model.p_rt_io_target_dc[b] = 0.0
                lp.model.p_rt_io_target_active_dc[b] = 0
        else:
            lp.model.p_rt_io_target_dc[b] = 0.0
            lp.model.p_rt_io_target_active_dc[b] = 0
