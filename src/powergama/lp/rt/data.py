"""Per-timestep real-time data: targets, deviation prices and the inter-area rules' DA flows.

The deviation prices are unit data (``rt_price_up``/``rt_price_down`` and their
``*_ref`` profile multipliers, resolved once in :class:`RtContext`); nothing here
prices a unit by its type or cost.

Called once per timestep by :meth:`RtExtension.update_timestep`, after the core
has written this timestep's generator limits and demand.

They write only ``p_rt_*`` parameters, so they are independent of the
single-period parameter writes and of each other.

Pinned generators and fixed demand are not locks any more: they are ordinary
grid data, ``pmin_ref = pmax_ref`` and ``demand_ref`` pointing at a profile
column, applied by the core. Caps win over ramps there, so a pinned generator
follows its pin whatever its ramp rates.
tests/test_sim_realtime.py covers both.

These take the LpProblem, for the same reason the single-period updater does:
they read mutable simulation state (_storage) alongside grid data, and an
eight-argument signature would be a longer way of spelling `self`. The RT
extension comes in as ``rt``, for its context and settings.
"""


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


SPLIT_EPS = 1e-6


def _enforce_split_sum_guard(p_pos: float, p_neg: float, eps: float = SPLIT_EPS) -> tuple[float, float]:
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


def _write_prices(lp, timestep, up, down, param_pos, param_neg, index):
    """Write one kind's prices: the tracked rows' up/down prices, 0 for every other row."""
    prices = {}
    if up is not None and down is not None:
        values_up = up.at(lp._grid.profiles, timestep)
        values_down = down.at(lp._grid.profiles, timestep)
        for r, p_up, p_down in zip(up.rows, values_up, values_down):
            prices[int(r)] = _enforce_split_sum_guard(p_up, p_down)
    for i in index:
        p_pos, p_neg = prices.get(int(i), (0.0, 0.0))
        param_pos[i] = p_pos
        param_neg[i] = p_neg


def update_deviation_prices(lp, rt, timestep):
    """Write this timestep's deviation prices from the unit data.

    Each tracked row has a price up (more than DA) and down (less than DA), in EUR/MWh:
    the constant ``rt_price_up``/``rt_price_down`` (``rt_pump_price_*`` for pumping) times
    its profile ``*_ref`` at this timestep (no profile: the constant). The only rule left
    here is the split-sum guard: a pair summing to less than 1e-6 is lifted to it.
    """
    m = lp.model
    ctx = rt.ctx
    _write_prices(
        lp,
        timestep,
        ctx.gen_price_up,
        ctx.gen_price_down,
        m.p_rt_deviation_price_gen_pos,
        m.p_rt_deviation_price_gen_neg,
        m.s_gen,
    )
    _write_prices(
        lp,
        timestep,
        ctx.pump_price_up,
        ctx.pump_price_down,
        m.p_rt_deviation_price_pump_pos,
        m.p_rt_deviation_price_pump_neg,
        m.s_gen_pump,
    )
    _write_prices(
        lp,
        timestep,
        ctx.cons_price_up,
        ctx.cons_price_down,
        m.p_rt_deviation_price_flex_pos,
        m.p_rt_deviation_price_flex_neg,
        m.s_load_flex,
    )


def update_rule_flows(lp, rt, timestep):
    """Write the DA summed flow (MW) of each inter-area rule RT holds to it, and its lock window."""
    if not rt.ctx.rules:
        return
    m = lp.model
    row = lp._grid.profiles.index.get_loc(timestep)
    for r, target in rt.ctx.rules.items():
        da_flow = float(lp._grid.profiles.iat[row, target.profile_position])
        m.p_rt_rule_da_flow[r] = da_flow
        if target.width_forward is not None:
            m.p_rt_rule_lock_ub[r] = da_flow + target.width_forward
        if target.width_backward is not None:
            m.p_rt_rule_lock_lb[r] = da_flow - target.width_backward
