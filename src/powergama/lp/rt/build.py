"""Real-time model construction: deviation tracking against a day-ahead schedule.

Everything here is additive to the single-period model except the objective. The
single-period builder calls these at the exact points the declarations used to
sit, which keeps Pyomo's component declaration order -- and therefore the column
order of the written LP -- unchanged.

Every function returns without declaring anything unless ``cfg.is_rt``, so a
normal run carries no RT components at all. That matters beyond tidiness: APPSI
hands every ``Var`` on the model to HiGHS, used or not, so unused deviation
variables would otherwise be solver columns in every run.

The tracking constraints are declared over the full index set and Skip the
entries with no DA target, rather than being declared over the targeted subset.
That is preserved from the original: it keeps the constraint index aligned with
the variable index, and Skip costs nothing in the written LP.
"""

import pyomo.environ as pyo


def add_params(model, cfg):
    """Mutable parameters holding this timestep's DA targets and deviation prices."""
    if not cfg.is_rt:
        return
    model.p_rt_target = pyo.Param(model.s_gen, within=pyo.Reals, default=0, mutable=True)
    model.p_rt_deviation_price_gen_pos = pyo.Param(model.s_gen, within=pyo.Reals, default=0, mutable=True)
    model.p_rt_deviation_price_gen_neg = pyo.Param(model.s_gen, within=pyo.Reals, default=0, mutable=True)
    # Consumer (flexible load) RT DA target for DA-target deviation penalties
    model.p_rt_flexload_target = pyo.Param(model.s_load_flex, within=pyo.NonNegativeReals, default=0, mutable=True)
    model.p_rt_deviation_price_flex_pos = pyo.Param(model.s_load_flex, within=pyo.Reals, default=0, mutable=True)
    model.p_rt_deviation_price_flex_neg = pyo.Param(model.s_load_flex, within=pyo.Reals, default=0, mutable=True)
    model.p_rt_storage_target = pyo.Param(model.s_gen_storage, within=pyo.NonNegativeReals, default=0, mutable=True)
    model.p_rt_storage_balance_rhs = pyo.Param(
        model.s_gen_storage, within=pyo.NonNegativeReals, default=0, mutable=True
    )
    model.p_rt_deviation_price_storage_pos = pyo.Param(model.s_gen_storage, within=pyo.Reals, default=0, mutable=True)
    model.p_rt_deviation_price_storage_neg = pyo.Param(model.s_gen_storage, within=pyo.Reals, default=0, mutable=True)
    model.p_rt_io_target_ac = pyo.Param(model.s_branch_ac, within=pyo.Reals, default=0, mutable=True)
    model.p_rt_io_target_dc = pyo.Param(model.s_branch_dc, within=pyo.Reals, default=0, mutable=True)
    model.p_rt_io_target_active_ac = pyo.Param(model.s_branch_ac, within=pyo.Binary, default=0, mutable=True)
    model.p_rt_io_target_active_dc = pyo.Param(model.s_branch_dc, within=pyo.Binary, default=0, mutable=True)


def add_generation_deviation_vars(model, cfg):
    """Generator DA target tracking deviations.

    Separate from the rest only to preserve declaration order: these sit between
    varPump and varFlexLoad in the single-period builder.
    """
    if not cfg.is_rt:
        return
    model.varRtTargetDevPos = pyo.Var(model.s_gen, within=pyo.NonNegativeReals)
    model.varRtTargetDevNeg = pyo.Var(model.s_gen, within=pyo.NonNegativeReals)


def add_remaining_deviation_vars(model, cfg):
    """Flexible load, storage trajectory and border flow tracking deviations."""
    if not cfg.is_rt:
        return
    # Flexible load DA target tracking deviations (soft penalty)
    model.varRtFlexLoadTargetDevPos = pyo.Var(model.s_load_flex, within=pyo.NonNegativeReals)
    model.varRtFlexLoadTargetDevNeg = pyo.Var(model.s_load_flex, within=pyo.NonNegativeReals)
    model.varRtStorageTargetDevPos = pyo.Var(model.s_gen_storage, within=pyo.NonNegativeReals)
    model.varRtStorageTargetDevNeg = pyo.Var(model.s_gen_storage, within=pyo.NonNegativeReals)
    model.varRtIoTargetDevPosAc = pyo.Var(model.s_branch_ac, within=pyo.NonNegativeReals)
    model.varRtIoTargetDevNegAc = pyo.Var(model.s_branch_ac, within=pyo.NonNegativeReals)
    model.varRtIoTargetDevPosDc = pyo.Var(model.s_branch_dc, within=pyo.NonNegativeReals)
    model.varRtIoTargetDevNegDc = pyo.Var(model.s_branch_dc, within=pyo.NonNegativeReals)


def add_target_tracking(model, grid, ctx, cfg):
    """Constraint: RT DA target dispatch deviation accounting."""

    if not cfg.rt_target_tracking_active:
        return

    def rt_target_rule(model, i):
        if int(i) not in ctx.rt_target_gen_indices:
            return pyo.Constraint.Skip
        return model.varGeneration[i] - model.p_rt_target[i] == model.varRtTargetDevPos[i] - model.varRtTargetDevNeg[i]

    model.cRtTargetTracking = pyo.Constraint(model.s_gen, rule=rt_target_rule)


def add_flexload_target_tracking(model, grid, ctx, cfg):
    """Constraint: RT DA flexible load (consumer) target deviation accounting.

    Flexible load can deviate from DA when other balancing channels are exhausted,
    but tracked deviations incur soft penalties to keep DA as the default solution.
    """
    if not cfg.rt_target_tracking_active:
        return

    def rt_flexload_target_rule(model, j):
        if int(j) not in ctx.rt_consumer_target_indices:
            return pyo.Constraint.Skip
        return (
            model.varFlexLoad[j] - model.p_rt_flexload_target[j]
            == model.varRtFlexLoadTargetDevPos[j] - model.varRtFlexLoadTargetDevNeg[j]
        )

    model.cRtFlexLoadTargetTracking = pyo.Constraint(model.s_load_flex, rule=rt_flexload_target_rule)


def add_storage_target_tracking(model, grid, ctx, cfg):
    """Constraint: RT DA storage filling trajectory deviation accounting."""

    if not cfg.rt_target_tracking_active:
        return

    def rt_storage_target_rule(model, i):
        if int(i) not in ctx.rt_storage_target_indices:
            return pyo.Constraint.Skip
        lhs = model.p_rt_storage_balance_rhs[i] - (grid.timeDelta * model.varGeneration[i])
        if int(i) in model.s_gen_pump:
            pump_eff = float(grid.generator.loc[i, "pump_efficiency"])
            lhs += grid.timeDelta * pump_eff * model.varPump[i]
        return (
            lhs - model.p_rt_storage_target[i] == model.varRtStorageTargetDevPos[i] - model.varRtStorageTargetDevNeg[i]
        )

    model.cRtStorageTargetTracking = pyo.Constraint(model.s_gen_storage, rule=rt_storage_target_rule)


def add_border_flow_target_tracking(model, grid, ctx, cfg):
    """Constraint: DA BE border flow target deviation accounting for AC/DC branches."""

    if not cfg.rt_target_tracking_active:
        return

    def _ac_rule(model, b):
        if int(b) not in ctx.idx_border_ac:
            return pyo.Constraint.Skip
        return (
            model.p_rt_io_target_active_ac[b] * (model.varAcBranchFlow[b] - model.p_rt_io_target_ac[b])
            == model.varRtIoTargetDevPosAc[b] - model.varRtIoTargetDevNegAc[b]
        )

    def _dc_rule(model, b):
        if int(b) not in ctx.idx_border_dc:
            return pyo.Constraint.Skip
        return (
            model.p_rt_io_target_active_dc[b] * (model.varDcBranchFlow[b] - model.p_rt_io_target_dc[b])
            == model.varRtIoTargetDevPosDc[b] - model.varRtIoTargetDevNegDc[b]
        )

    model.cRtIoTargetTrackingAc = pyo.Constraint(model.s_branch_ac, rule=_ac_rule)
    model.cRtIoTargetTrackingDc = pyo.Constraint(model.s_branch_dc, rule=_dc_rule)


def edit_objective_terms(model, ctx, cfg, terms):
    """Turn the cost-minimising objective into a deviation-minimising one.

    RT drops the operating cost -- the day-ahead run already settled it -- and
    prices deviations from the day-ahead schedule instead. Load shedding stays:
    otherwise the solver could bypass the tracked balancing channels by dropping
    demand instead of redispatching generation, storage or flexible load.
    """
    del terms["operating_cost"]
    terms["rt_deviation"] = deviation_cost(model, ctx, cfg)


def deviation_cost(model, ctx, cfg):
    """The RT deviation penalties added to the objective."""
    cost = sum(
        model.p_rt_deviation_price_gen_pos[i] * model.varRtTargetDevPos[i]
        + model.p_rt_deviation_price_gen_neg[i] * model.varRtTargetDevNeg[i]
        for i in model.s_gen
        if int(i) in ctx.rt_target_gen_indices
    )
    cost += sum(
        model.p_rt_deviation_price_flex_pos[j] * model.varRtFlexLoadTargetDevPos[j]
        + model.p_rt_deviation_price_flex_neg[j] * model.varRtFlexLoadTargetDevNeg[j]
        for j in model.s_load_flex
        if int(j) in ctx.rt_consumer_target_indices
    )
    cost += sum(
        model.p_rt_deviation_price_storage_pos[i] * model.varRtStorageTargetDevPos[i]
        + model.p_rt_deviation_price_storage_neg[i] * model.varRtStorageTargetDevNeg[i]
        for i in model.s_gen_storage
        if int(i) in ctx.rt_storage_target_indices
    )
    if cfg.rt_xborder_flow_penalty_eur_per_mwh > 0.0:
        cost += cfg.rt_xborder_flow_penalty_eur_per_mwh * (
            sum(
                model.varRtIoTargetDevPosAc[b] + model.varRtIoTargetDevNegAc[b]
                for b in model.s_branch_ac
                if int(b) in ctx.idx_border_ac
            )
            + sum(
                model.varRtIoTargetDevPosDc[b] + model.varRtIoTargetDevNegDc[b]
                for b in model.s_branch_dc
                if int(b) in ctx.idx_border_dc
            )
        )
    return cost
