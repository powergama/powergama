"""Real-time model construction: deviation tracking against a day-ahead schedule.

Everything here is additive to the single-period model except the objective, and
is reached only through :class:`~powergama.lp.rt.RtExtension`'s hooks, so a
normal run carries no RT components at all. That matters beyond tidiness: APPSI
hands every ``Var`` on the model to HiGHS, used or not, so unused deviation
variables would otherwise be solver columns in every run.

Each function takes the extension as ``rt``, for its own context (``rt.ctx``)
and settings; ``grid`` and ``ctx`` are the core's.

The tracking constraints are declared over the full index set and Skip the
entries with no DA target, rather than being declared over the targeted subset.
That is preserved from the original: it keeps the constraint index aligned with
the variable index, and Skip costs nothing in the written LP.
"""

import pyomo.environ as pyo


def add_params(model, rt):
    """Mutable parameters holding this timestep's DA targets and deviation prices."""
    model.p_rt_target = pyo.Param(model.s_gen, within=pyo.Reals, default=0, mutable=True)
    model.p_rt_deviation_price_gen_pos = pyo.Param(model.s_gen, within=pyo.Reals, default=0, mutable=True)
    model.p_rt_deviation_price_gen_neg = pyo.Param(model.s_gen, within=pyo.Reals, default=0, mutable=True)
    # Consumer (flexible load) RT DA target for DA-target deviation penalties
    model.p_rt_flexload_target = pyo.Param(model.s_load_flex, within=pyo.NonNegativeReals, default=0, mutable=True)
    model.p_rt_deviation_price_flex_pos = pyo.Param(model.s_load_flex, within=pyo.Reals, default=0, mutable=True)
    model.p_rt_deviation_price_flex_neg = pyo.Param(model.s_load_flex, within=pyo.Reals, default=0, mutable=True)
    model.p_rt_pump_target = pyo.Param(model.s_gen_pump, within=pyo.NonNegativeReals, default=0, mutable=True)
    model.p_rt_deviation_price_pump_pos = pyo.Param(model.s_gen_pump, within=pyo.Reals, default=0, mutable=True)
    model.p_rt_deviation_price_pump_neg = pyo.Param(model.s_gen_pump, within=pyo.Reals, default=0, mutable=True)
    add_rule_params(model, rt)


def add_deviation_vars(model):
    """Deviations from the DA targets: generators, flexible load, pumping, inter-area summed flows."""
    model.varRtTargetDevPos = pyo.Var(model.s_gen, within=pyo.NonNegativeReals)
    model.varRtTargetDevNeg = pyo.Var(model.s_gen, within=pyo.NonNegativeReals)
    model.varRtFlexLoadTargetDevPos = pyo.Var(model.s_load_flex, within=pyo.NonNegativeReals)
    model.varRtFlexLoadTargetDevNeg = pyo.Var(model.s_load_flex, within=pyo.NonNegativeReals)
    model.varRtPumpTargetDevPos = pyo.Var(model.s_gen_pump, within=pyo.NonNegativeReals)
    model.varRtPumpTargetDevNeg = pyo.Var(model.s_gen_pump, within=pyo.NonNegativeReals)
    add_rule_deviation_vars(model)


def add_target_tracking(model, grid, ctx, rt):
    """Constraint: RT DA target dispatch deviation accounting."""

    def rt_target_rule(model, i):
        if int(i) not in rt.ctx.target_gen_indices:
            return pyo.Constraint.Skip
        return (
            model.varGeneration[i, 0] - model.p_rt_target[i] == model.varRtTargetDevPos[i] - model.varRtTargetDevNeg[i]
        )

    model.cRtTargetTracking = pyo.Constraint(model.s_gen, rule=rt_target_rule)


def add_flexload_target_tracking(model, grid, ctx, rt):
    """Constraint: RT DA flexible load (consumer) target deviation accounting.

    Flexible load can deviate from DA when other balancing channels are exhausted,
    but tracked deviations incur soft penalties to keep DA as the default solution.
    """

    def rt_flexload_target_rule(model, j):
        if int(j) not in rt.ctx.consumer_target_indices:
            return pyo.Constraint.Skip
        return (
            model.varFlexLoad[j, 0] - model.p_rt_flexload_target[j]
            == model.varRtFlexLoadTargetDevPos[j] - model.varRtFlexLoadTargetDevNeg[j]
        )

    model.cRtFlexLoadTargetTracking = pyo.Constraint(model.s_load_flex, rule=rt_flexload_target_rule)


def add_pump_target_tracking(model, grid, ctx, rt):
    """Constraint: RT DA pumping deviation accounting.

    Pumped storage is settled on its power, like generation: what it pumps is
    tracked against the DA schedule, and the generation side by ``rt_target_ref``.
    Its filling level is not tracked: RT drops the terminal storage value, and the
    storage value enters through the deviation prices instead.
    """

    def rt_pump_target_rule(model, i):
        if int(i) not in rt.ctx.pump_target_indices:
            return pyo.Constraint.Skip
        return (
            model.varPump[i, 0] - model.p_rt_pump_target[i]
            == model.varRtPumpTargetDevPos[i] - model.varRtPumpTargetDevNeg[i]
        )

    model.cRtPumpTargetTracking = pyo.Constraint(model.s_gen_pump, rule=rt_pump_target_rule)


def edit_objective_terms(model, rt, terms):
    """Turn the cost-minimising objective into a deviation-minimising one.

    RT drops the operating cost and the storage value -- the day-ahead run already
    settled both -- and prices deviations from the day-ahead schedule instead. Load
    shedding stays: otherwise the solver could bypass the tracked balancing channels
    by dropping demand instead of redispatching generation, storage or flexible load.
    """
    del terms["operating_cost"]
    del terms["storage_value"]
    terms["rt_deviation"] = deviation_cost(model, rt)


def deviation_cost(model, rt):
    """The RT deviation penalties added to the objective."""
    cost = sum(
        model.p_rt_deviation_price_gen_pos[i] * model.varRtTargetDevPos[i]
        + model.p_rt_deviation_price_gen_neg[i] * model.varRtTargetDevNeg[i]
        for i in model.s_gen
        if int(i) in rt.ctx.target_gen_indices
    )
    cost += sum(
        model.p_rt_deviation_price_flex_pos[j] * model.varRtFlexLoadTargetDevPos[j]
        + model.p_rt_deviation_price_flex_neg[j] * model.varRtFlexLoadTargetDevNeg[j]
        for j in model.s_load_flex
        if int(j) in rt.ctx.consumer_target_indices
    )
    cost += sum(
        model.p_rt_deviation_price_pump_pos[i] * model.varRtPumpTargetDevPos[i]
        + model.p_rt_deviation_price_pump_neg[i] * model.varRtPumpTargetDevNeg[i]
        for i in model.s_gen_pump
        if int(i) in rt.ctx.pump_target_indices
    )
    cost += sum(
        rt.ctx.rules[int(r)].balancing_penalty * (model.varRtRuleDevPos[r] + model.varRtRuleDevNeg[r])
        for r in model.s_rt_rule_penalised
    )
    return cost


def _rule_sets(rt):
    rules = rt.ctx.rules
    return (
        sorted(rules),
        sorted(r for r, t in rules.items() if t.balancing_penalty > 0.0),
        sorted(r for r, t in rules.items() if t.lock_share is not None),
        sorted(r for r, t in rules.items() if t.width_forward is not None),
        sorted(r for r, t in rules.items() if t.width_backward is not None),
    )


def add_rule_params(model, rt):
    """The inter-area rules RT holds to their DA summed flow, and that flow (MW) this timestep."""
    tracked, penalised, locked, lock_ub, lock_lb = _rule_sets(rt)
    model.s_rt_rule = pyo.Set(initialize=tracked, ordered=True)
    model.s_rt_rule_penalised = pyo.Set(initialize=penalised, ordered=True)
    model.s_rt_rule_locked = pyo.Set(initialize=locked, ordered=True)
    # The locked rules with a bound on that side: an infinite NTC leaves a side open.
    model.s_rt_rule_lock_ub = pyo.Set(initialize=lock_ub, ordered=True)
    model.s_rt_rule_lock_lb = pyo.Set(initialize=lock_lb, ordered=True)
    model.p_rt_rule_da_flow = pyo.Param(model.s_rt_rule, within=pyo.Reals, default=0, mutable=True)
    model.p_rt_rule_lock_ub = pyo.Param(model.s_rt_rule_lock_ub, within=pyo.Reals, default=0, mutable=True)
    model.p_rt_rule_lock_lb = pyo.Param(model.s_rt_rule_lock_lb, within=pyo.Reals, default=0, mutable=True)


def add_rule_deviation_vars(model):
    """Deviations of a penalised rule's summed flow from DA, above and below."""
    model.varRtRuleDevPos = pyo.Var(model.s_rt_rule_penalised, within=pyo.NonNegativeReals)
    model.varRtRuleDevNeg = pyo.Var(model.s_rt_rule_penalised, within=pyo.NonNegativeReals)


def rule_transfer(model, row, h=0):
    """The summed flow of an inter-area rule, area_from -> area_to: the NTC constraint's expression."""
    return (
        sum(model.varAcBranchFlow[b, h] for b in row["ac_pos"])
        - sum(model.varAcBranchFlow[b, h] for b in row["ac_neg"])
        + sum(model.varDcBranchFlow[b, h] for b in row["dc_pos"])
        - sum(model.varDcBranchFlow[b, h] for b in row["dc_neg"])
    )


def add_rule_flow_tracking(model, grid, ctx, rt):
    """Constraint: deviation of a penalised rule's summed flow from its DA summed flow."""

    def rule(model, r):
        row = ctx.inter_area_rules_rows[int(r)]
        return (
            rule_transfer(model, row) - model.p_rt_rule_da_flow[r]
            == model.varRtRuleDevPos[r] - model.varRtRuleDevNeg[r]
        )

    model.cRtRuleFlowTracking = pyo.Constraint(model.s_rt_rule_penalised, rule=rule)


def add_rule_flow_lock(model, grid, ctx, rt):
    """Constraint: a locked rule's summed flow within DA - width_backward and DA + width_forward.

    The bounds are parameters, written each timestep by
    :func:`~powergama.lp.rt.data.update_rule_flows`; the widths are (1 - lock_share) x NTC.
    """

    def upper(model, r):
        return rule_transfer(model, ctx.inter_area_rules_rows[int(r)]) <= model.p_rt_rule_lock_ub[r]

    def lower(model, r):
        return rule_transfer(model, ctx.inter_area_rules_rows[int(r)]) >= model.p_rt_rule_lock_lb[r]

    model.cRtRuleLockForward = pyo.Constraint(model.s_rt_rule_lock_ub, rule=upper)
    model.cRtRuleLockBackward = pyo.Constraint(model.s_rt_rule_lock_lb, rule=lower)
