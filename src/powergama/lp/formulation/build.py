"""The LP over a window of H timesteps: single-period mode is H=1.

The model is built once per window length and then *mutated in place* for every
window by :mod:`.data`, which rewrites the mutable Params rather than rebuilding.
Everything that varies over time is indexed by hour ``h`` in the window, ``s_h``.

The window's first hour depends on the simulation state through Params --
``p_storage_prev``, ``p_flex_storage_prev``, and the ramp limits folded into the
generator bounds ``p_gen_pmin``/``p_gen_pmax`` -- exactly as the single-period
model always did. Later hours are coupled to the hour before through constraints
that only exist for ``h >= 1`` (the storage balances, availability from storage,
pump room, ramp limits), so a window of one timestep has none of them and is the
single-period LP.

Invariant for every rule below, enforced by tests/test_model_invariants.py:
Pyomo components (p_*, s_*, var*, c*, OBJ) are read off the rule's `model`
argument, never off anything else; grid data, configuration and the precomputed
lookups come from `grid`, `cfg` and `ctx`, closed over at build time.

Declaration order matters: Pyomo writes components to the LP in the order they
are added, so the sequence in build_model() is not arbitrary. With H=1 it gives
the same rows and columns in the same order as the single-period model before
the merge (#42), which keeps those results identical. Components that only exist
for H>1 are declared after the per-hour ones, and extensions
(:mod:`..extension`) declare theirs after the core's, in each phase.
"""

import numpy as np
import pyomo.environ as pyo

from ... import constants as const


def build_model(model, grid, ctx, cfg, H, extensions=()):
    """Construct the whole model for a window of *H* timesteps, in declaration order.

    Returns the objective per hour, after the extensions have edited its terms: a list
    of H pairs (term names, expression for the sum of the terms).
    """
    create_sets_and_parameters(model, grid, ctx, cfg, H)
    for ext in extensions:
        ext.add_parameters(model, grid, ctx)
    create_variables(model, grid, ctx, cfg)
    for ext in extensions:
        ext.add_variables(model, grid, ctx)
    objective_by_hour = create_objective(model, grid, ctx, cfg, H, extensions)
    powerbalance_rhs = get_powerbalance_rhs(model, ctx)
    create_constraint_powerflow_limit(model, grid, ctx, cfg)
    create_constraint_inter_area_ntc(model, grid, ctx, cfg)
    create_constraint_powerloss(model, grid, ctx, cfg)
    create_constraint_generator_output(model, grid, ctx, cfg)
    create_constraint_generator_pump(model, grid, ctx, cfg)
    create_constraint_load_flex(model, grid, ctx, cfg)
    create_constraint_storage(model, grid, ctx, cfg)
    create_constraint_powerbalance(model, grid, ctx, cfg, powerbalance_rhs)
    create_constraint_powerflow_equation(model, grid, ctx, cfg)
    create_constraint_window_coupling(model, grid, ctx, cfg)
    for ext in extensions:
        ext.add_constraints(model, grid, ctx)
    return objective_by_hour


def ramped_generators(ctx, ramp):
    """Generators with a ramp limit in *ramp* (ctx.ramp_up_pu or ctx.ramp_down_pu)."""
    if ramp is None:
        return []
    return [
        i for i in ctx.gen_ids if not ctx.disable_nuclear_ramp_profile_conflict[int(i)] and not np.isnan(float(ramp[i]))
    ]


def create_sets_and_parameters(model, grid, ctx, cfg, H):
    """Create pyomo model sets"""
    model.s_node = pyo.Set(ordered=True, initialize=grid.node["id"].tolist())
    model.s_branch_ac = pyo.Set(ordered=True, initialize=grid.branch.index.tolist())
    model.s_branch_dc = pyo.Set(ordered=True, initialize=grid.dcbranch.index.tolist())
    model.s_gen = pyo.Set(ordered=True, initialize=grid.generator.index.tolist())
    model.s_gen_pump = pyo.Set(ordered=True, initialize=grid.getIdxGeneratorsWithPumping())
    model.s_gen_storage = pyo.Set(ordered=True, initialize=grid.getIdxGeneratorsWithStorage())
    model.s_load = pyo.Set(ordered=True, initialize=grid.consumer.index.tolist())
    model.s_load_flex = pyo.Set(ordered=True, initialize=grid.getIdxConsumersWithFlexibleLoad())
    model.s_area = pyo.Set(ordered=True, initialize=grid.getAllAreas())
    # Hours in the window, and the pairs (generator, hour) whose ramp limit to the
    # hour before is a constraint: every hour but the first, which applies the limit
    # through the generator bounds instead.
    model.s_h = pyo.Set(ordered=True, initialize=range(H))
    model.s_ramp_up = pyo.Set(
        dimen=2, ordered=True, initialize=[(i, h) for i in ramped_generators(ctx, ctx.ramp_up_pu) for h in range(1, H)]
    )
    model.s_ramp_dn = pyo.Set(
        dimen=2,
        ordered=True,
        initialize=[(i, h) for i in ramped_generators(ctx, ctx.ramp_down_pu) for h in range(1, H)],
    )

    # Mutable parameters
    # Quantities that change from timestep to the next:
    model.p_gen_pmin = pyo.Param(
        model.s_gen,
        model.s_h,
        within=pyo.Reals,
        default=0,
        mutable=True,
        initialize={(i, h): v for i, v in enumerate(grid.generator["pmin"].values) for h in range(H)},
    )
    model.p_gen_pmax = pyo.Param(
        model.s_gen,
        model.s_h,
        within=pyo.Reals,
        default=0,
        mutable=True,
        initialize={(i, h): v for i, v in enumerate(grid.generator["pmax"].values) for h in range(H)},
    )
    # Per-MWh operating cost of every generator, storage generators included, and
    # per-MWh cost of pumping. Neither depends on the storage level: the value of
    # stored energy enters only through the terminal storage value, p_storage_value.
    model.p_gen_cost = pyo.Param(
        model.s_gen,
        within=pyo.Reals,
        default=0,
        mutable=True,
        initialize=grid.generator["fuelcost"].values,
    )
    model.p_genpump_cost = pyo.Param(
        model.s_gen,
        within=pyo.Reals,
        default=0,
        mutable=True,
        initialize={g: grid.generator.loc[g, "pump_deadband"] for g in model.s_gen_pump},
    )
    # Storage level at the start of the window (MWh), inflow in each hour (MW), and
    # the value of energy left in storage at the window's end (EUR/MWh).
    model.p_storage_prev = pyo.Param(model.s_gen_storage, within=pyo.Reals, default=0, mutable=True)
    model.p_storage_inflow = pyo.Param(model.s_gen_storage, model.s_h, within=pyo.Reals, default=0, mutable=True)
    model.p_storage_value = pyo.Param(model.s_gen_storage, within=pyo.Reals, default=0, mutable=True)
    model.p_pump_pmax = pyo.Param(
        model.s_gen_pump,
        model.s_h,
        within=pyo.NonNegativeReals,
        mutable=True,
        initialize={(g, h): grid.generator.loc[g, "pump_cap"] for g in model.s_gen_pump for h in range(H)},
    )
    model.p_demand = pyo.Param(model.s_load, model.s_h, within=pyo.Reals, default=0, mutable=True)
    # Value of energy delivered to a flexible-load reservoir (EUR/MWh), and its level
    # at the start of the window (MWh).
    model.p_loadflex_cost = pyo.Param(model.s_load_flex, model.s_h, within=pyo.Reals, default=0, mutable=True)
    model.p_flex_storage_prev = pyo.Param(model.s_load_flex, within=pyo.Reals, default=0, mutable=True)
    if cfg.lossmethod == 2:
        # for storing power losses until next timestep
        model.p_branch_ac_power_loss = pyo.Param(
            model.s_branch_ac, model.s_h, within=pyo.Reals, default=0, mutable=True
        )
        model.p_branch_dc_power_loss = pyo.Param(
            model.s_branch_dc, model.s_h, within=pyo.Reals, default=0, mutable=True
        )
    elif cfg.lossmethod == 1:
        # for storing power flow from previous timestep
        model.p_branch_ac_powerflow12 = pyo.Param(
            model.s_branch_ac, model.s_h, within=pyo.Reals, initialize=0, mutable=True
        )
        model.p_branch_ac_powerflow21 = pyo.Param(
            model.s_branch_ac, model.s_h, within=pyo.Reals, initialize=0, mutable=True
        )
        model.p_branch_dc_powerflow12 = pyo.Param(
            model.s_branch_dc, model.s_h, within=pyo.Reals, initialize=0, mutable=True
        )
        model.p_branch_dc_powerflow21 = pyo.Param(
            model.s_branch_dc, model.s_h, within=pyo.Reals, initialize=0, mutable=True
        )
    # Ramp limits between consecutive hours of the window, in MW. Lifted to twice the
    # installed capacity at midnight for generators with a daily reset, so that one
    # built model serves windows starting at any hour.
    model.p_ramp_up_mw = pyo.Param(model.s_ramp_up, within=pyo.Reals, default=0, mutable=True)
    model.p_ramp_dn_mw = pyo.Param(model.s_ramp_dn, within=pyo.Reals, default=0, mutable=True)


def create_variables(model, grid, ctx, cfg):
    """Create pyomo model variables"""
    model.varAcBranchFlow = pyo.Var(model.s_branch_ac, model.s_h, within=pyo.Reals)
    model.varDcBranchFlow = pyo.Var(model.s_branch_dc, model.s_h, within=pyo.Reals)
    if cfg.lossmethod == 1:
        # Bounds sit 1% above capacity so they never coincide with cMaxFlowAc/cMaxFlowDc. At exactly
        # the capacity, a branch at its rating also has flow12 (or flow21) at its bound, the solver
        # prices congestion on that bound, and the constraint dual reported as sensitivity is zero.
        # The margin is kept small because these bounds are also what limits two-way flow (#29).
        # Ref issue: https://github.com/powergama/powergama/issues/39
        def maxflow(model, j, h):
            return (0, 1.01 * grid.branch.loc[j, "capacity"])

        def maxflow_dc(model, j, h):
            return (0, 1.01 * grid.dcbranch.loc[j, "capacity"])

        # Ref issue: https://github.com/powergama/powergama/issues/29
        # Adding bounds on 12 and 21 flows reduces the problem with simultaneously large 12 and 21 flows
        # that are not physical (but hard to avoid in circumstances when branch loss is actually beneficial
        # for the optimisation). However, it doesn't eliminate the problem.
        model.varAcBranchFlow12 = pyo.Var(model.s_branch_ac, model.s_h, within=pyo.NonNegativeReals, bounds=maxflow)
        model.varAcBranchFlow21 = pyo.Var(model.s_branch_ac, model.s_h, within=pyo.NonNegativeReals, bounds=maxflow)
        model.varDcBranchFlow12 = pyo.Var(model.s_branch_dc, model.s_h, within=pyo.NonNegativeReals, bounds=maxflow_dc)
        model.varDcBranchFlow21 = pyo.Var(model.s_branch_dc, model.s_h, within=pyo.NonNegativeReals, bounds=maxflow_dc)
        model.varLossAc12 = pyo.Var(model.s_branch_ac, model.s_h, within=pyo.NonNegativeReals)
        model.varLossAc21 = pyo.Var(model.s_branch_ac, model.s_h, within=pyo.NonNegativeReals)
        model.varLossDc12 = pyo.Var(model.s_branch_dc, model.s_h, within=pyo.NonNegativeReals)
        model.varLossDc21 = pyo.Var(model.s_branch_dc, model.s_h, within=pyo.NonNegativeReals)
    model.varGeneration = pyo.Var(model.s_gen, model.s_h, within=pyo.NonNegativeReals)
    model.varPump = pyo.Var(model.s_gen_pump, model.s_h, within=pyo.NonNegativeReals)
    model.varFlexLoad = pyo.Var(model.s_load_flex, model.s_h, within=pyo.NonNegativeReals)
    model.varLoadShed = pyo.Var(model.s_load, model.s_h, within=pyo.NonNegativeReals)
    model.varDumpLoad = pyo.Var(model.s_load, model.s_h, within=pyo.NonNegativeReals)
    model.varStorage = pyo.Var(
        model.s_gen_storage,
        model.s_h,
        within=pyo.NonNegativeReals,
        bounds=lambda m, i, h: (0, grid.generator.loc[i, "storage_cap"]),
    )
    model.varSpill = pyo.Var(
        model.s_gen_storage, model.s_h, within=pyo.NonNegativeReals, bounds=lambda m, i, h: (0, ctx.spill_cap_mw[i])
    )
    model.varStorageOverflow = pyo.Var(model.s_gen_storage, model.s_h, within=pyo.NonNegativeReals)
    # Reservoir level at the end of the hour. Reals, not NonNegativeReals: the floor
    # is skipped for a consumer whose charge limit cannot keep up with its drain.
    model.varFlexStorage = pyo.Var(
        model.s_load_flex,
        model.s_h,
        within=pyo.Reals,
        bounds=lambda m, j, h: (0.0 if ctx.flex_has_floor[j] else None, ctx.flex_storage_cap[j]),
    )
    model.varVoltageAngle = pyo.Var(model.s_node, model.s_h, within=pyo.Reals, initialize=0.0)
    # Shortfall below the ramp-down limit after the window's first hour. The first
    # hour drops the limit where the generator cannot meet it (see data.py); later
    # hours' availability depends on the solve, so the limit is softened with a slack
    # priced like load shedding, and a window can never be infeasible.
    model.varRampSlack = pyo.Var(model.s_ramp_dn, within=pyo.NonNegativeReals)


def create_constraint_powerflow_limit(model, grid, ctx, cfg):
    """Constraint: Power flow limit"""

    def maxflowAc_rule(model, j, h):
        cap = grid.branch.loc[j, "capacity"]
        if not np.isinf(cap):
            expr = pyo.inequality(-cap, model.varAcBranchFlow[j, h], cap)
        else:
            expr = pyo.Constraint.Skip
        return expr

    def maxflowDc_rule(model, j, h):
        cap = grid.dcbranch.loc[j, "capacity"]
        if not np.isinf(cap):
            expr = pyo.inequality(-cap, model.varDcBranchFlow[j, h], cap)
        else:
            expr = pyo.Constraint.Skip
        return expr

    model.cMaxFlowAc = pyo.Constraint(model.s_branch_ac, model.s_h, rule=maxflowAc_rule)
    model.cMaxFlowDc = pyo.Constraint(model.s_branch_dc, model.s_h, rule=maxflowDc_rule)


def create_constraint_inter_area_ntc(model, grid, ctx, cfg):
    """Constraint: Directional NTC limits between areas.

    For each (area_from, area_to) pair, limit the net transfer
    from area_from -> area_to:
        -ntc_backward <= net_transfer <= ntc_forward
    """

    if ctx.inter_area_ntc is None or ctx.inter_area_ntc.shape[0] == 0:
        return

    model.s_inter_area_ntc = pyo.RangeSet(0, len(ctx.inter_area_ntc) - 1)

    def net_transfer(model, row, h):
        return (
            sum(model.varAcBranchFlow[b, h] for b in row["ac_pos"])
            - sum(model.varAcBranchFlow[b, h] for b in row["ac_neg"])
            + sum(model.varDcBranchFlow[b, h] for b in row["dc_pos"])
            - sum(model.varDcBranchFlow[b, h] for b in row["dc_neg"])
        )

    def ntc_forward_rule(model, cidx, h):
        row = ctx.inter_area_ntc_rows[int(cidx)]
        cap = row["ntc_forward"]
        if not np.isfinite(cap):
            return pyo.Constraint.Skip
        if not (row["ac_pos"] or row["ac_neg"] or row["dc_pos"] or row["dc_neg"]):
            return pyo.Constraint.Skip
        return net_transfer(model, row, h) <= cap

    def ntc_backward_rule(model, cidx, h):
        row = ctx.inter_area_ntc_rows[int(cidx)]
        cap = row["ntc_backward"]
        if not np.isfinite(cap):
            return pyo.Constraint.Skip
        if not (row["ac_pos"] or row["ac_neg"] or row["dc_pos"] or row["dc_neg"]):
            return pyo.Constraint.Skip
        return net_transfer(model, row, h) >= -cap

    model.cInterAreaNtcForward = pyo.Constraint(model.s_inter_area_ntc, model.s_h, rule=ntc_forward_rule)
    model.cInterAreaNtcBackward = pyo.Constraint(model.s_inter_area_ntc, model.s_h, rule=ntc_backward_rule)


def _powerloss_rules1(model, grid, ctx, cfg):
    """Power loss proportional to flow, proportionality factor given by previous timestep

    P_loss = alpha P
    alpha = P_loss0/P0 (straight line from origo to operating point)
            r_pu = grid.dcbranch.loc[b, "resistance"]
            p_pu = model.varDcBranchFlow[b] / const.baseMVA
            loss_pu = r_pu * p_pu**2
            lossMVA = loss_pu * const.baseMVA * dclossmultiplier

    NOTE:
    With losses included, flow is split, flow=flow12-flow21. But there is no constraint
    saying flow can be only one direction. In some cases, the optimisation may find it
    beneficial to have increases losses by having large values for both flow12 and flow21
    (with flow still below capacity). High losses on a branch may be beneficial by
    allowing more flow on another line (under certain combinations of branch impedances
    and capacities.)
    The penalty_twoway_flow parameter may be used to discourange simultaneous (unphysical)
    twoway flow by adding a penalty in the objective function. This is not generally advices
    as it may have other unwanted effects (not sufficiently tested.)
    """

    def make_lossAc_rule12(br):
        def rule(model, j, h):
            flow_abs = model.p_branch_ac_powerflow12[j, h] + model.p_branch_ac_powerflow21[j, h]
            alpha = br.loc[j, "resistance"] * flow_abs / const.baseMVA
            expr = model.varLossAc12[j, h] == alpha * model.varAcBranchFlow12[j, h]
            return expr

        model.cLossAc12 = pyo.Constraint(model.s_branch_ac, model.s_h, rule=rule)

    def make_lossAc_rule21(br):
        def rule(model, j, h):
            flow_abs = model.p_branch_ac_powerflow12[j, h] + model.p_branch_ac_powerflow21[j, h]
            alpha = br.loc[j, "resistance"] * flow_abs / const.baseMVA
            expr = model.varLossAc21[j, h] == alpha * model.varAcBranchFlow21[j, h]
            return expr

        model.cLossAc21 = pyo.Constraint(model.s_branch_ac, model.s_h, rule=rule)

    def make_lossDc_rule12(br):
        def rule(model, j, h):
            flow_abs = model.p_branch_dc_powerflow12[j, h] + model.p_branch_dc_powerflow21[j, h]
            alpha = br.loc[j, "resistance"] * flow_abs / const.baseMVA
            expr = model.varLossDc12[j, h] == alpha * model.varDcBranchFlow12[j, h]
            return expr

        model.cLossDc12 = pyo.Constraint(model.s_branch_dc, model.s_h, rule=rule)

    def make_lossDc_rule21(br):
        def rule(model, j, h):
            flow_abs = model.p_branch_dc_powerflow12[j, h] + model.p_branch_dc_powerflow21[j, h]
            alpha = br.loc[j, "resistance"] * flow_abs / const.baseMVA
            expr = model.varLossDc21[j, h] == alpha * model.varDcBranchFlow21[j, h]
            return expr

        model.cLossDc21 = pyo.Constraint(model.s_branch_dc, model.s_h, rule=rule)

    make_lossAc_rule12(grid.branch)
    make_lossAc_rule21(grid.branch)
    make_lossDc_rule12(grid.dcbranch)
    make_lossDc_rule21(grid.dcbranch)


def create_constraint_powerloss(model, grid, ctx, cfg):
    """Constraint: flow = flow12-flow21 & powerloss"""
    if cfg.lossmethod == 1:

        def flowAc_rule(model, j, h):
            expr = model.varAcBranchFlow[j, h] == model.varAcBranchFlow12[j, h] - model.varAcBranchFlow21[j, h]
            return expr

        def flowDc_rule(model, j, h):
            expr = model.varDcBranchFlow[j, h] == model.varDcBranchFlow12[j, h] - model.varDcBranchFlow21[j, h]
            return expr

        model.cFlowAc = pyo.Constraint(model.s_branch_ac, model.s_h, rule=flowAc_rule)
        model.cFlowDc = pyo.Constraint(model.s_branch_dc, model.s_h, rule=flowDc_rule)

    # 1b Losses vs flow
    if cfg.lossmethod == 1:
        _powerloss_rules1(model, grid, ctx, cfg)


def create_constraint_generator_output(model, grid, ctx, cfg):
    """Constraint: Generator output limit"""

    # Generator output constraint is not necessary, as lower and upper
    # bounds are set for each timestep in _update_progress. Should not
    # be specified as constraint with with pmax as limit, since e.g.
    # PV may have higher production than generator rating.

    # HGS: Doing it anyway, cf Espen Bødal and Martin Kristiansen
    # TODO: Check that there are no problems with this.

    def genMaxLimit_rule(model, i, h):
        return model.varGeneration[i, h] <= model.p_gen_pmax[i, h]

    def genMinLimit_rule(model, i, h):
        return model.varGeneration[i, h] >= model.p_gen_pmin[i, h]

    model.cGenMaxLimit = pyo.Constraint(model.s_gen, model.s_h, rule=genMaxLimit_rule)
    model.cGenMinLimit = pyo.Constraint(model.s_gen, model.s_h, rule=genMinLimit_rule)


def create_constraint_generator_pump(model, grid, ctx, cfg):
    """Constraint: Pump output limit (respects both hardware cap and remaining reservoir space)."""

    def pump_rule(model, g, h):
        expr = model.varPump[g, h] <= model.p_pump_pmax[g, h]
        return expr

    model.cPump = pyo.Constraint(model.s_gen_pump, model.s_h, rule=pump_rule)


def create_constraint_load_flex(model, grid, ctx, cfg):
    """Constraint: Flexible load limit"""

    def flexload_rule(model, i, h):
        flexLoadMax = (
            grid.consumer.loc[i, "demand_avg"]
            * grid.consumer.loc[i, "flex_fraction"]
            / grid.consumer.loc[i, "flex_on_off"]
        )
        expr = model.varFlexLoad[i, h] <= flexLoadMax
        return expr

    model.cFlexload = pyo.Constraint(model.s_load_flex, model.s_h, rule=flexload_rule)

    # Reservoir continuity: level = level_prev + (charge - constant drain) * dt,
    # with the level bounded to [0, capacity] by the variable's bounds.
    def flexstorage_rule(model, i, h):
        prev = model.p_flex_storage_prev[i] if h == 0 else model.varFlexStorage[i, h - 1]
        return model.varFlexStorage[i, h] == prev + grid.timeDelta * (model.varFlexLoad[i, h] - ctx.flex_drain_mw[i])

    model.cFlexStorCont = pyo.Constraint(model.s_load_flex, model.s_h, rule=flexstorage_rule)


def create_constraint_storage(model, grid, ctx, cfg):
    """Constraint: storage continuity over each hour.

    level = level_prev + (inflow - output + pumped*efficiency - spill - overflow) * dt

    The level is bounded to [0, storage_cap] and spill to spill_cap_frac of installed
    capacity. Overflow is what fits in neither, priced like load shedding so the
    window stays feasible.
    """
    dt = grid.timeDelta

    def storage_rule(model, i, h):
        pumped = 0
        if i in model.s_gen_pump:
            pumped = model.varPump[i, h] * grid.generator.loc[i, "pump_efficiency"]
        prev = model.p_storage_prev[i] if h == 0 else model.varStorage[i, h - 1]
        return model.varStorage[i, h] == prev + dt * (
            model.p_storage_inflow[i, h]
            - model.varGeneration[i, h]
            + pumped
            - model.varSpill[i, h]
            - model.varStorageOverflow[i, h]
        )

    model.cStorCont = pyo.Constraint(model.s_gen_storage, model.s_h, rule=storage_rule)


def create_constraint_powerbalance(model, grid, ctx, cfg, powerbalance_rhs):
    """ConstraintPower balance (power flow equation)  (Pnode = B theta)"""
    lossmethod = cfg.lossmethod

    def powerbalance_rule(model, n, h):
        lhs = 0
        for g in ctx.generators_at_node[n]:
            # this is a generator connected to node n
            lhs += model.varGeneration[g, h]
            if g in model.s_gen_pump:
                lhs -= model.varPump[g, h]
        for lod in ctx.loads_at_node[n]:
            lhs -= model.p_demand[lod, h]
            lhs += model.varLoadShed[lod, h]
            lhs -= model.varDumpLoad[lod, h]
            if lod in model.s_load_flex:
                lhs -= model.varFlexLoad[lod, h]
        for b in ctx.dcbranch_to_node[n]:
            lhs += model.varDcBranchFlow[b, h]
            if lossmethod == 1:
                lhs -= model.varLossDc12[b, h]
            elif lossmethod == 2:
                lhs -= model.p_branch_dc_power_loss[b, h] / 2
        for b in ctx.dcbranch_from_node[n]:
            lhs += -model.varDcBranchFlow[b, h]
            if lossmethod == 1:
                lhs -= model.varLossDc21[b, h]
            elif lossmethod == 2:
                lhs -= model.p_branch_dc_power_loss[b, h] / 2
        if lossmethod == 1:
            # we define flow12 as flow leaving node 1, and flow21 as flow leaving node 2
            # so subtract loss at "receiving" node but not at "sending" node
            # i.e. loss12 at to-node and loss21 at from-node
            for b in ctx.branch_to_node[n]:
                lhs += -model.varLossAc12[b, h]
            for b in ctx.branch_from_node[n]:
                lhs += -model.varLossAc21[b, h]
        elif lossmethod == 2:
            # add ac branch losses as load, equally split between sending and receiving node
            for b in ctx.branch_to_node[n]:
                # positive sign for flow into node
                lhs -= model.p_branch_ac_power_loss[b, h] / 2
            for b in ctx.branch_from_node[n]:
                lhs -= model.p_branch_ac_power_loss[b, h] / 2

        lhs = lhs / const.baseMVA

        rhs = powerbalance_rhs[n, h]

        expr = lhs == rhs
        # Skip constraint if it is trivial (otherwise run-time error)
        # TODO: Check if this is safe
        if (type(expr) is bool) and (expr is True):
            expr = pyo.Constraint.Skip
        return expr

    model.cPowerbalance = pyo.Constraint(model.s_node, model.s_h, rule=powerbalance_rule)


def create_constraint_powerflow_equation(model, grid, ctx, cfg):
    """Constraint: Power balance (power flow vs voltage angle)"""
    # Node and branch ids are strings/labels but the DA matrix needs positions, so
    # map them once here instead of inside the rule: the branch lookup was a linear
    # scan per branch, and the node lookup rebuilt the entire node list for every
    # non-zero of every row.
    node_list = list(model.s_node)
    branch_position = {b: i for i, b in enumerate(model.s_branch_ac)}

    # 1.
    def flowangle_rule(model, b, h):
        lhs = model.varAcBranchFlow[b, h]
        lhs = lhs / const.baseMVA
        rhs = 0
        row = ctx.DA[branch_position[b]]
        for i in range(len(row.indices)):
            n2 = node_list[row.indices[i]]
            rhs += row.data[i] * model.varVoltageAngle[n2, h] * const.baseAngle
        expr = lhs == rhs
        return expr

    model.cFlowAngle = pyo.Constraint(model.s_branch_ac, model.s_h, rule=flowangle_rule)

    # 2. Reference voltag angle)
    def referenceNode_rule(model, n, h):
        if n in ctx.refnodes:
            expr = model.varVoltageAngle[n, h] == 0
        else:
            expr = pyo.Constraint.Skip
        return expr

    model.cReferenceNode = pyo.Constraint(model.s_node, model.s_h, rule=referenceNode_rule)


def create_constraint_window_coupling(model, grid, ctx, cfg):
    """Constraints between consecutive hours of a window: none when H=1.

    The window's first hour gets these limits through its Params instead -- the
    generator bounds and the pump limit are computed from the simulation state before
    the solve (data.py). Later hours depend on the hour before, which is solved
    together with them, so the limits are constraints on its variables.
    """
    dt = grid.timeDelta

    # Generation from storage is limited by what the storage held at the end of the
    # hour before, plus inflow; the storage balance alone would let pumping in the
    # same hour pay for it.
    def storage_available_rule(model, i, h):
        if h == 0:
            return pyo.Constraint.Skip
        return model.varGeneration[i, h] <= model.p_storage_inflow[i, h] + model.varStorage[i, h - 1] / dt

    model.cGenUbStorAvail = pyo.Constraint(model.s_gen_storage, model.s_h, rule=storage_available_rule)

    # Pumping is limited by the room left in the storage at the end of the hour before.
    def pump_room_rule(model, i, h):
        if h == 0:
            return pyo.Constraint.Skip
        eff = grid.generator.loc[i, "pump_efficiency"]
        if eff <= 0:
            return model.varPump[i, h] == 0
        return model.varPump[i, h] * eff * dt <= grid.generator.loc[i, "storage_cap"] - model.varStorage[i, h - 1]

    model.cPumpRoom = pyo.Constraint(model.s_gen_pump, model.s_h, rule=pump_room_rule)

    def ramp_up_rule(model, i, h):
        return model.varGeneration[i, h] - model.varGeneration[i, h - 1] <= model.p_ramp_up_mw[i, h]

    def ramp_dn_rule(model, i, h):
        return (
            model.varGeneration[i, h - 1] - model.varGeneration[i, h] - model.varRampSlack[i, h]
            <= model.p_ramp_dn_mw[i, h]
        )

    model.cRampUp = pyo.Constraint(model.s_ramp_up, rule=ramp_up_rule)
    model.cRampDn = pyo.Constraint(model.s_ramp_dn, rule=ramp_dn_rule)


def objective_terms(model, dt, h, last):
    """The objective terms for hour *h*, by name: an extension may add to, drop or replace them.

    The objective is their sum over the window's hours, in insertion order. The last
    hour also carries the value of the energy left in storage when the window ends.

    Units: every term is a cost rate in EUR/h (MW times EUR/MWh), whatever the
    timestep length *dt*, so that the duals are prices in EUR/MWh. The value of the
    energy left in storage is an amount in EUR, which is why it is divided by dt.
    """
    terms = {}
    # Cost of generation and pumping, less the value of serving flexible load.
    terms["operating_cost"] = (
        sum(model.varGeneration[i, h] * model.p_gen_cost[i] for i in model.s_gen)
        + sum(model.varPump[i, h] * model.p_genpump_cost[i] for i in model.s_gen_pump)
        - sum(model.varFlexLoad[i, h] * model.p_loadflex_cost[i, h] for i in model.s_load_flex)
    )
    if last:
        # Value of the energy in storage at the end of the window. This is the only
        # place storage is valued: drawing 1 MWh from storage costs its storage value
        # here and nothing per MWh generated, and spilling costs the same.
        terms["storage_value"] = (
            -sum(model.p_storage_value[i] * model.varStorage[i, h] for i in model.s_gen_storage) / dt
        )
    # Penalty slack: keeps every window feasible.
    terms["load_shedding"] = (
        sum(model.varLoadShed[i, h] * const.loadshedcost for i in model.s_load)
        + sum(model.varDumpLoad[i, h] * const.loadshedcost for i in model.s_load)
        + sum(model.varStorageOverflow[i, h] * const.loadshedcost for i in model.s_gen_storage)
    )
    # Spill only when the reservoir is full, even where the storage value is zero.
    terms["spill"] = sum(model.varSpill[i, h] * const.spillcost for i in model.s_gen_storage)
    if h > 0:
        terms["ramp_slack"] = sum(
            model.varRampSlack[i, hh] * const.loadshedcost for i, hh in model.s_ramp_dn if hh == h
        )
    return terms


def create_objective(model, grid, ctx, cfg, H, extensions=()):
    """Create pyomo model objective function.

    Returns, for each hour, the names of its terms and the expression for their sum:
    the solve loop evaluates the latter to report the objective hour by hour.
    """
    objective_by_hour = []

    def cost_rule(model):
        objective_by_hour.clear()
        for h in range(H):
            terms = objective_terms(model, grid.timeDelta, h, last=h == H - 1)
            if cfg.penalty_twoway_flow > 0:
                # Discourage simultaneous flow in both directions on a branch with
                # losses. Not particularly elegant, and not normally to be used.
                # REF: Issue https://github.com/powergama/powergama/issues/29
                terms["twoway_flow_penalty"] = cfg.penalty_twoway_flow * sum(
                    model.varAcBranchFlow12[i, h] + model.varAcBranchFlow21[i, h] for i in model.s_branch_ac
                )
            for ext in extensions:
                ext.edit_objective_terms(model, terms, h)
            objective_by_hour.append((tuple(terms), sum(terms.values())))
        if H == 1:
            return objective_by_hour[0][1]
        return sum(expr for _, expr in objective_by_hour)

    if cfg.penalty_twoway_flow > 0:
        print("Adding a cost to penalise simultaneous branch flow in both directions - ")
    model.OBJ = pyo.Objective(rule=cost_rule, sense=pyo.minimize)
    return objective_by_hour


def get_powerbalance_rhs(model, ctx):
    """Get rhs expression in powerbalance constraint"""
    # This is taken out of the constraint creation function to speed up constraint creation.
    # Node ids are strings but Bbus needs matrix indices, so materialise the ordered set
    # once: looking a node's position up per node was O(n^2), and rebuilding the whole
    # list again for every non-zero in its row made that O(n^2) per node.
    node_list = list(model.s_node)
    rhs = dict()
    for h in model.s_h:
        for idx_node, n in enumerate(node_list):
            rhs[n, h] = 0
            row = ctx.Bbus[idx_node]
            for i in range(len(row.indices)):
                n2 = node_list[row.indices[i]]
                rhs[n, h] -= row.data[i] * model.varVoltageAngle[n2, h] * const.baseAngle
    return rhs
