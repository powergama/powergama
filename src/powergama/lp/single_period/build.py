"""The single-period formulation: one LP per timestep, solved in sequence.

The model is built once here and then *mutated in place* every timestep by
:mod:`.data`, which rewrites the mutable Params and the variable bounds rather
than rebuilding. That is the structural difference from the multi-period
formulation, whose ``data`` module instead returns a snapshot baked into a model
rebuilt per window.

Invariant for every rule below, enforced by tests/test_model_invariants.py:
Pyomo components (p_*, s_*, var*, c*, OBJ) are read off the rule's `model`
argument, never off anything else; grid data, configuration and the precomputed
lookups come from `grid`, `cfg` and `ctx`, closed over at build time.

Declaration order matters: Pyomo writes components to the LP in the order they
are added, so the sequence in build_model() is not arbitrary, and the RT seams
sit exactly where the RT declarations used to.
"""

import numpy as np
import pyomo.environ as pyo

from ... import constants as const
from ..rt import build as rt_build


def build_model(model, grid, ctx, cfg):
    """Construct the whole single-period model, in declaration order."""
    create_sets_and_parameters(model, grid, ctx, cfg)
    create_variables(model, grid, ctx, cfg)
    create_objective(model, grid, ctx, cfg)
    powerbalance_rhs = get_powerbalance_rhs(model, ctx)
    create_constraint_powerflow_limit(model, grid, ctx, cfg)
    create_constraint_inter_area_ntc(model, grid, ctx, cfg)
    create_constraint_powerloss(model, grid, ctx, cfg, penalty_twoway_flow=cfg.penalty_twoway_flow)
    create_constraint_generator_output(model, grid, ctx, cfg)
    rt_build.add_target_tracking(model, grid, ctx, cfg)
    create_constraint_generator_pump(model, grid, ctx, cfg)
    create_constraint_load_flex(model, grid, ctx, cfg)
    rt_build.add_flexload_target_tracking(model, grid, ctx, cfg)
    rt_build.add_storage_target_tracking(model, grid, ctx, cfg)
    rt_build.add_border_flow_target_tracking(model, grid, ctx, cfg)
    create_constraint_powerbalance(model, grid, ctx, cfg, powerbalance_rhs)
    create_constraint_powerflow_equation(model, grid, ctx, cfg)


def create_sets_and_parameters(model, grid, ctx, cfg):
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

    # Mutable parameters
    # Quantities that change from timestep to the next:
    model.p_gen_pmin = pyo.Param(
        model.s_gen,
        within=pyo.Reals,
        default=0,
        mutable=True,
        initialize=grid.generator["pmin"].values,
    )
    model.p_gen_pmax = pyo.Param(
        model.s_gen,
        within=pyo.Reals,
        default=0,
        mutable=True,
        initialize=grid.generator["pmax"].values,
    )
    model.p_gen_cost = pyo.Param(
        model.s_gen,
        within=pyo.Reals,
        default=0,
        mutable=True,
        initialize=grid.generator["fuelcost"].values,
    )
    model.p_genpump_cost = pyo.Param(model.s_gen, within=pyo.Reals, default=0, mutable=True)
    model.p_pump_pmax = pyo.Param(
        model.s_gen_pump,
        within=pyo.NonNegativeReals,
        mutable=True,
        initialize={g: grid.generator.loc[g, "pump_cap"] for g in model.s_gen_pump},
    )
    model.p_demand = pyo.Param(model.s_load, within=pyo.Reals, default=0, mutable=True)
    rt_build.add_params(model, cfg)
    # Written each timestep by _updateLpProblem from _flexload_storage_value. Only
    # the single-period path uses it; the multi-period path prices flexible load per
    # window in _build_day_joint_model instead.
    model.p_loadflex_cost = pyo.Param(model.s_load_flex, within=pyo.Reals, default=0, mutable=True)
    if cfg.lossmethod == 2:
        # for storing power losses until next timestep
        model.p_branch_ac_power_loss = pyo.Param(model.s_branch_ac, within=pyo.Reals, default=0, mutable=True)
        model.p_branch_dc_power_loss = pyo.Param(model.s_branch_dc, within=pyo.Reals, default=0, mutable=True)
    elif cfg.lossmethod == 1:
        # for storing power flow from previous timestep
        model.p_branch_ac_powerflow12 = pyo.Param(model.s_branch_ac, within=pyo.Reals, initialize=0, mutable=True)
        model.p_branch_ac_powerflow21 = pyo.Param(model.s_branch_ac, within=pyo.Reals, initialize=0, mutable=True)
        model.p_branch_dc_powerflow12 = pyo.Param(model.s_branch_dc, within=pyo.Reals, initialize=0, mutable=True)
        model.p_branch_dc_powerflow21 = pyo.Param(model.s_branch_dc, within=pyo.Reals, initialize=0, mutable=True)


def create_variables(model, grid, ctx, cfg):
    """Create pyomo model variables"""
    grid = grid
    model.varAcBranchFlow = pyo.Var(model.s_branch_ac, within=pyo.Reals)
    model.varDcBranchFlow = pyo.Var(model.s_branch_dc, within=pyo.Reals)
    if cfg.lossmethod == 1:
        # Bounds sit 1% above capacity so they never coincide with cMaxFlowAc/cMaxFlowDc. At exactly
        # the capacity, a branch at its rating also has flow12 (or flow21) at its bound, the solver
        # prices congestion on that bound, and the constraint dual reported as sensitivity is zero.
        # The margin is kept small because these bounds are also what limits two-way flow (#29).
        # Ref issue: https://github.com/powergama/powergama/issues/39
        def maxflow(model, j):
            return (0, 1.01 * grid.branch.loc[j, "capacity"])

        def maxflow_dc(model, j):
            return (0, 1.01 * grid.dcbranch.loc[j, "capacity"])

        # Ref issue: https://github.com/powergama/powergama/issues/29
        # Adding bounds on 12 and 21 flows reduces the problem with simultaneously large 12 and 21 flows
        # that are not physical (but hard to avoid in circumstances when branch loss is actually beneficial
        # for the optimisation). However, it doesn't eliminate the problem.
        model.varAcBranchFlow12 = pyo.Var(model.s_branch_ac, within=pyo.NonNegativeReals, bounds=maxflow)
        model.varAcBranchFlow21 = pyo.Var(model.s_branch_ac, within=pyo.NonNegativeReals, bounds=maxflow)
        model.varDcBranchFlow12 = pyo.Var(model.s_branch_dc, within=pyo.NonNegativeReals, bounds=maxflow_dc)
        model.varDcBranchFlow21 = pyo.Var(model.s_branch_dc, within=pyo.NonNegativeReals, bounds=maxflow_dc)
        model.varLossAc12 = pyo.Var(model.s_branch_ac, within=pyo.NonNegativeReals)
        model.varLossAc21 = pyo.Var(model.s_branch_ac, within=pyo.NonNegativeReals)
        model.varLossDc12 = pyo.Var(model.s_branch_dc, within=pyo.NonNegativeReals)
        model.varLossDc21 = pyo.Var(model.s_branch_dc, within=pyo.NonNegativeReals)
    model.varGeneration = pyo.Var(model.s_gen, within=pyo.NonNegativeReals)
    model.varPump = pyo.Var(model.s_gen_pump, within=pyo.NonNegativeReals)
    rt_build.add_generation_deviation_vars(model, cfg)
    model.varFlexLoad = pyo.Var(model.s_load_flex, within=pyo.NonNegativeReals)
    model.varLoadShed = pyo.Var(model.s_load, within=pyo.NonNegativeReals)
    model.varDumpLoad = pyo.Var(model.s_load, within=pyo.NonNegativeReals)
    rt_build.add_remaining_deviation_vars(model, cfg)
    model.varVoltageAngle = pyo.Var(model.s_node, within=pyo.Reals, initialize=0.0)


def create_constraint_powerflow_limit(model, grid, ctx, cfg):
    """Constraint: Power flow limit"""

    def maxflowAc_rule(model, j):
        cap = grid.branch.loc[j, "capacity"]
        if not np.isinf(cap):
            expr = pyo.inequality(-cap, model.varAcBranchFlow[j], cap)
        else:
            expr = pyo.Constraint.Skip
        return expr

    def maxflowDc_rule(model, j):
        cap = grid.dcbranch.loc[j, "capacity"]
        if not np.isinf(cap):
            expr = pyo.inequality(-cap, model.varDcBranchFlow[j], cap)
        else:
            expr = pyo.Constraint.Skip
        return expr

    model.cMaxFlowAc = pyo.Constraint(model.s_branch_ac, rule=maxflowAc_rule)
    model.cMaxFlowDc = pyo.Constraint(model.s_branch_dc, rule=maxflowDc_rule)


def create_constraint_inter_area_ntc(model, grid, ctx, cfg):
    """Constraint: Directional NTC limits between areas.

    For each (area_from, area_to) pair, limit the net transfer
    from area_from -> area_to:
        -ntc_backward <= net_transfer <= ntc_forward
    """

    if ctx.inter_area_ntc is None or ctx.inter_area_ntc.shape[0] == 0:
        return

    model.s_inter_area_ntc = pyo.RangeSet(0, len(ctx.inter_area_ntc) - 1)

    def ntc_forward_rule(model, cidx):
        row = ctx.inter_area_ntc_rows[int(cidx)]
        cap = row["ntc_forward"]
        if not np.isfinite(cap):
            return pyo.Constraint.Skip
        if not (row["ac_pos"] or row["ac_neg"] or row["dc_pos"] or row["dc_neg"]):
            return pyo.Constraint.Skip
        net_transfer = (
            sum(model.varAcBranchFlow[b] for b in row["ac_pos"])
            - sum(model.varAcBranchFlow[b] for b in row["ac_neg"])
            + sum(model.varDcBranchFlow[b] for b in row["dc_pos"])
            - sum(model.varDcBranchFlow[b] for b in row["dc_neg"])
        )
        return net_transfer <= cap

    def ntc_backward_rule(model, cidx):
        row = ctx.inter_area_ntc_rows[int(cidx)]
        cap = row["ntc_backward"]
        if not np.isfinite(cap):
            return pyo.Constraint.Skip
        if not (row["ac_pos"] or row["ac_neg"] or row["dc_pos"] or row["dc_neg"]):
            return pyo.Constraint.Skip
        net_transfer = (
            sum(model.varAcBranchFlow[b] for b in row["ac_pos"])
            - sum(model.varAcBranchFlow[b] for b in row["ac_neg"])
            + sum(model.varDcBranchFlow[b] for b in row["dc_pos"])
            - sum(model.varDcBranchFlow[b] for b in row["dc_neg"])
        )
        return net_transfer >= -cap

    model.cInterAreaNtcForward = pyo.Constraint(model.s_inter_area_ntc, rule=ntc_forward_rule)
    model.cInterAreaNtcBackward = pyo.Constraint(model.s_inter_area_ntc, rule=ntc_backward_rule)


def _powerloss_rules1(model, grid, ctx, cfg, penalty_twoway_flow=0):
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
        def rule(model, j):
            flow_abs = model.p_branch_ac_powerflow12[j] + model.p_branch_ac_powerflow21[j]
            alpha = br.loc[j, "resistance"] * flow_abs / const.baseMVA
            expr = model.varLossAc12[j] == alpha * model.varAcBranchFlow12[j]
            return expr

        model.cLossAc12 = pyo.Constraint(model.s_branch_ac, rule=rule)

    def make_lossAc_rule21(br):
        def rule(model, j):
            flow_abs = model.p_branch_ac_powerflow12[j] + model.p_branch_ac_powerflow21[j]
            alpha = br.loc[j, "resistance"] * flow_abs / const.baseMVA
            expr = model.varLossAc21[j] == alpha * model.varAcBranchFlow21[j]
            return expr

        model.cLossAc21 = pyo.Constraint(model.s_branch_ac, rule=rule)

    def make_lossDc_rule12(br):
        def rule(model, j):
            flow_abs = model.p_branch_dc_powerflow12[j] + model.p_branch_dc_powerflow21[j]
            alpha = br.loc[j, "resistance"] * flow_abs / const.baseMVA
            expr = model.varLossDc12[j] == alpha * model.varDcBranchFlow12[j]
            return expr

        model.cLossDc12 = pyo.Constraint(model.s_branch_dc, rule=rule)

    def make_lossDc_rule21(br):
        def rule(model, j):
            flow_abs = model.p_branch_dc_powerflow12[j] + model.p_branch_dc_powerflow21[j]
            alpha = br.loc[j, "resistance"] * flow_abs / const.baseMVA
            expr = model.varLossDc21[j] == alpha * model.varDcBranchFlow21[j]
            return expr

        model.cLossDc21 = pyo.Constraint(model.s_branch_dc, rule=rule)

    make_lossAc_rule12(grid.branch)
    make_lossAc_rule21(grid.branch)
    make_lossDc_rule12(grid.dcbranch)
    make_lossDc_rule21(grid.dcbranch)

    # Add penalty in objective function to discourage simultaneous flow in both directions
    # This is not particularly elegant and should not be normally used
    # REF: Issue https://github.com/powergama/powergama/issues/29
    if penalty_twoway_flow > 0:
        print("Adding a cost to penalise simultaneous branch flow in both directions - ")
        model.OBJ.expr += penalty_twoway_flow * sum(
            model.varAcBranchFlow12[i] + model.varAcBranchFlow21[i] for i in model.s_branch_ac
        )


def create_constraint_powerloss(model, grid, ctx, cfg, penalty_twoway_flow=0):
    """Constraint: flow = flow12-flow21 & powerloss"""
    if cfg.lossmethod == 1:

        def flowAc_rule(model, j):
            expr = model.varAcBranchFlow[j] == model.varAcBranchFlow12[j] - model.varAcBranchFlow21[j]
            return expr

        def flowDc_rule(model, j):
            expr = model.varDcBranchFlow[j] == model.varDcBranchFlow12[j] - model.varDcBranchFlow21[j]
            return expr

        model.cFlowAc = pyo.Constraint(model.s_branch_ac, rule=flowAc_rule)
        model.cFlowDc = pyo.Constraint(model.s_branch_dc, rule=flowDc_rule)

    # 1b Losses vs flow
    if cfg.lossmethod == 1:
        _powerloss_rules1(model, grid, ctx, cfg, penalty_twoway_flow)


def create_constraint_generator_output(model, grid, ctx, cfg):
    """Constraint: Generator output limit"""

    # Generator output constraint is not necessary, as lower and upper
    # bounds are set for each timestep in _update_progress. Should not
    # be specified as constraint with with pmax as limit, since e.g.
    # PV may have higher production than generator rating.

    # HGS: Doing it anyway, cf Espen Bødal and Martin Kristiansen
    # TODO: Check that there are no problems with this.

    def genMaxLimit_rule(model, i):
        return model.varGeneration[i] <= model.p_gen_pmax[i]

    def genMinLimit_rule(model, i):
        return model.varGeneration[i] >= model.p_gen_pmin[i]

    model.cGenMaxLimit = pyo.Constraint(model.s_gen, rule=genMaxLimit_rule)
    model.cGenMinLimit = pyo.Constraint(model.s_gen, rule=genMinLimit_rule)


def create_constraint_generator_pump(model, grid, ctx, cfg):
    """Constraint: Pump output limit (respects both hardware cap and remaining reservoir space)."""

    def pump_rule(model, g):
        expr = model.varPump[g] <= model.p_pump_pmax[g]
        return expr

    model.cPump = pyo.Constraint(model.s_gen_pump, rule=pump_rule)


def create_constraint_load_flex(model, grid, ctx, cfg):
    """Constraint: Flexible load limit"""

    def flexload_rule(model, i):
        flexLoadMax = (
            grid.consumer.loc[i, "demand_avg"]
            * grid.consumer.loc[i, "flex_fraction"]
            / grid.consumer.loc[i, "flex_on_off"]
        )
        expr = model.varFlexLoad[i] <= flexLoadMax
        return expr

    model.cFlexload = pyo.Constraint(model.s_load_flex, rule=flexload_rule)


def create_constraint_powerbalance(model, grid, ctx, cfg, powerbalance_rhs):
    """ConstraintPower balance (power flow equation)  (Pnode = B theta)"""
    lossmethod = cfg.lossmethod

    def powerbalance_rule(model, n):
        lhs = 0
        for g in ctx.generators_at_node[n]:
            # this is a generator connected to node n
            lhs += model.varGeneration[g]
            if g in model.s_gen_pump:
                lhs -= model.varPump[g]
        for lod in ctx.loads_at_node[n]:
            lhs -= model.p_demand[lod]
            lhs += model.varLoadShed[lod]
            lhs -= model.varDumpLoad[lod]
            if lod in model.s_load_flex:
                lhs -= model.varFlexLoad[lod]
        for b in ctx.dcbranch_to_node[n]:
            lhs += model.varDcBranchFlow[b]
            if lossmethod == 1:
                lhs -= model.varLossDc12[b]
            elif lossmethod == 2:
                lhs -= model.p_branch_dc_power_loss[b] / 2
        for b in ctx.dcbranch_from_node[n]:
            lhs += -model.varDcBranchFlow[b]
            if lossmethod == 1:
                lhs -= model.varLossDc21[b]
            elif lossmethod == 2:
                lhs -= model.p_branch_dc_power_loss[b] / 2
        if lossmethod == 1:
            # we define flow12 as flow leaving node 1, and flow21 as flow leaving node 2
            # so subtract loss at "receiving" node but not at "sending" node
            # i.e. loss12 at to-node and loss21 at from-node
            for b in ctx.branch_to_node[n]:
                lhs += -model.varLossAc12[b]
            for b in ctx.branch_from_node[n]:
                lhs += -model.varLossAc21[b]
        elif lossmethod == 2:
            # add ac branch losses as load, equally split between sending and receiving node
            for b in ctx.branch_to_node[n]:
                # positive sign for flow into node
                lhs -= model.p_branch_ac_power_loss[b] / 2
            for b in ctx.branch_from_node[n]:
                lhs -= model.p_branch_ac_power_loss[b] / 2

        lhs = lhs / const.baseMVA

        # powerbalance_rhs = get_powerbalance_rhs(model, ctx)
        rhs = powerbalance_rhs[n]

        expr = lhs == rhs
        # Skip constraint if it is trivial (otherwise run-time error)
        # TODO: Check if this is safe
        if (type(expr) is bool) and (expr is True):
            expr = pyo.Constraint.Skip
        return expr

    model.cPowerbalance = pyo.Constraint(model.s_node, rule=powerbalance_rule)


def create_constraint_powerflow_equation(model, grid, ctx, cfg):
    """Constraint: Power balance (power flow vs voltage angle)"""
    # Node and branch ids are strings/labels but the DA matrix needs positions, so
    # map them once here instead of inside the rule: the branch lookup was a linear
    # scan per branch, and the node lookup rebuilt the entire node list for every
    # non-zero of every row.
    node_list = list(model.s_node)
    branch_position = {b: i for i, b in enumerate(model.s_branch_ac)}

    # 1.
    def flowangle_rule(model, b):
        lhs = model.varAcBranchFlow[b]
        lhs = lhs / const.baseMVA
        rhs = 0
        row = ctx.DA[branch_position[b]]
        for i in range(len(row.indices)):
            n2 = node_list[row.indices[i]]
            rhs += row.data[i] * model.varVoltageAngle[n2] * const.baseAngle
        expr = lhs == rhs
        return expr

    model.cFlowAngle = pyo.Constraint(model.s_branch_ac, rule=flowangle_rule)

    # 2. Reference voltag angle)
    def referenceNode_rule(model, n):
        if n in ctx.refnodes:
            expr = model.varVoltageAngle[n] == 0
        else:
            expr = pyo.Constraint.Skip
        return expr

    model.cReferenceNode = pyo.Constraint(model.s_node, rule=referenceNode_rule)


def objective_terms(model):
    """The core objective, as named terms that an extension may add to, drop or replace.

    The objective is their sum, in insertion order.
    """
    terms = {}
    # Cost of generation, less the value of pumping and of serving flexible load.
    terms["operating_cost"] = (
        sum(model.varGeneration[i] * model.p_gen_cost[i] for i in model.s_gen)
        - sum(model.varPump[i] * model.p_genpump_cost[i] for i in model.s_gen_pump)
        - sum(model.varFlexLoad[i] * model.p_loadflex_cost[i] for i in model.s_load_flex)
    )
    # Penalty slack: keeps every timestep feasible.
    terms["load_shedding"] = sum(model.varLoadShed[i] * const.loadshedcost for i in model.s_load) + sum(
        model.varDumpLoad[i] * const.loadshedcost for i in model.s_load
    )
    return terms


def create_objective(model, grid, ctx, cfg):
    """Create pyomo model objective function"""

    def cost_rule(model):
        terms = objective_terms(model)
        if cfg.rt_deviation_objective_active:
            rt_build.edit_objective_terms(model, ctx, cfg, terms)
        return sum(terms.values())

    model.OBJ = pyo.Objective(rule=cost_rule, sense=pyo.minimize)


def get_powerbalance_rhs(model, ctx):
    """Get rhs expression in powerbalance constraint"""
    # This is taken out of the constraint creation function to speed up constraint creation.
    # Node ids are strings but Bbus needs matrix indices, so materialise the ordered set
    # once: looking a node's position up per node was O(n^2), and rebuilding the whole
    # list again for every non-zero in its row made that O(n^2) per node.
    node_list = list(model.s_node)
    rhs = dict()
    for idx_node, n in enumerate(node_list):
        rhs[n] = 0
        row = ctx.Bbus[idx_node]
        for i in range(len(row.indices)):
            n2 = node_list[row.indices[i]]
            rhs[n] -= row.data[i] * model.varVoltageAngle[n2] * const.baseAngle
    return rhs
