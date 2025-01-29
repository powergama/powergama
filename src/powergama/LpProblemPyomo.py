"""
Module containing PowerGAMA LpProblem class

 Power flow equations:

 Linearised ("DC") power flow equation
 Pinj - Bprime * theta = 0
           Bprime = (N-1)x(N-1) matrix (removed ref.bus row/column)
           theta = phase angles (at N-1 buses)
           Pinj = generation - load at node (cf makeSbus)

 Relationship between angles and power flow
 Pb - (D x A) x theta = 0
           theta_j = phase angle node j (excluding ref. node)
           Pb_k = power flow branch k
           D = diag(-b_k) (negative of susceptance on branch k)
           A = Mx(N-1) node-branch incidence (adjacency) matrix
"""

import sys
import warnings

import networkx as nx
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pyomo.opt

from . import constants as const


class LpProblem(pyo.ConcreteModel):
    """
    Class containing problem definition as a LP problem, and function calls
    to solve the problem

    """

    def _create_sets_and_parameters(self, grid_data):
        """Create pyomo model sets"""
        self.s_node = pyo.Set(ordered=True, initialize=grid_data.node["id"].tolist())
        self.s_branch_ac = pyo.Set(ordered=True, initialize=grid_data.branch.index.tolist())
        self.s_branch_dc = pyo.Set(ordered=True, initialize=grid_data.dcbranch.index.tolist())
        self.s_gen = pyo.Set(ordered=True, initialize=grid_data.generator.index.tolist())
        self.s_gen_pump = pyo.Set(ordered=True, initialize=grid_data.getIdxGeneratorsWithPumping())
        self.s_load = pyo.Set(ordered=True, initialize=grid_data.consumer.index.tolist())
        self.s_load_flex = pyo.Set(ordered=True, initialize=grid_data.getIdxConsumersWithFlexibleLoad())
        self.s_area = pyo.Set(ordered=True, initialize=grid_data.getAllAreas())

        # Mutable parameters
        # Quantities that change from timestep to the next:
        self.p_gen_pmin = pyo.Param(
            self.s_gen, within=pyo.Reals, default=0, mutable=True, initialize=grid_data.generator["pmin"].values
        )
        self.p_gen_pmax = pyo.Param(
            self.s_gen, within=pyo.Reals, default=0, mutable=True, initialize=grid_data.generator["pmax"].values
        )
        self.p_gen_cost = pyo.Param(
            self.s_gen, within=pyo.Reals, default=0, mutable=True, initialize=grid_data.generator["fuelcost"].values
        )
        self.p_genpump_cost = pyo.Param(self.s_gen, within=pyo.Reals, default=0, mutable=True)
        self.p_demand = pyo.Param(self.s_load, within=pyo.Reals, default=0, mutable=True)
        self.p_loadflex_cost = pyo.Param(
            self.s_load_flex,
            within=pyo.Reals,
            default=0,
            mutable=True,
            # initialize=grid_data.consumer.loc[self.s_load_flex, "flex_basevalue"].values,
        )
        if self._lossmethod == 2:
            self.p_branch_ac_power_loss = pyo.Param(self.s_branch_ac, within=pyo.Reals, default=0, mutable=True)
            self.p_branch_dc_power_loss = pyo.Param(self.s_branch_ac, within=pyo.Reals, default=0, mutable=True)

    def _create_variables(self):
        """Create pyomo model variables"""
        self.varAcBranchFlow = pyo.Var(self.s_branch_ac, within=pyo.Reals)
        self.varDcBranchFlow = pyo.Var(self.s_branch_ac, within=pyo.Reals)
        if self._lossmethod == 1:
            self.varAcBranchFlow12 = pyo.Var(self.s_branch_ac, within=pyo.NonNegativeReals)
            self.varAcBranchFlow21 = pyo.Var(self.s_branch_ac, within=pyo.NonNegativeReals)
            self.varDcBranchFlow12 = pyo.Var(self.s_branch_dc, within=pyo.NonNegativeReals)
            self.varDcBranchFlow21 = pyo.Var(self.s_branch_dc, within=pyo.NonNegativeReals)
            self.varLossAc12 = pyo.Var(self.s_branch_ac, within=pyo.NonNegativeReals)
            self.varLossAc21 = pyo.Var(self.s_branch_ac, within=pyo.NonNegativeReals)
            self.varLossDc12 = pyo.Var(self.s_branch_dc, within=pyo.NonNegativeReals)
            self.varLossDc21 = pyo.Var(self.s_branch_dc, within=pyo.NonNegativeReals)
        self.varGeneration = pyo.Var(self.s_gen, within=pyo.NonNegativeReals)
        self.varPump = pyo.Var(self.s_gen_pump, within=pyo.NonNegativeReals)
        self.varCurtailment = pyo.Var(self.s_gen, within=pyo.NonNegativeReals)
        self.varFlexLoad = pyo.Var(self.s_load_flex, within=pyo.NonNegativeReals)
        self.varLoadShed = pyo.Var(self.s_load, within=pyo.NonNegativeReals)
        self.varVoltageAngle = pyo.Var(self.s_node, within=pyo.Reals, initialize=0.0)

    def _create_constraint_powerflow_limit(self, grid_data):
        """Constraint: Power flow limit"""

        def maxflowAc_rule(model, j):
            cap = grid_data.branch.loc[j, "capacity"]
            if not np.isinf(cap):
                expr = pyo.inequality(-cap, model.varAcBranchFlow[j], cap)
            else:
                expr = pyo.Constraint.Skip
            return expr

        def maxflowDc_rule(model, j):
            cap = grid_data.dcbranch.loc[j, "capacity"]
            if not np.isinf(cap):
                expr = pyo.inequality(-cap, model.varDcBranchFlow[j], cap)
            else:
                expr = pyo.Constraint.Skip
            return expr

        self.cMaxFlowAc = pyo.Constraint(self.s_branch_ac, rule=maxflowAc_rule)
        self.cMaxFlowDc = pyo.Constraint(self.s_branch_dc, rule=maxflowDc_rule)

    def _create_constraint_powerloss(self, grid_data):
        """Constraint: flow = flow12-flow21 & powerloss"""
        if self._lossmethod == 1:

            def flowAc_rule(model, j):
                expr = model.varAcBranchFlow[j] == model.varAcBranchFlow12[j] - model.varAcBranchFlow21[j]
                return expr

            def flowDc_rule(model, j):
                expr = model.varDcBranchFlow[j] == model.varDcBranchFlow12[j] - model.varDcBranchFlow21[j]
                return expr

            self.cFlowAc = pyo.Constraint(self.s_branch_ac, rule=flowAc_rule)
            self.cFlowDc = pyo.Constraint(self.s_branch_dc, rule=flowDc_rule)

        # 1b Losses vs flow
        if self._lossmethod == 1:
            # Upper capacity limit, since capacity may be infinit
            clip_mw = 500
            br = grid_data.branch
            lossAcA = br["resistance"] * br["capacity"].clip(upper=clip_mw) / const.baseMVA
            lossAcB = 0

            br = grid_data.dcbranch
            lossDcA = br["resistance"] * br["capacity"].clip(upper=clip_mw) / const.baseMVA
            lossDcB = 0

            def lossAc_rule12(model, j):
                expr = model.varLossAc12[j] == model.varAcBranchFlow12[j] * lossAcA[j] + lossAcB
                return expr

            def lossAc_rule21(model, j):
                expr = model.varLossAc21[j] == model.varAcBranchFlow21[j] * lossAcA[j] + lossAcB
                return expr

            def lossDc_rule12(model, j):
                expr = model.varLossDc12[j] == model.varDcBranchFlow12[j] * lossDcA[j] + lossDcB
                return expr

            def lossDc_rule21(model, j):
                expr = model.varLossDc21[j] == model.varDcBranchFlow21[j] * lossDcA[j] + lossDcB
                return expr

            self.cLossAc12 = pyo.Constraint(self.s_branch_ac, rule=lossAc_rule12)
            self.cLossAc21 = pyo.Constraint(self.s_branch_ac, rule=lossAc_rule21)
            self.cLossDc12 = pyo.Constraint(self.s_branch_dc, rule=lossDc_rule12)
            self.cLossDc21 = pyo.Constraint(self.s_branch_dc, rule=lossDc_rule21)

    def _create_constraint_generator_output(self):
        """Constraint: Generator output limit"""

        # Generator output constraint is not necessary, as lower and upper
        # bounds are set for each timestep in _update_progress. Should not
        # be specified as constraint with with pmax as limit, since e.g.
        # PV may have higher production than generator rating.

        # HGS: Doing it anyway, cf Espen Bødal and Martin Kristiansen
        # TODO: Check that there are no problems with this.

        def genMaxLimit_rule(model, i):
            return model.varGeneration[i] <= self.p_gen_pmax[i]

        def genMinLimit_rule(model, i):
            if self.p_gen_pmin[i].value > 0:
                return model.varGeneration[i] >= self.p_gen_pmin[i]
            else:
                return pyo.Constraint.Skip

        self.cGenMaxLimit = pyo.Constraint(self.s_gen, rule=genMaxLimit_rule)
        self.cGenMinLimit = pyo.Constraint(self.s_gen, rule=genMinLimit_rule)

    def _create_constraint_generator_pump(self, grid_data):
        """Constraint: Pump output limit"""

        def pump_rule(model, g):
            expr = model.varPump[g] <= grid_data.generator.loc[g, "pump_cap"]
            return expr

        self.cPump = pyo.Constraint(self.s_gen_pump, rule=pump_rule)

    def _create_constraint_load_flex(self, grid_data):
        """Constraint: Flexible load limit"""

        def flexload_rule(model, i):
            flexLoadMax = (
                grid_data.consumer.loc[i, "demand_avg"]
                * grid_data.consumer.loc[i, "flex_fraction"]
                / grid_data.consumer.loc[i, "flex_on_off"]
            )
            expr = model.varFlexLoad[i] <= flexLoadMax
            return expr

        self.cFlexload = pyo.Constraint(self.s_load_flex, rule=flexload_rule)

    def _create_constraint_powerbalance(self, grid_data):
        """ConstraintPower balance (power flow equation)  (Pnode = B theta)"""

        def powerbalance_rule(model, n):
            lhs = 0
            for g in self._generators_at_node[n]:
                # this is a generator connected to node n
                lhs += model.varGeneration[g]
                if g in model.s_gen_pump:
                    lhs -= model.varPump[g]
            for lod in self._loads_at_node[n]:
                lhs -= self.p_demand[lod]
                lhs += model.varLoadShed[lod]
                if lod in model.s_load_flex:
                    lhs -= model.varFlexLoad[lod]
            for b in self._dcbranch_to_node[n]:
                lhs += model.varDcBranchFlow[b]
                if model._lossmethod == 1:
                    lhs -= model.varLossDc12[b]
                elif model._lossmethod == 2:
                    lhs -= model.p_branch_dc_power_loss[b] / 2
            for b in self._dcbranch_from_node[n]:
                lhs += -model.varDcBranchFlow[b]
                if model._lossmethod == 1:
                    lhs -= model.varLossDc21[b]
                elif model._lossmethod == 2:
                    lhs -= model.p_branch_dc_power_loss[b] / 2
            if self._lossmethod == 1:
                # add ac branch losses as load
                for b in self._branch_to_node[n]:
                    lhs += -model.varLossAc12[b]
                for b in self._branch_from_node[n]:
                    lhs += -model.varLossAc21[b]
            elif self._lossmethod == 2:
                for b in self._branch_to_node[n]:
                    # positive sign for flow into node
                    lhs -= model.p_branch_ac_power_loss[b] / 2
                for b in self._branch_from_node[n]:
                    lhs -= model.p_branch_ac_power_loss[b] / 2

            lhs = lhs / const.baseMVA

            # self._powerbalance_rhs = self._get_powerbalance_rhs()
            rhs = self._powerbalance_rhs[n]

            expr = lhs == rhs
            # Skip constraint if it is trivial (otherwise run-time error)
            # TODO: Check if this is safe
            if (type(expr) is bool) and (expr is True):
                expr = pyo.Constraint.Skip
            return expr

        self.cPowerbalance = pyo.Constraint(self.s_node, rule=powerbalance_rule)

    def _create_constraint_powerflow_equation(self, grid_data):
        """Constraint: Power balance (power flow vs voltage angle)"""

        # 1.
        def flowangle_rule(model, b):
            lhs = model.varAcBranchFlow[b]
            lhs = lhs / const.baseMVA
            rhs = 0
            # TODO: This can surely be simplified:
            # node id's are strings, but Bbus and DA matrices need matrix indices (int)
            idx_branch = list(self.s_branch_ac).index(b)
            # idx_branch = grid_data.branch.index.get_loc(b)  # Check if this works
            for i in range(len(self._DA[idx_branch].indices)):
                idx_node2 = self._DA[idx_branch].indices[i]
                DA_element = self._DA[idx_branch].data[i]
                n2 = list(model.s_node)[idx_node2]  # list since pyomo set is 1-based (could probably use +1 instead)
                rhs += DA_element * model.varVoltageAngle[n2] * const.baseAngle
            expr = lhs == rhs
            return expr

        self.cFlowAngle = pyo.Constraint(self.s_branch_ac, rule=flowangle_rule)

        # 2. Reference voltag angle)
        def referenceNode_rule(model, n):
            if n in self.refnodes:
                expr = model.varVoltageAngle[n] == 0
            else:
                expr = pyo.Constraint.Skip
            return expr

        self.cReferenceNode = pyo.Constraint(self.s_node, rule=referenceNode_rule)

    def _create_objective(self, grid_data):
        """Create pyomo model objective function"""

        def cost_rule(model):
            """Operational costs: cost of gen, load shed and curtailment"""

            # Operational costs phase 1 (if stage2DeltaTime>0)
            cost = sum(model.varGeneration[i] * self.p_gen_cost[i] for i in model.s_gen)
            cost -= sum(model.varPump[i] * self.p_genpump_cost[i] for i in model.s_gen_pump)
            cost -= sum(model.varFlexLoad[i] * self.p_loadflex_cost[i] for i in model.s_load_flex)
            cost += sum(model.varLoadShed[i] * const.loadshedcost for i in model.s_load)
            return cost

        self.OBJ = pyo.Objective(rule=cost_rule, sense=pyo.minimize)

    def _get_powerbalance_rhs(self):
        """Get rhs expression in powerbalance constraint"""
        # This is taken out of the constraint creation function to speed up constraint creation
        rhs = dict()
        for n in self.s_node:
            rhs[n] = 0
            # node id's are strings, but Bbus and DA matrices need matrix indices (int)
            idx_node = list(self.s_node).index(n)
            for i in range(len(self._Bbus[idx_node].indices)):
                idx_node2 = self._Bbus[idx_node].indices[i]
                B_element = self._Bbus[idx_node].data[i]
                n2 = list(self.s_node)[idx_node2]  # list since pyomo set is 1-based (could probably use +1 instead)
                rhs[n] -= B_element * self.varVoltageAngle[n2] * const.baseAngle
        return rhs

    def __init__(self, grid, lossmethod=0):
        """LP problem formulation

        Parameters
        ==========
        grid : GridData
            grid data object
        lossmethod : int
            loss method; 0=no losses, 1=linearised losses, 2=added as load
        """

        # 1.
        super().__init__()

        # 2. Compute matrices used in power flow equaions
        print("Computing B and DA matrices...")
        self._Bbus, self._DA = grid.compute_power_flow_matrices()

        print("Initialising LP problem...")

        # Helpers
        self._lossmethod = lossmethod
        self._grid = grid
        self.timeDelta = grid.timeDelta
        self._solver_persistent = False
        self._generators_at_node = grid.generator.groupby("node").groups
        self._loads_at_node = grid.consumer.groupby("node").groups
        self._branch_from_node = grid.branch.groupby("node_from").groups
        self._branch_to_node = grid.branch.groupby("node_to").groups
        self._dcbranch_from_node = grid.dcbranch.groupby("node_from").groups
        self._dcbranch_to_node = grid.dcbranch.groupby("node_to").groups
        for n in grid.node["id"]:
            # fill in so dict is defined for all nodes:
            if n not in self._generators_at_node:
                self._generators_at_node[n] = []
            if n not in self._loads_at_node:
                self._loads_at_node[n] = []
            if n not in self._branch_from_node:
                self._branch_from_node[n] = []
            if n not in self._branch_to_node:
                self._branch_to_node[n] = []
            if n not in self._dcbranch_from_node:
                self._dcbranch_from_node[n] = []
            if n not in self._dcbranch_to_node:
                self._dcbranch_to_node[n] = []

        self._idx_generatorsWithPumping = grid.getIdxGeneratorsWithPumping()
        self._idx_generatorsWithStorage = grid.getIdxGeneratorsWithStorage()
        self._idx_consumersWithFlexLoad = grid.getIdxConsumersWithFlexibleLoad()
        self._idx_branchesWithConstraints = grid.getIdxBranchesWithFlowConstraints()
        self._fancy_progressbar = False

        # Initial values of marginal costs, storage and storage values
        self._storage = (grid.generator["storage_ini"] * grid.generator["storage_cap"]).fillna(0)
        self._storage_flexload = (
            grid.consumer["flex_storagelevel_init"]
            * grid.consumer["flex_storage"]
            * grid.consumer["flex_fraction"]
            * grid.consumer["demand_avg"]
        ).fillna(0)
        self._energyspilled = grid.generator["storage_cap"].copy(deep=True)
        self._energyspilled[:] = 0

        # Find synchronous areas and specify reference node in each area
        G = nx.Graph()
        G.add_nodes_from(grid.node["id"])
        G.add_edges_from(zip(grid.branch["node_from"], grid.branch["node_to"]))
        G_subs = (G.subgraph(c) for c in nx.connected_components(G))
        self.refnodes = []
        for gr in G_subs:
            refnode = list(gr.nodes)[0]
            self.refnodes.append(refnode)
            print("Found synchronous area (size = {}), using ref node = {}".format(gr.order(), refnode))
        # use first node as voltage angle reference

        # 3. Create pyomo model
        self._create_sets_and_parameters(grid)
        self._create_variables()
        self._create_objective(grid)
        self._powerbalance_rhs = self._get_powerbalance_rhs()
        # 3b. Constraints:
        self._create_constraint_powerflow_limit(grid)
        self._create_constraint_powerloss(grid)
        self._create_constraint_generator_output()
        self._create_constraint_generator_pump(grid)
        self._create_constraint_load_flex(grid)
        self._create_constraint_powerbalance(grid)
        self._create_constraint_powerflow_equation(grid)

    def _get_timesteps_to_solve(self):
        numTimesteps = len(self._grid.timerange)
        return range(numTimesteps)

    def _relax_and_retry(self, opt, warmstart, count, solve_args):
        raise NotImplementedError

    # TODO: Update to allow persistent model solving
    # (remove+add constraint instead of mutable parameters)
    def _updateLpProblem(self, timestep):
        """
        Function that updates LP problem for a given timestep, due to changed
        power demand, power inflow and marginal generator costs
        """

        # 1. Generator output limits:
        #    -> power output constraints
        P_storage = self._storage / self.timeDelta
        P_max = self._grid.generator["pmax"]
        P_min = self._grid.generator["pmin"]
        for i in self.s_gen:
            inflow_factor = self._grid.generator.loc[i, "inflow_fac"]
            capacity = self._grid.generator.loc[i, "pmax"]
            inflow_profile = self._grid.generator.loc[i, "inflow_ref"]
            P_inflow = capacity * inflow_factor * self._grid.profiles.loc[timestep, inflow_profile]
            if i not in self._idx_generatorsWithStorage:
                """
                Don't let P_max limit the output (e.g. solar PV)
                This won't affect fuel based generators with zero storage,
                since these should have inflow=p_max in any case
                """
                if P_min[i] > 0:
                    self.p_gen_pmin[i] = min(P_inflow, P_min[i])
                self.p_gen_pmax[i] = P_inflow
            else:
                # generator has storage
                if P_min[i] > 0:
                    self.p_gen_pmin[i] = min(max(0, P_inflow + P_storage[i]), P_min[i])
                self.p_gen_pmax[i] = min(max(0, P_inflow + P_storage[i]), P_max[i])

        # TODO: re-create constraint - if persistent solver

        # 2. Update demand
        #    -> power balance constraint
        for i in self.s_load:
            average = self._grid.consumer.loc[i, "demand_avg"] * (1 - self._grid.consumer.loc[i, "flex_fraction"])
            profile_ref = self._grid.consumer.loc[i, "demand_ref"]
            demand_now = self._grid.profiles.loc[timestep, profile_ref] * average
            self.p_demand[i] = demand_now

        # 3. Cost parameters
        #    -> update objective function

        # 3a. generators with storage (storage value)
        for i in self._idx_generatorsWithStorage:
            this_type_filling = self._grid.generator.loc[i, "storval_filling_ref"]
            this_type_time = self._grid.generator.loc[i, "storval_time_ref"]
            storagecapacity = self._grid.generator.loc[i, "storage_cap"]
            fillinglevel = self._storage[i] / storagecapacity
            filling_col = int(round(fillinglevel * 100))
            storagevalue = (
                self._grid.generator.loc[i, "storage_price"]
                * self._grid.storagevalue_filling.loc[filling_col, this_type_filling]
                * self._grid.storagevalue_time.loc[timestep, this_type_time]
            )
            self.p_gen_cost[i] = storagevalue
            if i in self._idx_generatorsWithPumping:
                deadband = self._grid.generator.pump_deadband[i]
                self.p_genpump_cost[i] = storagevalue - deadband

        # 3b. flexible load (storage value)
        for i in self._idx_consumersWithFlexLoad:
            this_type_filling = self._grid.consumer.loc[i, "flex_storval_filling"]
            this_type_time = self._grid.consumer.loc[i, "flex_storval_time"]
            # Compute storage capacity in Mwh (from value in hours)
            storagecapacity_flexload = (
                self._grid.consumer.loc[i, "flex_storage"]  # h
                * self._grid.consumer.loc[i, "flex_fraction"]
                * self._grid.consumer.loc[i, "demand_avg"]
            )  # MW
            fillinglevel = self._storage_flexload[i] / storagecapacity_flexload
            filling_col = int(round(fillinglevel * 100))
            if fillinglevel > 1:
                storagevalue_flex = -const.flexload_outside_cost
            elif fillinglevel < 0:
                storagevalue_flex = const.flexload_outside_cost
            else:
                storagevalue_flex = (
                    self._grid.consumer.flex_basevalue[i]
                    * self._grid.storagevalue_filling.loc[filling_col, this_type_filling]
                    * self._grid.storagevalue_time.loc[timestep, this_type_time]
                )
            self.p_loadflex_cost[i] = storagevalue_flex

        return

    def _update_persistent_model(self, opt):
        """Update objective function, constraints and bounds in persistent model"""
        # Mutable parameters is not enough
        #
        # TODO: Code for persistent solver
        # https://pyomo.readthedocs.io/en/latest/solvers/persistent_solvers.html
        # to use pwersistent solvers, probably have to set instance at the
        # start, and then modify it in each iteration rather than giving
        # it as an argument to opt.solve:
        #    opt.set_instance(self.concretemodel)
        # and then use opt.solve()
        # To modify e.g. a constraint between solves, remove and add, e.g.:
        #    opt.remove_constraint(m.c)
        #    del m.c
        #    m.c = pe.Constraint(expr=m.y <= m.x)
        #    opt.add_constraint(m.c)
        # Variables can be updated without removing/adding
        #    m.x.setlb(1.0)
        #    opt.update_var(m.x)

        # TODO Check that it is correct - speed up possible?
        # TODO check if deleting and recreating constraint expression is necessary
        # (seems not)

        # 1. p_gen_pmin, p_gen_pmax => gen maxmin
        # 2. p_demand, p_branch_ac_power_loss, p_branch_dc_power_loss => powerbalance
        # 3. p_gen_cost, p_genpump_cost, p_loadflex_cost => OBJ

        # 1.
        for c in self.cGenMaxLimit.values():
            opt.remove_constraint(c)
        for c in self.cGenMinLimit.values():
            opt.remove_constraint(c)
        # del self.cGenMaxLimit
        # del self.cGenMinLimit
        # self._create_constraint_generator_output()
        for c in self.cGenMaxLimit.values():
            opt.add_constraint(c)
        for c in self.cGenMinLimit.values():
            opt.add_constraint(c)

        # 2.
        for c in self.cPowerbalance.values():
            opt.remove_constraint(c)
        # del self.cPowerbalance
        # self._create_constraint_powerbalance(self._grid)
        for c in self.cPowerbalance.values():
            opt.add_constraint(c)

        # 3.
        # no remove_object?
        # del self.OBJ
        # self._create_objective(self._grid)
        opt.set_objective(self.OBJ)

    def _updatePowerLosses(self, aclossmultiplier=1, dclossmultiplier=1):
        """Compute power losses from OPF solution and update parameters"""
        if self._lossmethod == 0:
            pass
        elif self._lossmethod == 1:
            # Use constant loss parameters
            # If loss parameters should change, they need to be declared
            # mutable=True
            pass
        elif self._lossmethod == 2:
            # Losses from previous timestep added as load
            for b in self.s_branch_ac:
                #                # r and x are given in pu; theta
                #                loss_pu = r * ((theta_to-theta_from)*const.baseAngle/x)**2
                #                # convert from p.u. to physical unit
                #                lossMVA = loss_pu*const.baseMVA
                # TODO: simpler (check and replace):
                r = self._grid.branch.loc[b, "resistance"]
                lossMVA = r * self.varAcBranchFlow[b] ** 2 / const.baseMVA
                # A multiplication factor to account for reactive current losses
                # (or more precicely, to get similar results as Giacomo in
                # the SmartNet project)
                lossMVA = lossMVA * aclossmultiplier
                self.p_branch_ac_power_loss[b] = lossMVA
            for b in self.s_branch_dc:
                # TODO: Test this before adding
                r_pu = self._grid.dcbranch.loc[b, "resistance"]
                p_pu = self.varDcBranchFlow[b] / const.baseMVA
                loss_pu = r_pu * p_pu**2
                lossMVA = loss_pu * const.baseMVA * dclossmultiplier
                self.p_branch_dc_power_loss[b] = lossMVA
        else:
            raise Exception("Loss method={} is not implemented".format(self._lossmethod))

    def _get_fault_start(self, timestep):
        # Used by LpFaultProblem
        return None

    def _storeResultsAndUpdateStorage(self, timestep, results):
        """Store timestep results in local arrays, and update storage"""

        # 1. Update generator storage:
        inflow_profile_refs = self._grid.generator["inflow_ref"]
        inflow_factor = self._grid.generator["inflow_fac"]
        capacity = self._grid.generator["pmax"]
        pumpedIn = np.zeros(len(capacity))
        energyIn = np.zeros(len(capacity))
        energyOut = np.zeros(len(capacity))
        for i in self.s_gen:
            genInflow = capacity[i] * inflow_factor[i] * self._grid.profiles[inflow_profile_refs[i]][timestep]
            energyIn[i] = genInflow * self.timeDelta
            energyOut[i] = self.varGeneration[i].value * self.timeDelta

        for i in self._idx_generatorsWithPumping:
            Ppump = self.varPump[i].value
            pumpedIn[i] = Ppump * self._grid.generator["pump_efficiency"][i] * self.timeDelta
        energyStorable = self._storage + energyIn + pumpedIn - energyOut
        storagecapacity = self._grid.generator["storage_cap"]
        # self._storage[i] = min(storagecapacity,energyStorable)
        self._storage = np.vstack((storagecapacity, energyStorable)).min(axis=0)
        self._energyspilled = energyStorable - self._storage

        # 2. Update flexible load storage
        for i in self._idx_consumersWithFlexLoad:
            energyIn_flexload = self.varFlexLoad[i].value * self.timeDelta
            energyOut_flexload = (
                self._grid.consumer["flex_fraction"][i] * self._grid.consumer["demand_avg"][i] * self.timeDelta
            )
            self._storage_flexload[i] += energyIn_flexload - energyOut_flexload

        # 3. Collect variable values from optimisation result
        F = self.OBJ()
        Pgen = [self.varGeneration[i].value for i in self.s_gen]
        Ppump = [self.varPump[i].value for i in self.s_gen_pump]
        Pflexload = [self.varFlexLoad[i].value for i in self.s_load_flex]
        Pb = [self.varAcBranchFlow[i].value for i in self.s_branch_ac]
        Pdc = [self.varDcBranchFlow[i].value for i in self.s_branch_dc]
        theta = [self.varVoltageAngle[i].value * const.baseAngle for i in self.s_node]
        # load shedding is aggregated to nodes (due to old code)
        Ploadshed = pd.Series(index=self._grid.node.id, data=[0] * len(self._grid.node.id), dtype=float)
        for j in self.s_load:
            node = self._grid.consumer["node"][j]
            Ploadshed[node] += self.varLoadShed[j].value

        # 4 Collect dual values
        # 4a. branch capacity sensitivity (whether pos or neg flow)
        senseB = []
        for j in self._idx_branchesWithConstraints:
            # for j in self.concretemodel.BRANCH_AC:
            c = self.cMaxFlowAc[j]
            senseB.append(-abs(self.dual[c] / const.baseMVA))
        senseDcB = []
        for j in self.s_branch_dc:
            c = self.cMaxFlowDc[j]
            senseDcB.append(-abs(self.dual[c] / const.baseMVA))

        # 4b. node demand sensitivity (energy balance)
        # TODO: Without abs(...) the value jumps between pos and neg. Why?
        senseN = []
        for j in self.s_node:
            c = self.cPowerbalance[j]
            senseN.append(abs(self.dual[c] / const.baseMVA))

        # consider spilled energy only for generators with storage<infinity
        # energyspilled = zeros(energyStorable.shape)
        # indx = self._grid.getIdxGeneratorsWithNonzeroInflow()
        # energyspilled[indx] = energyStorable[indx]-self._storage[indx]
        energyspilled = self._energyspilled
        storagelevel = self._storage[self._idx_generatorsWithStorage]
        storageprice = [self.p_gen_cost[i].value for i in self._idx_generatorsWithStorage]
        flexload_storagelevel = self._storage_flexload[self._idx_consumersWithFlexLoad]
        flexload_marginalprice = [self.p_loadflex_cost[i].value for i in self._idx_consumersWithFlexLoad]

        # TODO: Only keep track of inflow spilled for generators with
        # nonzero inflow

        # Extract power losses
        if self._lossmethod == 0:
            acPowerLoss = [0] * len(self.s_branch_ac)
            dcPowerLoss = [0] * len(self.s_branch_dc)
        elif self._lossmethod == 1:
            acPowerLoss = [self.varLossAc12[b].value + self.varLossAc21[b].value for b in self.s_branch_ac]
            dcPowerLoss = [self.varLossDc12[b].value + self.varLossDc21[b].value for b in self.s_branch_dc]
        elif self._lossmethod == 2:
            acPowerLoss = list(self.p_branch_ac_power_loss.extract_values().values())
            dcPowerLoss = list(self.p_branch_dc_power_loss.extract_values().values())
        else:
            raise Exception("Lossmethod must be 0,1 or 2")

        results.addResultsFromTimestep(
            timestep=self._grid.timerange[0] + timestep,
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
            marginalprice=storageprice,
            flexload_power=Pflexload,
            flexload_storage=flexload_storagelevel.tolist(),
            flexload_storagevalue=flexload_marginalprice,
            branch_ac_losses=acPowerLoss,
            branch_dc_losses=dcPowerLoss,
            fault_start=self._get_fault_start(timestep),
        )

        return

    def solve(
        self,
        results,
        solver="cbc",
        solver_path=None,
        warmstart=False,
        savefiles=False,
        aclossmultiplier=1,
        dclossmultiplier=1,
        solve_args=None,
    ):
        """
        Solve LP problem for each time step in the time range

        Parameters
        ----------
        results : Results
            PowerGAMA Results object reference
        solver : string (optional)
            name of solver to use ("cbc" or "gurobi"). Gurobi uses python
            interface, whilst CBC uses command line executable
        solver_path :string (optional, only relevant for cbc)
            path for solver executable
        warmstart : Boolean
            Use warmstart option (only some solvers, e.g. gurobi)
        savefiles : Boolean
            Save Pyomo model file and LP problem MPS file for each timestep
            This may be useful for debugging.
        aclossmultiplier : float
            Multiplier factor to scale computed AC losses, used with method 1
        dclossmultiplier : float
            Multiplier factor to scale computed DC losses, used with method 1
        logfile : string
            Name of log file for LP solver. Will keep only last iteration
        solve_args : dict
            Arguments passed on to pyomo.solve(...) in each iteration

        Returns
        -------
        results : Results
            PowerGAMA Results object reference
        """
        if solve_args is None:
            solve_args = {
                "tee": False,  # stream the solver output
                "keepfiles": False,  # print the LP file for examination
                "symbolic_solver_labels": True,  # use human readable names
                "logfile": "lpsolver_log.txt",
            }

        if "_persistent" in solver:
            self._solver_persistent = True

        # Initalise solver, and check it is available
        if solver == "gurobi":
            opt = pyo.SolverFactory("gurobi", solver_io="python")
            print(":) Using direct python interface to solver")
        elif solver == "gurobi_direct":  # think this is the same as gurobi above
            opt = pyo.SolverFactory(solver)
            print(":) Using gurobi_direct")
        elif solver == "gurobi_persistent":
            opt = pyo.SolverFactory("gurobi_persistent")
            print(":) Using persistent (in-memory) python interface to solver")
            print("-- Experimental --")
            symbolic_solver_labels = True
            if "symbolic_solver_labels" in solve_args:
                symbolic_solver_labels = solve_args["symbolic_solver_labels"]
                solve_args.pop("symbolic_solver_labels")
            opt.set_instance(self, symbolic_solver_labels=symbolic_solver_labels)
        elif solver == "appsi_highs":
            opt = pyo.SolverFactory(solver)
            if opt.available():
                print(":) Found solver")
            else:
                print(":( Could not find solver {}. Returning.".format(solver))
                raise Exception("Could not find LP solver {}".format(solver))
        else:
            solver_io = None
            # if solver=="cbc":
            # NL requres CBC with ampl interface built in
            #    solver_io="nl"
            opt = pyo.SolverFactory(solver, executable=solver_path, solver_io=solver_io)
            if opt.available():
                print(":) Found solver here: {}".format(opt.executable()))
            else:
                print(":( Could not find solver {}. Returning.".format(solver))
                raise Exception("Could not find LP solver {}".format(solver))

        # Enable access to dual values
        self.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        if self._lossmethod == 2:
            print("Computing losses in first timestep")
            self._updateLpProblem(timestep=0)
            res = opt.solve(self)
            # Now, power flow values are computed for the first timestep, and
            # power losses can be computed.

        print("Solving...")
        numTimesteps = len(self._get_timesteps_to_solve())
        count = 0
        warmstart_now = False
        for timestep in self._get_timesteps_to_solve():
            # update LP problem (inflow, storage, profiles)
            self._updateLpProblem(timestep)
            self._updatePowerLosses(aclossmultiplier, dclossmultiplier)
            if self._solver_persistent:
                self._update_persistent_model(opt=opt)

            # solve the LP problem
            if savefiles:
                # self.concretemodel.pprint('concretemodel_{}.txt'.format(timestep))
                self.write("LPproblem_{}.mps".format(timestep), io_options={"symbolic_solver_labels": True})
                # self.concretemodel.write("LPproblem_{}.nl".format(timestep))

            if warmstart and opt.warm_start_capable():
                # warmstart available (does not work with cbc)
                if count > 0:
                    warmstart_now = warmstart
                count = count + 1
                res = opt.solve(self, warmstart=warmstart_now, **solve_args)
            elif not warmstart:
                # no warmstart option
                res = opt.solve(self, **solve_args)
            else:
                raise Exception("Solver ({}) is not capable of warm start".format(opt.name))

            # store result for inspection if necessary
            self.solver_res = res

            # debugging:
            if False:
                print(
                    "Solver status = {}. Termination condition = {}".format(
                        res.solver.status, res.solver.termination_condition
                    )
                )

            if res.solver.status != pyomo.opt.SolverStatus.ok:
                warnings.warn("Something went wrong with LP solver: {}".format(res.solver.status))
                try:
                    self._relax_and_retry(opt, warmstart, count, solve_args)
                except NotImplementedError:
                    raise Exception("Something went wrong with LP solver: {}".format(res.solver.status))
            elif res.solver.termination_condition == pyomo.opt.TerminationCondition.infeasible:
                warnings.warn("t={}: No feasible solution found.".format(timestep))
                try:
                    self._relax_and_retry(opt, warmstart, count, solve_args)
                except NotImplementedError:
                    raise Exception("t={}: No feasible solution found.".format(timestep))

            self._update_progress(timestep, numTimesteps)

            # store results and update storage levels
            self._storeResultsAndUpdateStorage(timestep, results)

        return results

    def _update_progress(self, n, maxn):
        if self._fancy_progressbar:
            barLength = 20
            progress = float(n + 1) / maxn
            block = int(round(barLength * progress))
            text = "\rProgress: [{0}] {1} ({2}%)  ".format(
                "=" * block + " " * (barLength - block), n, int(progress * 100)
            )
            sys.stdout.write(text)
            sys.stdout.flush()
        else:
            if int(100 * (n + 1) / maxn) > int(100 * n / maxn):
                sys.stdout.write("%d%% " % (int(100 * (n + 1) / maxn)))
                sys.stdout.flush()

    def setProgressBar(self, value):
        """Specify how to show simulation progress

        Parameters
        ----------
        value : string
            'fancy' or 'default'
        """
        if value == "fancy":
            self._fancy_progressbar = True
        elif value == "default":
            self._fancy_progressbar = False
        else:
            raise Exception('Progress bar bust be either "default" or "fancy"')
