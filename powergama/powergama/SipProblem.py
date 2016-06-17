# -*- coding: utf-8 -*-
'''
Module containing PowerGAMA LpProblem class
'''

'''
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
'''

#import pulp
from pyomo.environ import *
from numpy import pi, asarray, vstack, zeros
import numpy as np
from datetime import datetime as datetime
from . import constants as const
import scipy.sparse
import sys
#needed for code to work both for python 2.7 and 3:
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass



class SipProblem(object):
    '''
    Class containing problem definition as a LP problem, and function calls
    to solve the problem
    '''
    solver = SolverFactory("gurobi")

    def __init__(self,grid):
        '''Create and initialise LpProblem object'''
        
        # Load dimensional data
        self._grid = grid
        self.timeDelta = grid.timeDelta
        self.num_nodes = grid.node.numNodes()
        self.num_generators = grid.generator.numGenerators()
        self.num_branches = grid.branch.numBranches()
        self.num_dc_branches = grid.dcbranch.numBranches()
        
        # Indexing of consumers and flexible consumers, i.e. elastic demand
        self._idx_load = [[]]*self.num_nodes        

        # Load initial values of marginal costs, storage and storage values
        self._marginalcosts = asarray(grid.generator.fuelcost)
        self._storage = (
                asarray(grid.generator.storagelevel_init)
                * asarray(grid.generator.storage) )

        range_nodes = range(self.num_nodes)
        range_generators = range(self.num_generators)
        range_branches = range(self.num_branches)
        range_dc_branches = range(self.num_dc_branches)
        range_time = grid.timerange

        self._fancy_progressbar = False
        
        idxBranchesConstr = self._grid.getIdxBranchesWithFlowConstraints()
        idxDcBranchesConstr = self._grid.getIdxDcBranchesWithFlowConstraints()
        
        nodeCostDcOffshore = 406*10**6
        nodeCostDcOnshore =  1
        dcFixedCost = 312*10**3
        dcDistanceCost = 1236*10**3
        dcDistancePowerCost = 578
        dcEndpointSea = 453123*10**3
        dcEndpointLand = 58209*10**3
        loss = 0.005/100
        AC_constraints = False          # False sets AC branch limits to 20 GW
        curtailment_cost = 0 # EUR/MWh
        
        ######################### Creating LP problem ########################
        
        model = ConcreteModel()
        
        # SETS
        model.NODES = Set(initialize=range_nodes)
        model.GEN = Set(initialize=range_generators)
        model.BRANCH = Set(initialize=range_branches)
        model.DC = Set(initialize=range_dc_branches)
        model.TIME = Set(initialize=range_time)
        
        # PARAMETERS
        def demand_max_rule(model,n):
            self._idx_load[n] = grid.getLoadsAtNode(n)
            idx_loads = self._idx_load[n]
            average=0
            for i in idx_loads:
                    average += self._grid.consumer.load[i]*(
                            1-self._grid.consumer.flex_fraction[i])
            return average
        model.demandMax = Param(model.NODES, initialize=demand_max_rule, mutable=True)
        
        # TODO: Have to make this demand consistent with "load indexing"        
        def demand_rule(model, n, t):
            self._idx_load[n] = grid.getLoadsAtNode(n)
            idx_loads = self._idx_load[n]
            demOutflow=0
            for i in idx_loads:
#                    average = self._grid.consumer.load[i]*(
#                            1-self._grid.consumer.flex_fraction[i])
                    profile_ref = self._grid.consumer.load_profile[i]
                    demOutflow += -self._grid.demandProfiles[profile_ref][t]
            return demOutflow
        model.demand = Param(model.NODES, model.TIME, initialize=demand_rule)
        
        def genMin_rule(model,g,t):
            P_storage = self._storage / self.timeDelta
            P_min = self._grid.generator.prodMin

            inflow_factor = self._grid.generator.inflow_factor[g]
            capacity = self._grid.generator.prodMax[g]
            inflow_profile = self._grid.generator.inflow_profile[g]
            P_inflow =  (capacity * inflow_factor
                * self._grid.inflowProfiles[inflow_profile][t])
            return min(P_inflow + P_storage[g], P_min[g])
        model.genMin = Param(model.GEN, model.TIME, initialize=genMin_rule)
        
        def genMax_rule(model,g):
            inflow_factor = self._grid.generator.inflow_factor[g]
            capacity = self._grid.generator.prodMax[g]
            return capacity * inflow_factor
        model.genMax = Param(model.GEN, initialize=genMax_rule, mutable=True)
                
        def genProfile_rule(model,g,t):
            P_storage = self._storage / self.timeDelta
            P_max = self._grid.generator.prodMax
            inflow_profile = self._grid.generator.inflow_profile[g]
            P_inflow =  self._grid.inflowProfiles[inflow_profile][t]
            if P_storage[g]==0:
                '''
                Don't let P_max limit the output (e.g. solar PV)
                This won't affect fuel based generators with zero storage,
                since these should have inflow=p_max in any case
                '''
                return P_inflow
            else:
                return min(P_inflow+P_storage[g]/P_max[g],1)
        model.genProfile = Param(model.GEN, model.TIME, initialize=genProfile_rule)
            
        def genMC_rule(model, g):
            return self._marginalcosts[g]
        model.genMC = Param(model.GEN, initialize=genMC_rule)
        model.shedCost = Param(model.NODES, initialize=const.loadshedcost)
        
        def branchMax_rule(model, j):
            if AC_constraints:
                return self._grid.branch.capacity[j]
            else:
                return 20000
        model.branchMax = Param(model.BRANCH, initialize=branchMax_rule)
        
        def dcbranchMax_rule(model,j):
            return self._grid.dcbranch.capacity[j]
        model.dcMax = Param(model.DC, initialize=dcbranchMax_rule)
        
        def dcNewbranchMax_rule(model, j):
            return self._grid.dcbranch.max_capacity[j]
        model.dcMaxNew = Param(model.DC, initialize=dcNewbranchMax_rule)
        
        # TODO: make a proper VC rule for DC branch
        DctoIdx = self._grid.dcbranch.node_toIdx(self._grid.node)
        DcfromIdx = self._grid.dcbranch.node_fromIdx(self._grid.node)
        
        def dcVarD_rule(model,j):
            lat = [self._grid.node.lat[DcfromIdx[j]], 
                   self._grid.node.lat[DctoIdx[j]]]
            lon = [self._grid.node.lon[DcfromIdx[j]], 
                   self._grid.node.lon[DctoIdx[j]]]
            distance = self._getDistance(lat, lon)
            return dcDistanceCost*distance    # dependent on number of branches
        model.dcVarD = Param(model.DC, initialize=dcVarD_rule)
        
        def dcVarDP_rule(model,j):
            lat = [self._grid.node.lat[DcfromIdx[j]], 
                   self._grid.node.lat[DctoIdx[j]]]
            lon = [self._grid.node.lon[DcfromIdx[j]], 
                   self._grid.node.lon[DctoIdx[j]]]
            distance = self._getDistance(lat, lon)
            return dcDistancePowerCost*distance    # dependent on power rating
        model.dcVarDP = Param(model.DC, initialize=dcVarDP_rule)
        
        # TODO: make a proper FC rule for DC branch (offshore/onshore nodes)
        def dcFC_rule(model, j):
            idx_nodes = [DctoIdx[j], DcfromIdx[j]]
            expr = dcFixedCost
            for ii in idx_nodes:    
                if self._grid.node.offshore[ii]:
                    expr += dcEndpointSea
                else:
                    expr += dcEndpointLand
            return expr
        model.dcFC = Param(model.DC, initialize=dcFC_rule)
        
        def nodeFC_rule(model, n):
            if self._grid.node.offshore[n]:
                return nodeCostDcOffshore
            else:
                return nodeCostDcOnshore
        model.nodeFC = Param(model.NODES, initialize=nodeFC_rule)
        
        # VARIABLES  
        model.dcVarInvest = Var(model.DC, domain = NonNegativeReals, initialize=0)
        model.dcFixInvest = Var(model.DC, domain = NonNegativeIntegers, initialize=0)
        model.nodeInvest = Var(model.NODES, domain = Binary)

        model.generation = Var(model.GEN, model.TIME, domain = NonNegativeReals, initialize=0)
        model.loadShed = Var(model.NODES, model.TIME, domain = NonNegativeReals, initialize=0)
        model.branchFlow = Var(model.BRANCH, model.TIME, domain = Reals, initialize=0)
        model.dcFlow = Var(model.DC, model.TIME, domain = Reals, initialize=0)
        
        model.curtailment = Var(model.GEN, model.TIME, domain = NonNegativeReals, initialize=0)
        
#        model.FirstStageCost = Var()
#        model.SecondStageCost = Var()
        

        # CONSTRAINTS
        print("Defining constraints...")
        
        def shed_rule(model, i, t):
            return model.loadShed[i,t] <= 1000000
        model.maxLoadShed = Constraint(model.NODES, model.TIME, rule=shed_rule)
         
        # Min and max power flow on AC branches  (could use idxBranchesConstr)
        def min_ACflow_rule(model, j, t):
            return model.branchFlow[j,t] >= -( model.branchMax[j] )
        model.minFlowAC = Constraint(model.BRANCH, model.TIME, rule=min_ACflow_rule)
        
        def max_ACflow_rule(model, j, t):
            return model.branchFlow[j,t] <= ( model.branchMax[j] )
        model.maxFlowAC = Constraint(model.BRANCH, model.TIME, rule=max_ACflow_rule)
        
        # Max and min power flow on DC branches (could use idxDcBranchesConstr)
        def min_DCflow_rule(model, j, t):
            return model.dcFlow[j,t] >= -( model.dcMax[j] + model.dcVarInvest[j] )
        model.minFlowDC = Constraint(model.DC, model.TIME, rule=min_DCflow_rule)
        
        def max_DCflow_rule(model, j, t):
            return model.dcFlow[j,t] <= ( model.dcMax[j] + model.dcVarInvest[j] )
        model.maxFlowDC = Constraint(model.DC, model.TIME, rule=max_DCflow_rule)
        
        # Investment constraint for capacity wrt lumped unit size (e.g. per 1200MW)
        def investDC_rule(model, j):
            return model.dcVarInvest[j] <= model.dcMaxNew[j]*model.dcFixInvest[j]
        model.dcInvest = Constraint(model.DC, rule=investDC_rule)
            
        # Bounds on maximum and minimum production (power inflow)
        def max_gen_rule(model,g,t):
            return model.generation[g,t] <= model.genMax[g]*model.genProfile[g,t]
        model.maxGeneration = Constraint(model.GEN, model.TIME, rule=max_gen_rule)
        
        def genEnergy_rule(model, g):
            E_max = self._grid.generator.energy
            if np.isnan(E_max[g]):
                return Constraint.Skip
            else:
                return sum(model.generation[g,t] for t in model.TIME)*8760/len(range_time) <= E_max[g]*10**6
        model.energyMax = Constraint(model.GEN, rule=genEnergy_rule)
        
        def curtailment_rule(model,g,t):
            if model.genMC[g] == 0:
                return model.curtailment[g,t] == model.genMax[g]*model.genProfile[g,t] - model.generation[g,t]
            else:
                return Constraint.Skip
        model.genCurtailment = Constraint(model.GEN, model.TIME, rule=curtailment_rule)
        
#        # TODO: find out why model was infeasible with minGen from datafile (and 0 MW is OK...)
#        def min_gen_rule(model,g,t):
#            return model.generation[g,t] >= 0 # model.genMin[g,t]
#        model.minGeneration = Constraint(model.GEN, model.TIME, rule=min_gen_rule)
        
        # TODO: couple "big M" with max number of new branches
        # TODO: figure out whether node is onshore or offshore
        def node_invest_rule(model, n):
            idx_dc_to = grid.getDcBranchesAtNode(n,'to')
            idx_dc_from = grid.getDcBranchesAtNode(n,'from')
            expr = sum(model.dcFixInvest[jj] for jj in idx_dc_to)
            expr += sum(model.dcFixInvest[ii] for ii in idx_dc_from)
            return expr <= 10*model.nodeInvest[n]
        model.nodeInvestBigM = Constraint(model.NODES, rule=node_invest_rule)
        
        def power_balance_rule(model, i, t):        
            # Find generators connected to this node:
            idx_gen = grid.getGeneratorsAtNode(i)

            # Find DC branches connected to node (direction is important)
            idx_dc_from = grid.getDcBranchesAtNode(i,'from')
            idx_dc_to = grid.getDcBranchesAtNode(i,'to')
            
            # Find AC branches connected to node (direction is important)
            idx_ac_from = grid.getAcBranchesAtNode(i,'from')
            idx_ac_to = grid.getAcBranchesAtNode(i,'to')

            # Find indices of loads connected to this node:
            self._idx_load[i] = grid.getLoadsAtNode(i)

            expr = sum(model.generation[ii,t] for ii in idx_gen)
            expr += model.loadShed[i,t]
            expr += sum(model.dcFlow[ii,t] for ii in idx_dc_to)*(1-loss)
            expr += sum(-model.dcFlow[ii,t] for ii in idx_dc_from)
            expr += sum(model.branchFlow[ii,t] for ii in idx_ac_to)*(1-loss)
            expr += sum(-model.branchFlow[ii,t] for ii in idx_ac_from)
            return expr == -model.demand[i,t]*model.demandMax[i]
        model.PowerBalance = Constraint(model.NODES, model.TIME, rule=power_balance_rule)             
        
        # STAGE SPESIFICS
        a = self._computeAnnuityFactor(rate=0.05, years=30)
        
        def ComputeFirstStageCost_rule(model):
            expr = summation(model.dcVarDP, model.dcVarInvest)          # DistancePower costs
            expr += summation(model.dcVarD, model.dcFixInvest)          # Distance costs per branch
            expr += summation(model.dcFC, model.dcFixInvest)            # Fixed cost per branch
            expr += summation(model.nodeFC, model.nodeInvest)
            return expr*(1 + 0.02*a)    # incl. O&M costs
        model.FirstStageCost = Expression(rule=ComputeFirstStageCost_rule)

        def ComputeSecondStageCost_rule(model):
            expr = sum(model.generation[i,t]*model.genMC[i] for i in model.GEN for t in model.TIME)
            expr += sum(model.loadShed[i,t]*model.shedCost[i] for i in model.NODES for t in model.TIME)
            expr += sum(model.curtailment[i,t]*curtailment_cost for i in model.GEN for t in model.TIME)
            return expr*8760/len(range_time)*a 
        model.SecondStageCost = Expression(rule=ComputeSecondStageCost_rule)
        
        # OBJECTIVE
        print("Objective function...")
        def Total_Cost_Objective_rule(model):
            return model.FirstStageCost + model.SecondStageCost
        model.Total_Cost_Objective = Objective(rule=Total_Cost_Objective_rule, sense=minimize)
        
        self.model = model
#        model.pprint()
        
#        opt = SolverFactory("gurobi")
#        results = opt.solve(model, tee=True)
#        instance.display()
        
#        # EXTENSIVE FORM        
#        ef = model.clone()
#        ef.scenario1 = scenario1._instance
#        ef.scenario2 = scenario2._instance
#        
#        ef.scenario1.objective.deactivate()
#        ef.scenario2.objective.deactivate()
#        ef.objective = Objective(expr=ef.scenario1.objective.expr*scenario1.probability
#                                    + ef.scenario2.objective.expr*scenario2.probability)
#        
#        ef.nonant_scenario1 = Constraint(expr=model.x == ef.scenario1.x)
#        ef.nonant_scenario2 = Constraint(expr=model.x == ef.scenario2.x)
        
#        instance = model.create_instance()
#        opt = SolverFactory("gurobi")
#        results = opt.solve(instance, 
#                            tee=True, #stream the solver output
#                            keepfiles=True, #print the LP file for examination
#                            symbolic_solver_labels=True) # use human readable names
#        print(results) #print solver status
#               
#        # model.pprint()
##        instance.load(results)
#        model.solutions.load_from(results)
#        print ([k for k in dir(instance)])
#        print (instance.Total_Cost_Objective)
        
        # TESTING
#        model.display()
#        model.pprint()
#        data = DataPortal()
#        data.load(filename='nordic2030_nodes.csv', set=model.NODES) # can be used to overwrite data
#        data.load(filename='nordic2030_branches.csv', set=model.BRANCH)
#        data.load(filename='nordic2030_branches.csv', param=(model.branchMax, model.branchMaxNew))
#        data.load(filename='nordic2030_branches.csv', param=(model.branchMax, model.branchMaxNew), index=model.BRANCH)
#        data.load(filename='nordic2030_branches.csv', select=('from', 'capacity'), param=model.branchMax, index=model.BRANCH)
        
#        instance = model.create_instance(data)
#        instance = model.create_instance(filename="C:\PYOMO\TEPmodel\concrete\scenariodata\RefModel.dat")
#        instance.display()
#        results = opt.solve(instance)
#        instance.display()        
#        
#        xg = []
#        d = []
#        xij = []
#        xkij = []
#        bname = []
#
#        for i in instance.NODES:
#            xg.append(value(instance.Gen[i]))
#            d.append(value(instance.Demand[i]))
        
        return
        ## END init




    def initialiseSolver(self,cbcpath):
        '''
        Initialise solver - normally not necessary
        '''
        solver = pulp.solvers.COIN_CMD(path=cbcpath)
        if solver.available():
            print (":) Found solver here: ", solver.available())
            self.solver = solver
        else:
            print(":( Could not find solver. Returning.")
            self.solver = None
            raise Exception("Could not find LP solver")
        return




    def _setLpGeneratorMaxMin(self,range_time):
        '''Specify constraints for generator output'''
        
        timestep=0
        P_storage = self._storage / self.timeDelta
        P_max = self._grid.generator.prodMax
        P_min = self._grid.generator.prodMin

        for i in range(self.num_generators):
            for t in range_time:
                inflow_factor = self._grid.generator.inflow_factor[i]
                capacity = self._grid.generator.prodMax[i]
                inflow_profile = self._grid.generator.inflow_profile[i]
                P_inflow =  (capacity * inflow_factor
                    * self._grid.inflowProfiles[inflow_profile][t])
                self._var_generation[i,t].lowBound = min(
                    P_inflow+P_storage[i],P_min[i])
                if P_storage[i]==0:
                    '''
                    Don't let P_max limit the output (e.g. solar PV)
                    This won't affect fuel based generators with zero storage,
                    since these should have inflow=p_max in any case
                    '''
                    self._var_generation[i,t].upBound = P_inflow
                else:
                    self._var_generation[i,t].upBound = min(P_inflow+P_storage[i],
                                                          P_max[i])

        return



    def _updateMarginalcosts(self,range_time):
        '''Marginal costs based on storage value for generators with storage'''
        timestep=0       
        for i in range(len(self._idx_generatorsWithStorage)):
            idx_gen = self._idx_generatorsWithStorage[i]
            this_type_filling = self._idx_generatorsStorageProfileFilling[i]
            this_type_time = self._idx_generatorsStorageProfileTime[i]
            storagecapacity = asarray(self._grid.generator.storage[idx_gen])
            fillinglevel = self._storage[idx_gen] / storagecapacity
            filling_col = int(round(fillinglevel*100))
            self._marginalcosts[idx_gen] = (
                self._grid.generator.storagevalue_abs[idx_gen]
                *self._grid.storagevalue_filling[this_type_filling][filling_col]
                *self._grid.storagevalue_time[this_type_time][timestep])

        # flexible load:
        for i in range(len(self._idx_consumersWithFlexLoad)):
            idx_cons = self._idx_consumersWithFlexLoad[i]
            this_type_filling = self._idx_consumersStorageProfileFilling[i]
            # Compute storage capacity in Mwh (from value in hours)
            storagecapacity_flexload = asarray(
                self._grid.consumer.flex_storage[idx_cons]      # h
                * self._grid.consumer.flex_fraction[idx_cons]   #
                * self._grid.consumer.load[idx_cons])           # MW
            fillinglevel = (
                self._storage_flexload[idx_cons] / storagecapacity_flexload  )
            filling_col = int(round(fillinglevel*100))
            if fillinglevel > 1:
                self._marginalcosts_flexload[idx_cons] = -const.flexload_outside_cost
            elif fillinglevel < 0:
                self._marginalcosts_flexload[idx_cons] = const.flexload_outside_cost
            else:
                self._marginalcosts_flexload[idx_cons] = (
                    self._grid.consumer.flex_basevalue[idx_cons]
                    *self._grid.storagevalue_filling[this_type_filling][filling_col]
                    )

        return

    def _computeAnnuityFactor(self,rate, years):
        '''Return the annuity factor that can be multiplied with yearly cashflows to calculate its present value (PV) '''
        annuity = ((1-1/((1+rate)**years))/rate)
        return annuity
        
    def _getDistance(self, lat, lon):
        from math import sin, cos, sqrt, atan2, radians
        # approximate radius of earth in km
        R = 6373.0
        lat1 = radians(lat[0])
        lon1 = radians(lon[0])
        lat2 = radians(lat[1])
        lon2 = radians(lon[1])
        
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        
        distance = R * c #km
        return distance

    def _updateLpProblem(self,range_time):
        '''
        Function that updates LP problem for a given timestep, due to changed
        power demand, power inflow and marginal generator costs
        '''
        
        range_nodes = range(self.num_nodes)
        range_generators = range(self.num_generators)
        range_dc_branches = range(self.num_dc_branches)

        # Update objective function
        self._updateMarginalcosts(range_time)
        annuityF = self._computeAnnuityFactor(rate=0.05,years=30)
            
        probObjective_gen = sum(\
            [self._marginalcosts[i]*self._var_generation[i,t]*8760/len(range_time)*annuityF \
                for i in range_generators for t in range_time]  )
        probSlack = pulp.lpSum(\
            [self._loadsheddingcosts[i]*self._var_loadshedding[i,t]*8760/len(range_time)*annuityF \
                for i in range_nodes for t in range_time]  )
        probObjective_finvest = pulp.lpSum(\
            [self._grid.dcbranch.fixed_cost[i]*self._var_finvestment[i] \
                for i in range_dc_branches])
        probObjective_vinvest = pulp.lpSum(\
            [self._grid.dcbranch.variable_cost[i]*self._var_vinvestment[i] \
                for i in range_dc_branches])
        

        self.prob.setObjective(probObjective_gen
                                +probObjective_finvest
                                +probObjective_vinvest)

        return




    def _storeResultsAndUpdateStorage(self,results):
        """Store timestep results in local arrays, and update storage"""
        range_time = self._grid.timerange
        timestep=0
        Pgen = np.empty([range_time.stop,],dtype=object)
        Pb = np.empty([range_time.stop,],dtype=object)
        Pdc = np.empty([range_time.stop,],dtype=object)
        loadshed = np.empty([range_time.stop,],dtype=object)   
        Xdc = np.empty([range_time.stop,],dtype=object)
        Ydc = np.empty([range_time.stop,],dtype=object)

        
        for t in range_time:
            Pgen[t] = [v[t].varValue for v in self._var_generation]
            Pb[t] = [v[t].varValue for v in self._var_branchflow]
            Pdc[t] = [v[t].varValue for v in self._var_dc]
            loadshed[t] = [v[t].varValue for v in self._var_loadshedding]
        Xdc = [v.varValue for v in self._var_vinvestment]
        Ydc = [v.varValue for v in self._var_finvestment]
            

        # Collect and store results
        F = pulp.value(self.prob.objective)

       #senseN = [cval.pi for cval in self._constraints_pf]
        for t in range_time:
            senseBranchCapacityUpper = [self.prob.constraints[ckey].pi
                if self.prob.constraints[ckey].pi!=None else None
                for ckey in self._constraints_branchUpperBounds[:,t]]
            senseBranchCapacityLower = [self.prob.constraints[ckey].pi
                if self.prob.constraints[ckey].pi!=None else None
                for ckey in self._constraints_branchLowerBounds[:,t]]
            senseDcBranchCapacityUpper = [self.prob.constraints[ckey].pi
                if self.prob.constraints[ckey].pi!=None else None
                for ckey in self._constraints_dcbranchUpperBounds[:,t]]
            senseDcBranchCapacityLower = [self.prob.constraints[ckey].pi
                if self.prob.constraints[ckey].pi!=None else None
                for ckey in self._constraints_dcbranchLowerBounds[:,t]]
            senseN = [self.prob.constraints[ckey].pi/const.baseMVA
                if self.prob.constraints[ckey].pi!=None else None
                for ckey in self._constraints_pf[:,t]]

        # TODO: Only keep track of inflow spilled for generators with
        # nonzero inflow

        results.addResultsFromTimestep(
            timestep = self._grid.timerange,
            objective_function = F,
            generator_power = Pgen,
            generator_pumped = [],
            branch_power = Pb,
            dcbranch_power = Pdc,
            node_angle = [],
            sensitivity_branch_capacity = senseBranchCapacityUpper,
            sensitivity_dcbranch_capacity = senseDcBranchCapacityUpper,
            sensitivity_node_power = senseN,
            storage = [],
            inflow_spilled = [],
            loadshed_power = loadshed,
            marginalprice = [],
            flexload_power = [],
            flexload_storage = [],
            flexload_storagevalue = [],
            fixed_investment = Ydc,
            variable_investment = Xdc
            )

        return


    def solve(self,results):
        '''
        Solve LP problem for each time step in the time range

        Arguments
        ---------
        results
            PowerGAMA Results object reference (optional)

        Returns
        ------
        results
            PowerGAMA Results object reference
        '''

        #if results == None:
        #    results = Results(self._grid)
        print("Creating instance....")
        instance = model.create('inputData.dat')
        instance.pprint()        
        print("Solving...")
#        self.prob.solve(self.solver)
        results = solver.solve(instance)
        results.write()
        print("Status:", pulp.LpStatus[self.prob.status])
        # store results and update storage levels
        self._storeResultsAndUpdateStorage(results)

        return results

    def _update_progress(self,n,maxn):
        if self._fancy_progressbar:
            barLength = 20
            progress = float(n+1)/maxn
            block = int(round(barLength*progress))
            text = "\rProgress: [{0}] {1} ({2}%)  " \
                .format( "="*block + " "*(barLength-block),
                        n, int(progress*100))
            sys.stdout.write(text)
            sys.stdout.flush()
        else:
            if int(100*(n+1)/maxn) > int(100*n/maxn):
                sys.stdout.write("%d%% "% (int(100*(n+1)/maxn)))
                sys.stdout.flush()

    def setProgressBar(self,value):
        if value=='fancy':
            self._fancy_progressbar=True
        elif value=='default':
            self._fancy_progressbar=False
        else:
            raise Exception('Progress bar bust be either "default" or "fancy"')