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

import pulp
#from pyomo.environ import *
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



class MipProblem(object):
    '''
    Class containing problem definition as a LP problem, and function calls
    to solve the problem
    '''
    solver = pulp.GUROBI()


    def __init__(self,grid):
        '''Create and initialise LpProblem object'''
        #
        #def lpProblemInitialise(self,grid):
        #
        self._grid = grid
        self.timeDelta = grid.timeDelta
        self.num_nodes = grid.node.numNodes()
        self.num_generators = grid.generator.numGenerators()
        self.num_branches = grid.branch.numBranches()
        self.num_dc_branches = grid.dcbranch.numBranches()
        # indexing of generators with different properties, such as pumping and storage
        self._idx_generatorsWithPumping = grid.getIdxGeneratorsWithPumping()
        self._idx_generatorsWithStorage = grid.getIdxGeneratorsWithStorage()
        self._idx_generatorsStorageProfileFilling = asarray(
            [grid.generator.storagevalue_profile_filling[i]
            for i in self._idx_generatorsWithStorage])
        self._idx_generatorsStorageProfileTime = asarray(
            [grid.generator.storagevalue_profile_time[i]
            for i in self._idx_generatorsWithStorage])
        # indexing of consumers that are flexible, i.e. elastic demand
        self._idx_consumersWithFlexLoad = grid.getIdxConsumersWithFlexibleLoad()

        self._fancy_progressbar = False
        
        # Collect initial values of marginal costs, storage and storage values
        self._storage = (
                asarray(grid.generator.storagelevel_init)
                * asarray(grid.generator.storage) )
        self._marginalcosts = asarray(grid.generator.fuelcost)
        self._storage_flexload = (
                asarray(grid.consumer.flex_storagelevel_init)
                * asarray(grid.consumer.flex_storage)
                * asarray(grid.consumer.flex_fraction)
                * asarray(grid.consumer.load))
        self._marginalcosts_flexload = asarray(grid.consumer.flex_basevalue)
        self._idx_consumersStorageProfileFilling = asarray(
            [grid.consumer.flex_storagevalue_profile_filling[i]
            for i in self._idx_consumersWithFlexLoad])

        range_nodes = range(self.num_nodes)
        range_generators = range(self.num_generators)
        range_pumps = range(len(self._idx_generatorsWithPumping))
        range_flexloads = range(len(self._idx_consumersWithFlexLoad))
        range_branches = range(self.num_branches)
        range_dc_branches = range(self.num_dc_branches)
        range_time = grid.timerange

        #
        # Creating LP problem
        #
        self.prob = pulp.LpProblem(
            "PowerGAMA_"+datetime.now().strftime("%Y-%m-%dT%H%M%S"),
            pulp.LpMinimize)

        # Define (and keep track of) LP problem variables        
        self._var_generation = [
            pulp.LpVariable("Pgen"+str(i)+"_"+str(t))
            for i in range_generators for t in range_time]
        self._var_branchflow = [
            pulp.LpVariable("Pbranch"+str(i)+"_"+str(t))
            for i in range_branches for t in range_time]
        self._var_dc = [
            pulp.LpVariable("Pdc"+str(i)+"_"+str(t))
            for i in range_dc_branches for t in range_time]
        self._var_loadshedding = [
            pulp.LpVariable("Pshed"+str(i)+"_"+str(t))
            for i in range_nodes for t in range_time]
        self._var_finvestment = [
            pulp.LpVariable("Ydc_invest"+"_"+str(i), lowBound=0, cat='Integer')
            for i in range_dc_branches] 
        self._var_vinvestment = [
            pulp.LpVariable("Xdc_invest"+"_"+str(i), lowBound=0)
            for i in range_dc_branches]
            
            
#        self._var_angle = [
#            pulp.LpVariable("theta"+str(i)+"_"+str(t))
#            for i in range_nodes for t in range_time]
#        self._var_pumping = [
#            pulp.LpVariable("Ppump"+str(i)+"_"+str(t))
#            for i in self._idx_generatorsWithPumping for t in range_time]
#        self._var_flexload = [
#            pulp.LpVariable("Pflexload"+str(i)+"_"+str(t))
#            for i in self._idx_consumersWithFlexLoad for t in range_time]
                
        self._idx_load = [[]]*self.num_nodes

        # Reshape the lists to get a 2D array (include time dependency)
        self._var_generation = np.reshape(self._var_generation, 
                                          (range_generators.stop,range_time.stop))
        self._var_branchflow = np.reshape(self._var_branchflow, 
                                                  (range_branches.stop,range_time.stop))
        self._var_dc = np.reshape(self._var_dc, 
                                 (range_dc_branches.stop,range_time.stop))
        self._var_loadshedding = np.reshape(self._var_loadshedding, 
                                     (range_nodes.stop,range_time.stop))
                                     
#        self._var_pumping = np.reshape(self._var_pumping, 
#                                          (len(self._idx_generatorsWithPumping),range_time.stop))
#        self._var_flexload = np.reshape(self._var_flexload, 
#                                          (len(self._idx_consumersWithFlexLoad),range_time.stop))
#        self._var_angle = np.reshape(self._var_angle, 
#                                     (range_nodes.stop,range_time.stop))
                                          
        # Compute matrices used in power flow equaions
#        print("Computing B and DA matrices...")
#        self._Bbus, self._DA = grid.computePowerFlowMatrices(const.baseZ)
#        print("Creating B.theta and DA.theta expressions")

         # Matrix * vector product -- Using coo_matrix
        # (http://stackoverflow.com/questions/4319014/
        #  iterating-through-a-scipy-sparse-vector-or-matrix)
        
#        cx = scipy.sparse.coo_matrix(self._DA)
#        DAtheta = [0]*cx.shape[0]
#        for i,j,v in zip(cx.row, cx.col, cx.data):
#            DAtheta[i] += v * self._var_angle[j,0]
#    
#        cx = scipy.sparse.coo_matrix(self._DA)
#        self._DAtheta = [0]*(cx.shape[0]*range_time.stop)
#        self._DAtheta = np.reshape(self._DAtheta, (cx.shape[0],range_time.stop))
#        for t in range_time:
#            for i,j,v in zip(cx.row, cx.col, cx.data):
#                self._DAtheta[i,t] += v * self._var_angle[j,t]
#
#        cx = scipy.sparse.coo_matrix(self._Bbus)
#        _Btheta = [0]*cx.shape[0]
#        for t in range_time:
#            for i,j,v in zip(cx.row, cx.col, cx.data):
#                _Btheta[i,t] += v * self._var_angle[j,t]


        # Variables upper and lower bounds (voltage angle and loadshed)
#        for i in range(self.num_nodes):
#            for t in range_time:
#                self._var_angle[i,t].lowBound = -pi
#                self._var_angle[i,t].upBound = pi

        for i in range(self.num_nodes):
            for t in range_time:
                self._var_loadshedding[i,t].lowBound = 0
                #self._var_loadshedding[i].upBound = inf
                # upper bound should not exceed total demand at load
                #TODO: Replace unlimited upper bound by real value

#        for i in range_pumps:
#            for t in range_time:
#                self._var_pumping[i,t].lowBound = 0
#                self._var_pumping[i,t].upBound = grid.generator.pump_cap[
#                                            self._idx_generatorsWithPumping[i]]

#        for i in range_flexloads:
#            for t in range_time:
#                idx_cons = self._idx_consumersWithFlexLoad[i]
#                self._var_flexload[i,t].lowBound = 0
#                self._var_flexload[i,t].upBound = (
#                    grid.consumer.load[idx_cons]
#                    * grid.consumer.flex_fraction[idx_cons]
#                    / grid.consumer.flex_on_off[idx_cons]
#                    )


        # TODO: Must add the time dimension for accurate number of constraints
        print("Defining constraints...")
        idxBranchesConstr = self._grid.getIdxBranchesWithFlowConstraints()
        idxDcBranchesConstr = self._grid.getIdxDcBranchesWithFlowConstraints()
        
        self._pfPload = np.empty([self.num_nodes, range_time.stop],dtype=object)
        # Initialise lists of constraints
        self._constraints_branchLowerBounds = np.empty([len(idxBranchesConstr), range_time.stop],dtype=object)
        self._constraints_branchUpperBounds = np.empty([len(idxBranchesConstr), range_time.stop],dtype=object)
        self._constraints_dcbranchLowerBounds = np.empty([len(idxDcBranchesConstr), range_time.stop],dtype=object)
        self._constraints_dcbranchUpperBounds = np.empty([len(idxDcBranchesConstr), range_time.stop],dtype=object)
        self._constraints_dcinvestMax = np.empty(len(idxDcBranchesConstr), dtype=object)       
        self._constraints_pf = np.full([self.num_nodes, range_time.stop],pulp.pulp.LpConstraint(),dtype=object)
#        self._constraints_pf = [pulp.pulp.LpConstraint()]*self.num_nodes
        

        # Swing bus angle = 0 (reference)
#        for t in range_time:        
#            probConstraintSwing = self._var_angle[0,t]==0
#            angl_name = "swingbus_angle_t="+str(t)
#            self.prob.addConstraint(probConstraintSwing,name=angl_name)


        # Max and min power flow on AC branches
        for i in idxBranchesConstr:
            for t in range_time:
                cl = self._var_branchflow[i,t] >= -self._grid.branch.capacity[i]
                cu = self._var_branchflow[i,t] <= self._grid.branch.capacity[i]
                cl_name = "branchflow_min_"+str(i)+"_"+str(t)
                cu_name = "branchflow_max_"+str(i)+"_"+str(t)
                self.prob.addConstraint(cl,name=cl_name)
                self.prob.addConstraint(cu,name=cu_name)
                # need to keep track of these constraints since we want to get
                # sensitivity information from solution:
                idx_branch_constr = idxBranchesConstr.index(i)
                self._constraints_branchLowerBounds[idx_branch_constr,t] = cl_name
                self._constraints_branchUpperBounds[idx_branch_constr,t] = cu_name

        # Max and min power flow on DC branches
        for i in idxDcBranchesConstr:
            for t in range_time:
                dc_cl = self._var_dc[i,t] >= -(self._grid.dcbranch.capacity[i] + self._var_vinvestment[i])
                dc_cu = self._var_dc[i,t] <= (self._grid.dcbranch.capacity[i] + self._var_vinvestment[i])
                dc_cl_name = "dcflow_min_"+str(i)+"_"+str(t)
                dc_cu_name = "dcflow_max_"+str(i)+"_"+str(t)
                self.prob.addConstraint(dc_cl,name=dc_cl_name)
                self.prob.addConstraint(dc_cu,name=dc_cu_name)
                # need to keep track of these constraints since we want to get
                # sensitivity information from solution:
                idx_dcbranch_constr = idxDcBranchesConstr.index(i)
                self._constraints_dcbranchLowerBounds[idx_dcbranch_constr,t] = dc_cl_name
                self._constraints_dcbranchUpperBounds[idx_dcbranch_constr,t] = dc_cu_name

        # Investment constraint for capacity wrt lumped unit size (e.g. per 1200MW)
        for i in idxDcBranchesConstr:
            dc_invconstr = self._var_vinvestment[i] <= self._grid.dcbranch.max_capacity[i]*self._var_finvestment[i]
            dc_invconstr_name = "dcinvest_max_"+str(i)
            self.prob.addConstraint(dc_invconstr, name=dc_invconstr_name)
            idx_dcinv_constr = idxDcBranchesConstr.index(i)
            self._constraints_dcinvestMax[idx_dcinv_constr] = dc_invconstr_name
            
#        # Equations giving the branch power flow from the nodal phase angles
#        for idx_branch in range_branches:
#            for t in range_time:
#                Pbr = self._var_branchflow[idx_branch,t]*(1/const.baseMVA)
#                pfb_name = "powerflow_vs_angle_eqn_"+str(idx_branch)+"_"+str(t)
#                self.prob.addConstraint(Pbr==self._DAtheta[idx_branch],name=pfb_name)

        # TODO: Make sure this timetep arrangement is ignored
        # Variable part (that is updated hour by hour)
        timestep = 0

        # Bounds on maximum and minimum production (power inflow)
        self._setLpGeneratorMaxMin(range_time)

        # Power flow equations (constraints)
        print("Power flow equations...")

        self._pfPload = np.empty([self.num_nodes, range_time.stop],dtype=object)
        self._pfPgen = np.empty([self.num_nodes, range_time.stop],dtype=object)
        self._pfPflow = np.empty([self.num_nodes, range_time.stop],dtype=object)
        self._pfPshed = np.empty([self.num_nodes, range_time.stop],dtype=object)
        self._pfPdc = np.empty([self.num_nodes, range_time.stop],dtype=object)
#        self._pfPpump= np.empty([self.num_nodes, range_time.stop],dtype=object)
#        self._pfPflexload= np.empty([self.num_nodes, range_time.stop],dtype=object)

        for idx_node in range_nodes:
            # Find generators connected to this node:
            idx_gen = grid.getGeneratorsAtNode(idx_node)

            # the idx_gen_pump has  indices referring to the list of generators
            # the number of pumps is equal to the length of this list
#            idx_gen_pump = grid.getGeneratorsWithPumpAtNode(idx_node)

            # Find DC branches connected to node (direction is important)
            idx_dc_from = grid.getDcBranchesAtNode(idx_node,'from')
            idx_dc_to = grid.getDcBranchesAtNode(idx_node,'to')
            
            # Find AC branches connected to node (direction is important)
            idx_ac_from = grid.getAcBranchesAtNode(idx_node,'from')
            idx_ac_to = grid.getAcBranchesAtNode(idx_node,'to')

            # Find indices of loads connected to this node:
            self._idx_load[idx_node] = grid.getLoadsAtNode(idx_node)

            # the idx_flexload has  indices referring to the list of loads
            # the number of flexible loads equals the length of this list
#            idx_flexload = grid.getLoadsFlexibleAtNode(idx_node)

            # TODO: label constraints with time step (this is the node constraint)
            # Constant part of power flow equations
            for t in range_time:
                self._pfPgen[idx_node,t] = [
                    self._var_generation[i,t]*(1/const.baseMVA) for i in idx_gen]
#                self._pfPpump[idx_node,t] = [
#                    -self._var_pumping[
#                    self._idx_generatorsWithPumping.index(i),t]*(1/const.baseMVA)
#                    for i in idx_gen_pump]
#                self._pfPflexload[idx_node,t] = [
#                    -self._var_flexload[
#                    self._idx_consumersWithFlexLoad.index(i),t]*(1/const.baseMVA)
#                    for i in idx_flexload]
                self._pfPshed[idx_node,t] = (
                    self._var_loadshedding[idx_node,t]*(1/const.baseMVA))
                self._pfPdc[idx_node,t] = (
                    [self._var_dc[i,t]*(1/const.baseMVA) for i in idx_dc_to]
                    +[ -self._var_dc[i,t]*(1/const.baseMVA) for i in idx_dc_from])
                self._pfPflow[idx_node,t] = (
                    [self._var_branchflow[i,t]*(1/const.baseMVA) for i in idx_ac_to]
                    +[ -self._var_dc[i,t]*(1/const.baseMVA) for i in idx_ac_from])
#                self._pfPflow[idx_node,t] = -_Btheta[idx_node]
                
                # this value will be updated later, so using zero for now:
                self._pfPload[idx_node,t] = pulp.lpSum(0)

                # Generation is positive
                # Pumping is negative
                # Demand is negative
                # Load shed is positive
                # Flow out of the node is positive
                cpf = pulp.lpSum(
                self._pfPgen[idx_node,t]
#                +self._pfPpump[idx_node,t]
#                +self._pfPflexload[idx_node,t]
                +self._pfPdc[idx_node,t]
                +self._pfPload[idx_node,t]
#                +self._pfPshed[idx_node,t] 
                +self._pfPflow[idx_node,t])== 0
#                cpf = pulp.lpSum(
#                self._pfPgen[idx_node,t]
#                +self._pfPpump[idx_node,t]
#                +self._pfPflexload[idx_node,t]
#                +self._pfPdc[idx_node,t]
#                +self._pfPload[idx_node,t]
#                +self._pfPshed[idx_node,t]) == self._pfPflow[idx_node,t]
                pf_name = "powerflow_eqn_"+str(idx_node)+"_"+str(t)
                self.prob.addConstraint(cpf,name=pf_name)
                self._constraints_pf[idx_node,t] = pf_name
                     

        print("Objective function...")

        print("  Using fixed load shedding cost of %f. One per node"
            % const.loadshedcost)
        self._loadsheddingcosts = [const.loadshedcost]*self.num_nodes

        self._updateLpProblem(range_time)
        self.prob.writeLP("PowerGIM.lp")
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

    def _updateLpProblem(self,range_time):
        '''
        Function that updates LP problem for a given timestep, due to changed
        power demand, power inflow and marginal generator costs
        '''
        timestep=0
        range_nodes = range(self.num_nodes)
        range_generators = range(self.num_generators)
        range_dc_branches = range(self.num_dc_branches)

        # Update power flow equations
        for t in range_time:
            for idx_node in range_nodes:

                # Update load
                idx_loads = self._idx_load[idx_node] #indices of loads at this node
                demOutflow=[]
                # Usually there is maximum one load per node, but it could be more
                for i in idx_loads:
                    average = self._grid.consumer.load[i]*(
                            1-self._grid.consumer.flex_fraction[i])
                    profile_ref = self._grid.consumer.load_profile[i]
                    demOutflow.append(
                            -self._grid.demandProfiles[profile_ref][t]
                            *average/const.baseMVA)

                self._pfPload[idx_node,t] = demOutflow


#                cpf = (
#                    self._pfPgen[idx_node,t]
#                    +self._pfPpump[idx_node,t]
#                    +self._pfPflexload[idx_node,t]
#                    +self._pfPdc[idx_node,t]
#                    +self._pfPload[idx_node,t]
#                    +self._pfPshed[idx_node,t] == self._pfPflow[idx_node,t])

                cpf = (
                    self._pfPgen[idx_node,t]
#                    +self._pfPpump[idx_node,t]
#                    +self._pfPflexload[idx_node,t]
                    +self._pfPdc[idx_node,t]
                    +self._pfPload[idx_node,t]
#                    +self._pfPshed[idx_node,t] 
                    +self._pfPflow[idx_node,t] == 0)

                # Find the associated constraint and modify it:
                key_constr = self._constraints_pf[idx_node,t]
                self.prob.constraints[key_constr] = cpf

        # Update objective function
        self._updateMarginalcosts(range_time)
        annuityF = self._computeAnnuityFactor(rate=0.05,years=30)
        
        probObjective_gen = pulp.lpSum(\
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
        


#        genpumpidx = self._idx_generatorsWithPumping;
#        probObjective_pump = pulp.lpSum([
#            max(0,(self._marginalcosts[genpumpidx[i]]
#            -self._grid.generator.pump_deadband[genpumpidx[i]]))
#            * (-self._var_pumping[i,t])
#            for i in range(len(genpumpidx)) for t in range_time
#            ])
#
#        flexloadidx = self._idx_consumersWithFlexLoad
#        probObjective_flexload = pulp.lpSum([
#            -self._marginalcosts_flexload[flexloadidx[i]]
#            * self._var_flexload[i,t]
#            for i in range(len(flexloadidx)) for t in range_time])

        self.prob.setObjective(probObjective_gen
#                                +probObjective_pump
#                                +probObjective_flexload
                                +probSlack
                                +probObjective_finvest
                                +probObjective_vinvest)

        return




    def _storeResultsAndUpdateStorage(self,results):
        """Store timestep results in local arrays, and update storage"""
        range_time = self._grid.timerange
        timestep=0
        Pgen = np.empty([range_time.stop,],dtype=object)
#        Ppump = np.empty([range_time.stop,],dtype=object)
#        Pflexload = np.empty([range_time.stop,],dtype=object)
        Pb = np.empty([range_time.stop,],dtype=object)
        Pdc = np.empty([range_time.stop,],dtype=object)
#        theta = np.empty([range_time.stop,],dtype=object)
        loadshed = np.empty([range_time.stop,],dtype=object)   
        Xdc = np.empty([range_time.stop,],dtype=object)
        Ydc = np.empty([range_time.stop,],dtype=object)

        
        for t in range_time:
            Pgen[t] = [v[t].varValue for v in self._var_generation]
#            Ppump[t] = [v[t].varValue for v in self._var_pumping]
#            Pflexload[t] = [v[t].varValue for v in self._var_flexload]
            Pb[t] = [v[t].varValue for v in self._var_branchflow]
            Pdc[t] = [v[t].varValue for v in self._var_dc]
#            theta[t] = [v[t].varValue for v in self._var_angle]
            loadshed[t] = [v[t].varValue for v in self._var_loadshedding]
        Xdc = [v.varValue for v in self._var_vinvestment]
        Ydc = [v.varValue for v in self._var_finvestment]
            
#        # Update storage:
#        inflow_profile_refs = self._grid.generator.inflow_profile
#        inflow_factor = self._grid.generator.inflow_factor
#        capacity= self._grid.generator.prodMax
#        genInflow = [capacity[i] * inflow_factor[i]
#                     * self._grid.inflowProfiles[inflow_profile_refs[i]][timestep]
#                        for i in range(len(capacity))]
#
#        energyIn = asarray(genInflow)*self.timeDelta
#        pumpedIn = zeros(len(capacity))
#        for i,x in enumerate(self._idx_generatorsWithPumping):
#            pumpedIn[x] = Ppump[i]*self._grid.generator.pump_efficiency[x]
#        pumpedIn = pumpedIn*self.timeDelta
#
#        energyOut = asarray(Pgen)*self.timeDelta
#        energyStorable = self._storage + energyIn + pumpedIn - energyOut
#        storagecapacity = asarray(self._grid.generator.storage)
#        self._storage = vstack((storagecapacity,energyStorable)).min(axis=0)
#
#        energyIn_flexload = zeros(len(self._grid.consumer.flex_fraction))
#        for i,x in enumerate(self._idx_consumersWithFlexLoad):
#            energyIn_flexload[x] = Pflexload[i]*self.timeDelta
#        energyOut_flexload = (
#            asarray(self._grid.consumer.flex_fraction)
#            * asarray(self._grid.consumer.load)
#            * self.timeDelta )
#        self._storage_flexload = (
#            self._storage_flexload + energyIn_flexload - energyOut_flexload )

        # Collect and store results
        F = pulp.value(self.prob.objective)

        #senseBranchCapacityUpper = [cval.pi if cval.pi!=None else 0 for cval in self._constraints_branchUpperBounds]
        #senseBranchCapacityLower = [cval.pi if cval.pi!=None else 0 for cval in self._constraints_branchLowerBounds]
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
#            senseB = [(i-j)/const.baseMVA if i!=None and j!=None else None
#                for i,j in zip(senseBranchCapacityUpper, senseBranchCapacityLower)]
#            senseDcB = [(i-j)/const.baseMVA  if i!=None and j!=None else None
#                for i,j in zip(senseDcBranchCapacityUpper, senseDcBranchCapacityLower)]

#        # TODO: This subtraction generates warning - because it includes nan and inf?
#        energyspilled = energyStorable-self._storage
#        storagelevel = self._storage[self._idx_generatorsWithStorage]
#        marginalprice = self._marginalcosts[self._idx_generatorsWithStorage]
#        flexload_storagelevel = self._storage_flexload[self._idx_consumersWithFlexLoad]
#        flexload_marginalprice = self._marginalcosts_flexload[self._idx_consumersWithFlexLoad]

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
        print("Solving...")
        self.prob.solve(self.solver)
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