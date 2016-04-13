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
from numpy import pi, asarray, vstack, zeros
from datetime import datetime as datetime
from . import constants as const
import scipy.sparse
import sys
#needed for code to work both for python 2.7 and 3:
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass



class LpProblem(object):
    '''
    Class containing problem definition as a LP problem, and function calls
    to solve the problem
    '''
    solver = None


    def __init__(self,grid):
        '''Create and initialise LpProblem object'''
        
        #def lpProblemInitialise(self,grid):

        self._grid = grid
        self.timeDelta = grid.timeDelta
        self.num_nodes = grid.node.numNodes()        
        self.num_generators = grid.generator.numGenerators()
        self.num_branches = grid.branch.numBranches()
        self.num_dc_branches = grid.dcbranch.numBranches()
        self._idx_generatorsWithPumping = grid.getIdxGeneratorsWithPumping()
        
        self._idx_generatorsWithStorage = grid.getIdxGeneratorsWithStorage()
        self._idx_generatorsStorageProfileFilling = asarray(
            [grid.generator.storagevalue_profile_filling[i] 
            for i in self._idx_generatorsWithStorage])
        self._idx_generatorsStorageProfileTime = asarray(
            [grid.generator.storagevalue_profile_time[i] 
            for i in self._idx_generatorsWithStorage])

        self._idx_consumersWithFlexLoad = grid.getIdxConsumersWithFlexibleLoad()                
        self._fancy_progressbar = False        

        # Initial values of marginal costs, storage and storage values      
        self._storage = (
            asarray(grid.generator.storagelevel_init)
            *asarray(grid.generator.storage) )
        self._marginalcosts = asarray(grid.generator.fuelcost)        
        
        self._storage_flexload = (
                asarray(grid.consumer.flex_storagelevel_init)
                * asarray(grid.consumer.flex_storage) 
                * asarray(grid.consumer.flex_fraction)
                * asarray(grid.consumer.load)
                )
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


        #print "Creating LP problem..."        
        self.prob = pulp.LpProblem(
            "PowerGAMA_"+datetime.now().strftime("%Y-%m-%dT%H%M%S"), 
            pulp.LpMinimize)

        # Define (and keep track of) LP problem variables
        self._var_generation = [
            pulp.LpVariable("Pgen%d" %(i)) for i in range_generators] 
        self._var_pumping = [
            pulp.LpVariable("Ppump%d" %(i)) 
            for i in self._idx_generatorsWithPumping] 
        self._var_flexload = [
            pulp.LpVariable("Pflexload%d" %(i)) 
            for i in self._idx_consumersWithFlexLoad] 
        self._var_branchflow = [
            pulp.LpVariable("Pbranch%d" %(i)) for i in range_branches] 
        self._var_dc = [
            pulp.LpVariable("Pdc%d" %(i)) for i in range_dc_branches] 
        self._var_angle = [
            pulp.LpVariable("theta%d" %(i)) for i in range_nodes] 
        self._var_loadshedding = [
            pulp.LpVariable("Pshed%d" %(i)) for i in range_nodes] 

        self._idx_load = [[]]*self.num_nodes
        
        # Compute matrices used in power flow equaions        
        print("Computing B and DA matrices...")        
        self._Bbus, self._DA = grid.computePowerFlowMatrices(const.baseZ)
        print("Creating B.theta and DA.theta expressions")

         # Matrix * vector product -- Using coo_matrix
        # (http://stackoverflow.com/questions/4319014/
        #  iterating-through-a-scipy-sparse-vector-or-matrix)
        
        cx = scipy.sparse.coo_matrix(self._DA)
        self._DAtheta = [0]*cx.shape[0]
        for i,j,v in zip(cx.row, cx.col, cx.data):
            self._DAtheta[i] += v * self._var_angle[j]

        cx = scipy.sparse.coo_matrix(self._Bbus)
        _Btheta = [0]*cx.shape[0]
        for i,j,v in zip(cx.row, cx.col, cx.data):
            _Btheta[i] += v * self._var_angle[j]
                
        # Variables upper and lower bounds (voltage angle and )        
        for i in range(self.num_nodes):
            self._var_angle[i].lowBound = -pi
            self._var_angle[i].upBound = pi

        for i in range(self.num_nodes):
            self._var_loadshedding[i].lowBound = 0
            #self._var_loadshedding[i].upBound = inf
            # upper bound should not exceed total demand at load
            #TODO: Replace unlimited upper bound by real value 

        for i in range_pumps:
            self._var_pumping[i].lowBound = 0
            self._var_pumping[i].upBound = grid.generator.pump_cap[
                                        self._idx_generatorsWithPumping[i]]

        for i in range_flexloads:
            idx_cons = self._idx_consumersWithFlexLoad[i]
            self._var_flexload[i].lowBound = 0
            self._var_flexload[i].upBound = (
                grid.consumer.load[idx_cons]
                * grid.consumer.flex_fraction[idx_cons]
                / grid.consumer.flex_on_off[idx_cons]
                )
 
        print("Defining constraints...")
        idxBranchesConstr = self._grid.getIdxBranchesWithFlowConstraints()
        idxDcBranchesConstr = self._grid.getIdxDcBranchesWithFlowConstraints()

        # Initialise lists of constraints
        self._constraints_branchLowerBounds = [[]]*len(idxBranchesConstr)
        self._constraints_branchUpperBounds = [[]]*len(idxBranchesConstr)
        self._constraints_dcbranchLowerBounds = [[]]*len(idxDcBranchesConstr)
        self._constraints_dcbranchUpperBounds = [[]]*len(idxDcBranchesConstr)
        self._constraints_pf = [pulp.pulp.LpConstraint()]*self.num_nodes

        # Swing bus angle = 0 (reference)
        probConstraintSwing = self._var_angle[0]==0
        self.prob.addConstraint(probConstraintSwing,name="swingbus_angle")
        #prob += probConstraintSwing,"swingbus_angle"
       
        # Max and min power flow on AC branches
        for i in idxBranchesConstr:
        #for i in range_branches:
            cl = self._var_branchflow[i] >= -self._grid.branch.capacity[i]
            cu = self._var_branchflow[i] <= self._grid.branch.capacity[i]
            cl_name = "branchflow_min_%d" %(i)            
            cu_name = "branchflow_max_%d" %(i)            
            self.prob.addConstraint(cl,name=cl_name)
            self.prob.addConstraint(cu,name=cu_name)
            # need to keep track of these constraints since we want to get
            # sensitivity information from solution:
            idx_branch_constr = idxBranchesConstr.index(i)
            self._constraints_branchLowerBounds[idx_branch_constr] = cl_name
            self._constraints_branchUpperBounds[idx_branch_constr] = cu_name

        # Max and min power flow on DC branches        
        for i in idxDcBranchesConstr:
            dc_cl = self._var_dc[i] >= -self._grid.dcbranch.capacity[i]
            dc_cu = self._var_dc[i] <= self._grid.dcbranch.capacity[i]
            dc_cl_name = "dcflow_min_%d" %(i)            
            dc_cu_name = "dcflow_max_%d" %(i)            
            self.prob.addConstraint(dc_cl,name=dc_cl_name)
            self.prob.addConstraint(dc_cu,name=dc_cu_name)
            # need to keep track of these constraints since we want to get
            # sensitivity information from solution:
            idx_dcbranch_constr = idxDcBranchesConstr.index(i)
            self._constraints_dcbranchLowerBounds[idx_dcbranch_constr] = dc_cl_name
            self._constraints_dcbranchUpperBounds[idx_dcbranch_constr] = dc_cu_name
        
        # Equations giving the branch power flow from the nodal phase angles
        for idx_branch in range_branches:
            Pbr = self._var_branchflow[idx_branch]*(1/const.baseMVA)
            pfb_name = "powerflow_vs_angle_eqn_%d"%(idx_branch)
            self.prob.addConstraint(Pbr==self._DAtheta[idx_branch],name=pfb_name)

        
        # Variable part (that is updated hour by hour)
        timestep = 0
        
        # Bounds on maximum and minimum production (power inflow)
        self._setLpGeneratorMaxMin(timestep)
        
        # Power flow equations (constraints)
        print("Power flow equations...")

        self._pfPload = [[]]*self.num_nodes
        self._pfPgen = [[]]*self.num_nodes
        self._pfPpump= [[]]*self.num_nodes
        self._pfPflexload= [[]]*self.num_nodes
        self._pfPflow = [[]]*self.num_nodes
        self._pfPshed = [[]]*self.num_nodes
        self._pfPdc = [[]]*self.num_nodes
        
        for idx_node in range_nodes:                        
            # Find generators connected to this node:            
            idx_gen = grid.getGeneratorsAtNode(idx_node)
            
            # the idx_gen_pump has  indices referring to the list of generators
            # the number of pumps is equal to the length of this list
            idx_gen_pump = grid.getGeneratorsWithPumpAtNode(idx_node)
            
            # Find DC branches connected to node (direction is important)
            idx_dc_from = grid.getDcBranchesAtNode(idx_node,'from')
            idx_dc_to = grid.getDcBranchesAtNode(idx_node,'to')

            # Find indices of loads connected to this node:
            self._idx_load[idx_node] = grid.getLoadsAtNode(idx_node)

            # the idx_flexload has  indices referring to the list of loads
            # the number of flexible loads equals the length of this list
            idx_flexload = grid.getLoadsFlexibleAtNode(idx_node)
                      
            # Constant part of power flow equations            
            self._pfPgen[idx_node] = [
                self._var_generation[i]*(1/const.baseMVA) for i in idx_gen]
            self._pfPpump[idx_node] = [
                -self._var_pumping[
                    self._idx_generatorsWithPumping.index(i)]*(1/const.baseMVA) 
                for i in idx_gen_pump]
            self._pfPflexload[idx_node] = [
                -self._var_flexload[
                    self._idx_consumersWithFlexLoad.index(i)]*(1/const.baseMVA) 
                for i in idx_flexload]
            self._pfPshed[idx_node] = ( 
                self._var_loadshedding[idx_node]*(1/const.baseMVA))
            self._pfPdc[idx_node] = (
                 [  self._var_dc[i]*(1/const.baseMVA) for i in idx_dc_to]
                +[ -self._var_dc[i]*(1/const.baseMVA) for i in idx_dc_from])
            self._pfPflow[idx_node] = -_Btheta[idx_node]
            
            # this value will be updated later, so using zero for now:            
            self._pfPload[idx_node] = pulp.lpSum(0) 

           # Generation is positive
            # Pumping is negative
            # Demand is negative
            # Load shed is positive
            # Flow out of the node is positive
            cpf = pulp.lpSum(
                self._pfPgen[idx_node]
                +self._pfPpump[idx_node]
                +self._pfPflexload[idx_node]
                +self._pfPdc[idx_node]
                +self._pfPload[idx_node]
                +self._pfPshed[idx_node]) == self._pfPflow[idx_node]
            pf_name = "powerflow_eqn_%d"%(idx_node)
            self.prob.addConstraint(cpf,name=pf_name)
            self._constraints_pf[idx_node] = pf_name
            

        
        print("Objective function...")

        print("  Using fixed load shedding cost of %f. One per node" 
            % const.loadshedcost)       
        self._loadsheddingcosts = [const.loadshedcost]*self.num_nodes

        self._updateLpProblem(timestep)
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

       
 
       
    def _setLpGeneratorMaxMin(self,timestep):
        '''Specify constraints for generator output'''       

        P_storage = self._storage / self.timeDelta
        P_max = self._grid.generator.prodMax
        P_min = self._grid.generator.prodMin
        
        for i in range(self.num_generators):
            inflow_factor = self._grid.generator.inflow_factor[i]
            capacity = self._grid.generator.prodMax[i]
            inflow_profile = self._grid.generator.inflow_profile[i]
            P_inflow =  (capacity * inflow_factor 
                * self._grid.inflowProfiles[inflow_profile][timestep])
            self._var_generation[i].lowBound = min(
                P_inflow+P_storage[i],P_min[i])            
            if P_storage[i]==0:
                '''
                Don't let P_max limit the output (e.g. solar PV)
                This won't affect fuel based generators with zero storage,
                since these should have inflow=p_max in any case
                '''
                self._var_generation[i].upBound = P_inflow
            else:
                self._var_generation[i].upBound = min(P_inflow+P_storage[i],
                                                      P_max[i])

        return


    
    def _updateMarginalcosts(self,timestep):
        '''Marginal costs based on storage value for generators with storage'''
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
                


    def _updateLpProblem(self,timestep):
        '''
        Function that updates LP problem for a given timestep, due to changed
        power demand, power inflow and marginal generator costs
        '''

        range_nodes = range(self.num_nodes)
        range_generators = range(self.num_generators)
        #range_branches = range(self.num_branches)

        # Update bounds on maximum and minimum production (power inflow)
        self._setLpGeneratorMaxMin(timestep)
                    
        # Update power flow equations
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
                    -self._grid.demandProfiles[profile_ref][timestep]
                    *average/const.baseMVA)
                
            self._pfPload[idx_node] = demOutflow

            
            cpf = (
                self._pfPgen[idx_node]
                +self._pfPpump[idx_node]
                +self._pfPflexload[idx_node]
                +self._pfPdc[idx_node]
                +self._pfPload[idx_node]
                +self._pfPshed[idx_node] == self._pfPflow[idx_node])
                
            # Find the associated constraint and modify it:            
            key_constr = self._constraints_pf[idx_node]
            self.prob.constraints[key_constr] = cpf
 
        # Update objective function      
        self._updateMarginalcosts(timestep)                                                 
        probObjective_gen = pulp.lpSum(\
            [self._marginalcosts[i]*self._var_generation[i]*self.timeDelta \
                for i in range_generators]  )       
        probSlack = pulp.lpSum(\
            [self._loadsheddingcosts[i]*self._var_loadshedding[i]*self.timeDelta \
                for i in range_nodes]  ) 
        
        genpumpidx = self._idx_generatorsWithPumping;
        probObjective_pump = pulp.lpSum([
            max(0,(self._marginalcosts[genpumpidx[i]]
            -self._grid.generator.pump_deadband[genpumpidx[i]])) 
            * (-self._var_pumping[i]) 
            for i in range(len(genpumpidx))
            ]  )
            
        flexloadidx = self._idx_consumersWithFlexLoad
        probObjective_flexload = pulp.lpSum([
            -self._marginalcosts_flexload[flexloadidx[i]]
            * self._var_flexload[i]
            for i in range(len(flexloadidx))
            ])
        
        self.prob.setObjective(probObjective_gen
                                +probObjective_pump
                                +probObjective_flexload                                
                                +probSlack)      
        
        return
        
        

    
    def _storeResultsAndUpdateStorage(self,timestep,results):
        """Store timestep results in local arrays, and update storage"""
                
        Pgen = [v.varValue for v in self._var_generation]
        Ppump = [v.varValue for v in self._var_pumping]
        Pflexload = [v.varValue for v in self._var_flexload]

        # Update storage:
        inflow_profile_refs = self._grid.generator.inflow_profile
        inflow_factor = self._grid.generator.inflow_factor
        capacity= self._grid.generator.prodMax
        genInflow = [capacity[i] * inflow_factor[i] 
        			 * self._grid.inflowProfiles[inflow_profile_refs[i]][timestep]
                        for i in range(len(capacity))]

        energyIn = asarray(genInflow)*self.timeDelta
        pumpedIn = zeros(len(capacity))
        for i,x in enumerate(self._idx_generatorsWithPumping):
            pumpedIn[x] = Ppump[i]*self._grid.generator.pump_efficiency[x]
        pumpedIn = pumpedIn*self.timeDelta

        energyOut = asarray(Pgen)*self.timeDelta
        energyStorable = self._storage + energyIn + pumpedIn - energyOut
        storagecapacity = asarray(self._grid.generator.storage)
        self._storage = vstack((storagecapacity,energyStorable)).min(axis=0)

        energyIn_flexload = zeros(len(self._grid.consumer.flex_fraction))        
        for i,x in enumerate(self._idx_consumersWithFlexLoad):
            energyIn_flexload[x] = Pflexload[i]*self.timeDelta
        energyOut_flexload = (
            asarray(self._grid.consumer.flex_fraction)
            * asarray(self._grid.consumer.load)
            * self.timeDelta )
        self._storage_flexload = (
            self._storage_flexload + energyIn_flexload - energyOut_flexload )
        
        # Collect and store results
        F = pulp.value(self.prob.objective)  
        Pb = [v.varValue for v in self._var_branchflow]
        Pdc = [v.varValue for v in self._var_dc]
        theta = [v.varValue for v in self._var_angle]
        #senseBranchCapacityUpper = [cval.pi if cval.pi!=None else 0 for cval in self._constraints_branchUpperBounds]
        #senseBranchCapacityLower = [cval.pi if cval.pi!=None else 0 for cval in self._constraints_branchLowerBounds]
        #senseN = [cval.pi for cval in self._constraints_pf]
        senseBranchCapacityUpper = [self.prob.constraints[ckey].pi
            if self.prob.constraints[ckey].pi!=None else None
            for ckey in self._constraints_branchUpperBounds]
        senseBranchCapacityLower = [self.prob.constraints[ckey].pi
            if self.prob.constraints[ckey].pi!=None else None
            for ckey in self._constraints_branchLowerBounds]
        senseDcBranchCapacityUpper = [self.prob.constraints[ckey].pi
            if self.prob.constraints[ckey].pi!=None else None
            for ckey in self._constraints_dcbranchUpperBounds]
        senseDcBranchCapacityLower = [self.prob.constraints[ckey].pi
            if self.prob.constraints[ckey].pi!=None else None
            for ckey in self._constraints_dcbranchLowerBounds]
        senseN = [self.prob.constraints[ckey].pi/const.baseMVA
            if self.prob.constraints[ckey].pi!=None else None
            for ckey in self._constraints_pf]
        senseB = [(i-j)/const.baseMVA if i!=None and j!=None else None 
            for i,j in zip(senseBranchCapacityUpper, senseBranchCapacityLower)]
        senseDcB = [(i-j)/const.baseMVA  if i!=None and j!=None else None 
            for i,j in zip(senseDcBranchCapacityUpper, senseDcBranchCapacityLower)]
            
        loadshed = [v.varValue for v in self._var_loadshedding]
        # TODO: This subtraction generates warning - because it includes nan and inf?
        energyspilled = energyStorable-self._storage
        storagelevel = self._storage[self._idx_generatorsWithStorage]
        marginalprice = self._marginalcosts[self._idx_generatorsWithStorage]
        flexload_storagelevel = self._storage_flexload[self._idx_consumersWithFlexLoad]
        flexload_marginalprice = self._marginalcosts_flexload[self._idx_consumersWithFlexLoad]
        
        # TODO: Only keep track of inflow spilled for generators with 
        # nonzero inflow
        
        results.addResultsFromTimestep(
            timestep = self._grid.timerange[0]+timestep,
            objective_function = F,
            generator_power = Pgen,
            generator_pumped = Ppump,
            branch_power = Pb,
            dcbranch_power = Pdc,
            node_angle = theta,
            sensitivity_branch_capacity = senseB,
            sensitivity_dcbranch_capacity = senseDcB,
            sensitivity_node_power = senseN,
            storage = storagelevel.tolist(),
            inflow_spilled = energyspilled.tolist(),
            loadshed_power = loadshed,
            marginalprice = marginalprice.tolist(),
            flexload_power = Pflexload,
            flexload_storage = flexload_storagelevel.tolist(),
            flexload_storagevalue = flexload_marginalprice.tolist())

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
        #prob0 = pulp.LpProblem("Grid Market Power - base", pulp.LpMinimize)
        numTimesteps = len(self._grid.timerange)
        for timestep in range(numTimesteps):
            # update LP problem (inflow, storage, profiles)                     
            self._updateLpProblem(timestep)
          
            # solve the LP problem
            #self.prob.solve(self.solver,use_mps=True)
            self.prob.solve(self.solver)
            
            # print result summary            
            #value_costfunction = pulp.value(self.prob.objective)
            self._update_progress(timestep,numTimesteps)
            #print "Timestep=",timestep, " => ",  \
            #    pulp.LpStatus[self.prob.status], \
            #    "<> cost=",value_costfunction
       
            # store results and update storage levels
            self._storeResultsAndUpdateStorage(timestep,results)
        
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