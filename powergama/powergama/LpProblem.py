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
from numpy import pi, asarray, vstack, inf
from datetime import datetime as datetime
import constants as const
import scipy.sparse
import itertools

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
        self._idx_generatorsWithStorage = grid.getIdxGeneratorsWithStorage()
        self._idx_generatorsStorageType = asarray([grid.generator.storagevalue_type[i] for i in self._idx_generatorsWithStorage])

                
        # Initial values of marginal costs, storage and storage values      
        self._storage = asarray(grid.generator.storagelevel_init)*asarray(grid.generator.storage)
        self._marginalcosts = asarray(self._grid.generator.marginalcost)        
        
        range_nodes = range(self.num_nodes)
        range_generators = range(self.num_generators)
        range_branches = range(self.num_branches)


        #print "Creating LP problem..."        
        self.prob = pulp.LpProblem("PowerGAMA "+str(datetime.now()), pulp.LpMinimize)

        # Define (and keep track of) LP problem variables
        self._var_generation = [pulp.LpVariable("Pgen%d" %(i)) for i in range_generators] 
        self._var_branchflow = [pulp.LpVariable("Pbranch%d" %(i)) for i in range_branches] 
        self._var_angle = [pulp.LpVariable("theta%d" %(i)) for i in range_nodes] 
        self._var_loadshedding = [pulp.LpVariable("Pshed%d" %(i)) for i in range_nodes] 

        self._idx_load = [[]]*self.num_nodes
        
        # Compute matrices used in power flow equaions        
        print "Computing B and DA matrices..."        
        self._Bbus, self._DA = grid.computePowerFlowMatrices(const.baseZ)
        print "Creating B.theta and DA.theta expressions"

         # Matrix * vector product -- Using coo_matrix
        # (http://stackoverflow.com/questions/4319014/iterating-through-a-scipy-sparse-vector-or-matrix)
        
        cx = scipy.sparse.coo_matrix(self._DA)
        self._DAtheta = [0]*cx.shape[0]
        for i,j,v in itertools.izip(cx.row, cx.col, cx.data):
            self._DAtheta[i] += v * self._var_angle[j]

        cx = scipy.sparse.coo_matrix(self._Bbus)
        _Btheta = [0]*cx.shape[0]
        for i,j,v in itertools.izip(cx.row, cx.col, cx.data):
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


 
        print "Defining constraints..."
        idxBranchesConstr = self._grid.getIdxBranchesWithFlowConstraints()

        # Initialise lists of constraints
        #self._constraints_branchLowerBounds = ['']*self.num_branches
        #self._constraints_branchUpperBounds = ['']*self.num_branches
        self._constraints_branchLowerBounds = [[]]*len(idxBranchesConstr)
        self._constraints_branchUpperBounds = [[]]*len(idxBranchesConstr)
        self._constraints_pf = [pulp.pulp.LpConstraint()]*self.num_nodes

        # Swing bus angle = 0 (reference)
        probConstraintSwing = self._var_angle[0]==0
        self.prob.addConstraint(probConstraintSwing,name="swingbus_angle")
        #prob += probConstraintSwing,"swingbus_angle"
       
        # Max and min power flow
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

        # Equations giving the branch power flow from the nodal phase angles
        for idx_branch in range_branches:
            Pbr = self._var_branchflow[idx_branch]/const.baseMVA
            pfb_name = "powerflow_vs_angle_eqn_%d"%(idx_branch)
            self.prob.addConstraint(Pbr==self._DAtheta[idx_branch],name=pfb_name)

        
        # Variable part (that is updated hour by hour)
        timestep = 0
        
        # Bounds on maximum and minimum production (power inflow)
        self._setLpGeneratorMaxMin(timestep)
        
        # Power flow equations (constraints)
        print "Power flow equations..."

        self._pfPload = [[]]*self.num_nodes
        self._pfPgen = [[]]*self.num_nodes
        self._pfPflow = [[]]*self.num_nodes
        self._pfPshed = [[]]*self.num_nodes
        
        #TODO Speed up this loop        
        for idx_node in range_nodes:                        
            # Find generators connected to this node:            
            idx_gen = grid.getGeneratorsAtNode(idx_node)
            # Find indices of loads connected to this node:
            self._idx_load[idx_node] = grid.getLoadsAtNode(idx_node)

            # Constant part of power flow equations            
            self._pfPgen[idx_node] = [self._var_generation[i]/const.baseMVA for i in idx_gen]
            self._pfPshed[idx_node] = self._var_loadshedding[idx_node]/const.baseMVA
            self._pfPflow[idx_node] = -_Btheta[idx_node]

            demOutflow=[]
            # Usually there is maximum one load per node, but it could be more,
            # so need a loop
            for i in self._idx_load[idx_node]:
                average = self._grid.consumer.load[i]
                profile_ref = self._grid.consumer.load_profile[i]
                demOutflow.append(-self._grid.demandProfiles[profile_ref][timestep]*average/const.baseMVA)
                
            self._pfPload[idx_node] = pulp.lpSum(demOutflow)
            
            cpf = pulp.lpSum(self._pfPgen[idx_node]+self._pfPload[idx_node]+self._pfPshed[idx_node]) == self._pfPflow[idx_node]
            pf_name = "powerflow_eqn_%d"%(idx_node)
            self.prob.addConstraint(cpf,name=pf_name)
            self._constraints_pf[idx_node] = pf_name


        
        print "Objective function..."

        #TODO: Get loadshedding cost from input
        print("  Using fixed load shedding cost of %f. One per node" % const.loadshedcost)       
        self._loadsheddingcosts = [const.loadshedcost]*self.num_nodes

        probObjective = pulp.lpSum([self._marginalcosts[i]*self._var_generation[i] for i in range_generators]  )       
        probSlack = pulp.lpSum([self._loadsheddingcosts[i]*self._var_loadshedding[i] for i in range_nodes]  ) 
        self.prob.setObjective(probObjective+probSlack)      

        return       
        ## END init




    def initialiseSolver(self,cbcpath):
        '''
        Initialise solver - normally not necessary
        '''
        solver = pulp.solvers.COIN_CMD(path=cbcpath)
        if solver.available():
            print ":) Found solver here: ", solver.available()
            self.solver = solver
        else:
            print ":( Could not find solver. Returning."            
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
            P_inflow =  capacity * inflow_factor * self._grid.inflowProfiles[inflow_profile][timestep]
            self._var_generation[i].lowBound = min(P_inflow+P_storage[i],P_min[i])
            self._var_generation[i].upBound = min(P_inflow+P_storage[i],P_max[i])
            #print " gen=%d: P_inflow=%g, P_storage=%g, Pmin=%g, Pmax=%g" %(i,P_inflow,P_storage[i],P_min[i],P_max[i])
        return


    
    def _updateMarginalcosts(self):
        '''Marginal costs based on storage value for generators with storage'''
        for i in range(len(self._idx_generatorsWithStorage)):
            idx_gen = self._idx_generatorsWithStorage[i]
            this_type = self._idx_generatorsStorageType[i]
            storagecapacity = asarray(self._grid.generator.storage[idx_gen])
            fillinglevel = self._storage[idx_gen] / storagecapacity       
            col = int(round(fillinglevel*100))
            self._marginalcosts[idx_gen] \
                = self._grid.generator.marginalcost[idx_gen] *self._grid.storagevalue[this_type][col]
                


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
                average = self._grid.consumer.load[i]
                profile_ref = self._grid.consumer.load_profile[i]
                demOutflow.append(-self._grid.demandProfiles[profile_ref][timestep]*average/const.baseMVA)
                
            self._pfPload[idx_node] = demOutflow

            
            cpf = pulp.lpSum(self._pfPgen[idx_node]+self._pfPload[idx_node]+self._pfPshed[idx_node]) == self._pfPflow[idx_node]
            # Find the associated constraint and modify it:            
            key_constr = self._constraints_pf[idx_node]
            self.prob.constraints[key_constr] = cpf
 
        # Update objective function      
        self._updateMarginalcosts()                                                 
        probObjective = pulp.lpSum(\
            [self._marginalcosts[i]*self._var_generation[i]*self.timeDelta \
                for i in range_generators]  )       
        probSlack = pulp.lpSum(\
            [self._loadsheddingcosts[i]*self._var_loadshedding[i]*self.timeDelta \
                for i in range_nodes]  ) 

        self.prob.setObjective(probObjective+probSlack)      
        
        return
        
        

    
    def _storeResultsAndUpdateStorage(self,timestep,results):
        """Store timestep results in local arrays, and update storage"""
                
        Pgen = [v.varValue for v in self._var_generation]
        
        # Update storage:
        inflow_profile_refs = self._grid.generator.inflow_profile
        inflow_factor = self._grid.generator.inflow_factor
        capacity= self._grid.generator.prodMax
        genInflow = [capacity[i] * inflow_factor[i] * self._grid.inflowProfiles[inflow_profile_refs[i]][timestep] \
                        for i in range(len(capacity))]

        energyIn = asarray(genInflow)*self.timeDelta
        energyOut = asarray(Pgen)*self.timeDelta
        energyStorable = self._storage + energyIn - energyOut
        storagecapacity = asarray(self._grid.generator.storage)
        self._storage = vstack((storagecapacity,energyStorable)).min(axis=0)

        # Collect and store results
        F = pulp.value(self.prob.objective)  
        Pb = [v.varValue for v in self._var_branchflow]
        theta = [v.varValue for v in self._var_angle]
        #senseBranchCapacityUpper = [cval.pi if cval.pi!=None else 0 for cval in self._constraints_branchUpperBounds]
        #senseBranchCapacityLower = [cval.pi if cval.pi!=None else 0 for cval in self._constraints_branchLowerBounds]
        #senseN = [cval.pi for cval in self._constraints_pf]
        senseBranchCapacityUpper = [self.prob.constraints[ckey].pi \
            if self.prob.constraints[ckey].pi!=None else 0 \
            for ckey in self._constraints_branchUpperBounds]
        senseBranchCapacityLower = [self.prob.constraints[ckey].pi \
            if self.prob.constraints[ckey].pi!=None else 0 \
            for ckey in self._constraints_branchLowerBounds]
        senseN = [self.prob.constraints[ckey].pi/const.baseMVA \
            if self.prob.constraints[ckey].pi!=None else 0 \
            for ckey in self._constraints_pf]
        senseB = [(i-j)/const.baseMVA for i,j in zip(senseBranchCapacityUpper, senseBranchCapacityLower)]
        loadshed = [v.varValue for v in self._var_loadshedding]
        energyspilled = energyStorable-self._storage
        storagelevel = self._storage[self._idx_generatorsWithStorage]
        
        results.addResultsFromTimestep(objective_function = F,
                                            generator_power = Pgen,
                                            branch_power = Pb,
                                            node_angle = theta,
                                            sensitivity_branch_capacity = senseB,
                                            sensitivity_node_power = senseN,
                                            storage = storagelevel,
                                            inflow_spilled = energyspilled,
                                            loadshed_power = loadshed,
                                            marginalprice = self._marginalcosts)

        return
    
        
    def solve(self,results):
        '''
        Solve LP problem for each time step in the time range
        
        Arguments
        ---------
        results
            PowerGAMA Results object reference
        '''
            
        print "Solving..."
        #prob0 = pulp.LpProblem("Grid Market Power - base", pulp.LpMinimize)
        numTimesteps = len(self._grid.timerange)
        for timestep in range(numTimesteps):
            # update LP problem (inflow, storage, profiles)                     
            self._updateLpProblem(timestep)
          
            # solve the LP problem
            self.prob.solve(self.solver,use_mps=False)
            
            # print result summary            
            value_costfunction = pulp.value(self.prob.objective)
            print "Timestep=",timestep, " => ",  \
                pulp.LpStatus[self.prob.status], \
                "<> cost=",value_costfunction
       
            # store results and update storage levels
            self._storeResultsAndUpdateStorage(timestep,results)
        

        
        