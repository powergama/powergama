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

import pyomo.environ as pyo
import pyomo.opt
import numpy as np
#from numpy import pi, array, asarray, vstack, zeros
import datetime
from . import constants as const
import scipy.sparse
import sys
import warnings
import pandas as pd
import networkx as nx

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

    def _createAbstractModel(self):    
        model = pyo.AbstractModel()
        model.name = 'PowerGAMA abstract model {}'.format(
            datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S"))
        
        # SETS ###############################################################        
        model.NODE = pyo.Set(ordered=True)
        model.GEN = pyo.Set(ordered=True)
        model.GEN_PUMP = pyo.Set(ordered=True)
        model.BRANCH_AC = pyo.Set(ordered=True)
        model.BRANCH_DC = pyo.Set(ordered=True)
        model.LOAD = pyo.Set(ordered=True)
        model.LOAD_FLEX = pyo.Set(ordered=True)
        #model.AREA = pyo.Set()        
        #model.GENTYPE = pyo.Set()        

        # PARAMETERS #########################################################
        model.genCost = pyo.Param(model.GEN, within=pyo.Reals, 
                                  mutable=True)
        model.pumpCost = pyo.Param(model.GEN_PUMP, within=pyo.Reals, 
                                  mutable=True)
        model.flexLoadCost = pyo.Param(model.LOAD_FLEX,
                                       within=pyo.NonNegativeReals,
                                       mutable=True)
        model.loadShedCost = pyo.Param(within=pyo.NonNegativeReals)
        model.branchAcCapacity = pyo.Param(model.BRANCH_AC, 
                                         within=pyo.NonNegativeReals)
        model.branchDcCapacity = pyo.Param(model.BRANCH_DC, 
                                         within=pyo.NonNegativeReals)    
        model.genPmaxLimit = pyo.Param(model.GEN,within=pyo.NonNegativeReals)
        model.pumpCapacity = pyo.Param(model.GEN_PUMP,within=pyo.Reals)
        model.flexloadMax = pyo.Param(model.LOAD_FLEX,
                                       within=pyo.NonNegativeReals)
        model.refNodes = pyo.Param(model.NODE, within=pyo.Boolean)

        # helpers:
        model.genNode = pyo.Param(model.GEN,within=model.NODE)
        model.demNode = pyo.Param(model.LOAD,within=model.NODE)
        model.branchNodeFrom = pyo.Param(model.BRANCH_AC,within=model.NODE)
        model.branchNodeTo = pyo.Param(model.BRANCH_AC,within=model.NODE)
        model.dcbranchNodeFrom = pyo.Param(model.BRANCH_AC,within=model.NODE)
        model.dcbranchNodeTo = pyo.Param(model.BRANCH_AC,within=model.NODE)
        model.demand = pyo.Param(model.LOAD,within=pyo.Reals, 
                                 mutable=True)
        model.coeff_B = pyo.Param(model.NODE,model.NODE,within=pyo.Reals)
        model.coeff_DA = pyo.Param(model.BRANCH_AC,model.NODE,within=pyo.Reals)

        
        # VARIABLES ##########################################################
            
        model.varAcBranchFlow = pyo.Var(model.BRANCH_AC,within = pyo.Reals)
        model.varDcBranchFlow = pyo.Var(model.BRANCH_DC,within = pyo.Reals)        
        model.varGeneration = pyo.Var(model.GEN,within = pyo.NonNegativeReals)
        model.varPump = pyo.Var(model.GEN_PUMP, within = pyo.NonNegativeReals)
        model.varCurtailment  = pyo.Var(model.GEN,
                                        within = pyo.NonNegativeReals)        
        model.varFlexLoad = pyo.Var(model.LOAD_FLEX, 
                                    within = pyo.NonNegativeReals)
        model.varLoadShed = pyo.Var(model.LOAD, within = pyo.NonNegativeReals) 
        model.varVoltageAngle = pyo.Var(model.NODE, within = pyo.Reals)
# I wonder if these bound on voltage angle creates infeasibility
# - is it really needed
# TODO: Verify voltage angle bounds required
#                                        bounds = (-np.pi,np.pi)) 
        
#        # not needed because limit is set by constraint        
#        def ubFlexLoad_rule(model,i):
#            ub =( model.flexload_demand_avg[i]
#                    * model.flexload_flex_fraction[i]
#                    / model.flexload_on_off[i] )
#            return ub

        # CONSTRAINTS ########################################################

        # 1 Power flow limit   (AC branches)   
        def maxflowAc_rule(model, j):
            cap = model.branchAcCapacity[j]
            if  not np.isinf(cap):
                expr = (-cap <= model.varAcBranchFlow[j] <= cap )
            else:
                expr = pyo.Constraint.Skip
            return expr
                        
        model.cMaxFlowAc = pyo.Constraint(model.BRANCH_AC, rule=maxflowAc_rule)
                                        
        # Power flow limit   (DC branches)   
        def maxflowDc_rule(model, j):
            cap = model.branchDcCapacity[j]
            expr = (-cap <= model.varDcBranchFlow[j] <= cap )
            return expr
                        
        model.cMaxFlowDc = pyo.Constraint(model.BRANCH_DC, rule=maxflowDc_rule)
        
        # 2 Generator output limit                                 
        def Pgen_rule(model,g):
            expr = model.varGeneration[g] <=  model.genPmaxLimit[g]
            return expr
        
        model.cMaxPgen = pyo.Constraint(model.GEN, rule=Pgen_rule)
                    
        # 3 Pump output limit                                 
        def pump_rule(model,g):
            expr = (model.varPump[g] <= model.pumpCapacity[g])
            return expr
        
        model.cPump = pyo.Constraint(model.GEN_PUMP, rule=pump_rule)

        # 4 Flexible load limit                                 
        def flexload_rule(model,i):
            expr = (model.varFlexLoad[i] <= model.flexloadMax[i])
            return expr
        
        model.cFlexload = pyo.Constraint(model.LOAD_FLEX,rule=flexload_rule)
               
        #5 Power balance (power flow equation)  (Pnode = B theta)
        def powerbalance_rule(model,n):
            lhs = 0            
            for g in model.GEN:
                if model.genNode[g]==n:
                    lhs += model.varGeneration[g]
            for p in model.GEN_PUMP:
                if model.genNode[p]==n:
                    lhs -= model.varPump[p]
            for l in model.LOAD:
                if model.demNode[l]==n:
                    lhs -= model.demand[l]
                    lhs += model.varLoadShed[l]
            for f in model.LOAD_FLEX:
                if model.demNode[f]==n:
                    lhs -= model.varFlexLoad[f]
            for b in model.BRANCH_DC:
                # positive sign for flow into node
                if model.dcbranchNodeTo[b]==n:
                    lhs += model.varDcBranchFlow[b]
                elif model.dcbranchNodeFrom[b]==n:
                    lhs -= model.varDcBranchFlow[b]

            lhs = lhs/const.baseMVA
            
            rhs = 0
            n2s = [k[1]  for k in model.coeff_B.keys() if k[0]==n]
            for n2 in n2s:
                rhs -= model.coeff_B[n,n2]*model.varVoltageAngle[n2]                
            expr = (lhs == rhs)
            #Skip constraint if it is trivial (otherwise run-time error)
            #TODO: Check if this is safe
            if ((type(expr) is bool) and (expr==True)):
                expr = pyo.Constraint.Skip
            return expr

        model.cPowerbalance = pyo.Constraint(model.NODE,rule=powerbalance_rule)
        
        #6 Power balance (power flow vs voltage angle)                               
        def flowangle_rule(model,b):
            lhs = model.varAcBranchFlow[b]
            lhs = lhs/const.baseMVA
            rhs = 0
            #TODO speed up- remove for loop
            n2s = [k[1]  for k in model.coeff_DA.keys() if k[0]==b]
            for n2 in n2s:
                rhs += model.coeff_DA[b,n2]*model.varVoltageAngle[n2]                
            #for n2 in model.NODE:
            #    if (b,n2) in model.coeff_DA.keys():
            #        rhs += model.coeff_DA[b,n2]*model.varVoltageAngle[n2]
            expr = (lhs==rhs)
            return expr

        model.cFlowAngle = pyo.Constraint(model.BRANCH_AC, rule=flowangle_rule)

        #7 Reference voltag angle)        
        def referenceNode_rule(model,n):
            if n in model.refNodes.keys():
                expr = (model.varVoltageAngle[n] == 0)
            else:
                expr = pyo.Constraint.Skip
            return expr
        
        model.cReferenceNode = pyo.Constraint(model.NODE,
                                              rule=referenceNode_rule)                       

        # OBJECTIVE ##########################################################
    
        def cost_rule(model):
            """Operational costs: cost of gen, load shed and curtailment"""

            # Operational costs phase 1 (if stage2DeltaTime>0)
            cost = sum(model.varGeneration[i]*model.genCost[i]
                            for i in model.GEN)
            cost -= sum(model.varPump[i]*model.pumpCost[i]
                            for i in model.GEN_PUMP)
            cost -= sum(model.varFlexLoad[i]*model.flexLoadCost[i]
                            for i in model.LOAD_FLEX )
            cost += sum(model.varLoadShed[i]*model.loadShedCost
                            for i in model.LOAD )
            #cost += sum(model.varCurtailment[i,t]*model.curtailmentCost 
            #            for i in model.GEN for t in model.TIME)
                                          
            return cost
    
        model.OBJ = pyo.Objective(rule=cost_rule, sense=pyo.minimize)
        
    
        return model



    def _createModelData(self,grid_data):
        '''Create model data in dictionary format

        Parameters
        ----------
        grid_data : powergama.GridData object
            contains grid model
        
        Returns
        --------
        dictionary with pyomo data (in pyomo format)
        '''
                
        #to see how the data format is:        
        #data = pyo.DataPortal(model=self.abstractmodel)
        #data.load(filename=datafile)
        
        di = {}
        #Sets:
        di['NODE'] = {None: grid_data.node['id'].tolist() }
        di['BRANCH_AC'] = {None: grid_data.branch.index.tolist() }
        di['BRANCH_DC'] = {None: grid_data.dcbranch.index.tolist() }
        di['GEN'] = {None: grid_data.generator.index.tolist() }
        di['GEN_PUMP'] = {None: grid_data.getIdxGeneratorsWithPumping() }
        di['LOAD'] = {None: grid_data.consumer.index.tolist() }
        di['LOAD_FLEX'] = {None: grid_data.getIdxConsumersWithFlexibleLoad() }
        di['AREA'] = {None: grid_data.getAllAreas() }
                
        # PARAMETERS #########################################################

        #self._marginalcosts_flexload = asarray(grid.consumer['flex_basevalue'])      
        di['branchAcCapacity'] = grid_data.branch['capacity'].to_dict()
        di['branchDcCapacity'] = grid_data.dcbranch['capacity'].to_dict()
        di['genPmaxLimit'] = grid_data.generator['pmax'].to_dict()
        di['genCost'] = grid_data.generator['fuelcost'].to_dict()
        di['pumpCost'] = {k: grid_data.generator['fuelcost'][k]
                            for k in di['GEN_PUMP'][None] }
        di['pumpCapacity'] = {k: grid_data.generator['pump_cap'][k] 
                                for k in di['GEN_PUMP'][None] }
        di['flexLoadCost'] = {i: grid_data.consumer['flex_basevalue'][i] 
                                for i in di['LOAD_FLEX'][None] }
        di['flexloadMax'] = {i: (grid_data.consumer['demand_avg'][i]
                                    * grid_data.consumer['flex_fraction'][i]
                                    / grid_data.consumer['flex_on_off'][i])
                                for i in di['LOAD_FLEX'][None] }
        # use fixed load shedding penalty of 1000 €/MWh
        di['loadShedCost'] = {None: 1000}

        di['genNode'] = grid_data.generator['node'].to_dict()
        di['demNode'] = grid_data.consumer['node'].to_dict()
        di['branchNodeFrom'] = grid_data.branch['node_from'].to_dict() 
        di['branchNodeTo'] = grid_data.branch['node_to'].to_dict() 
        di['dcbranchNodeFrom'] = grid_data.dcbranch['node_from'].to_dict() 
        di['dcbranchNodeTo'] = grid_data.dcbranch['node_to'].to_dict() 
        di['demand'] = grid_data.consumer['demand_avg'].to_dict()


        # Compute matrices used in power flow equaions        
        print("Computing B and DA matrices...")        
        Bbus, DA = grid_data.computePowerFlowMatrices(const.baseZ)

        n_i = di['NODE'][None]
        b_i = di['BRANCH_AC'][None]
        di['coeff_B'] = dict()
        di['coeff_DA'] = dict()
        
        print("Creating B and DA coefficients...")        
        cx = scipy.sparse.coo_matrix(Bbus)
        for i,j,v in zip(cx.row, cx.col, cx.data):
            di['coeff_B'][(n_i[i],n_i[j])] = v

        cx = scipy.sparse.coo_matrix(DA)
        for i,j,v in zip(cx.row, cx.col, cx.data):
            di['coeff_DA'][(b_i[i],n_i[j])] = v

        # Find synchronous areas and specify reference node in each area
        G = nx.Graph()
        G.add_nodes_from(grid_data.node['id'])
        G.add_edges_from(zip(grid_data.branch['node_from'],
                             grid_data.branch['node_to']))

        G_subs = nx.connected_component_subgraphs(G)
        refnodes = []
        for gr in G_subs:
            refnode = gr.nodes()[0]
            refnodes.append(refnode)
            print("Found synchronous area (size = {}), using ref node = {}"
                    .format(gr.order(),refnode))
        # use first node as voltage angle reference
        di['refNodes'] = {n:True for n in refnodes}

        return {'powergama':di}





    def __init__(self,grid):
        '''LP problem formulation
        
        Parameters
        ==========
        grid : GridData
            grid data object
        '''
        
        # Pyomo
        # 1 create abstract pyomo model        
        self.abstractmodel = self._createAbstractModel()

        # 2 create concrete instance using grid data
        modeldata_dict = self._createModelData(grid)
        print('Creating LP problem instance...')
        self.concretemodel = self.abstractmodel.create_instance(
                                data=modeldata_dict,
                                name="PowerGAMA Model",
                                namespace='powergama')

        print('Initialising LP problem...')
        
        # Creating local variables to keep track of storage
        self._grid = grid
        self.timeDelta = grid.timeDelta
        self._idx_generatorsWithPumping = grid.getIdxGeneratorsWithPumping()        
        self._idx_generatorsWithStorage = grid.getIdxGeneratorsWithStorage()

        self._idx_consumersWithFlexLoad = (
            grid.getIdxConsumersWithFlexibleLoad() )
        self._idx_branchesWithConstraints = (
            grid.getIdxBranchesWithFlowConstraints() )
        self._fancy_progressbar = False        

        # Initial values of marginal costs, storage and storage values      
        self._storage = (
            grid.generator['storage_ini']*grid.generator['storage_cap'] )

        self._storage_flexload = (
                grid.consumer['flex_storagelevel_init']
                * grid.consumer['flex_storage']
                * grid.consumer['flex_fraction']
                * grid.consumer['demand_avg']
                )
                
        self._energyspilled = grid.generator['storage_cap'].copy(deep=True)
        self._energyspilled[:]=0
        
        return       




    def _updateLpProblem(self,timestep):
        '''
        Function that updates LP problem for a given timestep, due to changed
        power demand, power inflow and marginal generator costs
        '''


        # 1. Update bounds on maximum and minimum production (power inflow)
        P_storage = self._storage / self.timeDelta
        P_max = self._grid.generator['pmax']
        P_min = self._grid.generator['pmin']        
        for i in self.concretemodel.GEN:
            inflow_factor = self._grid.generator['inflow_fac'][i]
            capacity = self._grid.generator['pmax'][i]
            inflow_profile = self._grid.generator['inflow_ref'][i]
            P_inflow =  (capacity * inflow_factor 
                * self._grid.profiles[inflow_profile][timestep])
            if i not in self._idx_generatorsWithStorage:
                '''
                Don't let P_max limit the output (e.g. solar PV)
                This won't affect fuel based generators with zero storage,
                since these should have inflow=p_max in any case
                '''
                #TODO: change from bounds to constraints
                # (which gives interesting dual value, cf max flow)
                self.concretemodel.varGeneration[i].setlb(
                        min(P_inflow,P_min[i]))
                self.concretemodel.varGeneration[i].setub(P_inflow)
            else:
                #generator has storage
                # max(...) is used to get non-negative value
                # (due to numerical effects, storage may be slightly <0)
                self.concretemodel.varGeneration[i].setlb(
                        min(max(0,P_inflow+P_storage[i]), P_min[i]) )
                self.concretemodel.varGeneration[i].setub(
                        min(max(0,P_inflow+P_storage[i]), P_max[i]) )

                    
        # 2. Update demand (which affects powr balance constraint)
        for i in self.concretemodel.LOAD:
            #dem = self.concretemodel.demand[i]
            average = self._grid.consumer['demand_avg'][i]*(
                            1-self._grid.consumer['flex_fraction'][i])
            profile_ref = self._grid.consumer['demand_ref'][i]
            dem_new = self._grid.profiles[profile_ref][timestep] * average
            self.concretemodel.demand[i] = dem_new
        
        # 3. Update cost parameters (which affect the objective function)
        # 3a. generators with storage (storage value)        
        for i in self._idx_generatorsWithStorage:
            this_type_filling = self._grid.generator['storval_filling_ref'][i]
            this_type_time = self._grid.generator['storval_time_ref'][i]      
            storagecapacity = self._grid.generator['storage_cap'][i]
            fillinglevel = self._storage[i] / storagecapacity       
            filling_col = int(round(fillinglevel*100))
            storagevalue = (
                self._grid.generator['storage_price'][i] 
                *self._grid.storagevalue_filling[this_type_filling][filling_col]
                *self._grid.storagevalue_time[this_type_time][timestep])
            self.concretemodel.genCost[i] = storagevalue
            if i in self._idx_generatorsWithPumping:
                deadband = self._grid.generator.pump_deadband[i]
                self.concretemodel.pumpCost[i] = storagevalue - deadband
        
        # 3b. flexible load (storage value)            
        for i in self._idx_consumersWithFlexLoad:
            this_type_filling = self._grid.consumer['flex_storval_filling'][i]
            this_type_time = self._grid.consumer['flex_storval_time'][i]
            # Compute storage capacity in Mwh (from value in hours)
            storagecapacity_flexload = (
                self._grid.consumer['flex_storage'][i]      # h
                * self._grid.consumer['flex_fraction'][i]   #
                * self._grid.consumer['demand_avg'][i])     # MW
            fillinglevel = (
                self._storage_flexload[i] / storagecapacity_flexload  )     
            filling_col = int(round(fillinglevel*100))
            if fillinglevel > 1:
                storagevalue_flex = -const.flexload_outside_cost
            elif fillinglevel < 0:
                storagevalue_flex = const.flexload_outside_cost
            else:
                storagevalue_flex = (
                    self._grid.consumer.flex_basevalue[i] 
                    *self._grid.storagevalue_filling[this_type_filling][filling_col]
                    *self._grid.storagevalue_time[this_type_time][timestep])
            self.concretemodel.flexLoadCost[i] = storagevalue_flex
        
        
        return
        
        

    
    def _storeResultsAndUpdateStorage(self,timestep,results):
        """Store timestep results in local arrays, and update storage"""

                        
        # 1. Update generator storage:
        inflow_profile_refs = self._grid.generator['inflow_ref']
        inflow_factor = self._grid.generator['inflow_fac']
        capacity= self._grid.generator['pmax']
        pumpedIn = np.zeros(len(capacity))
        energyIn = np.zeros(len(capacity))
        energyOut = np.zeros(len(capacity))
        for i in self.concretemodel.GEN:
            genInflow = (capacity[i] * inflow_factor[i]
        			 * self._grid.profiles[inflow_profile_refs[i]][timestep] )                        
            energyIn[i] = genInflow*self.timeDelta
            energyOut[i] = (self.concretemodel.varGeneration[i].value
                            *self.timeDelta)
        for i in self._idx_generatorsWithPumping:
            Ppump = self.concretemodel.varPump[i].value
            pumpedIn[i] = (Ppump*self._grid.generator['pump_efficiency'][i] 
                            *self.timeDelta)
        energyStorable = (self._storage + energyIn + pumpedIn - energyOut)
        storagecapacity = self._grid.generator['storage_cap']
        #self._storage[i] = min(storagecapacity,energyStorable)
        self._storage = np.vstack((storagecapacity,energyStorable)).min(axis=0)
        self._energyspilled = energyStorable-self._storage

        # 2. Update flexible load storage
        for i in self._idx_consumersWithFlexLoad:
            energyIn_flexload = (self.concretemodel.varFlexLoad[i]
                                 *self.timeDelta)
            energyOut_flexload = ( self._grid.consumer['flex_fraction'][i]
                                    * self._grid.consumer['demand_avg'][i]
                                    * self.timeDelta )
            self._storage_flexload[i] += energyIn_flexload-energyOut_flexload
        
        
        # 3. Collect variable values from optimisation result
        F = self.concretemodel.OBJ()
        Pgen = [self.concretemodel.varGeneration[i].value
                for i in self.concretemodel.GEN]
        Ppump = [self.concretemodel.varPump[i].value
                for i in self.concretemodel.GEN_PUMP]
        Pflexload = [self.concretemodel.varFlexLoad[i].value
                for i in self.concretemodel.LOAD_FLEX]
        Pb = [self.concretemodel.varAcBranchFlow[i].value
                for i in self.concretemodel.BRANCH_AC]
        Pdc = [self.concretemodel.varDcBranchFlow[i].value
                for i in self.concretemodel.BRANCH_DC]
        theta = [self.concretemodel.varVoltageAngle[i].value
                for i in self.concretemodel.NODE]
        #load shedding is aggregated to nodes (due to old code)
        Ploadshed = pd.Series(index=self._grid.node.id,
                                 data=[0]*len(self._grid.node.id),
                                 dtype=float)
        for j in self.concretemodel.LOAD:
            node = self._grid.consumer['node'][j]
            Ploadshed[node] += self.concretemodel.varLoadShed[j].value

        # 4 Collect dual values
        # 4a. branch capacity sensitivity (whether pos or neg flow)        
        senseB = []
        for j in self._idx_branchesWithConstraints:
        #for j in self.concretemodel.BRANCH_AC:
            c = self.concretemodel.cMaxFlowAc[j]            
            senseB.append(-abs(self.concretemodel.dual[c]/const.baseMVA ))
        senseDcB = []
        for j in self.concretemodel.BRANCH_DC:
            c = self.concretemodel.cMaxFlowDc[j]            
            senseDcB.append(-abs(self.concretemodel.dual[c]/const.baseMVA ))

        # 4b. node demand sensitivity (energy balance)
        # TODO: Without abs(...) the value jumps between pos and neg. Why?
        senseN = []
        for j in self.concretemodel.NODE:
            c = self.concretemodel.cPowerbalance[j]         
            senseN.append(abs(self.concretemodel.dual[c]/const.baseMVA ))
            
        # consider spilled energy only for generators with storage<infinity
        #energyspilled = zeros(energyStorable.shape)
        #indx = self._grid.getIdxGeneratorsWithNonzeroInflow()
        #energyspilled[indx] = energyStorable[indx]-self._storage[indx] 
        energyspilled = self._energyspilled
        storagelevel = self._storage[self._idx_generatorsWithStorage]
        storageprice = [self.concretemodel.genCost[i].value
                            for i in self._idx_generatorsWithStorage]
        flexload_storagelevel = self._storage_flexload[self._idx_consumersWithFlexLoad]
        flexload_marginalprice = [self.concretemodel.flexLoadCost[i].value
                                    for i in self._idx_consumersWithFlexLoad]
        
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
            loadshed_power = Ploadshed.tolist(),
            marginalprice = storageprice,
            flexload_power = Pflexload,
            flexload_storage = flexload_storagelevel.tolist(),
            flexload_storagevalue = flexload_marginalprice)

        return
               
        
    def solve(self,results,solver='cbc',solver_path=None,warmstart=False,
              savefiles=False):
        '''
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
            
        Returns
        -------
        results : Results
            PowerGAMA Results object reference
        '''

        # Initalise solver, and check it is available
        if solver=="gurobi":
            opt = pyo.SolverFactory('gurobi',solver_io='python')
            print(":) Using direct python interface to solver")
        else:
            opt = pyo.SolverFactory(solver,executable=solver_path)                    
            if opt.available():
                print (":) Found solver here: {}".format(opt.executable()))
            else:
                print(":( Could not find solver {}. Returning."
                        .format(self.solver))     
                raise Exception("Could not find LP solver {}"
                                .format(self.solver))
        
        #Enable access to dual values
        self.concretemodel.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
       
        print("Solving...")
        numTimesteps = len(self._grid.timerange)
        count = 0
        warmstart_now=False
        for timestep in range(numTimesteps):
            # update LP problem (inflow, storage, profiles)                     
            self._updateLpProblem(timestep)
          
            # solve the LP problem
            if savefiles:
                self.concretemodel.pprint('concretemodel_{}.txt'.format(timestep))        
                self.concretemodel.write("LPproblem_{}.mps".format(timestep))

            if warmstart and opt.warm_start_capable():  
                #warmstart available (does not work with cbc)
                if count>0:
                    warmstart_now=warmstart
                count = count+1
                res = opt.solve(self.concretemodel, 
                        tee=False, #stream the solver output
                        keepfiles=False, #print the LP file for examination
                        warmstart=warmstart_now,
                        symbolic_solver_labels=True) # use human readable names 
            elif not warmstart:
                #no warmstart option
                res = opt.solve(self.concretemodel, 
                        tee=False, #stream the solver output
                        keepfiles=False, #print the LP file for examination
                        symbolic_solver_labels=True) # use human readable names 
            else:
                raise Exception("Solver ({}) is not capable of warm start"
                                    .format(opt.name))
                
            
            if (res.solver.status != pyomo.opt.SolverStatus.ok):
                warnings.warn("Something went wrong with LP solver: {}"
                                .format(res.solver.status))
            elif (res.solver.termination_condition 
                    == pyomo.opt.TerminationCondition.infeasible):
                warnings.warn("t={}: No feasible solution found."
                                .format(timestep))
                            
            self._update_progress(timestep,numTimesteps)
       
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
        '''Specify how to show simulation progress
		
        Parameters
        ----------
        value : string
            'fancy' or 'default'
        '''
        if value=='fancy':
            self._fancy_progressbar=True
        elif value=='default':
            self._fancy_progressbar=False
        else:
            raise Exception('Progress bar bust be either "default" or "fancy"')
        