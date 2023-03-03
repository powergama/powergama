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
import powergama

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

        # PARAMETERS #########################################################
        model.genCost = pyo.Param(model.GEN, within=pyo.Reals,
                                  mutable=True)
        model.pumpCost = pyo.Param(model.GEN_PUMP, within=pyo.Reals,
                                  mutable=True)
        model.flexLoadCost = pyo.Param(model.LOAD_FLEX,
                                       within=pyo.Reals,
                                       mutable=True)
        model.loadShedCost = pyo.Param(within=pyo.NonNegativeReals)
        model.branchAcCapacity = pyo.Param(model.BRANCH_AC,
                                         within=pyo.NonNegativeReals)
        model.branchDcCapacity = pyo.Param(model.BRANCH_DC,
                                         within=pyo.NonNegativeReals)
        model.genPmaxLimit = pyo.Param(model.GEN,within=pyo.NonNegativeReals,
                                       mutable = True)
        model.genPminLimit = pyo.Param(model.GEN,within=pyo.NonNegativeReals,
                                       mutable = True)
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

        if self._lossmethod==1:
            model.lossAcA = pyo.Param(model.BRANCH_AC,within=pyo.Reals,
                                        default=0,mutable=False)
            model.lossAcB = pyo.Param(model.BRANCH_AC,within=pyo.Reals,
                                        default=0,mutable=False)
            model.lossDcA = pyo.Param(model.BRANCH_DC,within=pyo.Reals,
                                        default=0,mutable=False)
            model.lossDcB = pyo.Param(model.BRANCH_DC,within=pyo.Reals,
                                        default=0,mutable=False)
        elif self._lossmethod==2:
            model.branchAcPowerLoss = pyo.Param(model.BRANCH_AC,within=pyo.Reals,
                                                default=0, mutable=True)
            model.branchDcPowerLoss = pyo.Param(model.BRANCH_AC,within=pyo.Reals,
                                                default=0, mutable=True)
            
        # VARIABLES ##########################################################
        model.varAcBranchFlow = pyo.Var(model.BRANCH_AC,within = pyo.Reals)
        model.varDcBranchFlow = pyo.Var(model.BRANCH_AC,within = pyo.Reals)
        if self._lossmethod==1:
            model.varAcBranchFlow12 = pyo.Var(model.BRANCH_AC,
                                              within = pyo.NonNegativeReals)
            model.varAcBranchFlow21 = pyo.Var(model.BRANCH_AC,
                                              within = pyo.NonNegativeReals)
            model.varDcBranchFlow12 = pyo.Var(model.BRANCH_DC,
                                              within = pyo.NonNegativeReals)
            model.varDcBranchFlow21 = pyo.Var(model.BRANCH_DC,
                                              within = pyo.NonNegativeReals)
            model.varLossAc12 = pyo.Var(model.BRANCH_AC,
                                              within = pyo.NonNegativeReals)
            model.varLossAc21 = pyo.Var(model.BRANCH_AC,
                                              within = pyo.NonNegativeReals)
            model.varLossDc12 = pyo.Var(model.BRANCH_DC,
                                              within = pyo.NonNegativeReals)
            model.varLossDc21 = pyo.Var(model.BRANCH_DC,
                                              within = pyo.NonNegativeReals)
        model.varGeneration = pyo.Var(model.GEN,within = pyo.NonNegativeReals)
        model.varPump = pyo.Var(model.GEN_PUMP, within = pyo.NonNegativeReals)
        model.varCurtailment  = pyo.Var(model.GEN,
                                        within = pyo.NonNegativeReals)
        model.varFlexLoad = pyo.Var(model.LOAD_FLEX,
                                    within = pyo.NonNegativeReals)
        model.varLoadShed = pyo.Var(model.LOAD, within = pyo.NonNegativeReals)
        model.varVoltageAngle = pyo.Var(model.NODE, within = pyo.Reals,
                                        initialize=0.0)
# I wonder if these bound on voltage angle creates infeasibility
# - is it really needed
# TODO: Verify voltage angle bounds required
#                                        bounds = (-np.pi,np.pi))


        # CONSTRAINTS ########################################################

        # 1 Power flow limit
        def maxflowAc_rule(model, j):
            cap = model.branchAcCapacity[j]
            if  not np.isinf(cap):
                expr = pyo.inequality(-cap, model.varAcBranchFlow[j], cap )
            else:
                expr = pyo.Constraint.Skip
            return expr
        def maxflowDc_rule(model, j):
            cap = model.branchDcCapacity[j]
            if  not np.isinf(cap):
                expr = pyo.inequality(-cap, model.varDcBranchFlow[j], cap )
            else:
                expr = pyo.Constraint.Skip
            return expr
        model.cMaxFlowAc = pyo.Constraint(model.BRANCH_AC, rule=maxflowAc_rule)
        model.cMaxFlowDc = pyo.Constraint(model.BRANCH_DC, rule=maxflowDc_rule)

        if self._lossmethod==1:
            def flowAc_rule(model, j):
                expr = (model.varAcBranchFlow[j] == 
                        model.varAcBranchFlow12[j]- model.varAcBranchFlow21[j])
                return expr
            def flowDc_rule(model, j):
                expr = (model.varDcBranchFlow[j] == 
                        model.varDcBranchFlow12[j] - model.varDcBranchFlow21[j])
                return expr
            model.cFlowAc = pyo.Constraint(model.BRANCH_AC, rule=flowAc_rule)
            model.cFlowDc = pyo.Constraint(model.BRANCH_DC, rule=flowDc_rule)

        # 1b Losses vs flow
        if self._lossmethod==1:
            def lossAc_rule12(model,j):
                expr = (model.varLossAc12[j] == 
                        model.varAcBranchFlow12[j] * model.lossAcA[j] 
                        + model.lossAcB[j])
                return expr
            def lossAc_rule21(model,j):
                expr = (model.varLossAc21[j] == 
                        model.varAcBranchFlow21[j] * model.lossAcA[j] 
                        + model.lossAcB[j])
                return expr
            def lossDc_rule12(model,j):
                expr = (model.varLossDc12[j] == 
                        model.varDcBranchFlow12[j] * model.lossDcA[j] 
                        + model.lossDcB[j])
                return expr
            def lossDc_rule21(model,j):
                expr = (model.varLossDc21[j] == 
                        model.varDcBranchFlow21[j] * model.lossDcA[j] 
                        + model.lossDcB[j])
                return expr
            model.cLossAc12 = pyo.Constraint(model.BRANCH_AC, rule=lossAc_rule12)
            model.cLossAc21 = pyo.Constraint(model.BRANCH_AC, rule=lossAc_rule21)    
            model.cLossDc12 = pyo.Constraint(model.BRANCH_DC, rule=lossDc_rule12)
            model.cLossDc21 = pyo.Constraint(model.BRANCH_DC, rule=lossDc_rule21)


        # 2 Generator output limit
        # Generator output constraint is not necessary, as lower and upper
        # bounds are set for each timestep in _update_progress. Should not
        # be specified as constraint with with pmax as limit, since e.g.
        # PV may have higher production than generator rating.
        
        #HGS: Doing it anyway, cf Espen Bødal and Martin Kristiansen
        #TODO: Check that there are no problems with this.
        def genMaxLimit_rule(model, i):
            return model.varGeneration[i] <= model.genPmaxLimit[i]
        model.cGenMaxLimit = pyo.Constraint(model.GEN, rule=genMaxLimit_rule)

        def genMinLimit_rule(model, i):
            if model.genPminLimit[i].value > 0:
                return model.varGeneration[i] >= model.genPminLimit[i]
            else:
                return pyo.Constraint.Skip
        model.cGenMinLimit = pyo.Constraint(model.GEN, rule=genMinLimit_rule)

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
        #TODO: Speed this up by not looping over all generators etc for each
        # node, use instead a dictionary with key=node, value=list of generators
        # make it from grp=pg_data.generator.groupby('node')
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
                    if self._lossmethod==1:
                        lhs -= model.varLossDc12[b]
                    elif self._lossmethod==2:
                        lhs -= model.branchDcPowerLoss[b]/2
                elif model.dcbranchNodeFrom[b]==n:
                    lhs += -model.varDcBranchFlow[b] 
                    if self._lossmethod==1:
                        lhs -= model.varLossDc21[b]
                    elif self._lossmethod==2:
                        lhs -= model.branchDcPowerLoss[b]/2
            if self._lossmethod==1:
                for b in model.BRANCH_AC:
                    # positive sign for flow into node
                    if model.branchNodeTo[b]==n:
                        lhs += -model.varLossAc12[b]
                    elif model.branchNodeFrom[b]==n:
                        lhs += -model.varLossAc21[b]
            elif self._lossmethod==2:
                for b in model.BRANCH_AC:
                    # positive sign for flow into node
                    if model.branchNodeTo[b]==n:
                        lhs -= model.branchAcPowerLoss[b]/2
                    elif model.branchNodeFrom[b]==n:
                        lhs -= model.branchAcPowerLoss[b]/2

            lhs = lhs/const.baseMVA

            rhs = 0
            n2s = [k[1]  for k in model.coeff_B.keys() if k[0]==n]
            for n2 in n2s:
                rhs -= model.coeff_B[n,n2]*(
                            model.varVoltageAngle[n2]*const.baseAngle)
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
                rhs += model.coeff_DA[b,n2]*(
                            model.varVoltageAngle[n2]*const.baseAngle)
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
        di['genPminLimit'] = grid_data.generator['pmin'].to_dict()
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
        di['loadShedCost'] = {None: powergama.constants.loadshedcost}

        di['genNode'] = grid_data.generator['node'].to_dict()
        di['demNode'] = grid_data.consumer['node'].to_dict()
        di['branchNodeFrom'] = grid_data.branch['node_from'].to_dict()
        di['branchNodeTo'] = grid_data.branch['node_to'].to_dict()
        di['dcbranchNodeFrom'] = grid_data.dcbranch['node_from'].to_dict()
        di['dcbranchNodeTo'] = grid_data.dcbranch['node_to'].to_dict()
        di['demand'] = grid_data.consumer['demand_avg'].to_dict()

        if self._lossmethod==1:
            # Upper capacity limit, since capacity may be infinit
            clip_mw = 500
            br = grid_data.branch
            di['lossAcA'] = (
                br['resistance']*br['capacity'].clip(upper=clip_mw)
                /const.baseMVA).to_dict()
            di['lossAcB'] = {b: 0 for b in di['BRANCH_AC'][None] }

            br = grid_data.dcbranch
            di['lossDcA'] = ( 
                br['resistance']*br['capacity'].clip(upper=clip_mw)
                /const.baseMVA).to_dict()
            di['lossDcB'] = {b: 0 for b in di['BRANCH_DC'][None] }

        # Compute matrices used in power flow equaions
        print("Computing B and DA matrices...")
        Bbus, DA = grid_data.compute_power_flow_matrices()

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

        G_subs = (G.subgraph(c) for c in nx.connected_components(G))
        #deprecated:
        #G_subs = nx.connected_component_subgraphs(G)
        refnodes = []
        for gr in G_subs:
            refnode = list(gr.nodes)[0]
            refnodes.append(refnode)
            print("Found synchronous area (size = {}), using ref node = {}"
                    .format(gr.order(),refnode))
        # use first node as voltage angle reference
        di['refNodes'] = {n:True for n in refnodes}

        return {'powergama':di}


    def __init__(self,grid,lossmethod=0):
        '''LP problem formulation

        Parameters
        ==========
        grid : GridData
            grid data object
        lossmethod : int
            loss method; 0=no losses, 1=linearised losses, 2=added as load
        '''
        self._lossmethod = lossmethod

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
            grid.generator['storage_ini']*grid.generator['storage_cap'] 
            ).fillna(0)

        self._storage_flexload = (
                grid.consumer['flex_storagelevel_init']
                * grid.consumer['flex_storage']
                * grid.consumer['flex_fraction']
                * grid.consumer['demand_avg']
                ).fillna(0)

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
                if P_min[i] > 0:
                    self.concretemodel.genPminLimit[i] = min(
                        P_inflow,P_min[i]) # Espen
                self.concretemodel.genPmaxLimit[i] = P_inflow # Espen
            else:
                #generator has storage
                if P_min[i] > 0:
                    self.concretemodel.genPminLimit[i] = min(
                        max(0,P_inflow+P_storage[i]), P_min[i]) # Espen
                self.concretemodel.genPmaxLimit[i] = min(
                    max(0,P_inflow+P_storage[i]), P_max[i]) # Espen


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


    def _updatePowerLosses(self,aclossmultiplier=1,dclossmultiplier=1):
        '''Compute power losses from OPF solution and update parameters'''
        if self._lossmethod==0:
            pass
        elif self._lossmethod==1:            
            # Use constant loss parameters
            # If loss parameters should change, they need to be declared
            # mutable=True
            pass
        elif self._lossmethod==2:
            # Losses from previous timestep added as load
            for b in self.concretemodel.BRANCH_AC:
#                # r and x are given in pu; theta
#                loss_pu = r * ((theta_to-theta_from)*const.baseAngle/x)**2
#                # convert from p.u. to physical unit
#                lossMVA = loss_pu*const.baseMVA
                #TODO: simpler (check and replace):
                r = self._grid.branch.loc[b,'resistance']
                lossMVA = r * self.concretemodel.varAcBranchFlow[b]**2/const.baseMVA
                # A multiplication factor to account for reactive current losses
                # (or more precicely, to get similar results as Giacomo in 
                # the SmartNet project)
                lossMVA = lossMVA * aclossmultiplier
                self.concretemodel.branchAcPowerLoss[b] = lossMVA           
            for b in self.concretemodel.BRANCH_DC:
                #TODO: Test this before adding
                r_pu = self._grid.dcbranch.loc[b,'resistance']
                p_pu = self.concretemodel.varDcBranchFlow[b]/const.baseMVA
                loss_pu = r_pu * p_pu**2
                lossMVA = loss_pu * const.baseMVA * dclossmultiplier
                self.concretemodel.branchDcPowerLoss[b] = lossMVA
                #self.concretemodel.branchDcPowerLoss[b] = 0.0
        else:
            raise Exception("Loss method={} is not implemented"
                            .format(self._lossmethod))
        



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
            energyIn_flexload = (self.concretemodel.varFlexLoad[i].value
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
        theta = [self.concretemodel.varVoltageAngle[i].value*const.baseAngle
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

        # Extract power losses
        if self._lossmethod==0:
            acPowerLoss = [0]*len(self.concretemodel.BRANCH_AC)
            dcPowerLoss = [0]*len(self.concretemodel.BRANCH_DC)
        elif self._lossmethod==1:
            acPowerLoss = [self.concretemodel.varLossAc12[b].value
                           +self.concretemodel.varLossAc21[b].value
                            for b in self.concretemodel.BRANCH_AC]
            dcPowerLoss = [self.concretemodel.varLossDc12[b].value
                           +self.concretemodel.varLossDc21[b].value
                            for b in self.concretemodel.BRANCH_DC]
#            acPowerLoss = [pyo.value(self.concretemodel.varAcBranchFlow12[b]
#                            *self.concretemodel.lossAcA[b]
#                                + self.concretemodel.lossAcB[b]
#                           +self.concretemodel.varAcBranchFlow21[b]
#                            *self.concretemodel.lossAcA[b]
#                                + self.concretemodel.lossAcB[b])
#                            for b in self.concretemodel.BRANCH_AC]
#            dcPowerLoss = [pyo.value(self.concretemodel.varDcBranchFlow12[b]
#                            *self.concretemodel.lossDcA[b] 
#                                + self.concretemodel.lossDcB[b]
#                           +self.concretemodel.varDcBranchFlow21[b]
#                            *self.concretemodel.lossDcA[b] 
#                                + self.concretemodel.lossDcB[b])
#                            for b in self.concretemodel.BRANCH_DC]
        elif self._lossmethod==2:
            acPowerLoss = list(self.concretemodel.branchAcPowerLoss.extract_values().values())
            dcPowerLoss = list(self.concretemodel.branchDcPowerLoss.extract_values().values())
        else:
            raise Exception("Lossmethod must be 0,1 or 2")

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
            flexload_storagevalue = flexload_marginalprice,
            branch_ac_losses = acPowerLoss,
            branch_dc_losses = dcPowerLoss
            )

        return


    def solve(self,results,solver='cbc',solver_path=None,warmstart=False,
              savefiles=False,aclossmultiplier=1,
              dclossmultiplier=1,logfile="lpsolver_log.txt"):
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
        aclossmultiplier : float
            Multiplier factor to scale computed AC losses, used with method 1
        dclossmultiplier : float
            Multiplier factor to scale computed DC losses, used with method 1
        logfile : string
            Name of log file for LP solver. Will keep only last iteration

        Returns
        -------
        results : Results
            PowerGAMA Results object reference
        '''

        # Initalise solver, and check it is available
        if solver=="gurobi":
            opt = pyo.SolverFactory('gurobi',solver_io='python')
            print(":) Using direct python interface to solver")
        elif solver=="gurobi_persistent":
            opt = pyo.SolverFactory('gurobi_persistent')
            print(":) Using persistent (in-memory) python interface to solver")
            print(" => remember to notify solver of model changes!")
            print("    https://pyomo.readthedocs.io/en/latest/solvers/persistent_solvers.html")
        else:
            solver_io = None
            #if solver=="cbc":
            # NL requres CBC with ampl interface built in
            #    solver_io="nl" 
            opt = pyo.SolverFactory(solver,executable=solver_path,
                                    solver_io=solver_io) 
            if opt.available():
                print (":) Found solver here: {}".format(opt.executable()))
            else:
                print(":( Could not find solver {}. Returning."
                        .format(solver))
                raise Exception("Could not find LP solver {}"
                                .format(solver))

        #TODO: Code for persistent solver
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

        #Enable access to dual values
        self.concretemodel.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        if self._lossmethod==2:
            print("Computing losses in first timestep")
            self._updateLpProblem(timestep=0)
            res = opt.solve(self.concretemodel,
                        tee=False, #stream the solver output
                        keepfiles=False, #print the LP file for examination
                        symbolic_solver_labels=True) # use human readable names
                         
            #Now, power flow values are computed for the first timestep, and
            # power losses can be computed.
            
        print("Solving...")
        numTimesteps = len(self._grid.timerange)
        count = 0
        warmstart_now=False
        for timestep in range(numTimesteps):
            # update LP problem (inflow, storage, profiles)
            self._updateLpProblem(timestep)
            self._updatePowerLosses(aclossmultiplier,dclossmultiplier)

            # solve the LP problem
            if savefiles:
                #self.concretemodel.pprint('concretemodel_{}.txt'.format(timestep))
                self.concretemodel.write("LPproblem_{}.mps".format(timestep),
                                 io_options={'symbolic_solver_labels':True})
                #self.concretemodel.write("LPproblem_{}.nl".format(timestep))

            if warmstart and opt.warm_start_capable():
                #warmstart available (does not work with cbc)
                if count>0:
                    warmstart_now=warmstart
                count = count+1
                res = opt.solve(self.concretemodel,
                        tee=False, #stream the solver output
                        keepfiles=False, #print the LP file for examination
                        warmstart=warmstart_now,
                        symbolic_solver_labels=True, # use human readable names
                        logfile=logfile)
            elif not warmstart:
                #no warmstart option
                res = opt.solve(self.concretemodel,
                        tee=False, #stream the solver output
                        keepfiles=False, #print the LP file for examination
                        symbolic_solver_labels=True, # use human readable names
                        logfile=logfile)
            else:
                raise Exception("Solver ({}) is not capable of warm start"
                                    .format(opt.name))

            # Results loaded automatically, so this is not required
            # self.concretemodel.solutions.load_from(res)

            # store result for inspection if necessary
            self.solver_res = res

            #debugging:
            if False:
                print("Solver status = {}. Termination condition = {}"
                    .format(res.solver.status,
                            res.solver.termination_condition))

            if (res.solver.status != pyomo.opt.SolverStatus.ok):
                warnings.warn("Something went wrong with LP solver: {}"
                                .format(res.solver.status))
                raise Exception("Something went wrong with LP solver: {}"
                                .format(res.solver.status))
            elif (res.solver.termination_condition
                    == pyomo.opt.TerminationCondition.infeasible):
                warnings.warn("t={}: No feasible solution found."
                                .format(timestep))
                raise Exception("t={}: No feasible solution found."
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
