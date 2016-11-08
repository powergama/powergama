# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 13:21:21 2016

@author: Martin Kristiansen, Harald Svendsen
"""


import pyomo.environ as pyo
import pandas as pd
import numpy as np


class SipModel():
    '''
    Power Grid Investment Module - stochastic investment problem
    '''
    
    _NUMERICAL_THRESHOLD_ZERO = 1e-6
    _HOURS_PER_YEAR = 8760
    
    def __init__(self, maxNewBranchNum, M_const = 1000, CO2price=False):
        """Create Abstract Pyomo model for PowerGIM"""
        self.abstractmodel = self._createAbstractModel(maxNewBranchNum,
                                                       CO2price)
        self.M_const = M_const

        
        
    
    def _createAbstractModel(self,maxNewBranchNum,CO2price):    
        model = pyo.AbstractModel()
        model.name = 'PowerGIM abstract model'
        
        # SETS ###############################################################
        
        model.NODE = pyo.Set()
        model.GEN = pyo.Set()
        model.BRANCH = pyo.Set()
        model.LOAD = pyo.Set()
        model.AREA = pyo.Set()
        model.TIME = pyo.Set()
        
        #TODO: could simplify if we had a set for phase/stage - instead of
        # having double set of variables, e.g. generation1 and generation2
        #generation(g,t,stage)
        
        #A set for each stage i.e. a list with two sets
        model.NODE_EXPAND1 = pyo.Set()
        model.NODE_EXPAND2 = pyo.Set()
        model.BRANCH_EXPAND1 = pyo.Set()
        model.BRANCH_EXPAND2 = pyo.Set()
        model.GEN_EXPAND1 = pyo.Set()
        model.GEN_EXPAND2 = pyo.Set()
        
        model.BRANCHTYPE = pyo.Set()
        model.BRANCHCOSTITEM = pyo.Set(initialize=['B','Bd', 'Bdp', 
                                                   'CLp','CL','CSp','CS'])        
        model.NODETYPE = pyo.Set()
        model.NODECOSTITEM = pyo.Set(initialize=['L','S'])
        model.LINEAR = pyo.Set(initialize=['fix','slope'])
        
        model.GENTYPE = pyo.Set()
        

        # PARAMETERS #########################################################
        model.samplefactor = pyo.Param(within=pyo.NonNegativeReals)
        model.financeInterestrate = pyo.Param(within=pyo.Reals)
        model.financeYears = pyo.Param(within=pyo.Reals)
        model.omRate = pyo.Param(within=pyo.Reals)
        model.curtailmentCost = pyo.Param(within=pyo.NonNegativeReals)
        model.CO2price = pyo.Param(within=pyo.NonNegativeReals)
        model.VOLL = pyo.Param(within=pyo.NonNegativeReals)
        model.stage2TimeDelta = pyo.Param(within=pyo.NonNegativeReals)
        
        #investment costs and limits:        
        model.branchtypeMaxCapacity = pyo.Param(model.BRANCHTYPE,
                                                within=pyo.Reals)
        model.branchMaxNewCapacity = pyo.Param(model.BRANCH,within=pyo.Reals)
        model.branchtypeCost = pyo.Param(model.BRANCHTYPE, 
                                         model.BRANCHCOSTITEM,
                                         within=pyo.Reals)
        model.branchLossfactor = pyo.Param(model.BRANCHTYPE,model.LINEAR,
                                     within=pyo.Reals)
        model.nodetypeCost = pyo.Param(model.NODETYPE, model.NODECOSTITEM,
                                       within=pyo.Reals)
        model.genTypeCost = pyo.Param(model.GENTYPE, within=pyo.Reals)
        model.nodeCostScale = pyo.Param(model.NODE,within=pyo.Reals)
        model.branchCostScale = pyo.Param(model.BRANCH,within=pyo.Reals)
        model.genCostScale = pyo.Param(model.GEN, within=pyo.Reals)
        model.genNewCapMax = pyo.Param(model.GEN, within=pyo.Reals)
        
        #branches:
        model.branchExistingCapacity = pyo.Param(model.BRANCH, 
                                                 within=pyo.NonNegativeReals)
        model.branchExistingCapacity2 = pyo.Param(model.BRANCH, 
                                                 within=pyo.NonNegativeReals)
        model.branchExpand = pyo.Param(model.BRANCH,
                                       within=pyo.NonNegativeIntegers)         
        model.branchDistance = pyo.Param(model.BRANCH, 
                                         within=pyo.NonNegativeReals)                                             
        model.branchType = pyo.Param(model.BRANCH,within=model.BRANCHTYPE)
        model.branchOffshoreFrom = pyo.Param(model.BRANCH,within=pyo.Binary)
        model.branchOffshoreTo = pyo.Param(model.BRANCH,within=pyo.Binary)
    
        #nodes:
        model.nodeExistingNumber = pyo.Param(model.NODE, 
                                             within=pyo.NonNegativeIntegers)
        model.nodeOffshore = pyo.Param(model.NODE, within=pyo.Binary)
        model.nodeType = pyo.Param(model.NODE, within=model.NODETYPE)
        
        #generators
        model.genCostAvg = pyo.Param(model.GEN, within=pyo.Reals)
        model.genCostProfile = pyo.Param(model.GEN,model.TIME,
                                         within=pyo.Reals)
        model.genCapacity = pyo.Param(model.GEN,within=pyo.Reals)
        model.genCapacity2 = pyo.Param(model.GEN,within=pyo.Reals)
        model.genCapacityProfile = pyo.Param(model.GEN,model.TIME,
                                          within=pyo.Reals)
        model.genPAvg = pyo.Param(model.GEN,within=pyo.Reals)
        model.genType = pyo.Param(model.GEN, within=model.GENTYPE)
        model.genExpand = pyo.Param(model.GEN, 
                                    within=pyo.NonNegativeIntegers)
        model.genTypeEmissionRate = pyo.Param(model.GENTYPE, within=pyo.Reals)
        
        #helpers:
        model.genNode = pyo.Param(model.GEN,within=model.NODE)
        model.demNode = pyo.Param(model.LOAD,within=model.NODE)
        model.branchNodeFrom = pyo.Param(model.BRANCH,within=model.NODE)
        model.branchNodeTo = pyo.Param(model.BRANCH,within=model.NODE)
        model.nodeArea = pyo.Param(model.NODE,within=model.AREA)
        
        #consumers
        # the split int an average value, and a profile is to make it easier
        # to generate scenarios (can keep profile, but adjust demandAvg)
        model.demandAvg = pyo.Param(model.LOAD,within=pyo.Reals)
        model.demandProfile = pyo.Param(model.LOAD,model.TIME,
                                        within=pyo.Reals)
        model.emissionCap = pyo.Param(model.LOAD, within=pyo.NonNegativeReals)
        
        # VARIABLES ##########################################################
    
        # investment: new branch capacity
        def branchNewCapacity_bounds(model,j):
            return (0,model.branchMaxNewCapacity[j])
        model.branchNewCapacity1 = pyo.Var(model.BRANCH_EXPAND1, 
                                          within = pyo.NonNegativeReals,
                                          bounds = branchNewCapacity_bounds)                                  
        model.branchNewCapacity2 = pyo.Var(model.BRANCH_EXPAND2, 
                                          within = pyo.NonNegativeReals,
                                          bounds = branchNewCapacity_bounds)
        # investment: new branch cables
        def branchNewCables_bounds(model,j):
            return (0,maxNewBranchNum)                                  
        model.branchNewCables1 = pyo.Var(model.BRANCH_EXPAND1, 
                                        within = pyo.NonNegativeIntegers,
                                        bounds = branchNewCables_bounds)
                                        
        model.branchNewCables2 = pyo.Var(model.BRANCH_EXPAND2, 
                                        within = pyo.NonNegativeIntegers,
                                        bounds = branchNewCables_bounds)

        # investment: new nodes
        model.newNodes1 = pyo.Var(model.NODE_EXPAND1, within = pyo.Binary)
        model.newNodes2 = pyo.Var(model.NODE_EXPAND2, within = pyo.Binary)
        
        # investment: generation capacity
        def genNewCapacity_bounds(model,g):
            return (0,model.genNewCapMax[g])
        model.genNewCapacity1 = pyo.Var(model.GEN_EXPAND1,
                                       within = pyo.NonNegativeReals,
                                       bounds = genNewCapacity_bounds)
        model.genNewCapacity2 = pyo.Var(model.GEN_EXPAND2,
                                       within = pyo.NonNegativeReals,
                                       bounds = genNewCapacity_bounds)
        
        # branch power flow (phase 1 and phase 2)
        def branchFlow_bounds(model,j,t):
            ub = (model.branchExistingCapacity[j]
                    +branchNewCapacity_bounds(model,j)[1])
            return (0,ub)
        model.branchFlow12_1 = pyo.Var(model.BRANCH, model.TIME, 
                                     within = pyo.NonNegativeReals,
                                     bounds = branchFlow_bounds)
        model.branchFlow12_2 = pyo.Var(model.BRANCH, model.TIME, 
                                     within = pyo.NonNegativeReals,
                                     bounds = branchFlow_bounds)
        model.branchFlow21_1 = pyo.Var(model.BRANCH, model.TIME, 
                                     within = pyo.NonNegativeReals,
                                     bounds = branchFlow_bounds)
        model.branchFlow21_2 = pyo.Var(model.BRANCH, model.TIME, 
                                     within = pyo.NonNegativeReals,
                                     bounds = branchFlow_bounds)

        
        # generator output (bounds set by constraint)
        model.generation1 = pyo.Var(model.GEN, model.TIME, 
                                   within = pyo.NonNegativeReals)
        model.generation2 = pyo.Var(model.GEN, model.TIME, 
                                   within = pyo.NonNegativeReals)

        # load shedding (cf gen)
        #model.loadShed = pyo.Var(model.NODE, model.TIME, 
        #                         domain = pyo.NonNegativeReals) 
        
        # bounds are given in the curtailment restriction
        model.curtailment1  = pyo.Var(model.GEN, model.TIME, 
                                    domain = pyo.NonNegativeReals)
        model.curtailment2  = pyo.Var(model.GEN, model.TIME, 
                                    domain = pyo.NonNegativeReals)
        
        
        # CONSTRAINTS ########################################################

        # Power flow limitations (in both directions, in phase 1 and 2)        
#        def maxflow12_rule(model, j, t, phase):
#            cap=model.branchExistingCapacity[j]
#            if j in model.BRANCH_EXPAND[phase]:
#                cap += model.branchNewCapacity[phase][j]                
#            expr = (model.branchFlow12[phase][j,t] <= cap )
#            return expr
#        model.cMaxFlow12 = [pyo.Constraint(model.BRANCH, model.TIME, 0, 
#                                         rule=maxflow12_rule),
#                            pyo.Constraint(model.BRANCH, model.TIME, 1,
#                                         rule=maxflow12_rule)]
        #phase 1        
        def maxflow12_rule1(model, j, t):
            cap = model.branchExistingCapacity[j]
            if j in model.BRANCH_EXPAND1:
                cap += model.branchNewCapacity1[j]
            expr = (model.branchFlow12_1[j,t] <= cap )
            return expr
        
        #phase 2
        def maxflow12_rule2(model, j, t):
            cap = (model.branchExistingCapacity[j]
                    +model.branchExistingCapacity2[j])
            if j in model.BRANCH_EXPAND1:
                cap += model.branchNewCapacity1[j]
            if j in model.BRANCH_EXPAND2:
                cap += model.branchNewCapacity2[j]
            expr = (model.branchFlow12_2[j,t] <= cap)
            return expr
        
        #phase 1
        def maxflow21_rule1(model, j, t):
            cap = model.branchExistingCapacity[j]
            if j in model.BRANCH_EXPAND1:
                cap += model.branchNewCapacity1[j]
            expr = (model.branchFlow21_1[j,t] <= cap )
            return expr
        
        #phase 2
        def maxflow21_rule2(model, j, t):
            cap = (model.branchExistingCapacity[j]
                    +model.branchExistingCapacity2[j])
            if j in model.BRANCH_EXPAND1:
                cap += model.branchNewCapacity1[j]
            if j in model.BRANCH_EXPAND2:
                cap += model.branchNewCapacity2[j]
            expr = (model.branchFlow21_2[j,t] <= cap)
            return expr
        model.cMaxFlow12_1 = pyo.Constraint(model.BRANCH, model.TIME, 
                                         rule=maxflow12_rule1)
        model.cMaxFlow12_2 = pyo.Constraint(model.BRANCH, model.TIME, 
                                         rule=maxflow12_rule2)
        model.cMaxFlow21_1 = pyo.Constraint(model.BRANCH, model.TIME, 
                                         rule=maxflow21_rule1)
        model.cMaxFlow21_2 = pyo.Constraint(model.BRANCH, model.TIME, 
                                         rule=maxflow21_rule2)
                                         
        # No new branch capacity without new cables
        def maxNewCap_rule1(model,j):
            typ = model.branchType[j]
            expr = (model.branchNewCapacity1[j] 
                    <= model.branchtypeMaxCapacity[typ]
                        *model.branchNewCables1[j])
            return expr
        def maxNewCap_rule2(model,j):
            typ = model.branchType[j]
            expr = (model.branchNewCapacity2[j] 
                    <= model.branchtypeMaxCapacity[typ]
                        *model.branchNewCables2[j])
            return expr
        model.cmaxNewCapacity1 = pyo.Constraint(model.BRANCH_EXPAND1,
                                               rule=maxNewCap_rule1)
        model.cmaxNewCapacity2 = pyo.Constraint(model.BRANCH_EXPAND2,
                                               rule=maxNewCap_rule2)

# #Not required anymore since using BRANCH_EXPAND set instead of BRANCH set        
#        def newBranches_rule(model,j):
#            if model.branchExpand[j]==0:
#                expr = model.branchNewCables[j]==0
#                return expr
#            else:
#                return pyo.Constraint.Skip
#        model.cNewBranches = pyo.Constraint(model.BRANCH,
#                                            rule=newBranches_rule)
                                            
        # A node required at each branch endpoint
        def newNodes_rule1(model,n):
            expr = 0
            numnodes = model.nodeExistingNumber[n]
            if n in model.NODE_EXPAND1:
                numnodes += model.newNodes1[n]
            for j in model.BRANCH_EXPAND1:
                if model.branchNodeFrom[j]==n or model.branchNodeTo[j]==n:
                    expr += model.branchNewCables1[j]
            expr = expr <= self.M_const * numnodes
            if ((type(expr) is bool) and (expr==True)):
                expr = pyo.Constraint.Skip
            return expr
                    

        def newNodes_rule2(model,n):
            expr = 0
            numnodes = model.nodeExistingNumber[n]
            if n in model.NODE_EXPAND1:
                numnodes += model.newNodes1[n]
            if n in model.NODE_EXPAND2:
                numnodes += model.newNodes2[n]
            for j in model.BRANCH_EXPAND2:
                if model.branchNodeFrom[j]==n or model.branchNodeTo[j]==n:
                    expr += model.branchNewCables2[j]
            expr = expr <= self.M_const * numnodes
            if ((type(expr) is bool) and (expr==True)):
                expr = pyo.Constraint.Skip
            return expr
        
        model.cNewNodes1 = pyo.Constraint(model.NODE,
                                          rule=newNodes_rule1)
        model.cNewNodes2 = pyo.Constraint(model.NODE,
                                          rule=newNodes_rule2)
        
#        def newGenCapacity_rule1(model,g):
#            if model.genExpand[g] == 0:
#                expr = model.genNewCapacity[g] == 0
#            else:
#                expr = pyo.Constraint.Skip
#            return expr
#        model.cNewGenCapacity = pyo.Constraint(model.GEN_EXPAND1, 
#                                               rule=newGenCapacity_rule1)
                            
        
        # Generator output limitations

#        def maxPgen_rule(model,g,t,phase):
#            cap = model.genCapacity[g]
#            for ph in range(phase+1):
#                if g in model.GEN_EXPAND[ph]:
#                    cap += model.genNewCapacity[ph][g]
#            expr = model.generationTEST[g,t] <= (
#                model.genCapacityProfile[g,t] * cap)
#            #expr = model.generation[ph][g,t] <= (
#            #    model.genCapacityProfile[g,t] * cap)
#            return expr
            
        # phase 1
        def maxPgen_rule1(model,g,t):
            cap = model.genCapacity[g]
            if g in model.GEN_EXPAND1:
                cap += model.genNewCapacity1[g]
            expr = model.generation1[g,t] <= (
                model.genCapacityProfile[g,t] * cap)
            return expr
        
        # phase 2
        def maxPgen_rule2(model,g,t):
            cap = model.genCapacity[g] + model.genCapacity2[g]
            if g in model.GEN_EXPAND1:
                cap += model.genNewCapacity1[g]
            if g in model.GEN_EXPAND2:
                cap += model.genNewCapacity2[g]
            expr = model.generation2[g,t] <= (
                model.genCapacityProfile[g,t] * cap)
            return expr
            

        model.cMaxPgen1 = pyo.Constraint(model.GEN,model.TIME,
                                        rule=maxPgen_rule1)
        model.cMaxPgen2 = pyo.Constraint(model.GEN,model.TIME,
                                        rule=maxPgen_rule2)
#        if False:
#            def maxPgen_rule(model,g,t):
#                expr = model.generation[g,t] <= (model.genCapacityProfile[g,t]*
#                            model.genCapacity[g])
#                return expr
#            model.cMaxPgen = pyo.Constraint(model.GEN,model.TIME,
#                                            rule=maxPgen_rule)
        
        
        # Generator maximum average output (energy sum) 
        #(e.g. for hydro with storage)
        def maxPavg_rule1(model,g):
            if model.genPAvg[g]>0:
                expr = (sum(model.generation_phase1[g,t] for t in model.TIME) 
                            <= model.genPAvg[g]*len(model.TIME))
            else:
                expr = pyo.Constraint.Skip
            return expr
        def maxPavg_rule2(model,g):
            if model.genPAvg[g]>0:
                expr = (sum(model.generation_phase2[g,t] for t in model.TIME) 
                            <= model.genPAvg[g]*len(model.TIME))
            else:
                expr = pyo.Constraint.Skip
            return expr
        model.cMaxPavg1 = pyo.Constraint(model.GEN,
                                               rule=maxPavg_rule1)
        model.cMaxPavg2 = pyo.Constraint(model.GEN,
                                               rule=maxPavg_rule2)

        # Emissions restriction per country/load
        # TODO: Update in line with 2-stage/2-phase model
        if CO2price:
            def emissionCap_rule(model,a):
                expr = 0
                for n in model.NODE:
                    if model.nodeArea[n]==a:
                        expr += sum(model.generation[g,t]*model.genTypeEmissionRate[model.genType[g]] 
                                    for t in model.TIME for g in model.GEN 
                                    if model.genNode[g]==n)
                samplefactor = 8760/len(model.TIME)
                expr = (expr*samplefactor <= sum(model.emissionCap[c] for c in model.LOAD if model.nodeArea[model.demNode[c]]==a))
                return expr
            model.cEmissionCap = pyo.Constraint(model.AREA, rule=emissionCap_rule)
            

        def curtailment_rule1(model,g,t):
            # Only consider curtailment cost for zero cost generators
            if model.genCostAvg[g] == 0:
                gencap = model.genCapacity[g] 
                if g in model.GEN_EXPAND1:
                    gencap += + model.genNewCapacity1[g]
                expr =  (model.curtailment1[g,t] == (
                        model.genCapacityProfile[g,t]*gencap 
                        - model.generation1[g,t]))
                return expr
            else:
                return pyo.Constraint.Skip

        def curtailment_rule2(model,g,t):
            # Only consider curtailment cost for zero cost generators
            if model.genCostAvg[g] == 0:
                gencap = model.genCapacity[g] + model.genCapacity2[g] 
                if g in model.GEN_EXPAND1:
                    gencap += model.genNewCapacity1[g]
                if g in model.GEN_EXPAND2:
                    gencap += model.genNewCapacity2[g]
                expr =  (model.curtailment2[g,t] == (
                        model.genCapacityProfile[g,t]*gencap 
                            - model.generation2[g,t]))
                return expr
            else:
                return pyo.Constraint.Skip

        model.genCurtailment_phase1 = pyo.Constraint(model.GEN, model.TIME, 
                                              rule=curtailment_rule1)
        model.genCurtailment_phase2 = pyo.Constraint(model.GEN, model.TIME, 
                                              rule=curtailment_rule2)
#        else:
#            def curtailment_rule(model,g,t):
#                # Only consider curtailment cost for zero cost generators
#                if model.genCostAvg[g] == 0:
#                    expr =  (model.curtailment[g,t] 
#                        == model.genCapacityProfile[g,t]*(model.genCapacity[g]) 
#                            - model.generation[g,t])
#                    return expr
#                else:
#                    return pyo.Constraint.Skip
#            model.genCurtailment = pyo.Constraint(model.GEN, model.TIME, 
#                                                  rule=curtailment_rule)
       
        # Power balance in nodes : gen+demand+flow into node=0
        def powerbalance_rule1(model,n,t):
            expr = 0

            # flow of power into node (subtrating losses)
            for j in model.BRANCH:
                if model.branchNodeFrom[j]==n:
                    # branch out of node
                    typ = model.branchType[j]
                    dist = model.branchDistance[j]
                    expr += -model.branchFlow12_1[j,t]
                    expr += model.branchFlow21_1[j,t] * (1-(
                                model.branchLossfactor[typ,'fix']
                                +model.branchLossfactor[typ,'slope']*dist))
                if model.branchNodeTo[j]==n:
                    # branch into node
                    typ = model.branchType[j]
                    dist = model.branchDistance[j]
                    expr += model.branchFlow12_1[j,t] * (1-(
                                model.branchLossfactor[typ,'fix']
                                +model.branchLossfactor[typ,'slope']*dist))
                    expr += -model.branchFlow21_1[j,t] 

            # generated power 
            for g in model.GEN:
                if model.genNode[g]==n:
                    expr += model.generation1[g,t]

            # consumed power
            for c in model.LOAD:
                if model.demNode[c]==n:
                    expr += -model.demandAvg[c]*model.demandProfile[c,t]
            
            expr = (expr == 0)
            if ((type(expr) is bool) and (expr==True)):
                # Trivial constraint
                expr = pyo.Constraint.Skip
            return expr
            
        def powerbalance_rule2(model,n,t):
            expr = 0

            # flow of power into node (subtrating losses)
            for j in model.BRANCH:
                if model.branchNodeFrom[j]==n:
                    # branch out of node
                    typ = model.branchType[j]
                    dist = model.branchDistance[j]
                    expr += -model.branchFlow12_2[j,t]
                    expr += model.branchFlow21_2[j,t] * (1-(
                                model.branchLossfactor[typ,'fix']
                                +model.branchLossfactor[typ,'slope']*dist))
                if model.branchNodeTo[j]==n:
                    # branch into node
                    typ = model.branchType[j]
                    dist = model.branchDistance[j]
                    expr += model.branchFlow12_2[j,t] * (1-(
                                model.branchLossfactor[typ,'fix']
                                +model.branchLossfactor[typ,'slope']*dist))
                    expr += -model.branchFlow21_2[j,t] 

            # generated power 
            for g in model.GEN:
                if model.genNode[g]==n:
                    expr += model.generation2[g,t]

            # consumed power
            for c in model.LOAD:
                if model.demNode[c]==n:
                    expr += -model.demandAvg[c]*model.demandProfile[c,t]
            
            expr = (expr == 0)
            if ((type(expr) is bool) and (expr==True)):
                # Trivial constraint
                expr = pyo.Constraint.Skip
            return expr
            
        model.cPowerbalance_phase1 = pyo.Constraint(model.NODE,model.TIME,
                                             rule=powerbalance_rule1)
        model.cPowerbalance_phase2 = pyo.Constraint(model.NODE,model.TIME,
                                             rule=powerbalance_rule2)
        
        # COST PARAMETERS ############
        def costBranch(model,b,var_num,var_cap):
            b_cost = 0
            typ = model.branchType[b]
            b_cost += (model.branchtypeCost[typ,'B']
                        *var_num[b])
            b_cost += (model.branchtypeCost[typ,'Bd']
                        *model.branchDistance[b]
                        *var_num[b])
            b_cost += (model.branchtypeCost[typ,'Bdp']
                        *model.branchDistance[b]
                        *var_cap[b])
            
            #endpoints offshore (N=1) or onshore (N=0) ?
            N1 = model.branchOffshoreFrom[b]
            N2 = model.branchOffshoreTo[b]
            for N in [N1,N2]:
                b_cost += N*(model.branchtypeCost[typ,'CS']
                            *var_num[b]
                        +model.branchtypeCost[typ,'CSp']
                        *var_cap[b])            
                b_cost += (1-N)*(model.branchtypeCost[typ,'CL']
                            *var_num[b]
                        +model.branchtypeCost[typ,'CLp']
                        *var_cap[b])
            
            return model.branchCostScale[b]*b_cost

        def costNode(model,n,var_num):
            n_cost = 0
            N = model.nodeOffshore[n]
            n_cost += N*(model.nodetypeCost[model.nodeType[n],'S']
                        *var_num[n])
            n_cost += (1-N)*(model.nodetypeCost[model.nodeType[n],'L']
                        *var_num[n])
            return model.nodeCostScale[n]*n_cost
            
        def costGen(model,g,var_cap):
            g_cost = 0
            typ = model.genType[g]
            g_cost += model.genTypeCost[typ]*var_cap[g]
            return model.genCostScale[g]*g_cost

        #model.branchCost = pyo.Param(model.BRANCH, 
        #                                 within=pyo.NonNegativeReals,
        #                                 initialize=costBranch)                                             
        #model.nodeCost = pyo.Param(model.NODE, within=pyo.NonNegativeReals,
        #                           initialize=costNode)

        # OBJECTIVE ##############################################################
            
        
        def firstStageCost_rule(model):
            """Investment cost, including lifetime O&M costs (NPV)"""
            investment = 0

            # add branch, node and generator investment costs:
            for b in model.BRANCH_EXPAND1:
                investment += costBranch(model,b,
                                   var_num=model.branchNewCables1,
                                   var_cap=model.branchNewCapacity1)
            for n in model.NODE_EXPAND1:
                investment += costNode(model,n,var_num=model.newNodes1)            
            for g in model.GEN_EXPAND1:
                investment += costGen(model, g,var_cap=model.genNewCapacity1)

            # add O&M costs:
            omcost = investment * model.omRate * annuityfactor(
                            model.financeInterestrate,
                            model.financeYears)
            expr = investment + omcost                            
            return   expr  
        model.firstStageCost = pyo.Expression(rule=firstStageCost_rule)
    
        def secondStageCost_rule(model):
            """Operational costs: cost of gen, load shed and curtailment (NPV)"""

            # Operational costs phase 1 (if stage2DeltaTime>0)
            opcost1 = sum(model.generation1[i,t]*(
                            model.genCostAvg[i]*model.genCostProfile[i,t]
                            +model.genTypeEmissionRate[model.genType[i]]*model.CO2price)
                            for i in model.GEN for t in model.TIME)
            opcost1 += sum(model.curtailment1[i,t]*model.curtailmentCost 
                        for i in model.GEN for t in model.TIME)
            opcost1 = model.samplefactor*opcost1
            opcost1 = opcost1*annuityfactor(model.financeInterestrate,
                                          model.stage2TimeDelta)
                                          
            # Operational costs phase 2 (disounted)
            opcost2 = sum(model.generation2[i,t]*(
                            model.genCostAvg[i]*model.genCostProfile[i,t]
                            +model.genTypeEmissionRate[model.genType[i]]*model.CO2price)
                            for i in model.GEN for t in model.TIME)
            opcost2 += sum(model.curtailment2[i,t]*model.curtailmentCost 
                        for i in model.GEN for t in model.TIME)           
            opcost2 = model.samplefactor*opcost2
            opcost2 = opcost2*(annuityfactor(model.financeInterestrate,
                               model.financeYears)
                               -annuityfactor(model.financeInterestrate,
                                  model.stage2TimeDelta)
                                  )
                                 

            #loadshedding=0 by powerbalance constraint
            #expr += sum(model.loadShed[i,t]*model.shedCost[i] 
            #            for i in model.NODE for t in model.TIME)
            # annual costs of operation
            
            # 2nd stage investment costs
            investment = 0
            for b in model.BRANCH_EXPAND2:
                investment += costBranch(model,b,
                                   var_num=model.branchNewCables2,
                                   var_cap=model.branchNewCapacity2)
            for n in model.NODE_EXPAND2:
                investment += costNode(model,n,var_num=model.newNodes2)            
            for g in model.GEN_EXPAND2:
                investment += costGen(model, g,var_cap=model.genNewCapacity2)
            
            # add O&M costs (NPV of lifetime costs)
            omcost = investment*model.omRate*(
                annuityfactor(model.financeInterestrate,model.financeYears)
                -annuityfactor(model.financeInterestrate,model.stage2TimeDelta))

            expr = opcost1 + opcost2 + investment + omcost         
            return expr
        model.secondStageCost = pyo.Expression(rule=secondStageCost_rule)
    
        def total_Cost_Objective_rule(model):
            return model.firstStageCost + model.secondStageCost
        model.OBJ = pyo.Objective(rule=total_Cost_Objective_rule, 
                                  sense=pyo.minimize)
        
    
        return model

#    def _costBranch(self,model,b):
#        '''compute branch cost'''
#        b_cost = 0
#        typ = model.branchType[b]
#        b_cost += (model.branchtypeCost[typ,'B']
#                    *model.branchNewCables[b])
#        b_cost += (model.branchtypeCost[typ,'Bd']
#                    *model.branchDistance[b]
#                    *model.branchNewCables[b])
#        b_cost += (model.branchtypeCost[typ,'Bdp']
#                *model.branchDistance[b]*model.branchNewCapacity[b])
#        
#        #endpoints offshore (N=1) or onshore (N=0) ?
#        N1 = model.branchOffshoreFrom[b]
#        N2 = model.branchOffshoreTo[b]
#        for N in [N1,N2]:
#            b_cost += N*(model.branchtypeCost[typ,'CS']
#                        *model.branchNewCables[b]
#                    +model.branchtypeCost[typ,'CSp']
#                    *model.branchNewCapacity[b])            
#            b_cost += (1-N)*(model.branchtypeCost[typ,'CL']
#                        *model.branchNewCables[b]
#                    +model.branchtypeCost[typ,'CLp']
#                    *model.branchNewCapacity[b])
#        
#        return model.branchCostScale[b]*b_cost
        

    def _offshoreBranch(self,grid_data):
        '''find out whether branch endpoints are offshore or onshore
        
        Returns 1 for offshore and 0 for onsore from/to endpoints
        '''
        d={'from':[],'to':[]}
        
        d['from'] = [grid_data.node[grid_data.node['id']==n]['offshore']
                    .tolist()[0] for n in grid_data.branch['node_from']]
        d['to'] = [grid_data.node[grid_data.node['id']==n]['offshore']
                    .tolist()[0] for n in grid_data.branch['node_to']]
        return d

        
    def createConcreteModel(self,dict_data):
        """Create Concrete Pyomo model for PowerGIM
        
        Parameters
        ----------
        dict_data : dictionary
            dictionary containing the model data. This can be created with
            the createModelData(...) method
        
        Returns
        -------
            Concrete pyomo model
        """

        concretemodel = self.abstractmodel.create_instance(data=dict_data,
                               name="PowerGIM Model",
                               namespace='powergim')
        return concretemodel


    def createModelData(self,grid_data,datafile,maxNewBranchCap):
        '''Create model data in dictionary format

        Parameters
        ----------
        grid_data : powergama.GridData object
            contains grid model
        datafile : string
            name of XML file containing additional parameters
        
        Returns
        --------
        dictionary with pyomo data (in pyomo format)
        '''
        
        branch_distances = grid_data.branchDistances()
        
        #to see how the data format is:        
        #data = pyo.DataPortal(model=self.abstractmodel)
        #data.load(filename=datafile)
        
        di = {}
        #Sets:
        di['NODE'] = {None: grid_data.node['id'].tolist()}
        di['BRANCH'] = {None: [i for i in range(grid_data.branch.shape[0]) ]}
        di['GEN'] = {None: [i for i in range(grid_data.generator.shape[0]) ]}
        di['LOAD'] = {None: [i for i in range(grid_data.consumer.shape[0]) ]}
        di['AREA'] = {None: grid_data.node.area.unique().tolist()}
        di['TIME'] = {None: grid_data.timerange}
        
        br_expand1 = grid_data.branch[
                        grid_data.branch['expand']==1].index.tolist()
        br_expand2 = grid_data.branch[
                        grid_data.branch['expand']==2].index.tolist()
        gen_expand1 = grid_data.generator[
                        grid_data.generator['expand']==1].index.tolist()
        gen_expand2 = grid_data.generator[
                        grid_data.generator['expand']==2].index.tolist()
        # Convert from numpy.int64 (pandas) to int in order to work with PySP
        # (pprint function error otherwise)
        br_expand1 = [int(i) for i in br_expand1]
        br_expand2 = [int(i) for i in br_expand2]
        gen_expand1 = [int(i) for i in gen_expand1]
        gen_expand2 = [int(i) for i in gen_expand2]
        # Determine which nodes should be considered upgraded in each stage,
        # depending on whether any generators or branches are connected
        node_expand1=[]
        node_expand2=[]
        for n in grid_data.node['id'][grid_data.node['existing']==0]:
            if (n in grid_data.generator['node'][grid_data.generator['expand']==1].tolist()
                or n in grid_data.branch['node_to'][grid_data.branch['expand']==1].tolist() 
                or n in grid_data.branch['node_from'][grid_data.branch['expand']==1].tolist()):
                #stage one generator  or branch expansion connected to node
                node_expand1.append(n)
            if (n in grid_data.generator['node'][grid_data.generator['expand']==2].tolist()
                or n in grid_data.branch['node_to'][grid_data.branch['expand']==2].tolist() 
                or n in grid_data.branch['node_from'][grid_data.branch['expand']==2].tolist()):
                #stage two generator or branch expansion connected to node
                node_expand2.append(n)
#        node_expand1 = grid_data.node[
#                        grid_data.node['expand1']==1].index.tolist()
#        node_expand2 = grid_data.node[
#                        grid_data.node['expand2']==2].index.tolist()
        
        di['BRANCH_EXPAND1'] = {None: br_expand1}
        di['BRANCH_EXPAND2'] = {None: br_expand2}
        di['GEN_EXPAND1'] = {None:gen_expand1}
        di['GEN_EXPAND2'] = {None:gen_expand2}
        di['NODE_EXPAND1'] = {None:node_expand1}
        di['NODE_EXPAND2'] = {None:node_expand2}
        
        #Parameters:
        di['samplefactor'] = {None: self._HOURS_PER_YEAR/len(grid_data.timerange)}
        di['nodeOffshore'] = {}
        di['nodeType'] = {}
        di['nodeExistingNumber'] = {}
        di['nodeCostScale']={}
        di['nodeArea']={}
        for k,row in grid_data.node.iterrows():
            n=grid_data.node['id'][k]
            di['nodeOffshore'][n] = row['offshore']
            di['nodeType'][n] = row['type']
            di['nodeExistingNumber'][n] = row['existing']
            di['nodeCostScale'][n] = row['cost_scaling']
            di['nodeArea'][n] = row['area']
            
        di['branchExistingCapacity'] = {}
        di['branchExistingCapacity2'] = {}
        di['branchExpand'] = {}
        di['branchDistance'] = {}
        di['branchType'] = {}
        di['branchCostScale'] = {}
        di['branchOffshoreFrom'] = {}
        di['branchOffshoreTo'] = {}
        di['branchNodeFrom'] = {}
        di['branchNodeTo'] = {}
        di['branchMaxNewCapacity'] = {}
        offsh = self._offshoreBranch(grid_data)
        for k,row in grid_data.branch.iterrows():
            di['branchExistingCapacity'][k] = row['capacity']
            di['branchExistingCapacity2'][k] = row['capacity2']
            if row['max_newCap'] >0:
                di['branchMaxNewCapacity'][k] = row['max_newCap']
            else:
                di['branchMaxNewCapacity'][k] = maxNewBranchCap
            di['branchExpand'][k] = row['expand']
            if row['distance'] >= 0:
                di['branchDistance'][k] = row['distance']
            else:
                di['branchDistance'][k] = branch_distances[k]                    
            di['branchType'][k] = row['type']
            di['branchCostScale'][k] = row['cost_scaling']
            di['branchOffshoreFrom'][k] = offsh['from'][k]
            di['branchOffshoreTo'][k] = offsh['to'][k]
            di['branchNodeFrom'][k] = row['node_from']
            di['branchNodeTo'][k] = row['node_to']
            
        di['genCapacity']={}
        di['genCapacity2']={}
        di['genCapacityProfile']={}
        di['genNode']={}
        di['genCostAvg'] = {}
        di['genCostProfile'] = {}
        di['genPAvg'] = {}
        di['genExpand'] = {}
        di['genNewCapMax'] = {}
        di['genType'] = {}
        di['genCostScale'] = {}
        for k,row in grid_data.generator.iterrows():
            di['genCapacity'][k] = row['pmax']
            di['genCapacity2'][k] = row['pmax2']
            di['genNode'][k] = row['node']
            di['genCostAvg'][k] = row['fuelcost']
            di['genPAvg'][k] = row['pavg']
            di['genExpand'][k] = row['expand']
            di['genNewCapMax'][k] = row['p_maxNew']
            di['genType'][k] = row['type']
            di['genCostScale'][k] = row['cost_scaling']
            ref = row['fuelcost_ref']
            ref2 = row['inflow_ref']
            for i,t in enumerate(grid_data.timerange):
                di['genCostProfile'][(k,t)] = grid_data.profiles[ref][i]
                di['genCapacityProfile'][(k,t)] = (grid_data.profiles[ref2][i]
                            * row['inflow_fac'])
           
        di['demandAvg'] = {}
        di['demandProfile'] ={}
        di['demNode'] = {}
        di['emissionCap'] = {}
        for k,row in grid_data.consumer.iterrows():
            di['demNode'][k] = row['node']
            di['demandAvg'][k] = row['demand_avg']
            di['emissionCap'][k] = row['emission_cap']
            ref = row['demand_ref']
            for i,t in enumerate(grid_data.timerange):
                di['demandProfile'][(k,t)] = grid_data.profiles[ref][i]
        

        # Read input data from XML file
        import xml
        tree=xml.etree.ElementTree.parse(datafile)
        root = tree.getroot()
        
        di['NODETYPE'] = {None:[]}
        di['nodetypeCost'] = {}
        for i in root.findall('./nodetype/item'):
            name = i.attrib['name']
            di['NODETYPE'][None].append(name)
            di['nodetypeCost'][(name,'L')] = float(i.attrib['L'])
            di['nodetypeCost'][(name,'S')] = float(i.attrib['S'])

        di['BRANCHTYPE'] = {None:[]}
        di['branchtypeCost'] = {}
        di['branchtypeMaxCapacity'] = {}
        di['branchLossfactor'] = {}
        for i in root.findall('./branchtype/item'):
            name = i.attrib['name']
            di['BRANCHTYPE'][None].append(name)
            di['branchtypeCost'][(name,'B')] = float(i.attrib['B'])
            di['branchtypeCost'][(name,'Bd')] = float(i.attrib['Bd'])
            di['branchtypeCost'][(name,'Bdp')] = float(i.attrib['Bdp'])
            di['branchtypeCost'][(name,'CL')] = float(i.attrib['CL'])
            di['branchtypeCost'][(name,'CLp')] = float(i.attrib['CLp'])
            di['branchtypeCost'][(name,'CS')] = float(i.attrib['CS'])
            di['branchtypeCost'][(name,'CSp')] = float(i.attrib['CSp'])
            di['branchtypeMaxCapacity'][name] = float(i.attrib['maxCap'])
            di['branchLossfactor'][(name,'fix')] = float(i.attrib['lossFix'])
            di['branchLossfactor'][(name,'slope')] = float(i.attrib['lossSlope'])
                                                        
        di['GENTYPE'] = {None:[]}
        di['genTypeCost'] = {}
        di['genTypeEmissionRate'] = {}
        for i in root.findall('./gentype/item'):
            name = i.attrib['name']
            di['GENTYPE'][None].append(name)
            di['genTypeCost'][name] = float(i.attrib['CX']) 
            di['genTypeEmissionRate'][name] = float(i.attrib['CO2'])                                               
        
        for i in root.findall('./parameters'):
            di['curtailmentCost'] = {None: 
                float(i.attrib['curtailmentCost'])}
            di['financeInterestrate'] = {None: 
                float(i.attrib['financeInterestrate'])}
            di['financeYears'] = {None: 
                float(i.attrib['financeYears'])}
            di['omRate'] = {None: 
                float(i.attrib['omRate'])}
            di['CO2price'] = {None: 
                float(i.attrib['CO2price'])}
            di['VOLL'] = {None: 
                float(i.attrib['VOLL'])}
            di['stage2TimeDelta'] = {None: 
                float(i.attrib['stage2TimeDelta'])}

        return {'powergim':di}

        
    def writeStochasticProblem(self,path,dict_data):
        '''create input files for solving stochastic problem
        
        Parameters
        ----------
        path : string
            Where to put generated files
        dict_data : dictionary
            Pyomo data model in dictionary format. Output from 
            createModelData method
            
        Returns
        -------
        string that can be written to .dat file (reference model data)
        '''
        
 
        NL="\n"
        TAB="\t"
        dat_str = "#PowerGIM data file"+NL   
        str_set=""
        str_param=""
        str_sprm=""
        
        for key,val in dict_data['powergim'].items():
            v = getattr(self.abstractmodel,key)
            
            if type(v)==pyo.base.sets.SimpleSet:
                str_set += "set " + key + " := "
                for x in val[None]:
                    str_set += str(x) + " "
                str_set += ";" + NL
                
            elif type(v)==pyo.base.param.IndexedParam:
                print("PARAM: ",v)
                str_param += "param " + key + ":= " + NL
                for k2,v2 in val.items():
                    str_param += TAB
                    if isinstance(k2,tuple):
                        #multiple keys (table data)
                        for ki in k2:
                            str_param += str(ki) + TAB
                    else:
                        #single key (list data)
                        str_param += str(k2) + TAB
                    str_param += str(v2) + NL
                str_param += TAB+";"+NL
                
            elif type(v)==pyo.base.param.SimpleParam:
                # single value data
                str_sprm += "param " + key + " := " + str(val[None]) + ";" + NL
                
            else:
                print("Unknown data  ",key,v,type(v))
                raise Exception("Unknown data")
            
        dat_str += NL + str_set + NL + str_sprm + NL + str_param    
        
        #scenario structure data file:
        #TODO: export scenario structure data file
        
        #root node scenario data:
        with open("{}/RootNode.dat".format(path), "w") as text_file:
            text_file.write(dat_str)
        print("Root node data written to {}/RootNode.dat".format(path))
        
        return dat_str
        
        
    def createScenarioTreeModel(self,num_scenarios):
        '''Generate model instance with data. Alternative to .dat files
        
        Parameters
        ----------
        num_scenarios : int
            number of scenarios. Each with the same probability
        
        Returns
        -------
        PySP scenario tree model
        
        This method may be called by "pysp_scenario_tree_model_callback()" in
        the model input file instead of using input .dat files
        '''
        from pyomo.pysp.scenariotree.tree_structure_model \
            import CreateConcreteTwoStageScenarioTreeModel
    
        st_model = CreateConcreteTwoStageScenarioTreeModel(num_scenarios)
    
        first_stage = st_model.Stages.first()
        second_stage = st_model.Stages.last()
    
        # First Stage
        st_model.StageCost[first_stage] = 'firstStageCost'
        st_model.StageVariables[first_stage].add('branchNewCables1')
        st_model.StageVariables[first_stage].add('branchNewCapacity1')
        st_model.StageVariables[first_stage].add('newNodes1')
        st_model.StageVariables[first_stage].add('genNewCapacity1')
    
        # Second Stage
        st_model.StageCost[second_stage] = 'secondStageCost'
        st_model.StageVariables[second_stage].add('generation1')
        st_model.StageVariables[second_stage].add('curtailment1')
        st_model.StageVariables[second_stage].add('branchFlow12_1')
        st_model.StageVariables[second_stage].add('branchFlow21_1')
        st_model.StageVariables[second_stage].add('generation2')
        st_model.StageVariables[second_stage].add('curtailment2')
        st_model.StageVariables[second_stage].add('branchFlow12_2')
        st_model.StageVariables[second_stage].add('branchFlow21_2')
        st_model.StageVariables[second_stage].add('genNewCapacity2')
        st_model.StageVariables[second_stage].add('branchNewCables2')
        st_model.StageVariables[second_stage].add('branchNewCapacity2')
        st_model.StageVariables[second_stage].add('newNodes2')
            
        st_model.ScenarioBasedData=False
    
        # Alternative, using networkx to create scenario tree:
        # TODO: implement this alternative
        if False:
            from pyomo.pysp.scenariotree.tree_structure_model  \
               import ScenarioTreeModelFromNetworkX
            import networkx
           
            G = networkx.DiGraph()
            G.add_node("R")
            G.add_node("u0")
            G.add_edge("R", "u0", probability=0.1)
            G.add_node("u1")
            G.add_edge("R", "u1", probability=0.5)
            G.add_node("u2")
            G.add_edge("R", "u2", probability=0.4)
    
            stm = ScenarioTreeModelFromNetworkX(G,
                             edge_probability_attribute="probability",
                             stage_names=["T1", "T2"])
                             
            # Declare the variables for each node (or stage)
            stm.StageVariables["T1"].add("x")
            stm.StageDerivedVariables["T1"].add("z")
            # for this example, variables in the second and
            # third time-stage change for each node
            stm.NodeVariables["u0"].add("y0")
            stm.NodeDerivedVariables["u0"].add("xu0")
            
            # Declare the Var or Expression object that
            # reports the cost at each time stage
            stm.StageCost["T1"] = "firstStageCost"
            stm.StageCost["T2"] = "secondStageCost"
            
            st_model = stm
            
        return st_model

    def computeCostBranch(self,model,b,include_om=False):
        '''Investment cost of single branch
        
        corresponds to  firstStageCost in abstract model'''
        
        ar = 1
        #TODO is it not enough to check if b in BRANCH_EXPAND?
        if b in model.BRANCH_EXPAND1:
            br_num = model.branchNewCables1[b].value
            br_cap = model.branchNewCapacity1[b].value
            ar = annuityfactor(model.financeInterestrate,model.financeYears)
        elif b in model.BRANCH_EXPAND2:
            br_num = model.branchNewCables2[b].value
            br_cap = model.branchNewCapacity2[b].value
            ar = (annuityfactor(model.financeInterestrate,model.financeYears)
                  -annuityfactor(model.financeInterestrate,
                                 model.stage2TimeDelta))
        else:
            br_num=0
            br_cap=0
                 
        b_cost = 0
        typ = model.branchType[b]
        b_cost += (model.branchtypeCost[typ,'B']
                    *br_num)
        b_cost += (model.branchtypeCost[typ,'Bd']
                    *model.branchDistance[b]*br_num)
        b_cost += (model.branchtypeCost[typ,'Bdp']
                    *model.branchDistance[b]*br_cap)
        
        #endpoints offshore (N=1) or onshore (N=0) ?
        N1 = model.branchOffshoreFrom[b]
        N2 = model.branchOffshoreTo[b]
        for N in [N1,N2]:
            b_cost += N*(model.branchtypeCost[typ,'CS']*br_num
                        +model.branchtypeCost[typ,'CSp']*br_cap)            
            b_cost += (1-N)*(model.branchtypeCost[typ,'CL']*br_num
                        +model.branchtypeCost[typ,'CLp']*br_cap)
    
        cost = model.branchCostScale[b]*b_cost
        
        if include_om:
            cost = cost*(1 + model.omRate * ar)
        return cost

    def computeCostNode(self,model,n,include_om=False):
        '''Investment cost of single node
        
        corresponds to cost in abstract model'''
        
        # node may be expanded in stage 1 or stage 2, but will never be 
        # expanded in both
        
        ar = 1
        n_num = 0
        if n in model.NODE_EXPAND1:
            n_num = model.newNodes1[n].value
            ar = annuityfactor(model.financeInterestrate,model.financeYears)
        # only need to consider second stage investment if node not expanded 
        # in first stage (i.e. n_num=0)
        if n in model.NODE_EXPAND2 and n_num==0:
            n_num = model.newNodes2[n].value
            ar = (annuityfactor(model.financeInterestrate,model.financeYears)
                  -annuityfactor(model.financeInterestrate,
                                 model.stage2TimeDelta))
        n_cost = 0
        N = model.nodeOffshore[n]
        n_cost += N*(model.nodetypeCost[model.nodeType[n],'S']*n_num)
        n_cost += (1-N)*(model.nodetypeCost[model.nodeType[n],'L']*n_num)
        cost = model.nodeCostScale[n]*n_cost
        if include_om:
            cost = cost*(1 + model.omRate * ar)
        return cost


    def computeCostGenerator(self,model,g,include_om=False):
        '''Investment cost of generator
        '''
        ar = 1
        g_cap = 0
        if g in model.GEN_EXPAND1:
            g_cap = model.genNewCapacity1[g].value
            ar = annuityfactor(model.financeInterestrate,model.financeYears)
        elif g in model.GEN_EXPAND2:
            g_cap = model.genNewCapacity2[g].value
            ar = (annuityfactor(model.financeInterestrate,model.financeYears)
                  -annuityfactor(model.financeInterestrate,
                                 model.stage2TimeDelta))
            
        typ = model.genType[g]
        cost = model.genTypeCost[typ]*g_cap
        if include_om:
            cost = cost*(1 + model.omRate * ar)
        return cost


    def computeGenerationCost(self,model,g,phase):
        '''compute NPV cost of generation (+ curtailment and CO2 emissions)
        
        This corresponds to secondStageCost in abstract model        
        '''
        ar = 1
        if phase == 1:
            gen = model.generation1
            curt = model.curtailment1
            ar = annuityfactor(model.financeInterestrate,
                               model.stage2TimeDelta)
        elif phase==2:
            gen = model.generation2
            curt = model.curtailment2
            ar = (
                annuityfactor(model.financeInterestrate,model.financeYears)
                -annuityfactor(model.financeInterestrate,model.stage2TimeDelta)
                )
        expr = sum(gen[g,t].value
                    *model.genCostAvg[g]*model.genCostProfile[g,t] 
                     for t in model.TIME)
        expr += sum(curt[g,t].value*model.curtailmentCost for t in model.TIME)
        expr += sum(gen[g,t].value*
                            model.genTypeEmissionRate[model.genType[g]]*model.CO2price
                            for t in model.TIME)
        # lifetime cost
        samplefactor = model.samplefactor.value
        expr = expr*samplefactor*ar
        return expr
     
                
    def computeDemand(self,model,c,t):
        '''compute demand at specified load ant time'''
        return model.demandAvg[c]*model.demandProfile[c,t]

        
    def computeAreaEmissions(self,model,c, phase, cost=False):
        '''compute total emissions from a load/country'''
        # TODO: ensure that all nodes are mapped to a country/load
        n = model.demNode[c]
        expr = 0
        samplefactor = model.samplefactor.value
        if phase==1:
            gen=model.generation1
            ar = annuityfactor(model.financeInterestrate,model.stage2TimeDelta)
        elif phase==2:
            gen=model.generation2
            ar = (annuityfactor(model.financeInterestrate,model.financeYears)
                -annuityfactor(model.financeInterestrate,model.stage2TimeDelta))
                
        for g in model.GEN:
            if model.genNode[g]==n:
                expr += sum(gen[g,t].value*
                            model.genTypeEmissionRate[model.genType[g]]
                            for t in model.TIME)
                    
        expr = expr*samplefactor
        if cost:
            expr = expr * model.CO2price * ar
        return expr
        
                                
    def computeAreaRES(self, model,j,phase, shareof):
        '''compute renewable share of demand or total generation capacity'''
        node = model.demNode[j]
        area = model.nodeArea[node]
        Rgen = 0
        costlimit_RES=1 # limit for what to consider renewable generator
        
        if phase==1:
            gen_p=model.generation1
        elif phase==2:
            gen_p=model.generation2
        gen = 0
        dem = sum(model.demandAvg[j]*model.demandProfile[j,t] for t in model.TIME)
        for g in model.GEN:
            if model.nodeArea[model.genNode[g]]==area:
                if model.genCostAvg[g] <= costlimit_RES:
                    Rgen += sum(gen_p[g,t].value for t in model.TIME)
                else:
                    gen += sum(gen_p[g,t].value for t in model.TIME)

        if shareof=='dem':
           return Rgen/dem
        elif shareof=='gen':
           return Rgen/(gen+Rgen)
        else:
           print('Choose shareof dem or gen')

    
    def computeAreaPrice(self, model, area, t, phase):
        '''cumpute the approximate area price based on max marginal cost'''
        mc = []
        for g in model.GEN:
            if phase==1:
                gen = model.generation1[g,t].value
            elif phase==2:
                gen = model.generation2[g,t].value                
            if gen > 0:
                if model.nodeArea[model.genNode[g]]==area:
                    mc.append(model.genCostAvg[g]*model.genCostProfile[g,t]
                                +model.genTypeEmissionRate[model.genType[g]]*model.CO2price)
        price = max(mc)
        return price

        
    def computeAreaWelfare(self, model, c, t, phase):
        '''compute social welfare for a given area and time step
        
        Returns: Welfare, ProducerSurplus, ConsumerSurplus, 
                 CongestionRent, IMport, eXport
        '''
        node = model.demNode[c]
        area = model.nodeArea[node]
        PS = 0; CS = 0; CR = 0; GC = 0; gen = 0; 
        dem = model.demandAvg[c]*model.demandProfile[c,t]
        price = self.computeAreaPrice(model, area, t, phase)
        #branch_capex = self.computeAreaCostBranch(model,c,include_om=True) #npv
        #gen_capex = self.computeAreaCostGen(model,c) #annualized
        
        #TODO: check phase1 vs phase2
        if phase==1:
            gen_p = model.generation1
            flow12 = model.branchFlow12_1
            flow21 = model.branchFlow21_1
        elif phase==2:
            gen_p = model.generation2
            flow12 = model.branchFlow12_2
            flow21 = model.branchFlow21_2
            
        for g in model.GEN:
            if model.nodeArea[model.genNode[g]]==area:
                gen += gen_p[g,t].value
                GC += gen_p[g,t].value*(
                    model.genCostAvg[g]*model.genCostProfile[g,t]
                    +model.genTypeEmissionRate[model.genType[g]]*model.CO2price)
        CS = (model.VOLL-price)*dem
        CC = price*dem
        PS = price*gen - GC
        if gen > dem:
            X = price*(gen-dem)
            IM = 0
            flow = []
            price2 = []
            for j in model.BRANCH:
                if (model.nodeArea[model.branchNodeFrom[j]]==area and
                    model.nodeArea[model.branchNodeTo[j]]!=area):
                        flow.append(flow12[j,t].value)
                        price2.append(self.computeAreaPrice(model,
                                        model.nodeArea[model.branchNodeTo[j]],
                                        t,phase))
                if (model.nodeArea[model.branchNodeTo[j]]==area and
                    model.nodeArea[model.branchNodeFrom[j]]!=area):
                        flow.append(flow21[j,t].value)
                        price2.append(self.computeAreaPrice(model,
                                        model.nodeArea[model.branchNodeFrom[j]],
                                        t,phase))
            CR = sum(flow[i]*(price2[i]-price) for i in range(len(flow)))/2
        elif gen < dem:
            X = 0
            IM = price*(dem-gen)
            flow = []
            price2 = []
            for j in model.BRANCH:
                if (model.nodeArea[model.branchNodeFrom[j]]==area and
                    model.nodeArea[model.branchNodeTo[j]]!=area):
                        flow.append(flow21[j,t].value)
                        price2.append(self.computeAreaPrice(model,
                                        model.nodeArea[model.branchNodeTo[j]],
                                        t,phase))
                if (model.nodeArea[model.branchNodeTo[j]]==area and
                    model.nodeArea[model.branchNodeFrom[j]]!=area):
                        flow.append(flow12[j,t].value)
                        price2.append(self.computeAreaPrice(model,
                                        model.nodeArea[model.branchNodeFrom[j]],
                                        t,phase))
            CR = sum(flow[i]*(price-price2[i]) for i in range(len(flow)))/2
        else:
            X = 0
            IM = 0
            flow = [0]
            price2 = [0]
            CR = 0
        W = PS + CS + CR
        return {'W':W, 'PS':PS, 'CS':CS, 'CC':CC, 'GC':GC, 'CR':CR, 'IM':IM, 'X':X}

            
    def computeAreaCostBranch(self,model,c,include_om=False):
        '''Investment cost of single branch
        
        corresponds to  firstStageCost in abstract model'''
        node = model.demNode[c]
        area = model.nodeArea[node]
        cost = 0
        
        for b in model.BRANCH:
            if model.nodeArea[model.branchNodeTo[b]]==area:
                cost += self.computeCostBranch(model,b,include_om)
            elif model.nodeArea[model.branchNodeFrom[b]]==area:
                cost += self.computeCostBranch(model,b,include_om)
        
        if include_om:
             cost = cost*(1 + model.omRate*annuityfactor(
                            model.financeInterestrate,
                            model.financeYears)) 
        
        return cost/2
        
        
    def computeAreaCostGen(self,model,c):
        '''compute capital costs for new generator capacity'''
        node = model.demNode[c]
        area = model.nodeArea[node]
        gen_capex = 0
        
        for g in model.GEN:
            if model.nodeArea[model.genNode[g]]==area:
                typ = model.genType[g]
                gen_capex += model.genTypeCost[typ]*model.genNewCapacity[g].value*model.genCostScale[g]
                
        return gen_capex  
        
    def saveDeterministicResults(self,model,excel_file):
        '''export results to excel file
        
        Parameters
        ==========
        model : Pyomo model
            concrete instance of optimisation model
        excel_file : string
            name of Excel file to create
        
        '''
        df_branches = pd.DataFrame()
        df_nodes = pd.DataFrame()
        df_gen = pd.DataFrame()
        df_load = pd.DataFrame()
        # Specifying columns is not necessary, but useful to get the wanted 
        # ordering
        df_branches = pd.DataFrame(columns=['from','to','fArea','tArea','type',
                                           'existingCapacity',
                                           'existingCapacity2','expand',
                                           'newCables','newCapacity'])
#        df_nodes = pd.DataFrame(columns=['num','area','newNodes1',
#                                         'cost','cost_withOM'])
        df_gen = pd.DataFrame(columns=['num','node','area',
                                       'type','pmax','pmax2','expand',
                                       'newCapacity'])
#        df_load = pd.DataFrame(columns=['num','node','area','Pavg','Pmin','Pmax',
#                                        'emissions','emissionCap', 'emission_cost',
#                                        'price_avg','RES%dem','RES%gen', 'IM', 'EX',
#                                        'CS', 'PS', 'CR', 'CAPEX', 'Welfare'])
        samplefactor=model.samplefactor.value
        
        for j in model.BRANCH:
            df_branches.loc[j,'num'] = j
            df_branches.loc[j,'from'] = model.branchNodeFrom[j]
            df_branches.loc[j,'to'] = model.branchNodeTo[j]
            df_branches.loc[j,'fArea'] = model.nodeArea[model.branchNodeFrom[j]]
            df_branches.loc[j,'tArea'] = model.nodeArea[model.branchNodeTo[j]]
            df_branches.loc[j,'existingCapacity'] = model.branchExistingCapacity[j]
            df_branches.loc[j,'existingCapacity2'] = model.branchExistingCapacity2[j]
            df_branches.loc[j,'expand'] = model.branchExpand[j]
            df_branches.loc[j,'type'] = model.branchType[j]
            cap1 = 0 #model.branchExistingCapacity[j]
            cap2 = cap1
            if j in model.BRANCH_EXPAND1:
                df_branches.loc[j,'newCables'] = model.branchNewCables1[j].value
                df_branches.loc[j,'newCapacity'] = model.branchNewCapacity1[j].value
                cap1 = cap1 + model.branchNewCapacity1[j].value
                cap2 = cap1
            if j in model.BRANCH_EXPAND2:
                df_branches.loc[j,'newCables'] = model.branchNewCables2[j].value
                df_branches.loc[j,'newCapacity'] = model.branchNewCapacity2[j].value
                cap2 = cap2 + model.branchNewCapacity2[j].value                
            #branch costs
            df_branches.loc[j,'cost'] = self.computeCostBranch(model,j)
            df_branches.loc[j,'cost_withOM'] = self.computeCostBranch(
                        model,j,include_om=True)
            df_branches.loc[j,'congestion_rent'] = self.computeCostBranch(
                        model,j)
            # Phase 1
            df_branches.loc[j,'flow12avg_1'] = np.mean([
                model.branchFlow12_1[(j,t)].value for t in model.TIME])
            df_branches.loc[j,'flow12%_1'] = (
                df_branches.loc[j,'flow12avg_1']/cap1)
            df_branches.loc[j,'flow21avg_1'] = np.mean([
                model.branchFlow21_1[(j,t)].value for t in model.TIME])
            df_branches.loc[j,'flow21%_1'] = (
                df_branches.loc[j,'flow21avg_1']/cap1)

            # Phase 2
            df_branches.loc[j,'flow12avg_2'] = np.mean([
                model.branchFlow12_2[(j,t)].value for t in model.TIME])
            df_branches.loc[j,'flow12%_2'] = (
                df_branches.loc[j,'flow12avg_2']/cap2)
            df_branches.loc[j,'flow21avg_2'] = np.mean([
                model.branchFlow21_2[(j,t)].value for t in model.TIME])
            df_branches.loc[j,'flow21%_2'] = (
                df_branches.loc[j,'flow21avg_2']/cap2)

                                    
        for j in model.NODE:
            df_nodes.loc[j,'num'] = j
            df_nodes.loc[j,'area'] = model.nodeArea[j]
            if j in model.NODE_EXPAND1:
                df_nodes.loc[j,'newNodes1'] = model.newNodes1[j].value
            if j in model.NODE_EXPAND2:
                df_nodes.loc[j,'newNodes2'] = model.newNodes2[j].value
            df_nodes.loc[j,'cost'] = self.computeCostNode(model,j)
            df_nodes.loc[j,'cost_withOM'] = self.computeCostNode(model,j,
                include_om=True)
            
        
        for j in model.GEN:
            df_gen.loc[j,'num'] = j
            df_gen.loc[j,'area'] = model.nodeArea[model.genNode[j]]
            df_gen.loc[j,'node'] = model.genNode[j]
            df_gen.loc[j,'type'] = model.genType[j]
            df_gen.loc[j,'pmax'] = model.genCapacity[j]
            df_gen.loc[j,'pmax2'] = model.genCapacity2[j]
            df_gen.loc[j,'expand'] = model.genExpand[j]
            df_gen.loc[j,'emission_rate'] = model.genTypeEmissionRate[model.genType[j]]
            #Phase 1:
            df_gen.loc[j,'emission1'] = (model.genTypeEmissionRate[model.genType[j]]*sum(
                                        model.generation1[j,t].value for t in model.TIME)
                                        *samplefactor)
            df_gen.loc[j,'Pavg1'] = np.mean([
                model.generation1[(j,t)].value for t in model.TIME])
            df_gen.loc[j,'Pmin1'] = np.min([
                model.generation1[(j,t)].value for t in model.TIME])
            df_gen.loc[j,'Pmax1'] = np.max([
                model.generation1[(j,t)].value for t in model.TIME])
            if model.genCostAvg[j] == 0:
                df_gen.loc[j,'curtailed_avg1'] = np.mean([
                    model.curtailment1[(j,t)].value for t in model.TIME])
            #Phase 2:
            df_gen.loc[j,'emission2'] = (model.genTypeEmissionRate[model.genType[j]]*sum(
                                        model.generation2[j,t].value for t in model.TIME)
                                        *samplefactor)
            df_gen.loc[j,'Pavg2'] = np.mean([
                model.generation2[(j,t)].value for t in model.TIME])
            df_gen.loc[j,'Pmin2'] = np.min([
                model.generation2[(j,t)].value for t in model.TIME])
            df_gen.loc[j,'Pmax2'] = np.max([
                model.generation2[(j,t)].value for t in model.TIME])
            if model.genCostAvg[j] == 0:
                df_gen.loc[j,'curtailed_avg2'] = np.mean([
                    model.curtailment2[(j,t)].value for t in model.TIME])
                
            if j in model.GEN_EXPAND1:                
                df_gen.loc[j,'newCapacity'] = model.genNewCapacity1[j].value
            elif j in model.GEN_EXPAND2:                
                df_gen.loc[j,'newCapacity'] = model.genNewCapacity2[j].value
            df_gen.loc[j,'cost_NPV1'] = self.computeGenerationCost(model,j,
                phase=1)
            df_gen.loc[j,'cost_NPV2'] = self.computeGenerationCost(model,j,
                phase=2)
            df_gen.loc[j,'cost_investment'] = self.computeCostGenerator(model,j)
            df_gen.loc[j,'cost_investment_withOM'] = self.computeCostGenerator(model,j,
                include_om=True)

        print("TODO: powergim.saveDeterministicResults LOAD:"
                "only showing phase 2 (after 2nd stage investments)")
        phase=2
        def _n(n,p):
            return n+str(p)
            
        for j in model.LOAD:
            df_load.loc[j,'num'] = j
            df_load.loc[j,'node'] = model.demNode[j]
            df_load.loc[j,'area'] = model.nodeArea[model.demNode[j]]
            df_load.loc[j,'price_avg'] = np.mean(
                [self.computeAreaPrice(model,df_load.loc[j,'area'], t,phase=phase)
                for t in model.TIME])
            df_load.loc[j,'Pavg'] = np.mean([self.computeDemand(model,j,t)
                for t in model.TIME])
            df_load.loc[j,'Pmin'] = np.min([self.computeDemand(model,j,t)
                for t in model.TIME])
            df_load.loc[j,'Pmax'] = np.max([self.computeDemand(model,j,t)
                for t in model.TIME])
            df_load.loc[j,'emissionCap'] = model.emissionCap[j]
            df_load.loc[j,_n('emissions',phase)] = self.computeAreaEmissions(model,j,
                                                phase=phase)
            df_load.loc[j,_n('emission_cost',phase)] = self.computeAreaEmissions(model,j, 
                                                phase=phase,cost=True)
            df_load.loc[j,_n('RES%dem',phase)] = self.computeAreaRES(model,j,phase=phase,
                                                            shareof='dem')
            df_load.loc[j,_n('RES%gen',phase)] = self.computeAreaRES(model,j,phase=phase,
                                                            shareof='gen')
            df_load.loc[j,_n('Welfare',phase)] = sum(self.computeAreaWelfare(model,j,t,phase=phase)['W']
                                            for t in model.TIME)*samplefactor
            df_load.loc[j,_n('PS',phase)] = sum(self.computeAreaWelfare(model,j,t,phase=phase)['PS']
                                            for t in model.TIME)*samplefactor
            df_load.loc[j,_n('CS',phase)] = sum(self.computeAreaWelfare(model,j,t,phase=phase)['CS']
                                            for t in model.TIME)*samplefactor
            df_load.loc[j,_n('CR',phase)] = sum(self.computeAreaWelfare(model,j,t,phase=phase)['CR']
                                            for t in model.TIME)*samplefactor
            df_load.loc[j,_n('IM',phase)] = sum(self.computeAreaWelfare(model,j,t,phase=phase)['IM']
                                            for t in model.TIME)*samplefactor 
            df_load.loc[j,_n('EX',phase)] = sum(self.computeAreaWelfare(model,j,t,phase=phase)['X']
                                            for t in model.TIME)*samplefactor 
            df_load.loc[j,_n('CAPEX1',phase)] = self.computeAreaCostBranch(model,j,
                                            include_om=False)
        
        df_cost = pd.DataFrame(columns=['value','unit'])
        df_cost.loc['firstStageCost','value'] = (
            pyo.value(model.firstStageCost)/10**9)
        df_cost.loc['secondStageCost','value'] = (
            pyo.value(model.secondStageCost)/10**9)
        df_cost.loc['firstStageCost','unit'] = '10^9 EUR'
        df_cost.loc['secondStageCost','unit'] = '10^9 EUR'
            
        #model.solutions.load_from(results)
        #print('First stage costs: ', 
        #      pyo.value(model.firstStageCost)/10**9, 'bnEUR')
        #print('Second stage costs: ', 
        #      pyo.value(model.secondStageCost)/10**9, 'bnEUR')

        writer = pd.ExcelWriter(excel_file) 
        df_cost.to_excel(excel_writer=writer,sheet_name="cost") 
        df_branches.to_excel(excel_writer=writer,sheet_name="branches")     
        df_nodes.to_excel(excel_writer=writer,sheet_name="nodes") 
        df_gen.to_excel(excel_writer=writer,sheet_name="generation") 
        df_load.to_excel(excel_writer=writer,sheet_name="demand") 


    def extractResultingGridData(self,grid_data,
                                 model=None,file_ph=None,
                                 stage=1,scenario=None):
        '''Extract resulting optimal grid layout from simulation results
        
        Parameters
        ==========
        grid_data : powergama.GridData
            grid data class
        model : Pyomo model
            concrete instance of optimisation model containing det. results
        file_ph : string
            CSV file containing results from stochastic solution
        stage : int
            Which stage to extract data for (1 or 2).
            1: only stage one investments included (default)
            2: both stage one and stage two investments included
        scenario : int
            which stage 2 scenario to get data for (only relevant when stage=2)
            
        Use either model or file_ph parameter        
        
        Returns
        =======
        GridData object reflecting optimal solution
        '''
        import copy

        grid_res = copy.deepcopy(grid_data)
        res_brC = pd.DataFrame(data=grid_res.branch['capacity']) 
        res_N = pd.DataFrame(data=grid_res.node['existing'])
        res_G = pd.DataFrame(data=grid_res.generator['pmax'])
        if model is not None:
            #res_brC = pd.DataFrame(0,index=model.BRANCH,columns=['val'])  
            #res_N = pd.DataFrame(0,index=model.NODE,columns=['val'])
            #res_G = pd.DataFrame(0,index=model.GEN,columns=['val'])
            if stage >= 1:
                for j in model.BRANCH_EXPAND1:
                    res_brC['capacity'][j] += model.branchNewCapacity1[j].value
                for j in model.NODE_EXPAND1:
                    res_N['existing'][j] += int(model.newNodes1[j].value)
                for j in model.GEN_EXPAND1:
                    res_G['pmax'][j] += model.genNewCapacity1[j].value
            if stage >= 2:
                #add to investments in stage 1
                res_brC['capacity'] += grid_res.branch['capacity2']
                res_G['pmax'] += grid_res.generator['pmax2']
                for j in model.BRANCH_EXPAND2:
                    res_brC['capacity'][j] += model.branchNewCapacity2[j].value
                for j in model.NODE_EXPAND2:
                    res_N['existing'][j] += int(model.newNodes2[j].value)
                for j in model.GEN_EXPAND2:
                    res_G['pmax'][j] += model.genNewCapacity2[j].value
        elif file_ph is not None:
            #res_brC = pd.DataFrame(data=grid_res.branch['capacity'],
            #                       columns=['val']) 
            #res_N = pd.DataFrame(data=grid_res.node['existing'],
            #                       columns=['val'])
            #res_G = pd.DataFrame(data=grid_res.generator['pmax'],
            #                       columns=['val'])
            df_ph = pd.read_csv(file_ph,header=None,skipinitialspace=True,
                                names=['stage','node',
                                       'var','var_indx','value'])
            if stage>=1:
                df_branchNewCables = df_ph[df_ph['var']=='branchNewCables1']
                df_branchNewCapacity = df_ph[df_ph['var']=='branchNewCapacity1']
                df_newNodes = df_ph[df_ph['var']=='newNodes1']
                df_newGen = df_ph[df_ph['var']=='genNewCapacity1']
                for k,row in df_branchNewCapacity.iterrows():
                    res_brC['capacity'][int(row['var_indx'])] += float(row['value'])
                for k,row in df_newNodes.iterrows():
                    res_N['existing'][row['var_indx']] += int(row['value'])
                for k,row in df_newGen.iterrows():
                    res_G['pmax'][int(row['var_indx'])] += float(row['value'])                
            if stage>=2:
                if scenario is None:
                    raise Exception('Missing input "scenario"')
                    
                df_branchNewCapacity = df_ph[
                    (df_ph['var']=='branchNewCapacity2') &
                    (df_ph['node']=='LeafNode_Scenario{}'.format(scenario))]
                df_newNodes = df_ph[(df_ph['var']=='newNodes2') &
                    (df_ph['node']=='LeafNode_Scenario{}'.format(scenario))]
                df_newGen = df_ph[(df_ph['var']=='genNewCapacity2') &
                    (df_ph['node']=='LeafNode_Scenario{}'.format(scenario))]
                #TODO fix: this will add up from all scenarios:
                for k,row in df_branchNewCapacity.iterrows():
                    res_brC['capacity'][int(row['var_indx'])] += float(row['value'])
                for k,row in df_newNodes.iterrows():
                    res_N['existing'][row['var_indx']] += int(row['value'])
                for k,row in df_newGen.iterrows():
                    res_G['pmax'][int(row['var_indx'])] += float(row['value'])                
        else:
            raise Exception('Missing input parameter')
            
        grid_res.branch['capacity'] = res_brC['capacity']
        grid_res.node['existing'] = res_N['existing']
        grid_res.generator['pmax'] = res_G['pmax']
        grid_res.branch = grid_res.branch[grid_res.branch['capacity'] 
            > self._NUMERICAL_THRESHOLD_ZERO]
        grid_res.node = grid_res.node[grid_res.node['existing'] 
            > self._NUMERICAL_THRESHOLD_ZERO]
        return grid_res
        

    def presentResults(self,csvfile):
        '''load  results and present plots etc
        
        Parameters
        ----------
        csvfile : string
            name of csv file holding RUNPH results (ph.cvs)
        
        Returns
        -------
        pandas.Dataframe results
        '''
        
        df = pd.read_csv(csvfile,header=None, 
                         names=['r_stage','r_node','r_var','r_indx','r_val'])
                         
        #TODO: present results
        # Martin's functions...
                         
        return df
        
        
def annuityfactor(rate,years):
    '''Net present value factor for fixed payments per year at fixed rate'''
    if rate==0:
        annuity = years
    else:
        annuity = (1-1/((1+rate)**years))/rate
    return annuity
        
            

def sample_kmeans(X, samplesize):
    """K-means sampling

    Parameters
    ==========
    X : matrix
        data matrix to sample from
    samplesize : int
        size of sample
        
    This method relies on sklearn.cluster.KMeans

    """
    """
    TODO: Have to weight the importance, i.e. multiply timeseries with
    installed capacities in order to get proper clustering. 
    """
    from sklearn.cluster import KMeans
    #from sklearn.metrics.pairwise import pairwise_distances_argmin
    #from sklearn.datasets.samples_generator import make_blobs
    
    n_clusters=samplesize
    k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    k_means.fit(X)
    # which cluster nr it belongs to:
    k_means_labels = k_means.labels_    
    k_means_cluster_centers = k_means.cluster_centers_
    k_means_labels_unique, X_indecies = np.unique(k_means_labels, 
                                                  return_index=True)
    #k_means_predict = k_means.predict(X)
        
    return k_means_cluster_centers

    
def sample_mmatching(X, samplesize):
    """
    The idea is to make e.g. 10000 randomsample-sets of size=samplesize 
    from the originial datasat X. 
    
    Choose the sampleset with the lowest objective:
    MINIMIZE [(meanSample - meanX)^2 + (stdvSample - stdvX)^2...]
    
    in terms of stitistical measures
    """
    
    return


def sample_meanshift(X, samplesize):
    """M matching sampling

    Parameters
    ==========
    X : matrix
        data matrix to sample from
    samplesize : int
        size of sample
        
    This method relies on sklearn.cluster.MeanShift

    It is a centroid-based algorithm, which works by updating candidates 
    for centroids to be the mean of the points within a given region. 
    These candidates are then filtered in a post-processing stage to 
    eliminate near-duplicates to form the final set of centroids.
    """
    from sklearn.cluster import MeanShift, estimate_bandwidth
    #from sklearn.datasets.samples_generator import make_blobs

    # The following bandwidth can be automatically detected using
    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=samplesize)
    
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    #labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    
    #labels_unique = np.unique(labels)
    #n_clusters_ = len(labels_unique)
    
    #print("number of estimated clusters : %d" % n_clusters_)
    
    return cluster_centers


def sample_latinhypercube(X, samplesize):
    """Latin hypercube sampling

    Parameters
    ==========
    X : matrix
        data matrix to sample from
    samplesize : int
        size of sample
        
    This method relies on pyDOE.lhs(n, [samples, criterion, iterations])

    """
    """
    lhs(n, [samples, criterion, iterations])
    
    n:an integer that designates the number of factors (required)
    samples: an integer that designates the number of sample points to generate 
        for each factor (default: n) 
    criterion: a string that tells lhs how to sample the points (default: None,  
        which simply randomizes the points within the intervals):
        “center” or “c”: center the points within the sampling intervals
        “maximin” or “m”: maximize the minimum distance between points, but place 
                          the point in a randomized location within its interval
        “centermaximin” or “cm”: same as “maximin”, but centered within the intervals
        “correlation” or “corr”: minimize the maximum correlation coefficient
    """
    from pyDOE import lhs
    from scipy.stats.distributions import norm
    X_rows = X.shape[0]; X_cols = X.shape[1]
    X_mean = X.mean(); X_std = X.std()
    X_sample = lhs( X_cols , samples=samplesize , criterion='center' )
    kernel=False
    if kernel:
        # Fit data w kernel density
        from sklearn.neighbors.kde import KernelDensity
        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
        kde.score_samples(X)
        # random sampling (TODO: fit to latin hypercube sample):
        kde_sample = kde.sample(samplesize)     
    else:
        # Fit data w normal distribution
        for i in range(X_cols):
            X_sample[:,i] = norm(loc=X_mean[i] , scale=X_std[i]).ppf(X_sample[:,i])
    
    return X_sample
  
def sampleProfileData(data, samplesize, sampling_method):
        """ Sample data from full-year time series
        
        Parameters
        ==========
        X : matrix
            data matrix to sample from
        samplesize : int
            size of sample
        sampling_method : str
            'kmeans', 'lhs', 'uniform', ('mmatching', 'meanshift')
        
        Returns:
            reduced data matrix according to sample size and method
        """
        X = data.profiles.copy()
        
        if sampling_method == 'kmeans':
            print("Using k-means...")
            # Multiply time series for load and VRES with their respective 
            # maximum capacities in order to get the correct clusters/samples

            for k in data.profiles.columns.values.tolist():
                ref = k
                pmax = sum(data.generator.pmax[g] for g in range(len(data.generator)) if data.generator.inflow_ref[g] == ref)
                if pmax > 0:
                    X[ref] = data.profiles[ref] * pmax
#            for k,row in self.generator.iterrows():
#                pmax = row['pmax']
#                ref = row['inflow_ref']
#                if X[ref].mean()<1:
#                    X[ref] = self.profiles[ref] * pmax
            X['const'] = 1
            
            for k,row in data.consumer.iterrows():
                pmax = row['demand_avg']
                ref = row['demand_ref']
                X[ref] = data.profiles[ref] * pmax
                
                X_sample = sample_kmeans(X, samplesize)
                X_sample = pd.DataFrame(data=X_sample,
                        columns=X.columns)

            # convert back to relative values
            for k in data.profiles.columns.values.tolist():
                ref = k
                pmax = sum(data.generator.pmax[g] for g in range(len(data.generator)) if data.generator.inflow_ref[g] == ref)
                if pmax > 0:
                    X_sample[ref] = X_sample[ref] / pmax
#            for k,row in self.generator.iterrows():
#                pmax = row['pmax']
#                ref = row['inflow_ref']
#                if pmax == 0:
#                    centroids[ref] = 0
#                else:
#                    if X[ref].mean()>1:
#                        centroids[ref] = centroids[ref] / pmax
            for k,row in data.consumer.iterrows():
                pmax = row['demand_avg']
                ref = row['demand_ref']
                if pmax == 0:
                    X_sample[ref] = 0
                else:
                    X_sample[ref] = X_sample[ref] / pmax
            X_sample['const'] = 1
            return X_sample

        elif sampling_method == 'mmatching':
            print("Using moment matching...")
        elif sampling_method == 'meanshift':
            print("Using Mean-Shift...")
            X_sample = sample_meanshift(X, samplesize)
            return X_sample
        elif sampling_method == 'lhs':
            print("Using Latin-Hypercube...")
            X_sample = sample_latinhypercube(X, samplesize)
            X_sample = pd.DataFrame(data=X_sample,
                        columns=X.columns)
            X_sample['const'] = 1
            X_sample[(X_sample < 0)] = 0
            return X_sample
        elif sampling_method == 'uniform':
            print("Using uniform sampling (consider changing sampling method!)...")
            import random
            timerange = random.sample(range(8760),samplesize)
            X_sample = data.profiles.loc[timerange, :]
            X_sample.index = list(range(len(X_sample.index)))
            return X_sample
                             
        return