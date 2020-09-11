# -*- coding: utf-8 -*-
"""
Module for power grid investment analyses

"""

import pandas as pd
import numpy as np
import pyomo.environ as pyo
import pyomo.pysp.scenariotree.tree_structure_model as tsm
import pyomo.pysp.util.rapper as rapper
import networkx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import xml
import copy
import math
from . import constants as const


__version__ = 0.1
__version_date__ = '2020-09-11'

def annuityfactor(rate,years):
    '''Net present value factor for fixed payments per year at fixed rate'''
    if rate==0:
        annuity = years
    else:
        annuity = (1-1/((1+rate)**years))/rate
    return annuity


def _myround(x, base=1,method='round'):
    '''Round to nearest multiple of base'''
    if method=='round':
        return int(base * round(float(x)/base))
    elif method=='floor':
        return int(base * math.floor(float(x)/base))
    elif method=='ceil':
        return int(base * math.ceil(float(x)/base))
    else:
        raise


class SipModel():
    '''
    Power Grid Investment Module - Stochastic Investment Problem
    '''

    _NUMERICAL_THRESHOLD_ZERO = 1e-6
    _HOURS_PER_YEAR = 8760

    def __init__(self, M_const = 1000):
        """Create Abstract Pyomo model for PowerGIM

        Parameters
        ----------
        M_const : int
            large constant
        """
        self.abstractmodel = self._createAbstractModel()
        self.M_const = M_const


    def costNode(self,model,n,stage):
        '''Expression for cost of node, investment cost no discounting'''
        n_cost = 0
        var_num = model.newNodes
        N = model.nodeOffshore[n]
        n_cost += N*(model.nodetypeCost[model.nodeType[n],'S']
                    *var_num[n,stage])
        n_cost += (1-N)*(model.nodetypeCost[model.nodeType[n],'L']
                    *var_num[n,stage])
        return model.nodeCostScale[n]*n_cost

    def costBranch(self,model,b,stage):
        '''Expression for cost of branch, investment cost no discounting'''
        b_cost = 0

        var_num=model.branchNewCables
        var_cap=model.branchNewCapacity
        typ = model.branchType[b]
        b_cost += (model.branchtypeCost[typ,'B']
                    *var_num[b,stage])
        b_cost += (model.branchtypeCost[typ,'Bd']
                    *model.branchDistance[b]
                    *var_num[b,stage])
        b_cost += (model.branchtypeCost[typ,'Bdp']
                    *model.branchDistance[b]
                    *var_cap[b,stage])

        #endpoints offshore (N=1) or onshore (N=0) ?
        N1 = model.branchOffshoreFrom[b]
        N2 = model.branchOffshoreTo[b]
        for N in [N1,N2]:
            b_cost += N*(model.branchtypeCost[typ,'CS']
                        *var_num[b,stage]
                    +model.branchtypeCost[typ,'CSp']
                    *var_cap[b,stage])
            b_cost += (1-N)*(model.branchtypeCost[typ,'CL']
                        *var_num[b,stage]
                    +model.branchtypeCost[typ,'CLp']
                    *var_cap[b,stage])

        return model.branchCostScale[b]*b_cost


    def costGen(self,model,g,stage):
        '''Expression for cost of generator, investment cost no discounting'''
        g_cost = 0
        var_cap=model.genNewCapacity
        typ = model.genType[g]
        g_cost += model.genTypeCost[typ]*var_cap[g,stage]
        return model.genCostScale[g]*g_cost

    def npvInvestment(self,model,stage,investment,
                includeOM=True,subtractSalvage=True):
        """NPV of investment cost including lifetime O&M and salvage value

        Parameters
        ----------
        model : object
            Pyomo model
        stage : int
            Investment or operation stage (1 or 2)
        investment :
            cost of e.g. node, branch or gen
        """
        omfactor = 0
        salvagefactor = 0
        if subtractSalvage:
            salvagefactor = (
                int(stage-1)*model.stage2TimeDelta/model.financeYears)*(
                    1/((1+model.financeInterestrate)
                    **(model.financeYears-model.stage2TimeDelta*int(stage-1))))
        if includeOM:
            omfactor = model.omRate * (
                annuityfactor(model.financeInterestrate,
                              model.financeYears)
                -annuityfactor(model.financeInterestrate,
                               int(stage-1)*model.stage2TimeDelta))

        # discount costs that come in stage 2 (the future)
        # present value vs future value: pv = fv/(1+r)^n
        discount_t0 = (1/((1+model.financeInterestrate)
                        **(model.stage2TimeDelta*int(stage-1))))

        investment = investment*discount_t0
        pv_cost = investment*(1 + omfactor - salvagefactor)
        return pv_cost

    def costInvestments(self,model,stage,
                        includeOM=True,subtractSalvage=True):
        """Investment cost, including lifetime O&M costs (NPV)"""
        investment = 0
        # add branch, node and generator investment costs:
        for b in model.BRANCH:
            investment += self.costBranch(model,b,stage)
        for n in model.NODE:
            investment += self.costNode(model,n,stage)
        for g in model.GEN:
            investment += self.costGen(model,g,stage)
        # add O&M costs and compute NPV:
        cost = self.npvInvestment(model,stage,investment,
                                  includeOM,subtractSalvage)
        return cost

    def costOperation(self,model, stage):
        """Operational costs: cost of gen, load shed (NPV)"""
        opcost = 0
        #discount_t0 = (1/((1+model.financeInterestrate)
        #    **(model.stage2TimeDelta*int(stage-1))))

        # operation cost per year:
        opcost = sum(model.generation[i,t,stage]*model.samplefactor[t]*(
                    model.genCostAvg[i]*model.genCostProfile[i,t]
                    +model.genTypeEmissionRate[model.genType[i]]*model.CO2price)
                    for i in model.GEN for t in model.TIME)
        opcost += sum(model.loadShed[c,t,stage]*model.VOLL*model.samplefactor[t]
                        for c in model.LOAD for t in model.TIME)

        # compute present value of future annual costs
        if stage == len(model.STAGE):
            #from year stage2TimeDelta until financeYears
            opcost = opcost*(
                annuityfactor(model.financeInterestrate,model.financeYears)
                - annuityfactor(model.financeInterestrate,
                                int(stage-1)*model.stage2TimeDelta))
        else:
            #from year 0
            opcost = opcost*annuityfactor(model.financeInterestrate,
                                          model.stage2TimeDelta)

        #Harald: this is already discounted back to year 0 from the present
        # value calculation above
        #opcost = opcost*discount_t0

        return opcost

    def costOperationSingleGen(self,model, g, stage):
        """Operational costs: cost of gen, load shed (NPV)"""
        opcost = 0
        #discount_t0 = (1/((1+model.financeInterestrate)
        #    **(model.stage2TimeDelta*int(stage-1))))

        # operation cost per year:
        opcost = sum(model.generation[g,t,stage]*model.samplefactor[t]*(
                    model.genCostAvg[g]*model.genCostProfile[g,t]
                    +model.genTypeEmissionRate[model.genType[g]]*model.CO2price)
                    for t in model.TIME)

        # compute present value of future annual costs
        if stage == len(model.STAGE):
            opcost = opcost*(
                annuityfactor(model.financeInterestrate,model.financeYears)
                - annuityfactor(model.financeInterestrate,
                                int(stage-1)*model.stage2TimeDelta))
        else:
            opcost = opcost*annuityfactor(model.financeInterestrate,
                                          model.stage2TimeDelta)
        #opcost = opcost*discount_t0
        return opcost


    def _createAbstractModel(self):
        model = pyo.AbstractModel()
        model.name = 'PowerGIM abstract model'

        # SETS ###############################################################

        model.NODE = pyo.Set()
        model.GEN = pyo.Set()
        model.BRANCH = pyo.Set()
        model.LOAD = pyo.Set()
        model.AREA = pyo.Set()
        model.TIME = pyo.Set()
        model.STAGE = pyo.Set()

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
        model.samplefactor = pyo.Param(model.TIME, within=pyo.NonNegativeReals)
        model.financeInterestrate = pyo.Param(within=pyo.Reals)
        model.financeYears = pyo.Param(within=pyo.Reals)
        model.omRate = pyo.Param(within=pyo.Reals)
        model.CO2price = pyo.Param(within=pyo.NonNegativeReals)
        model.VOLL = pyo.Param(within=pyo.NonNegativeReals)
        model.stage2TimeDelta = pyo.Param(within=pyo.NonNegativeReals)
        model.maxNewBranchNum = pyo.Param(within=pyo.NonNegativeReals)

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
        model.branchReactance = pyo.Param(model.BRANCH,
                                          within=pyo.NonNegativeReals)
        model.branchExistingCapacity = pyo.Param(model.BRANCH,
                                                 within=pyo.NonNegativeReals)
        model.branchExistingCapacity2 = pyo.Param(model.BRANCH,
                                                 within=pyo.NonNegativeReals)
        model.branchExpand = pyo.Param(model.BRANCH,
                                       within=pyo.Binary)
        model.branchExpand2 = pyo.Param(model.BRANCH,
                                       within=pyo.Binary)
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
        model.refNodes = pyo.Param(model.NODE, within=pyo.Boolean)


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
                                    within=pyo.Binary)
        model.genExpand2 = pyo.Param(model.GEN,
                                    within=pyo.Binary)
        model.genTypeEmissionRate = pyo.Param(model.GENTYPE, within=pyo.Reals)

        #helpers:
        model.genNode = pyo.Param(model.GEN,within=model.NODE)
        model.demNode = pyo.Param(model.LOAD,within=model.NODE)
        model.branchNodeFrom = pyo.Param(model.BRANCH,within=model.NODE)
        model.branchNodeTo = pyo.Param(model.BRANCH,within=model.NODE)
        model.nodeArea = pyo.Param(model.NODE,within=model.AREA)
        model.coeff_B = pyo.Param(model.NODE,model.NODE,within=pyo.Reals)
        model.coeff_DA = pyo.Param(model.BRANCH,model.NODE,within=pyo.Reals)

        #consumers
        # the split int an average value, and a profile is to make it easier
        # to generate scenarios (can keep profile, but adjust demandAvg)
        model.demandAvg = pyo.Param(model.LOAD,within=pyo.Reals)
        model.demandProfile = pyo.Param(model.LOAD,model.TIME,
                                        within=pyo.Reals)
        model.emissionCap = pyo.Param(model.LOAD, within=pyo.NonNegativeReals)
        model.maxShed = pyo.Param(model.LOAD, model.TIME, within=pyo.NonNegativeReals)

        # VARIABLES ##########################################################

        # investment: new branch capacity
        def branchNewCapacity_bounds(model,j,h):
            if h>1:
                return (0,model.branchMaxNewCapacity[j]*model.branchExpand2[j])
            else:
                return (0,model.branchMaxNewCapacity[j]*model.branchExpand[j])
        model.branchNewCapacity = pyo.Var(model.BRANCH, model.STAGE,
                                          within = pyo.NonNegativeReals,
                                          bounds = branchNewCapacity_bounds)

        # investment: new branch cables
        def branchNewCables_bounds(model,j,h):
            if h>1:
                return (0,model.maxNewBranchNum*model.branchExpand2[j])
            else:
                return (0,model.maxNewBranchNum*model.branchExpand[j])
        model.branchNewCables = pyo.Var(model.BRANCH, model.STAGE,
                                        within = pyo.NonNegativeIntegers,
                                        bounds = branchNewCables_bounds)


        # investment: new nodes
        model.newNodes = pyo.Var(model.NODE, model.STAGE, within = pyo.Binary)


        # investment: generation capacity
        def genNewCapacity_bounds(model,g,h):
            if h>1:
                return (0,model.genNewCapMax[g]*model.genExpand2[g])
            else:
                return (0,model.genNewCapMax[g]*model.genExpand[g])
        model.genNewCapacity = pyo.Var(model.GEN, model.STAGE,
                                       within = pyo.NonNegativeReals,
                                       bounds = genNewCapacity_bounds)


        # branch power flow (also given by constraints??)
        def branchFlow_bounds(model,j,t,h):
            if h == 1:
                ub = (model.branchExistingCapacity[j]+
                        branchNewCapacity_bounds(model,j,h)[1])
            elif h == 2:
                ub = (model.branchExistingCapacity[j]+
                        model.branchExistingCapacity2[j]+
                        branchNewCapacity_bounds(model,j,h-1)[1]+
                        branchNewCapacity_bounds(model,j,h)[1])
            return (0,ub)
        model.branchFlow12 = pyo.Var(model.BRANCH, model.TIME,model.STAGE,
                                     within = pyo.NonNegativeReals,
                                     bounds = branchFlow_bounds)
        model.branchFlow21 = pyo.Var(model.BRANCH, model.TIME, model.STAGE,
                                     within = pyo.NonNegativeReals,
                                     bounds = branchFlow_bounds)

        # voltage angle
        model.voltageAngle = pyo.Var(model.NODE,model.TIME,model.STAGE, within=pyo.Reals)

        # generator output (bounds set by constraint)
        model.generation = pyo.Var(model.GEN, model.TIME,model.STAGE,
                                   within = pyo.NonNegativeReals)
        # load shedding
        def loadShed_bounds(model, c, t, h):
            ub = model.maxShed[c,t]
            #ub = 0
            #for c in model.LOAD:
            #    if model.demNode[c]==n:
            #        ub += model.demandAvg[c]*model.demandProfile[c,t]
            return (0,ub)
        model.loadShed = pyo.Var(model.LOAD, model.TIME, model.STAGE,
                                 domain = pyo.NonNegativeReals,
                                 bounds = loadShed_bounds)



        # CONSTRAINTS ########################################################

        # Power flow limitations (in both directions)
        def maxflow12_rule(model,j,t,h):
            cap = model.branchExistingCapacity[j]
            if h >1:
                cap += model.branchExistingCapacity2[j]
            for x in range(h):
                cap += model.branchNewCapacity[j,x+1]
            expr = (model.branchFlow12[j,t,h] <= cap)
            return expr

        def maxflow21_rule(model,j,t,h):
            cap = model.branchExistingCapacity[j]
            if h >1:
                cap += model.branchExistingCapacity2[j]
            for x in range(h):
                cap += model.branchNewCapacity[j,x+1]
            expr = (model.branchFlow21[j,t,h] <= cap)
            return expr

        model.cMaxFlow12 = pyo.Constraint(model.BRANCH, model.TIME,model.STAGE,
                                         rule=maxflow12_rule)
        model.cMaxFlow21 = pyo.Constraint(model.BRANCH, model.TIME,model.STAGE,
                                         rule=maxflow21_rule)

        # No new branch capacity without new cables
        def maxNewCap_rule(model,j,h):
            typ = model.branchType[j]
            expr = (model.branchNewCapacity[j,h]
                    <= model.branchtypeMaxCapacity[typ]
                        *model.branchNewCables[j,h])
            return expr
        model.cmaxNewCapacity = pyo.Constraint(model.BRANCH, model.STAGE,
                                               rule=maxNewCap_rule)


        # A node required at each branch endpoint
        def newNodes_rule(model,n,h):
            expr = 0
            numnodes = model.nodeExistingNumber[n]
            for x in range(h):
                numnodes += model.newNodes[n,x+1]
            for j in model.BRANCH:
                if model.branchNodeFrom[j]==n or model.branchNodeTo[j]==n:
                    expr += model.branchNewCables[j,h]
            expr = (expr <= self.M_const * numnodes)
            if ((type(expr) is bool) and (expr==True)):
                expr = pyo.Constraint.Skip
            return expr
        model.cNewNodes = pyo.Constraint(model.NODE,model.STAGE,
                                          rule=newNodes_rule)

        # Generator output limitations
        # TODO: add option to set minimum output = timeseries for renewable,
        # i.e. disallov curtaliment (could be global parameter)
        def maxPgen_rule(model,g,t,h):
            cap = model.genCapacity[g]
            if h>1:
                cap += model.genCapacity2[g]
            for x in range(h):
                if (g in model.GEN_EXPAND1) or (g in model.GEN_EXPAND2):
                    cap += model.genNewCapacity[g,x+1]
            allowCurtailment = True
            #TODO: make this limit a parameter (global or per generator?)
#            if model.genCostAvg[g]*model.genCostProfile[g,t]<1:
            if model.genCostAvg[g] < 1:
                # don't allow curtailment of cheap generators (renewables)
                allowCurtailment = False
            if allowCurtailment:
                expr = model.generation[g,t,h] <= (
                    model.genCapacityProfile[g,t] * cap)
            else:
                # don't allow curtailment of generator output
                expr = model.generation[g,t,h] == (
                    model.genCapacityProfile[g,t] * cap)

            return expr
        model.cMaxPgen = pyo.Constraint(model.GEN,model.TIME,model.STAGE,
                                        rule=maxPgen_rule)


        # Generator maximum average output (energy sum)
        # (e.g. for hydro with storage)
        def maxEnergy_rule(model,g,h):
            cap = model.genCapacity[g]
            if h>1:
                cap += model.genCapacity2[g]
            for x in range(h):
                cap += model.genNewCapacity[g,x+1]
            if model.genPAvg[g]>0:
                expr = (sum(model.generation[g,t,h] for t in model.TIME)
                            <= (model.genPAvg[g]*cap*len(model.TIME)))
            else:
                expr = pyo.Constraint.Skip
            return expr
        model.cMaxEnergy = pyo.Constraint(model.GEN,model.STAGE,
                                               rule=maxEnergy_rule)


        # Emissions restriction per country/load
        # TODO: deal with situation when no emission cap has been given (-1)
        def emissionCap_rule(model,a,h):
            if model.CO2price > 0:
                expr = 0
                for n in model.NODE:
                    if model.nodeArea[n]==a:
                        expr += sum(model.generation[g,t,h]*model.genTypeEmissionRate[model.genType[g]]*model.samplefactor[t]
                                    for t in model.TIME for g in model.GEN
                                    if model.genNode[g]==n)
                expr = (expr <= sum(model.emissionCap[c]
                    for c in model.LOAD if model.nodeArea[model.demNode[c]]==a))
            else:
                expr = pyo.Constraint.Skip
            return expr
        model.cEmissionCap = pyo.Constraint(model.AREA,model.STAGE,
                                             rule=emissionCap_rule)


        # Power balance in nodes : gen+demand+flow into node=0
        def powerbalance_rule(model,n,t,h):
            expr = 0
            # flow of power into node (subtrating losses)
            for j in model.BRANCH:
                if model.branchNodeFrom[j]==n:
                    # branch out of node
                    typ = model.branchType[j]
                    dist = model.branchDistance[j]
                    expr += -model.branchFlow12[j,t,h]
                    expr += model.branchFlow21[j,t,h] * (1-(
                                model.branchLossfactor[typ,'fix']
                                +model.branchLossfactor[typ,'slope']*dist))
                if model.branchNodeTo[j]==n:
                    # branch into node
                    typ = model.branchType[j]
                    dist = model.branchDistance[j]
                    expr += model.branchFlow12[j,t,h] * (1-(
                                model.branchLossfactor[typ,'fix']
                                +model.branchLossfactor[typ,'slope']*dist))
                    expr += -model.branchFlow21[j,t,h]

            # generated power
            for g in model.GEN:
                if model.genNode[g]==n:
                    expr += model.generation[g,t,h]

            # load shedding
            for c in model.LOAD:
                if model.demNode[c]==n:
                    expr += model.loadShed[c,t,h]

            # consumed power
            for c in model.LOAD:
                if model.demNode[c]==n:
                    expr += -model.demandAvg[c]*model.demandProfile[c,t]

            if not any([model.branchReactance[b] for b in model.BRANCH])>0:
                expr = (expr == 0)
            else:
                expr = expr/const.baseMVA

                rhs = 0
                n2s = [k[1]  for k in model.coeff_B.keys() if k[0]==n]
                for n2 in n2s:
                    rhs -= model.coeff_B[n,n2]*model.voltageAngle[n2,t,h]
                expr = (expr == rhs)

            if ((type(expr) is bool) and (expr==True)):
                # Trivial constraint
                expr = pyo.Constraint.Skip
            return expr
        model.cPowerbalance = pyo.Constraint(model.NODE,model.TIME,model.STAGE,
                                             rule=powerbalance_rule)

        # Power balance (power flow vs voltage angle)
        def flowangle_rule(model,b,t,h):
            if not any([model.branchReactance[b] for b in model.BRANCH])>0:
                return pyo.Constraint.Skip
            else:
                lhs = model.branchFlow12[b,t,h]+model.branchFlow21[b,t,h]
                lhs = lhs/const.baseMVA
                rhs = 0
                #TODO speed up- remove for loop
                n2s = [k[1]  for k in model.coeff_DA.keys() if k[0]==b]
                for n2 in n2s:
                    rhs += model.coeff_DA[b,n2]*model.voltageAngle[n2,t,h]
                #for n2 in model.NODE:
                #    if (b,n2) in model.coeff_DA.keys():
                #        rhs += model.coeff_DA[b,n2]*model.varVoltageAngle[n2]
                expr = (lhs==rhs)
                return expr
        model.cFlowAngle = pyo.Constraint(model.BRANCH, model.TIME, model.STAGE, rule=flowangle_rule)

        # Reference voltag angle
        def referenceNode_rule(model,n,t,h):
            if not any([model.branchReactance[b] for b in model.BRANCH])>0:
                return pyo.Constraint.Skip
            else:
                if n in model.refNodes.keys():
                    expr = (model.voltageAngle[n,t,h] == 0)
                else:
                    expr = pyo.Constraint.Skip
                return expr
        model.cReferenceNode = pyo.Constraint(model.NODE,model.TIME,model.STAGE,
                                              rule=referenceNode_rule)


        # OBJECTIVE ##############################################################
        model.investmentCost = pyo.Var(model.STAGE, within=pyo.Reals)
        model.opCost = pyo.Var(model.STAGE, within=pyo.Reals)

        def investmentCost_rule(model,stage):
            """Investment cost, including lifetime O&M costs (NPV)"""
            expr = self.costInvestments(model,stage)
            return model.investmentCost[stage] == expr
        model.cInvestmentCost = pyo.Constraint(model.STAGE,rule=investmentCost_rule)


        def opCost_rule(model, stage):
            """Operational costs: cost of gen, load shed (NPV)"""
            opcost = self.costOperation(model,stage)
            return model.opCost[stage] == opcost
        model.cOperationalCosts = pyo.Constraint(model.STAGE, rule=opCost_rule)


        def total_Cost_Objective_rule(model):
            investment = pyo.summation(model.investmentCost)
            operation = pyo.summation(model.opCost)
            return investment + operation
        model.OBJ = pyo.Objective(rule=total_Cost_Objective_rule,
                                  sense=pyo.minimize)


        #TODO Harald check - seem to need this for stochastic problem
        # these are referred to in scenario tree creation
        def costPerStage(model,stage):
            sumcost = model.opCost[stage] + model.investmentCost[stage]
            return sumcost
        model.stageCost = pyo.Expression(model.STAGE,rule=costPerStage)

        return model



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


    def createModelData(self,grid_data,datafile,
                        maxNewBranchNum, maxNewBranchCap):
        '''Create model data in dictionary format

        Parameters
        ----------
        grid_data : powergama.GridData object
            contains grid model
        datafile : string
            name of XML file containing additional parameters
        maxNewBranchNum : int
            upper limit on parallel branches to consider (e.g. 10)
        maxNewBranchCap : float (MW)
            upper limit on new capacity to consider (e.g. 10000)

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
        di['NODE'] = {None: grid_data.node['id'].tolist() }
        di['BRANCH'] = {None: grid_data.branch.index.tolist() }
        di['GEN'] = {None: grid_data.generator.index.tolist() }
        di['LOAD'] = {None: grid_data.consumer.index.tolist() }
        di['AREA'] = {None: grid_data.getAllAreas() }
        di['TIME'] = {None: grid_data.timerange}
        #di['STAGE'] = {None: grid_data.branch.expand[grid_data.branch['expand']>0].unique().tolist()}

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
        di['maxNewBranchNum'] = {None: maxNewBranchNum}
        di['samplefactor'] = {}
        if hasattr(grid_data.profiles, 'frequency'):
            di['samplefactor'] = grid_data.profiles['frequency']
        else:
            for t in grid_data.timerange:
                di['samplefactor'][t] = self._HOURS_PER_YEAR/len(grid_data.timerange)
        di['nodeOffshore'] = {}
        di['nodeType'] = {}
        di['nodeExistingNumber'] = {}
        di['nodeCostScale']={}
        di['nodeArea']={}
        for k,row in grid_data.node.iterrows():
            n=grid_data.node['id'][k]
            #n=grid_data.node.index[k] #or simply =k
            di['nodeOffshore'][n] = row['offshore']
            di['nodeType'][n] = row['type']
            di['nodeExistingNumber'][n] = row['existing']
            di['nodeCostScale'][n] = row['cost_scaling']
            di['nodeArea'][n] = row['area']

        di['branchExistingCapacity'] = {}
        di['branchExistingCapacity2'] = {}
        di['branchExpand'] = {}
        di['branchExpand2'] = {}
        di['branchDistance'] = {}
        di['branchType'] = {}
        di['branchCostScale'] = {}
        di['branchOffshoreFrom'] = {}
        di['branchOffshoreTo'] = {}
        di['branchNodeFrom'] = {}
        di['branchNodeTo'] = {}
        di['branchMaxNewCapacity'] = {}
        di['branchReactance'] = {}
        offsh = self._offshoreBranch(grid_data)
        for k,row in grid_data.branch.iterrows():
            di['branchExistingCapacity'][k] = row['capacity']
            di['branchExistingCapacity2'][k] = row['capacity2']
            if row['max_newCap'] >0:
                di['branchMaxNewCapacity'][k] = row['max_newCap']
            else:
                di['branchMaxNewCapacity'][k] = maxNewBranchCap
            di['branchExpand'][k] = row['expand']
            di['branchExpand2'][k] = row['expand2']
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
            di['branchReactance'][k] = row['reactance']

        di['genCapacity']={}
        di['genCapacity2']={}
        di['genCapacityProfile']={}
        di['genNode']={}
        di['genCostAvg'] = {}
        di['genCostProfile'] = {}
        di['genPAvg'] = {}
        di['genExpand'] = {}
        di['genExpand2'] = {}
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
            di['genExpand2'][k] = row['expand2']
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
        di['maxShed'] = {}
        for k,row in grid_data.consumer.iterrows():
            di['demNode'][k] = row['node']
            di['demandAvg'][k] = row['demand_avg']
            di['emissionCap'][k] = row['emission_cap']
            ref = row['demand_ref']
            for i,t in enumerate(grid_data.timerange):
                di['demandProfile'][(k,t)] = grid_data.profiles[ref][i]
                di['maxShed'][(k,t)] = grid_data.profiles[ref][i]*row['demand_avg']


        # Read input data from XML file
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
            di['STAGE'] = {None:
                list(range(1,int(i.attrib['stages'])+1))}

         # Compute matrices used in power flow equaions
#        import scipy.sparse
#        import networkx as nx
        print("Computing B and DA matrices...")
#        Bbus, DA = grid_data.computePowerFlowMatrices(const.baseZ)

        n_i = di['NODE'][None]
        b_i = di['BRANCH'][None]
        di['coeff_B'] = dict()
        di['coeff_DA'] = dict()

        print("Creating B and DA coefficients...")
#        cx = scipy.sparse.coo_matrix(Bbus)
#        for i,j,v in zip(cx.row, cx.col, cx.data):
#            di['coeff_B'][(n_i[i],n_i[j])] = v
#
#        cx = scipy.sparse.coo_matrix(DA)
#        for i,j,v in zip(cx.row, cx.col, cx.data):
#            di['coeff_DA'][(b_i[i],n_i[j])] = v
#
#        # Find synchronous areas and specify reference node in each area
#        G = nx.Graph()
#        G.add_nodes_from(grid_data.node['id'])
#        G.add_edges_from(zip(grid_data.branch['node_from'],
#                             grid_data.branch['node_to']))
#
#        G_subs = nx.connected_component_subgraphs(G)
        refnodes = []
#        for gr in G_subs:
#            refnode = gr.nodes()[0]
#            refnodes.append(refnode)
#            print("Found synchronous area (size = {}), using ref node = {}"
#                    .format(gr.order(),refnode))
        # use first node as voltage angle reference
        di['refNodes'] = {n:True for n in refnodes}



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
        #root node scenario data:
        with open("{}/RootNode.dat".format(path), "w") as text_file:
            text_file.write(dat_str)
        print("Root node data written to {}/RootNode.dat".format(path))

        return dat_str


    def createScenarioTreeModel(self,
        num_scenarios,probabilities=None,stages=[1,2]):
        '''Generate model instance with data. Alternative to .dat files

        Parameters
        ----------
        num_scenarios : int
            number of scenarios. Each with the same probability
        probabilities : list of float
            probabilities of each scenario (must sum to 1). Number of elements
            determine number of scenarios
        stages : list of stage names
            NOTE: Presently only works with default value=[1,2]

        Returns
        -------
        PySP 2-stage scenario tree model

        This method may be called by "pysp_scenario_tree_model_callback()" in
        the model input file instead of using input .dat files
        '''

        if probabilities is None:
            # equal probability:
            probabilities = [1/num_scenarios]*num_scenarios
        #if probabilities is None:
        #    st_model = tsm.CreateConcreteTwoStageScenarioTreeModel(
        #                        num_scenarios)

        G = networkx.DiGraph()
        G.add_node("root")
        for i in range(len(probabilities)):
            G.add_edge("root","Scenario{}".format(i+1),
                       probability=probabilities[i])
        stage_names=stages

        st_model = tsm.ScenarioTreeModelFromNetworkX(G,
                         edge_probability_attribute="probability",
                         stage_names=stage_names)

        first_stage = st_model.Stages.first()
        second_stage = st_model.Stages.last()

        # First Stage
        st_model.StageCost[first_stage] = 'stageCost[1]'
        st_model.StageVariables[first_stage].add('branchNewCables[*,1]')
        st_model.StageVariables[first_stage].add('branchNewCapacity[*,1]')
        st_model.StageVariables[first_stage].add('newNodes[*,1]')
        st_model.StageVariables[first_stage].add('genNewCapacity[*,1]')

        # Second Stage
        st_model.StageCost[second_stage] = 'stageCost[2]'
        st_model.StageVariables[second_stage].add('generation')
        st_model.StageVariables[second_stage].add('branchFlow12')
        st_model.StageVariables[second_stage].add('branchFlow21')
        st_model.StageVariables[second_stage].add('genNewCapacity[*,2]')
        st_model.StageVariables[second_stage].add('branchNewCables[*,2]')
        st_model.StageVariables[second_stage].add('branchNewCapacity[*,2]')
        st_model.StageVariables[second_stage].add('newNodes[*,2]')

        st_model.ScenarioBasedData=False

        return st_model


    def computeBranchCongestionRent(self, model, b, stage=1):
        '''
        Compute annual congestion rent for a given branch
        '''
        # TODO: use nodal price, not area price.
        N1 = model.branchNodeFrom[b]
        N2 = model.branchNodeTo[b]

        area1 = model.nodeArea[N1]
        area2 = model.nodeArea[N2]

        flow = []
        deltaP = []

        for t in model.TIME:
            deltaP.append(abs(self.computeAreaPrice(model, area1, t,stage)
                         - self.computeAreaPrice(model, area2, t,stage))*model.samplefactor[t])
            flow.append(model.branchFlow21[b,t,stage].value + model.branchFlow12[b,t,stage].value)

        return sum(deltaP[i]*flow[i] for i in range(len(deltaP)))

    def computeCostBranch(self,model,b,stage=2,include_om=False):
        '''Investment cost of single branch NPV

        corresponds to  firstStageCost in abstract model'''
        cost1 = self.costBranch(model,b,stage=stage)
        cost1npv = self.npvInvestment(model,stage=stage,investment=cost1,
                                      includeOM=include_om,
                                      subtractSalvage=True)
        cost_value = pyo.value(cost1npv)
        return cost_value

#        ar = 1
#        br_num=0
#        br_cap=0
#        b_cost = 0
#
#        salvagefactor = (int(stage-1)*model.stage2TimeDelta/model.financeYears)*(
#                1/((1+model.financeInterestrate)
#                **(model.financeYears-model.stage2TimeDelta*int(stage-1))))
#        discount_t0 = (1/((1+model.financeInterestrate)
#                        **(model.stage2TimeDelta*int(stage-1))))
#        if stage==1:
#            ar = annuityfactor(model.financeInterestrate,model.financeYears)
#        else:
#            ar = (annuityfactor(model.financeInterestrate,model.financeYears)
#                  -annuityfactor(model.financeInterestrate,
#                                 int(stage-1)*model.stage2TimeDelta))
#        br_num += model.branchNewCables[b,stage].value
#        br_cap += model.branchNewCapacity[b,stage].value
#        typ = model.branchType[b]
#        b_cost += (model.branchtypeCost[typ,'B']
#                    *br_num)
#        b_cost += (model.branchtypeCost[typ,'Bd']
#                    *model.branchDistance[b]*br_num)
#        b_cost += (model.branchtypeCost[typ,'Bdp']
#                    *model.branchDistance[b]*br_cap)
#        #endpoints offshore (N=1) or onshore (N=0) ?
#        N1 = model.branchOffshoreFrom[b]
#        N2 = model.branchOffshoreTo[b]
#        for N in [N1,N2]:
#            b_cost += N*(model.branchtypeCost[typ,'CS']*br_num
#                        +model.branchtypeCost[typ,'CSp']*br_cap)
#            b_cost += (1-N)*(model.branchtypeCost[typ,'CL']*br_num
#                        +model.branchtypeCost[typ,'CLp']*br_cap)
#        cost = model.branchCostScale[b]*b_cost
#        if include_om:
#            cost = cost*(1 + model.omRate * ar)
#        cost -= cost*salvagefactor
#        cost = cost*discount_t0
#        return cost

    def computeCostNode(self,model,n,include_om=False):
        '''Investment cost of single node

        corresponds to cost in abstract model'''

        # Re-use method used in optimisation
        #NOTE: adding "()" after the expression gives the value
        cost={}
        costnpv={}
        cost_value = 0
        for stage in model.STAGE:
            cost[stage] = self.costNode(model,n,stage=stage)
            costnpv[stage] = self.npvInvestment(model,stage=stage,
                               investment=cost[stage],
                               includeOM=include_om,
                               subtractSalvage=True)
            cost_value = cost_value + pyo.value(costnpv[stage])
        return cost_value


#        # node may be expanded in stage 1 or stage 2, but will never be
#        # expanded in both
#        # TODO: cope with stages
#
#        ar = 1
#        n_num = 0
#        stage=2
#        salvagefactor = (int(stage-1)*model.stage2TimeDelta/model.financeYears)*(
#                1/((1+model.financeInterestrate)
#                **(model.financeYears-model.stage2TimeDelta*int(stage-1))))
#        discount_t0 = (1/((1+model.financeInterestrate)
#                        **(model.stage2TimeDelta*int(stage-1))))
#        for s in model.STAGE:
#            if s<2:
#                n_num = model.newNodes[n,s].value
#                ar = (annuityfactor(model.financeInterestrate,model.financeYears))
#            else:
#                n_num = model.newNodes[n,s-1].value+model.newNodes[n,s].value
#                ar = (annuityfactor(model.financeInterestrate,model.financeYears)
#                      -annuityfactor(model.financeInterestrate,
#                                     (s-1)*model.stage2TimeDelta))
#        n_cost = 0
#        N = model.nodeOffshore[n]
#        n_cost += N*(model.nodetypeCost[model.nodeType[n],'S']*n_num)
#        n_cost += (1-N)*(model.nodetypeCost[model.nodeType[n],'L']*n_num)
#        cost = model.nodeCostScale[n]*n_cost
#        if include_om:
#            cost = cost*(1 + model.omRate * ar)
#        cost -= cost*salvagefactor
#        cost = cost*discount_t0
#        return cost


    def computeCostGenerator(self,model,g,stage=2,include_om=False):
        '''Investment cost of generator NPV
        '''
        cost1 = self.costGen(model,g,stage=stage)
        cost1npv = self.npvInvestment(model,stage=stage,investment=cost1,
                                      includeOM=include_om,
                                      subtractSalvage=True)
        cost_value = pyo.value(cost1npv)
        return cost_value

#        ar = 1
#        g_cap = 0
#        salvagefactor = (int(stage-1)*model.stage2TimeDelta/model.financeYears)*(
#                1/((1+model.financeInterestrate)
#                **(model.financeYears-model.stage2TimeDelta*int(stage-1))))
#        discount_t0 = (1/((1+model.financeInterestrate)
#                        **(model.stage2TimeDelta*int(stage-1))))
#        if stage==1:
#            ar = annuityfactor(model.financeInterestrate,model.financeYears)
#        elif stage==2:
#            ar = (annuityfactor(model.financeInterestrate,model.financeYears)
#                      -annuityfactor(model.financeInterestrate,
#                                     model.stage2TimeDelta))
#        g_cap = model.genNewCapacity[g,stage].value
#        typ = model.genType[g]
#        cost = model.genTypeCost[typ]*g_cap*ar
#        if include_om:
#            cost = cost*(1 + model.omRate * ar)
#        cost -= cost*salvagefactor
#        cost = cost*discount_t0
#        return cost


    def computeGenerationCost(self,model,g,stage):
        '''compute NPV cost of generation (+ CO2 emissions)

        This corresponds to secondStageCost in abstract model
        '''
        cost1 = self.costOperationSingleGen(model,g,stage=stage)
        cost_value = pyo.value(cost1)
        return cost_value

#        return cost_value
#        ar = 1
#        if stage == 1:
#            gen = model.generation
#            ar = annuityfactor(model.financeInterestrate,
#                               model.stage2TimeDelta)
#        elif stage==2:
#            gen = model.generation
#            ar = (
#                annuityfactor(model.financeInterestrate,model.financeYears)
#                -annuityfactor(model.financeInterestrate,model.stage2TimeDelta)
#                )
#        expr = sum(gen[g,t,stage].value*model.samplefactor[t]*
#                    model.genCostAvg[g]*model.genCostProfile[g,t]
#                     for t in model.TIME)
#        expr2 = sum(gen[g,t,stage].value*model.samplefactor[t]*
#                            model.genTypeEmissionRate[model.genType[g]]
#                            for t in model.TIME)
#        expr2 = expr2*model.CO2price.value
#        # lifetime cost
#        expr = (expr+expr2)*ar
#        return expr


    def computeDemand(self,model,c,t):
        '''compute demand at specified load ant time'''
        return model.demandAvg[c]*model.demandProfile[c,t]



    def computeCurtailment(self, model, g, t, stage=2):
        '''compute curtailment [MWh] per generator per hour'''
        cur = 0
        gen_max = 0
        if model.generation[g,t,stage].value >0 and model.genCostAvg[g]*model.genCostProfile[g,t]<1:
            if stage == 1:
                gen_max = model.genCapacity[g]
                if model.genNewCapacity[g,stage].value is not None:
                    gen_max = gen_max + model.genNewCapacity[g,stage].value
                cur = gen_max*model.genCapacityProfile[g,t] - model.generation[g,t,stage].value
            if stage == 2:
                gen_max = (model.genCapacity[g] + model.genCapacity2[g])
                if model.genNewCapacity[g,stage-1].value is not None:
                    gen_max = gen_max + model.genNewCapacity[g,stage-1].value
                if model.genNewCapacity[g,stage].value is not None:
                    gen_max = gen_max + model.genNewCapacity[g,stage].value
                cur = gen_max*model.genCapacityProfile[g,t] - model.generation[g,t,stage].value
        return cur




    def computeAreaEmissions(self,model,c, stage=2, cost=False):
        '''compute total emissions from a load/country'''
        # TODO: ensure that all nodes are mapped to a country/load
        n = model.demNode[c]
        expr = 0
        gen=model.generation
        if stage==1:
            ar = annuityfactor(model.financeInterestrate,model.stage2TimeDelta)
        elif stage==2:
            ar = (annuityfactor(model.financeInterestrate,model.financeYears)
                -annuityfactor(model.financeInterestrate,model.stage2TimeDelta))

        for g in model.GEN:
            if model.genNode[g]==n:
                expr += sum(gen[g,t,stage].value*model.samplefactor[t]*
                            model.genTypeEmissionRate[model.genType[g]]
                            for t in model.TIME)
        if cost:
            expr = expr * model.CO2price.value * ar
        return expr


    def computeAreaRES(self, model,j, shareof, stage=2):
        '''compute renewable share of demand or total generation capacity'''
        node = model.demNode[j]
        area = model.nodeArea[node]
        Rgen = 0
        costlimit_RES=1 # limit for what to consider renewable generator
        gen_p=model.generation
        gen = 0
        dem = sum(model.demandAvg[j]*model.demandProfile[j,t] for t in model.TIME)
        for g in model.GEN:
            if model.nodeArea[model.genNode[g]]==area:
                if model.genCostAvg[g] <= costlimit_RES:
                    Rgen += sum(gen_p[g,t,stage].value for t in model.TIME)
                else:
                    gen += sum(gen_p[g,t,stage].value for t in model.TIME)

        if shareof=='dem':
           return Rgen/dem
        elif shareof=='gen':
            if gen+Rgen>0:
                return Rgen/(gen+Rgen)
            else:
                return np.nan
        else:
           print('Choose shareof dem or gen')


    def computeAreaPrice(self, model, area, t, stage=2):
        '''cumpute the approximate area price based on max marginal cost'''
        mc = []
        for g in model.GEN:
            gen = model.generation[g,t,stage].value
            if gen > 0:
                if model.nodeArea[model.genNode[g]]==area:
                    mc.append(model.genCostAvg[g]*model.genCostProfile[g,t]
                                +model.genTypeEmissionRate[model.genType[g]]*model.CO2price.value)
        if len(mc)==0:
            # no generators in area, so no price...
            price=np.nan
        else:
            price = max(mc)
        return price


    def computeAreaWelfare(self, model, c, t, stage=2):
        '''compute social welfare for a given area and time step

        Returns: Welfare, ProducerSurplus, ConsumerSurplus,
                 CongestionRent, IMport, eXport
        '''
        node = model.demNode[c]
        area = model.nodeArea[node]
        PS = 0; CS = 0; CR = 0; GC = 0; gen = 0;
        dem = model.demandAvg[c]*model.demandProfile[c,t]
        price = self.computeAreaPrice(model, area, t, stage)
        #branch_capex = self.computeAreaCostBranch(model,c,include_om=True) #npv
        #gen_capex = self.computeAreaCostGen(model,c) #annualized

        #TODO: check phase1 vs phase2
        gen_p = model.generation
        flow12 = model.branchFlow12
        flow21 = model.branchFlow21

        for g in model.GEN:
            if model.nodeArea[model.genNode[g]]==area:
                gen += gen_p[g,t,stage].value
                GC += gen_p[g,t,stage].value*(
                    model.genCostAvg[g]*model.genCostProfile[g,t]
                    +model.genTypeEmissionRate[model.genType[g]]*model.CO2price.value)
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
                        flow.append(flow12[j,t,stage].value)
                        price2.append(self.computeAreaPrice(model,
                                        model.nodeArea[model.branchNodeTo[j]],
                                        t,stage))
                if (model.nodeArea[model.branchNodeTo[j]]==area and
                    model.nodeArea[model.branchNodeFrom[j]]!=area):
                        flow.append(flow21[j,t,stage].value)
                        price2.append(self.computeAreaPrice(model,
                                        model.nodeArea[model.branchNodeFrom[j]],
                                        t,stage))
            CR = sum(flow[i]*(price2[i]-price) for i in range(len(flow)))/2
        elif gen < dem:
            X = 0
            IM = price*(dem-gen)
            flow = []
            price2 = []
            for j in model.BRANCH:
                if (model.nodeArea[model.branchNodeFrom[j]]==area and
                    model.nodeArea[model.branchNodeTo[j]]!=area):
                        flow.append(flow21[j,t,stage].value)
                        price2.append(self.computeAreaPrice(model,
                                        model.nodeArea[model.branchNodeTo[j]],
                                        t,stage))
                if (model.nodeArea[model.branchNodeTo[j]]==area and
                    model.nodeArea[model.branchNodeFrom[j]]!=area):
                        flow.append(flow12[j,t,stage].value)
                        price2.append(self.computeAreaPrice(model,
                                        model.nodeArea[model.branchNodeFrom[j]],
                                        t,stage))
            CR = sum(flow[i]*(price-price2[i]) for i in range(len(flow)))/2
        else:
            X = 0
            IM = 0
            flow = [0]
            price2 = [0]
            CR = 0
        W = PS + CS + CR
        return {'W':W, 'PS':PS, 'CS':CS, 'CC':CC, 'GC':GC, 'CR':CR, 'IM':IM, 'X':X}


    def computeAreaCostBranch(self,model,c,stage,include_om=False):
        '''Investment cost for branches connected to an given area'''
        node = model.demNode[c]
        area = model.nodeArea[node]
        cost = 0

        for b in model.BRANCH:
            if model.nodeArea[model.branchNodeTo[b]]==area:
                cost += self.computeCostBranch(model,b,stage,include_om)
            elif model.nodeArea[model.branchNodeFrom[b]]==area:
                cost += self.computeCostBranch(model,b,stage,include_om)

        # assume 50/50 cost sharing
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
                                           'existingCapacity','expand','newCables','newCapacity',
                                           'flow12avg_1','flow21avg_1',
                                           'existingCapacity2','expand2','newCables2','newCapacity2',
                                           'flow12avg_2', 'flow21avg_2',
                                           'cost_withOM','congestion_rent'])
        df_nodes = pd.DataFrame(columns=['num','area','newNodes1','newNodes2',
                                         'cost','cost_withOM'])
        df_gen = pd.DataFrame(columns=['num','node','area','type',
                                       'pmax','expand','newCapacity',
                                       'pmax2','expand2','newCapacity2'])
        df_load = pd.DataFrame(columns=['num','node','area','Pavg','Pmin','Pmax',
                                        'emissions','emissionCap', 'emission_cost',
                                        'price_avg','RES%dem','RES%gen', 'IM', 'EX',
                                        'CS', 'PS', 'CR', 'CAPEX', 'Welfare'])

        for j in model.BRANCH:
            df_branches.loc[j,'num'] = j
            df_branches.loc[j,'from'] = model.branchNodeFrom[j]
            df_branches.loc[j,'to'] = model.branchNodeTo[j]
            df_branches.loc[j,'fArea'] = model.nodeArea[model.branchNodeFrom[j]]
            df_branches.loc[j,'tArea'] = model.nodeArea[model.branchNodeTo[j]]
            df_branches.loc[j,'type'] = model.branchType[j]
            df_branches.loc[j,'existingCapacity'] = model.branchExistingCapacity[j]
            df_branches.loc[j,'expand'] = model.branchExpand[j]
            df_branches.loc[j,'existingCapacity2'] = model.branchExistingCapacity2[j]
            df_branches.loc[j,'expand2'] = model.branchExpand2[j]
            for s in model.STAGE:
                if s == 1:
                    df_branches.loc[j,'newCables'] = model.branchNewCables[j,s].value
                    df_branches.loc[j,'newCapacity'] = model.branchNewCapacity[j,s].value
                    df_branches.loc[j,'flow12avg_1'] = np.mean([
                        model.branchFlow12[(j,t,s)].value for t in model.TIME])
                    df_branches.loc[j,'flow21avg_1'] = np.mean([
                        model.branchFlow21[(j,t,s)].value for t in model.TIME])
                    cap1 = model.branchExistingCapacity[j] + model.branchNewCapacity[j,s].value
                    df_branches.loc[j,'flow12%_1'] = (
                            df_branches.loc[j,'flow12avg_1']/cap1)
                    df_branches.loc[j,'flow21%_1'] = (
                            df_branches.loc[j,'flow21avg_1']/cap1)
                elif s==2:
                    df_branches.loc[j,'newCables2'] = model.branchNewCables[j,s].value
                    df_branches.loc[j,'newCapacity2'] = model.branchNewCapacity[j,s].value
                    df_branches.loc[j,'flow12avg_2'] = np.mean([
                        model.branchFlow12[(j,t,s)].value for t in model.TIME])
                    df_branches.loc[j,'flow21avg_2'] = np.mean([
                        model.branchFlow21[(j,t,s)].value for t in model.TIME])
                    cap1 = model.branchExistingCapacity[j] + model.branchNewCapacity[j,s-1].value
                    cap2 = cap1 + model.branchExistingCapacity2[j] + model.branchNewCapacity[j,s].value
                    df_branches.loc[j,'flow12%_2'] = (
                            df_branches.loc[j,'flow12avg_2']/cap2)
                    df_branches.loc[j,'flow21%_2'] = (
                            df_branches.loc[j,'flow21avg_2']/cap2)
            #branch costs
            df_branches.loc[j,'cost'] = sum(self.computeCostBranch(model,j,stage) for stage in model.STAGE)
            df_branches.loc[j,'cost_withOM'] = sum(self.computeCostBranch(
                        model,j,stage,include_om=True) for stage in model.STAGE)
            df_branches.loc[j,'congestion_rent'] = self.computeBranchCongestionRent(
                        model,j,stage=len(model.STAGE))*annuityfactor(model.financeInterestrate,model.financeYears)

        for j in model.NODE:
            df_nodes.loc[j,'num'] = j
            df_nodes.loc[j,'area'] = model.nodeArea[j]
            for s in model.STAGE:
                if s==1:
                    df_nodes.loc[j,'newNodes1'] = model.newNodes[j,s].value
                elif s==2:
                    df_nodes.loc[j,'newNodes2'] = model.newNodes[j,s].value
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
            df_gen.loc[j,'expand2'] = model.genExpand2[j]
            df_gen.loc[j,'emission_rate'] = model.genTypeEmissionRate[model.genType[j]]
            for s in model.STAGE:
                if s==1:
                    df_gen.loc[j,'emission1'] = (model.genTypeEmissionRate[model.genType[j]]*sum(
                        model.generation[j,t,s].value*model.samplefactor[t] for t in model.TIME))
                    df_gen.loc[j,'Pavg1'] = np.mean([
                        model.generation[(j,t,s)].value for t in model.TIME])
                    df_gen.loc[j,'Pmin1'] = np.min([
                        model.generation[(j,t,s)].value for t in model.TIME])
                    df_gen.loc[j,'Pmax1'] = np.max([
                        model.generation[(j,t,s)].value for t in model.TIME])
                    df_gen.loc[j,'curtailed_avg1'] = np.mean([
                        self.computeCurtailment(model,j,t,stage=1) for t in model.TIME])
                    df_gen.loc[j,'newCapacity'] = model.genNewCapacity[j,s].value
                    df_gen.loc[j,'cost_NPV1'] = self.computeGenerationCost(model,j,stage=1)
                elif s==2:
                    df_gen.loc[j,'emission2'] = (model.genTypeEmissionRate[model.genType[j]]*sum(
                        model.generation[j,t,s].value*model.samplefactor[t] for t in model.TIME))
                    df_gen.loc[j,'Pavg2'] = np.mean([
                        model.generation[(j,t,s)].value for t in model.TIME])
                    df_gen.loc[j,'Pmin2'] = np.min([
                        model.generation[(j,t,s)].value for t in model.TIME])
                    df_gen.loc[j,'Pmax2'] = np.max([
                        model.generation[(j,t,s)].value for t in model.TIME])
                    df_gen.loc[j,'curtailed_avg2'] = np.mean([
                        self.computeCurtailment(model,j,t,stage=2) for t in model.TIME])
                    df_gen.loc[j,'newCapacity2'] = model.genNewCapacity[j,s].value
                    df_gen.loc[j,'cost_NPV2'] = self.computeGenerationCost(model,j,stage=2)

            df_gen.loc[j,'cost_investment'] = sum(self.computeCostGenerator(model,j,stage) for stage in model.STAGE)
            df_gen.loc[j,'cost_investment_withOM'] = sum(self.computeCostGenerator(model,j,stage,
                include_om=True) for stage in model.STAGE)

        print("TODO: powergim.saveDeterministicResults LOAD:"
                "only showing phase 2 (after 2nd stage investments)")
        #stage=2
        stage=max(model.STAGE)
        def _n(n,p):
            return n+str(p)

        for j in model.LOAD:
            df_load.loc[j,'num'] = j
            df_load.loc[j,'node'] = model.demNode[j]
            df_load.loc[j,'area'] = model.nodeArea[model.demNode[j]]
            df_load.loc[j,'price_avg'] = np.mean(
                [self.computeAreaPrice(model,df_load.loc[j,'area'], t,stage=stage)
                for t in model.TIME])
            df_load.loc[j,'Pavg'] = np.mean([self.computeDemand(model,j,t)
                for t in model.TIME])
            df_load.loc[j,'Pmin'] = np.min([self.computeDemand(model,j,t)
                for t in model.TIME])
            df_load.loc[j,'Pmax'] = np.max([self.computeDemand(model,j,t)
                for t in model.TIME])
            if model.CO2price.value>0:
                df_load.loc[j,'emissionCap'] = model.emissionCap[j]
            df_load.loc[j,_n('emissions',stage)] = self.computeAreaEmissions(model,j,
                                                stage=stage)
            df_load.loc[j,_n('emission_cost',stage)] = self.computeAreaEmissions(model,j,
                                                stage=stage,cost=True)
            df_load.loc[j,_n('RES%dem',stage)] = self.computeAreaRES(model,j,stage=stage,
                                                            shareof='dem')
            df_load.loc[j,_n('RES%gen',stage)] = self.computeAreaRES(model,j,stage=stage,
                                                            shareof='gen')
            df_load.loc[j,_n('Welfare',stage)] = sum(self.computeAreaWelfare(model,j,t,stage=stage)['W']*model.samplefactor[t]
                                            for t in model.TIME)
            df_load.loc[j,_n('PS',stage)] = sum(self.computeAreaWelfare(model,j,t,stage=stage)['PS']*model.samplefactor[t]
                                            for t in model.TIME)
            df_load.loc[j,_n('CS',stage)] = sum(self.computeAreaWelfare(model,j,t,stage=stage)['CS']*model.samplefactor[t]
                                            for t in model.TIME)
            df_load.loc[j,_n('CR',stage)] = sum(self.computeAreaWelfare(model,j,t,stage=stage)['CR']*model.samplefactor[t]
                                            for t in model.TIME)
            df_load.loc[j,_n('IM',stage)] = sum(self.computeAreaWelfare(model,j,t,stage=stage)['IM']*model.samplefactor[t]
                                            for t in model.TIME)
            df_load.loc[j,_n('EX',stage)] = sum(self.computeAreaWelfare(model,j,t,stage=stage)['X']*model.samplefactor[t]
                                            for t in model.TIME)
            df_load.loc[j,_n('CAPEX1',stage)] = self.computeAreaCostBranch(model,j,stage,
                                            include_om=False)

        df_cost = pd.DataFrame(columns=['value','unit'])
        df_cost.loc['InvestmentCosts','value'] = sum(
            model.investmentCost[s].value for s in model.STAGE)/1e9
        df_cost.loc['OperationalCosts','value'] = sum(
            model.opCost[s].value for s in model.STAGE)/1e9
        df_cost.loc['newTransmission','value'] = sum(
            self.computeCostBranch(model,b,stage,include_om=True)
            for b in model.BRANCH for stage in model.STAGE)/1e9
        df_cost.loc['newGeneration','value'] = sum(
            self.computeCostGenerator(model,g,stage,include_om=True)
            for g in model.GEN for stage in model.STAGE)/1e9
        df_cost.loc['newNodes','value'] = sum(
            self.computeCostNode(model,n,include_om=True)
            for n in model.NODE)/1e9
        df_cost.loc['InvestmentCosts','unit'] = '10^9 EUR'
        df_cost.loc['OperationalCosts','unit'] = '10^9 EUR'
        df_cost.loc['newTransmission','unit'] = '10^9 EUR'
        df_cost.loc['newGeneration','unit'] = '10^9 EUR'
        df_cost.loc['newNodes','unit'] = '10^9 EUR'

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
        writer.save()


    def extractResultingGridData(self,grid_data,
                                 model=None,file_ph=None,
                                 stage=1,scenario=None, newData=False):
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
        newData : Boolean
            Choose whether to use only new data (True) or add new data to
            existing data (False)

        Use either model or file_ph parameter

        Returns
        =======
        GridData object reflecting optimal solution
        '''

        grid_res = copy.deepcopy(grid_data)
        res_brC = pd.DataFrame(data=grid_res.branch['capacity'])
        res_N = pd.DataFrame(data=grid_res.node['existing'])
        res_G = pd.DataFrame(data=grid_res.generator['pmax'])
        if newData:
            res_brC[res_brC>0] = 0
#            res_N[res_N>0] = 0
            res_G[res_G>0] = 0

        if model is not None:
            # Deterministic optimisation results
            if stage >= 1:
                stage1=1
                for j in model.BRANCH:
                    res_brC.loc[j,'capacity'] += model.branchNewCapacity[j,stage1].value
                for j in model.NODE:
                    res_N.loc[j,'existing'] += int(model.newNodes[j,stage1].value)
                for j in model.GEN:
                    newgen=model.genNewCapacity[j,stage1].value
                    if newgen is not None:
                        res_G.loc[j,'pmax'] += model.genNewCapacity[j,stage1].value
            if stage >= 2:
                #add to investments in stage 1
                res_brC['capacity'] += grid_res.branch['capacity2']
                res_G['pmax'] += grid_res.generator['pmax2']
                for j in model.BRANCH:
                    res_brC.loc[j,'capacity'] += model.branchNewCapacity[j,stage].value
                for j in model.NODE:
                    res_N.loc[j,'existing'] += int(model.newNodes[j,stage].value)
                for j in model.GEN:
                    newgen=model.genNewCapacity[j,stage].value
                    if newgen is not None:
                        res_G.loc[j,'pmax'] += model.genNewCapacity[j,stage].value
        elif file_ph is not None:
            # Stochastic optimisation results
            df_ph = pd.read_csv(file_ph,header=None,skipinitialspace=True,
                names=['stage','node','var','var_indx','value'],
                na_values=['None']).fillna(0)
            # var_indx consists of e.g. [ind,stage], or [ind,time,stage]
            # expand into three columns [ind,time,stage]
            ind_split = df_ph['var_indx'].str.split(':',expand=True)
            if (ind_split.shape[1]!=3):
                raise Exception("Was assuming different format"
                                " - implementation error")
            mask_only2= ind_split[2]==None
            ind_split.loc[mask_only2,2]=ind_split.loc[mask_only2,2]
            ind_split.loc[mask_only2,1]=ind_split.loc[mask_only2,2]
            df_ph[['ind_i','ind_time','ind_stage']]= ind_split
            df_ph['ind_i']=df_ph['ind_i'].str.replace("'","")
            if stage>=1:
                stage1=1
                df_branchNewCapacity = df_ph[
                    (df_ph['var']=='branchNewCapacity') &
                    (df_ph['stage']==stage1)]
                df_newNodes = df_ph[
                    (df_ph['var']=='newNodes') &
                    (df_ph['stage']==stage1)]
                df_newGen = df_ph[
                    (df_ph['var']=='genNewCapacity') &
                    (df_ph['stage']==stage1)]
                for k,row in df_branchNewCapacity.iterrows():
                    res_brC.loc[int(row['ind_i']),'capacity'] += float(row['value'])
                for k,row in df_newNodes.iterrows():
                    res_N.loc[row['ind_i'],'existing'] += int(float(row['value']))
                for k,row in df_newGen.iterrows():
                    res_G.loc[int(row['ind_i']),'pmax'] += float(row['value'])
            if stage>=2:
                if scenario is None:
                    raise Exception('Missing input "scenario"')
                res_brC['capacity'] += grid_res.branch['capacity2']
                res_G['pmax'] += grid_res.generator['pmax2']

                df_branchNewCapacity = df_ph[
                    (df_ph['var']=='branchNewCapacity') &
                    (df_ph['stage']==stage) &
                    (df_ph['node']=='Scenario{}'.format(scenario))]
                df_newNodes = df_ph[(df_ph['var']=='newNodes') &
                    (df_ph['stage']==stage) &
                    (df_ph['node']=='Scenario{}'.format(scenario))]
                df_newGen = df_ph[(df_ph['var']=='genNewCapacity') &
                    (df_ph['stage']==stage) &
                    (df_ph['node']=='Scenario{}'.format(scenario))]

                for k,row in df_branchNewCapacity.iterrows():
                    res_brC.loc[int(row['ind_i']),'capacity'] += float(row['value'])
                for k,row in df_newNodes.iterrows():
                    res_N.loc[row['ind_i'],'existing'] += int(float(row['value']))
                for k,row in df_newGen.iterrows():
                    res_G.loc[int(row['ind_i']),'pmax'] += float(row['value'])
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

    def loadResults(self, filename,sheet):
        '''load results from excel into pandas dataframe'''
        df_res = pd.read_excel(filename,sheetname=sheet)
        return df_res

    def plotEnergyMix(self, model, areas=None,timeMaxMin=None,relative=False,
                      showTitle=True,variable="energy",gentypes=None, stage=1):
        '''
        Plot energy, generation capacity or spilled energy as stacked bars

        Parameters
        ----------
        areas : list of sting
            Which areas to include, default=None means include all
        timeMaxMin : list of two integers
            Time range, [min,max]
        relative : boolean
            Whether to plot absolute (false) or relative (true) values
        variable : string ("energy","capacity","spilled")
            Which variable to plot (default is energy production)
        gentypes : list
            List of generator types to include. None gives all.
        '''

        s = stage
        if areas is None:
            areas = list(model.AREA)
        if timeMaxMin is None:
            timeMaxMin = list(model.TIME)
        if gentypes is None:
            gentypes = list(model.GENTYPE)

        gen_output = []
        if variable=="energy":
            print("Getting energy output from all generators...")
            for g in model.GEN:
                gen_output.append(sum(model.generation[g,t,s].value for t in timeMaxMin))
            title = "Energy mix"
        elif variable=="capacity":
            print("Getting capacity from all generators...")
            for g in model.GEN:
                gen_output.append(model.genCapacity[g])
            title = "Capacity mix"
        elif variable=="spilled":
            print("Getting curatailed energy from all generators...")
            for g in model.GEN:
                gen_output.append(sum(self.computeCurtailment(model,g,t,s) for t in timeMaxMin))
            title = "Energy spilled"
        else:
            print("Variable not valid")
            return
        #all_generators = self.grid.getGeneratorsPerAreaAndType()

        if relative:
            prodsum={}
            for ar in areas:
                prodsum[ar] = 0
                for i in model.GEN:
                    if ar == model.nodeArea[model.genNode[i]]:
                        prodsum[ar] += gen_output[i]

        plt.figure()
        ax = plt.subplot(111)
        width = 0.8
        previous = [0]*len(areas)
        numCurves = len(gentypes)+1
        colours = cm.hsv(np.linspace(0, 1, numCurves))
        #colours = cm.viridis(np.linspace(0, 1, numCurves))
        #colours = cm.Set3(np.linspace(0, 1, numCurves))
        #colours = cm.Grays(np.linspace(0, 1, numCurves))
        #colours = cm.Dark2(np.linspace(0, 1, numCurves))
        count=0
        ind = range(len(areas))
        for typ in gentypes:
            A=[]
            for ar in model.AREA:
                prod = 0
                for g in model.GEN:
                    if (typ==model.genType[g])&(ar==model.nodeArea[model.genNode[g]]):
                        prod += gen_output[g]
                    else:
                        prod += 0
                if relative:
                    if prodsum[ar]>0:
                        prod = prod/prodsum[ar]
                        A.append(prod)
                    else:
                        A.append(prod)
                else:
                    A.append(prod)
            plt.bar(ind,A, width,label=typ,
                    bottom=previous,color=colours[count])
            previous = [previous[i]+A[i] for i in range(len(A))]
            count = count+1

        handles, labels = ax.get_legend_handles_labels()
        handles.reverse()
        labels.reverse()
        plt.legend(handles, labels, loc='upper right', fontsize='medium')
#        plt.legend(handles, labels, loc='best',
#                   bbox_to_anchor=(1.05,1), borderaxespad=0.0)
        plt.xticks(np.arange(len(areas))+width/2., tuple(areas) )
        if showTitle:
            plt.title(title)
        plt.show()
        return

    def plotAreaPrice(self, model, boxplot=False, areas=None,timeMaxMin=None,showTitle=False,stage=1):
        '''Show area price(s)
        TODO: incoporate samplefactor

        Parameters
        ----------
        areas (list)
            list of areas to show
        timeMaxMin (list) (default = None)
            [min, max] - lower and upper time interval
        '''

        s = stage
        if areas is None:
            areas = list(model.AREA)
        if timeMaxMin is None:
            timeMaxMin = list(model.TIME)
        timerange = range(timeMaxMin[0],timeMaxMin[-1])

        numCurves = len(areas)+1
        #colours = cm.viridis(np.linspace(0, 1, numCurves))
        colours = cm.hsv(np.linspace(0, 1, numCurves))
        count = 0
        if boxplot:
            areaprice = {}
            factor = {}
            for a in areas:
                areaprice[a] = {}
                factor[a] = {}
                areaprice[a] = [self.computeAreaPrice(model,a,t,s) for t in timerange]
                factor[a] = [model.samplefactor[t] for t in timerange]
            df = pd.DataFrame.from_dict(areaprice)
            props = dict(whiskers='DarkOrange', medians='lime', caps='Gray')
            boxprops = dict(linestyle='--', linewidth=3, color='DarkOrange', facecolor='k')
            flierprops = dict(marker='o', markerfacecolor='none', markersize=8,linestyle='none')
            meanpointprops = dict(marker='D', markeredgecolor='red',markerfacecolor='red')
            medianprops = dict(linestyle='-', linewidth=4, color='red')
            df.plot.box(color=props, boxprops=boxprops, flierprops=flierprops,
                        meanprops=meanpointprops, medianprops=medianprops, patch_artist=True, showmeans=True)
            #plt.legend(areas)
        else:
            plt.figure()
            for a in areas:
                areaprice = [self.computeAreaPrice(model,a,t,s) for t in timerange]
                plt.plot(timerange,areaprice,label=a, color=colours[count], lw=2.0)
                count += 1
                if showTitle:
                    plt.title("Area price")
        if showTitle:
            plt.title('Area Price')
        plt.legend(loc='upper right',fontsize='medium')
        plt.ylabel('Price [EUR/MWh]')
        plt.show()
        return

    def plotWelfare(self, model, areas=None,timeMaxMin=None,relative=False,
                      showTitle=False,variable="energy",gentypes=None, stage=2):
        '''
        Plot welfare

        Parameters
        ----------
        areas : list of sting
            Which areas to include, default=None means include all
        timeMaxMin : list of two integers
            Time range, [min,max]
        relative : boolean
            Whether to plot absolute (false) or relative (true) values
        variable : string ("energy","capacity","spilled")
            Which variable to plot (default is energy production)
        gentypes : list
            List of generator types to include. None gives all.
        '''

        s = stage
        if areas is None:
            areas=[]
            for c in model.LOAD:
                areas.append(model.nodeArea[model.demNode[c]])
        if timeMaxMin is None:
            timeMaxMin = list(model.TIME)
        if gentypes is None:
            gentypes = list(model.GENTYPE)

        welfare={}
        if variable=="all":
            print("Getting welfare from all nodes...")
            types = ['PS','CS','CR']
            for typ in types:
                welfare[typ] = {}
                for c in model.LOAD:
                    welfare[typ][c] = sum([self.computeAreaWelfare(model,c,t,s)[typ]
                                            *model.samplefactor[t] for t in model.TIME])/10**9
            title = "Total welfare"
        else:
            print("Variable not valid")
            return

        if relative:
            total={}
            for c in model.LOAD:
                total[c]=0
                for typ in types:
                    total[c] += sum([self.computeAreaWelfare(model,c,t,s)[typ]
                                    *model.samplefactor[t] for t in model.TIME])

        plt.figure()
        ax = plt.subplot(111)
        width = 0.8
        previous = [0]*len(areas)
        numCurves = len(types)+1
        colours = cm.hsv(np.linspace(0, 1, numCurves))
        #colours = cm.viridis(np.linspace(0, 1, numCurves))
        #colours = cm.Set3(np.linspace(0, 1, numCurves))
        #colours = cm.Grays(np.linspace(0, 1, numCurves))
        #colours = cm.Dark2(np.linspace(0, 1, numCurves))
        count=0
        ind = range(len(model.LOAD))
        for typ in types:
            A=[]
            for c in model.LOAD:
                if relative:
                    if total[c]>0:
                        welfare[typ][c] = welfare[typ][c]/total[c]
                        A.append(welfare[typ][c])
                else:
                    A.append(welfare[typ][c])
            plt.bar(ind,A, width,label=typ,bottom=previous,color=colours[count])
            previous = [previous[i]+A[i] for i in range(len(A))]
            count = count+1

        handles, labels = ax.get_legend_handles_labels()
        handles.reverse()
        labels.reverse()
        plt.legend(handles, labels, loc='upper right', fontsize='medium')
#        plt.legend(handles, labels, loc='best',
#                   bbox_to_anchor=(1.05,1), borderaxespad=0.0)
        plt.xticks(np.arange(len(areas))+width/2., tuple(areas) )
        plt.ylabel('Annual welfare [bn€]')
        if showTitle:
            plt.title(title)
        plt.show()

        return

    def plotInvestments(self,filename, variable, unit='capacity'):
        '''
        Plot investment bar plots

        filename: string
            excel-file generated by 'saveDeterministicResults'
        variable: string
            dcbranch, acbranch, node, generator
        unit: string
            capacity, monetary
        '''
        figsize=(8,6)
        width = 0.8
        if variable=='dcbranch':
            df_res = self.loadResults(filename, sheet='branches')
            df_res = df_res[df_res['type']=='dcdirect']
            df_res = df_res.groupby(['fArea','tArea']).sum()
            numCurves = len(df_res['newCapacity'][df_res['newCapacity']>0])+1
            colours = cm.hsv(np.linspace(0, 1, numCurves))
            if not df_res['newCapacity'][df_res['newCapacity']>0].empty:
                ax1 = df_res['newCapacity'][df_res['newCapacity']>0].plot(kind='bar',
                            title='new capacity', figsize=figsize, color=colours, width=width)
                ax1.set_xlabel('Interconnector', fontsize=12)
                ax1.set_xticklabels(ax1.xaxis.get_majorticklabels(), rotation=0)
                ax1.set_ylabel('New capacity [MW]', fontsize=12)
                ax2 = df_res[['cost_withOM','congestion_rent']][df_res['newCapacity']>0].divide(10**9).plot(
                            kind='bar', title='costs and benefits', figsize=figsize,
                            legend=True, fontsize=11,color=colours, width=width)
                ax2.set_xlabel('Interconnector', fontsize=12)
                ax2.set_xticklabels(ax2.xaxis.get_majorticklabels(), rotation=0)
                ax2.set_ylabel('Net present value [bn€]', fontsize=12)
        elif variable=='acbranch':
            df_res = self.loadResults(filename, sheet='branches')
            df_res = df_res[df_res['type']=='ac']
            df_res = df_res.groupby(['fArea','tArea']).sum()
            plt.figure()
            df_res['newCapacity'][df_res['newCapacity']>0].plot.bar()
        elif variable=='node':
            df_res = self.loadResults(filename, sheet='nodes')
        elif variable=='generator':
            df_res = self.loadResults(filename, sheet='generation')
            df_res = df_res.groupby(['area'])
            plt.figure()
            df_res['newCapacity'].plot.bar(stacked=True)
        else:
            print('A variable has to be chosen: dcbranch, acbranch, node, generator')

        return

    def plotBranchData(self, model,stage=2):
        '''
        Plot branch data
        '''
        s = stage
        df_branch = pd.DataFrame()
        i=0
        for b in model.BRANCH:
            for t in model.TIME:
                i+=1
                df_branch.loc[i,'branch'] = b
                df_branch.loc[i,'fArea'] = model.nodeArea[model.branchNodeFrom[b]]
                df_branch.loc[i,'tArea'] = model.nodeArea[model.branchNodeTo[b]]
                df_branch.loc[i,'type'] = model.branchType[b]
                df_branch.loc[i,'hour'] = t
                df_branch.loc[i,'weight'] = model.samplefactor[t]
                df_branch.loc[i,'flow12'] = model.branchFlow12[b,t,s].value
                df_branch.loc[i,'flow21'] = model.branchFlow21[b,t,s].value
                df_branch.loc[i,'utilization'] = (model.branchFlow12[b,t,s].value+model.branchFlow21[b,t,s].value)/(
                    model.branchExistingCapacity[b]+model.branchExistingCapacity2[b]
                    +sum(model.branchNewCapacity[b,h+1].value for h in range(s)))

        df_branch.groupby('branch')['flow12']

        return


def computeSTOcosts(grid_data,dict_data,generation=None,include_om=True):
    """
    Compute costs as in objective function of the optimisation
    This function is used to analyse optimisation results.

    PARAMETERS
    ==========
    grid_data : powergama.grid_data
        grid object
    dict_data : dict
        dictionary holding the optimisation input data (as dictionary)
    generation : list of dataframes, one per stage
        generator operational costs, dataframe with columns ['gen','time','value']
    """

    print("Warning! powergim.computeSTOcosts is not fully implemented yet.")

    # BRANCHES
    distances = grid_data.branchDistances()
    f_rate = dict_data['powergim']['financeInterestrate'][None]
    f_years = dict_data['powergim']['financeYears'][None]
    stage2delta = dict_data['powergim']['stage2TimeDelta'][None]
    cost={(1,'invest'):0.0,
          (1,'op'):0.0,
          (2,'invest'):0.0,
          (2,'op'):0.0}
    for count in range(grid_data.branch.shape[0]):
        br_indx = grid_data.branch.index[count]
        b = grid_data.branch.loc[br_indx]
        br_dist = distances[count]
        br_type=b['type']
        br_cap = b['capacity']
        br_num = math.ceil(br_cap/
            dict_data['powergim']['branchtypeMaxCapacity'][br_type])
        ar = 0
        salvagefactor=0
        discount_t0=1
        # default is that branch has not been expanded (b_stage=0)
        b_stage=0
        if b['expand']==1:
            b_stage=1
            ar = annuityfactor(f_rate,f_years)
        elif b['expand2']==1:
            b_stage=2
            ar = (annuityfactor(f_rate,f_years)
                  -annuityfactor(f_rate, stage2delta))
            salvagefactor = (stage2delta/f_years)*(
                            1/((1+f_rate)**(f_years-stage2delta)))
            discount_t0 = (1/((1+f_rate)**stage2delta))

        b_cost = 0
        b_cost += (dict_data['powergim']['branchtypeCost'][(br_type,'B')]
                    *br_num)
        b_cost += (dict_data['powergim']['branchtypeCost'][(br_type,'Bd')]
                    *br_dist*br_num)
        b_cost += (dict_data['powergim']['branchtypeCost'][(br_type,'Bdp')]
                    *br_dist*br_cap)

        #endpoints offshore (N=1) or onshore (N=0)
        N1 = dict_data['powergim']['branchOffshoreFrom'][br_indx]
        N2 = dict_data['powergim']['branchOffshoreTo'][br_indx]
        #print(br_indx,N1,N2,br_num,br_cap,br_type,b['expand'])
        for N in [N1,N2]:
            b_cost += N*(dict_data['powergim']['branchtypeCost']
                                    [(br_type,'CS')]*br_num
                        +dict_data['powergim']['branchtypeCost']
                                    [(br_type,'CSp')]*br_cap)
            b_cost += (1-N)*(dict_data['powergim']['branchtypeCost']
                                    [(br_type,'CL')]*br_num
                        +dict_data['powergim']['branchtypeCost']
                                    [(br_type,'CLp')]*br_cap)

        b_cost = dict_data['powergim']['branchCostScale'][br_indx]*b_cost

        # discount  back to t=0
        b_cost = b_cost*discount_t0
        # O&M
        omcost = 0
        if include_om:
            omcost =dict_data['powergim']['omRate'][None] * ar * b_cost

        # subtract salvage value and add om cost
        b_cost = b_cost*(1-salvagefactor) + omcost

        if b_stage>0:
            cost[(b_stage,'invest')] += b_cost

    # NODES (not yet)

    # GENERATION (op cost)
    if not generation is None:
        df_gen1=generation[0]
        df_gen2=generation[1]
        ar1 = annuityfactor(f_rate,stage2delta)
        ar2 = (annuityfactor(f_rate,f_years)
                  -annuityfactor(f_rate, stage2delta))
        samplefactor = pd.Series(dict_data['powergim']['samplefactor'])
        for count in range(grid_data.generator.shape[0]):
            g_indx = grid_data.generator.index[count]
            gen1 = sum(df_gen1[(df_gen1['gen']==g_indx)
                                & (df_gen1['time']==t)]['value'].iloc[0]
                        *dict_data['powergim']['genCostAvg'][g_indx]
                        *dict_data['powergim']['genCostProfile'][(g_indx,t)]
                        *samplefactor[t]
                         for t in grid_data.timerange)
            gen2 = sum(df_gen2[(df_gen2['gen']==g_indx)
                                & (df_gen2['time']==t)]['value'].iloc[0]
                        *dict_data['powergim']['genCostAvg'][g_indx]
                        *dict_data['powergim']['genCostProfile'][(g_indx,t)]
                        *samplefactor[t]
                         for t in grid_data.timerange)
            #if model.curtailmentCost.value >0:
            #    expr += sum(curt[g,t].value*model.curtailmentCost for t in model.TIME)
            #expr += sum(gen[g,t].value*
            #                    model.genTypeEmissionRate[model.genType[g]]*model.CO2price
            #                    for t in model.TIME)
            # lifetime cost

            #samplefactor = dict_data['powergim']['samplefactor'][None]
            cost[(2,'op')] += (gen1*ar1 + gen2*ar2)
    return cost
