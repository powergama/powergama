# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 13:21:21 2016

@author: hsven
"""


import pyomo.environ as pyo

class SipModel():
    '''
    Power Grid Investment Module - stochastic investment problem
    '''
    
    def __init__(self, M_const = 1000):
        """Create Abstract Pyomo model for PowerGIM"""
        self.abstractmodel = self._createAbstractModel()
        self.M_const = M_const
        
        
    
    def _createAbstractModel(self):    
        model = pyo.AbstractModel()
        model.name = 'PowerGIM abstract model'
        
        # SETS ###############################################################
        
        model.NODE = pyo.Set()
        model.GEN = pyo.Set()
        model.BRANCH = pyo.Set()
        model.LOAD = pyo.Set()
        model.TIME = pyo.Set()
        
        model.BRANCHTYPE = pyo.Set()
        model.BRANCHCOSTITEM = pyo.Set(initialize=['B','Bd', 'Bdp', 
                                                   'CLp','CL','CSp','CS'])        
        model.NODETYPE = pyo.Set()
        model.NODECOSTITEM = pyo.Set(initialize=['L','S'])
        model.LINEAR = pyo.Set(initialize=['fix','slope'])
        
        # PARAMETERS #########################################################
        
        model.financeInterestrate = pyo.Param(within=pyo.Reals)
        model.financeYears = pyo.Param(within=pyo.Reals)
        model.omRate = pyo.Param(within=pyo.Reals)
        model.curtailmentCost = pyo.Param(within=pyo.NonNegativeReals)
        
        #investment costs and limits:        
        model.branchtypeMaxCapacity = pyo.Param(model.BRANCHTYPE,
                                                within=pyo.Reals)
        model.branchtypeCost = pyo.Param(model.BRANCHTYPE, 
                                         model.BRANCHCOSTITEM,
                                         within=pyo.Reals)
        model.branchLossfactor = pyo.Param(model.BRANCHTYPE,model.LINEAR,
                                     within=pyo.Reals)
        model.nodetypeCost = pyo.Param(model.NODETYPE, model.NODECOSTITEM,
                                       within=pyo.Reals)
        model.nodeCostScale = pyo.Param(model.NODE,within=pyo.Reals)
        model.branchCostScale = pyo.Param(model.BRANCH,within=pyo.Reals)
        
        #branches:
        model.branchExistingCapacity = pyo.Param(model.BRANCH, 
                                                 within=pyo.NonNegativeReals)
        model.branchExpand = pyo.Param(model.BRANCH,within=pyo.Binary)         
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
        model.genCapacityProfile = pyo.Param(model.GEN,model.TIME,
                                          within=pyo.Reals)
        model.genPAvg = pyo.Param(model.GEN,within=pyo.Reals)
        
        #helpers:
        model.genNode = pyo.Param(model.GEN,within=model.NODE)
        model.demNode = pyo.Param(model.LOAD,within=model.NODE)
        model.branchNodeFrom = pyo.Param(model.BRANCH,within=model.NODE)
        model.branchNodeTo = pyo.Param(model.BRANCH,within=model.NODE)
        
        #consumers
        # the split int an average value, and a profile is to make it easier
        # to generate scenarios (can keep profile, but adjust demandAvg)
        model.demandAvg = pyo.Param(model.LOAD,within=pyo.Reals)
        model.demandProfile = pyo.Param(model.LOAD,model.TIME,
                                        within=pyo.Reals)
    
        # VARIABLES ##########################################################
    
        # investment: new dcbranch capacity [dcVarInvest]
        model.branchNewCapacity = pyo.Var(model.BRANCH, 
                                          within = pyo.NonNegativeReals)
        # investment: new dcbranch number of cables [dcFixInvest]
        model.branchNewCables = pyo.Var(model.BRANCH, 
                                        within = pyo.NonNegativeIntegers)
        # investment: new nodes
        model.newNodes = pyo.Var(model.NODE, within = pyo.Binary)
        # generator output
        model.generation = pyo.Var(model.GEN, model.TIME, 
                                   within = pyo.NonNegativeReals)
        # branch power flow (ac and dc):
        model.branchFlow12 = pyo.Var(model.BRANCH, model.TIME, 
                                     within = pyo.NonNegativeReals)
        model.branchFlow21 = pyo.Var(model.BRANCH, model.TIME, 
                                     within = pyo.NonNegativeReals)
        
        # load shedding (cf gen)
        #model.loadShed = pyo.Var(model.NODE, model.TIME, 
        #                         domain = pyo.NonNegativeReals)
        model.curtailment = pyo.Var(model.GEN, model.TIME, 
                                    domain = pyo.NonNegativeReals)
        
        
        # CONSTRAINTS ########################################################
        # Power flow limitations
        
        def maxflow12_rule(model, j, t):
            expr = (model.branchFlow12[j,t] <= model.branchExistingCapacity[j] 
                        + model.branchNewCapacity[j] )
            return expr
        model.cMaxFlow12 = pyo.Constraint(model.BRANCH, model.TIME, 
                                         rule=maxflow12_rule)
        def maxflow21_rule(model, j, t):
            expr = (model.branchFlow21[j,t] <= model.branchExistingCapacity[j] 
                        + model.branchNewCapacity[j] )
            return expr
        model.cMaxFlow21 = pyo.Constraint(model.BRANCH, model.TIME, 
                                         rule=maxflow21_rule)
                                         
        # No new branch capacity without new cables
        def maxNewCap_rule(model,j):
            typ = model.branchType[j]
            expr = (model.branchNewCapacity[j] 
                    <= model.branchtypeMaxCapacity[typ]
                        *model.branchNewCables[j])
            return expr
        model.cmaxNewCapacity = pyo.Constraint(model.BRANCH,
                                               rule=maxNewCap_rule)
        def newBranches_rule(model,j):
            if model.branchExpand[j]==0:
                expr = model.branchNewCables[j]==0
                return expr
            else:
                return pyo.Constraint.Skip
        model.cNewBranches = pyo.Constraint(model.BRANCH,
                                            rule=newBranches_rule)
                                            
        # A node required at each branch endpoint
        def newNodes_rule(model,n):
            expr = 0
            for j in model.BRANCH:
                if model.branchNodeFrom[j]==n or model.branchNodeTo[j]==n:
                    expr += model.branchNewCables[j]
            #for j in model.branchNewCables:
            #    expr += model.branchNewCables[j]
            expr = expr <= self.M_const*(model.newNodes[n]
                                         +model.nodeExistingNumber[n])
            return expr
        model.cNewNodes = pyo.Constraint(model.NODE,rule=newNodes_rule)
        
        # Generator output limitations
        def maxPgen_rule(model,g,t):
            expr = model.generation[g,t] <= (model.genCapacity[g]
                *model.genCapacityProfile[g,t])
            
            return expr
        model.cMaxPgen = pyo.Constraint(model.GEN,model.TIME,
                                        rule=maxPgen_rule)
        
        
        # Generator maximum average output (energy sum) 
        #(e.g. for hydro with storage)
        def maxPavg_rule(model,g):
            if model.genPAvg[g]>0:
                expr = (sum(model.generation[g,t] for t in model.TIME) 
                            <= model.genPAvg[g]*len(model.TIME))
            else:
                expr = pyo.Constraint.Skip
            return expr
        model.cMaxPavg = pyo.Constraint(model.GEN,rule=maxPavg_rule)


        def curtailment_rule(model,g,t):
            if model.genCostAvg[g] == 0:
                expr =  (model.curtailment[g,t] 
                    == model.genCapacity[g]*model.genCapacityProfile[g,t] 
                        - model.generation[g,t])
                return expr
            else:
                return pyo.Constraint.Skip
        model.genCurtailment = pyo.Constraint(model.GEN, model.TIME, 
                                              rule=curtailment_rule)

       
        # Power balance in nodes : gen+demand+flow into node=0
        def powerbalance_rule(model,n,t):
            expr = 0

            # flow of power into node (subtrating losses)
            for j in model.BRANCH:
                if model.branchNodeFrom[j]==n:
                    # branch out of node
                    typ = model.branchType[j]
                    dist = model.branchDistance[j]
                    expr += -model.branchFlow12[j,t]
                    expr += model.branchFlow21[j,t] * (1-(
                                model.branchLossfactor[typ,'fix']
                                +model.branchLossfactor[typ,'slope']*dist))
                if model.branchNodeTo[j]==n:
                    # branch into node
                    typ = model.branchType[j]
                    dist = model.branchDistance[j]
                    expr += model.branchFlow12[j,t]
                    expr += -model.branchFlow21[j,t] * (1-(
                                model.branchLossfactor[typ,'fix']
                                +model.branchLossfactor[typ,'slope']*dist))

            # generated power 
            for g in model.GEN:
                if model.genNode[g]==n:
                    expr += model.generation[g,t]

            # consumed power
            for c in model.LOAD:
                if model.demNode[c]==n:
                    expr += -model.demandAvg[c]*model.demandProfile[c,t]
            
            
            
            expr = (expr == 0)
            return expr
        model.cPowerbalance = pyo.Constraint(model.NODE,model.TIME,
                                             rule=powerbalance_rule)
        
        # OBJECTIVE ##############################################################
        def annuityfactor(rate,years):
            annuity = (1-1/((1+rate)**years))/rate
            return annuity
        
        def firstStageCost_rule(model):
            """Investment cost, including lifetime O&M costs"""
            expr = 0

            # add branch costs:
            for b in model.BRANCH:
                b_cost = 0
                typ = model.branchType[b]
                b_cost += (model.branchtypeCost[typ,'B']
                            *model.branchNewCables[b])
                b_cost += (model.branchtypeCost[typ,'Bd']
                            *model.branchDistance[b]
                            *model.branchNewCables[b])
                b_cost += (model.branchtypeCost[typ,'Bdp']
                        *model.branchDistance[b]*model.branchNewCapacity[b])
                
                #endpoints offshore (N=1) or onshore (N=0) ?
                N1 = model.branchOffshoreFrom[b]
                N2 = model.branchOffshoreTo[b]
                for N in [N1,N2]:
                    b_cost += N*(model.branchtypeCost[typ,'CS']
                                *model.branchNewCables[b]
                            +model.branchtypeCost[typ,'CSp']
                            *model.branchNewCapacity[b])            
                    b_cost += (1-N)*(model.branchtypeCost[typ,'CL']
                                *model.branchNewCables[b]
                            +model.branchtypeCost[typ,'CLp']
                            *model.branchNewCapacity[b])
                
                expr += model.branchCostScale[b]*b_cost
                        
            # add node costs:
            for n in model.NODE:
                n_cost = 0
                typ = model.nodeType[n]
                N = model.nodeOffshore[n]
                n_cost += N*(model.nodetypeCost[model.nodeType[n],'S']
                            *model.newNodes[n])
                n_cost += (1-N)*(model.nodetypeCost[model.nodeType[n],'L']
                            *model.newNodes[n])
                expr += model.nodeCostScale[n]*n_cost
            
            # add O&M costs:
            expr = expr*(1 + model.omRate*annuityfactor(
                            model.financeInterestrate,
                            model.financeYears)) 
            return   expr  
        model.firstStageCost = pyo.Expression(rule=firstStageCost_rule)
    
        def secondStageCost_rule(model):
            """Operational costs: cost of gen, load shed and curtailment"""
            expr = sum(model.generation[i,t]
                        *model.genCostAvg[i]*model.genCostProfile[i,t] 
                        for i in model.GEN for t in model.TIME)
            #expr += sum(model.loadShed[i,t]*model.shedCost[i] 
            #            for i in model.NODE for t in model.TIME)
            expr += sum(model.curtailment[i,t]*model.curtailmentCost 
                        for i in model.GEN for t in model.TIME)
            # lifetime cost
            samplefactor = 8760/len(model.TIME)
            expr = samplefactor*expr*annuityfactor(model.financeInterestrate,
                                                   model.financeYears)
            return expr
        model.secondStageCost = pyo.Expression(rule=secondStageCost_rule)
    
        def total_Cost_Objective_rule(model):
            return model.firstStageCost + model.secondStageCost
        model.OBJ = pyo.Objective(rule=total_Cost_Objective_rule, 
                                  sense=pyo.minimize)
        
    
        return model


        
    def createConcreteModel(self,dict_data):
        """Create Concrete Pyomo model for PowerGIM"""

        concretemodel = self.abstractmodel.create_instance(data=dict_data,
                               name="PowerGIM Model",
                               namespace='powergim')
        return concretemodel


    def _offshoreBranch(self,grid_data):
        '''find out whether branch endpoints are offshore or onshore
        
        Returns 1 for offshore and 1 for onsore from/to endpoints
        '''
        d={'from':[],'to':[]}
        
        d['from'] = [grid_data.node[grid_data.node['id']==n]['offshore']
                    .tolist()[0] for n in grid_data.branch['node_from']]
        d['to'] = [grid_data.node[grid_data.node['id']==n]['offshore']
                    .tolist()[0] for n in grid_data.branch['node_to']]
        return d


    def createModelData(self,grid_data,datafile):
        '''Create model data in dictionary format

        Parameters
        ----------
        grid_data : powergama.GridData object
            contains grid model
        datafile : name of XML file containing additional parameters
        
        Returns
        '''
        
       
       
        print('TODO: Compute distances')        
        #grid_data.branch['distance']=99
        branch_distances = grid_data.branchDistances()
        
        #data = pyo.DataPortal(model=self.abstractmodel)
        #data.load(filename=datafile)
        
        di = {}
        #Sets:
        di['NODE'] = {None: grid_data.node['id'].tolist()}
        di['BRANCH'] = {None: [i for i in range(grid_data.branch.shape[0]) ]}
        di['GEN'] = {None: [i for i in range(grid_data.generator.shape[0]) ]}
        di['LOAD'] = {None: [i for i in range(grid_data.consumer.shape[0]) ]}
        di['TIME'] = {None: grid_data.timerange}        
        
        #Parameters:
        di['nodeOffshore'] = {}
        di['nodeType'] = {}
        di['nodeExistingNumber'] = {}
        di['nodeCostScale']={}
        for k,row in grid_data.node.iterrows():
            n=grid_data.node['id'][k]
            di['nodeOffshore'][n] = row['offshore']
            di['nodeType'][n] = row['type']
            di['nodeExistingNumber'][n] = row['existing']
            di['nodeCostScale'][n] = row['cost_scaling']
            
        di['branchExistingCapacity'] = {}
        di['branchExpand'] = {}
        di['branchDistance'] = {}
        di['branchType'] = {}
        di['branchCostScale'] = {}
        di['branchOffshoreFrom'] = {}
        di['branchOffshoreTo'] = {}
        di['branchNodeFrom'] = {}
        di['branchNodeTo'] = {}
        offsh = self._offshoreBranch(grid_data)
        for k,row in grid_data.branch.iterrows():
            di['branchExistingCapacity'][k] = row['capacity']
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
        di['genCapacityProfile']={}
        di['genNode']={}
        di['genCostAvg'] = {}
        di['genCostProfile'] = {}
        di['genPAvg'] = {}
        for k,row in grid_data.generator.iterrows():
            di['genCapacity'][k] = row['pmax']
            di['genNode'][k] = row['node']
            di['genCostAvg'][k] = row['fuelcost']
            di['genPAvg'][k] = row['pavg']
            ref = row['fuelcost_ref']
            ref2 = row['inflow_ref']
            for i,t in enumerate(grid_data.timerange):
                di['genCostProfile'][(k,t)] = grid_data.profiles[ref][i]
                di['genCapacityProfile'][(k,t)] = (grid_data.profiles[ref2][i]
                            * row['inflow_fac'])
           
        di['demandAvg'] = {}
        di['demandProfile'] ={}
        di['demNode'] = {}
        for k,row in grid_data.consumer.iterrows():
            di['demNode'][k] = row['node']
            di['demandAvg'][k] = row['demand_avg']
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
            di['branchLossfactor'][(name,'slope')] = float(
                                                        i.attrib['lossSlope'])
        for i in root.findall('./parameters'):
            di['curtailmentCost'] = {None: 
                float(i.attrib['curtailmentCost'])}
            di['financeInterestrate'] = {None: 
                float(i.attrib['financeInterestrate'])}
            di['financeYears'] = {None: 
                float(i.attrib['financeYears'])}
            di['omRate'] = {None: 
                float(i.attrib['omRate'])}

        return {'powergim':di}

        
    def createStochasticProblem(self,path):
        '''create input files for solving stochastic problem
        
        Generates Referencedata.dat        
        
        Parameters
        ----------
        path : where to put generated files
        '''
        
        #TODO: Export data dictionary to ReferenceModel.dat
        '''
        Idea
        Export data to
        
        loop through dict_data
            type(key)==pyo.base.sets.SimpleSet / SimpleParam
            loop through elements            
                print key value pairs
        '''
        
        raise Exception('Not implemented')
        return