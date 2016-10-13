# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 13:21:21 2016

@author: hsven
"""


import pyomo.environ as pyo
import pandas as pd

class SipModel():
    '''
    Power Grid Investment Module - stochastic investment problem
    '''
    
    def __init__(self, maxNewBranchCap,maxNewBranchNum,M_const = 1000):
        """Create Abstract Pyomo model for PowerGIM"""
        self.abstractmodel = self._createAbstractModel(maxNewBranchCap,
                                                       maxNewBranchNum)
        self.M_const = M_const
        
        
    
    def _createAbstractModel(self,maxNewBranchCap,maxNewBranchNum):    
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
        def branchNewCapacity_bounds(model,j):
            return (0,maxNewBranchCap)
        model.branchNewCapacity = pyo.Var(model.BRANCH, 
                                          within = pyo.NonNegativeReals,
                                          bounds = branchNewCapacity_bounds)
        # investment: new dcbranch number of cables [dcFixInvest]
        def branchNewCables_bounds(model,j):
            return (0,maxNewBranchNum)                                  
        model.branchNewCables = pyo.Var(model.BRANCH, 
                                        within = pyo.NonNegativeIntegers,
                                        bounds = branchNewCables_bounds)
        # investment: new nodes
        model.newNodes = pyo.Var(model.NODE, within = pyo.Binary)
        
        # branch power flow (ac and dc):
        def branchFlow_bounds(model,j,t):
            ub = (model.branchExistingCapacity[j]
                    +branchNewCapacity_bounds(model,j)[1])
            return (0,ub)
        model.branchFlow12 = pyo.Var(model.BRANCH, model.TIME, 
                                     within = pyo.NonNegativeReals,
                                     bounds = branchFlow_bounds)
        model.branchFlow21 = pyo.Var(model.BRANCH, model.TIME, 
                                     within = pyo.NonNegativeReals,
                                     bounds = branchFlow_bounds)
        
        # generator output
        def generation_bounds(model,j,t):
            # could use available capacity instead of capacity, and get rid
            # of generation limit constraint
            ub = model.genCapacity[j]
            return (0,ub)
        model.generation = pyo.Var(model.GEN, model.TIME, 
                                   within = pyo.NonNegativeReals,
                                   bounds = generation_bounds)
        # load shedding (cf gen)
        #model.loadShed = pyo.Var(model.NODE, model.TIME, 
        #                         domain = pyo.NonNegativeReals)       
        model.curtailment = pyo.Var(model.GEN, model.TIME, 
                                    domain = pyo.NonNegativeReals,
                                    bounds = generation_bounds)
        
        
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
            #loadshedding=0 by powerbalance constraint
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


    def createModelData(self,grid_data,datafile):
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
        st_model.StageVariables[first_stage].add('branchNewCables')
        st_model.StageVariables[first_stage].add('branchNewCapacity')
        st_model.StageVariables[first_stage].add('newNodes')
    
        # Second Stage
        st_model.StageCost[second_stage] = 'secondStageCost'
        st_model.StageVariables[second_stage].add('generation')
        st_model.StageVariables[second_stage].add('curtailment')
        st_model.StageVariables[second_stage].add('branchFlow12')
        st_model.StageVariables[second_stage].add('branchFlow21')
            
        st_model.ScenarioBasedData=False
    
        # Alternative, using networkx to create scenario tree:
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
        
        