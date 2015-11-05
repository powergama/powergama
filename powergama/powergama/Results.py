# -*- coding: utf-8 -*-
'''
Module containing the PowerGAMA Results class
'''

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import numpy as np
from mpl_toolkits.basemap import Basemap
import math
import powergama.database as db
import csv

class Results(object):
    '''
    Class for storing and analysing/presenting results from PowerGAMA
    '''    
    
    
    def __init__(self,grid,databasefile,replace=True):
        '''
        Create a PowerGAMA Results object
        
        Parameters
        ----------
        grid
            GridData - object reference
            databasefile - name of sqlite3 file for storage of results
            replace - replace existing sqlite file (default=true). 
                        replace=false is useful to analyse previously
                        generated results
            
            
        '''
        self.grid = grid
        self.timerange = grid.timerange
        self.storage_idx_generators = grid.getIdxGeneratorsWithStorage()
        self.pump_idx_generators = grid.getIdxGeneratorsWithPumping()
        self.flex_idx_consumers = grid.getIdxConsumersWithFlexibleLoad()
        self.idxConstrainedBranchCapacity \
            = grid.getIdxBranchesWithFlowConstraints()
        
        self.db = db.Database(databasefile)
        if replace:
            self.db.createTables(grid)
        else:
            # check that the length of the specified timerange matches the 
            # database
            timerange_db = self.db.getTimerange()
            if timerange_db != list(self.timerange):
                print("Database time range = [%d,%d]\n" %
                      (timerange_db[0],timerange_db[-1]))
                raise Exception('Database time range mismatch')

        '''
        self.objectiveFunctionValue=[]    
        self.generatorOutput=[]
        self.branchFlow=[]
        self.nodeAngle=[]
        self.sensitivityBranchCapacity=[]
        self.sensitivityDcBranchCapacity=[]
        self.sensitivityNodePower=[]
        self.storage=[]
        self.marginalprice=[]
        self.inflowSpilled=[]
        self.loadshed=[]
        '''    
        
    def addResultsFromTimestep(self,timestep,objective_function,
                               generator_power,
                               generator_pumped,
                               branch_power,dcbranch_power,node_angle,
                               sensitivity_branch_capacity,
                               sensitivity_dcbranch_capacity,
                               sensitivity_node_power,
                               storage,
                               inflow_spilled,
                               loadshed_power,
                               marginalprice,
                               flexload_power,
                               flexload_storage,
                               flexload_storagevalue):
        '''Store results from optimal power flow for a new timestep'''
        
        # Store results in sqlite database on disk (to avoid memory problems)
        self.db.appendResults(
            timestep = timestep,
            objective_function = objective_function,
            generator_power = generator_power,
            generator_pumped = generator_pumped,
            branch_flow = branch_power,
            dcbranch_flow = dcbranch_power,
            node_angle = node_angle,
            sensitivity_branch_capacity = sensitivity_branch_capacity,
            sensitivity_dcbranch_capacity = sensitivity_dcbranch_capacity,
            sensitivity_node_power = sensitivity_node_power,
            storage = storage,
            inflow_spilled = inflow_spilled,
            loadshed_power = loadshed_power,
            marginalprice = marginalprice,
            flexload_power = flexload_power,
            flexload_storage = flexload_storage,
            flexload_storagevalue = flexload_storagevalue,
            idx_storagegen = self.storage_idx_generators,
            idx_branchsens = self.idxConstrainedBranchCapacity,
            idx_pumpgen = self.pump_idx_generators,
            idx_flexload = self.flex_idx_consumers)
       
        '''
        self.objectiveFunctionValue.append(objective_function)
        self.generatorOutput.append(generator_power)
        self.branchFlow.append(branch_power)
        self.nodeAngle.append(node_angle)
        self.sensitivityBranchCapacity.append(sensitivity_branch_capacity)
        self.sensitivityDcBranchCapacity.append(sensitivity_dcbranch_capacity)
        self.sensitivityNodePower.append(sensitivity_node_power)
        self.storage.append(storage)
        self.inflowSpilled.append(inflow_spilled)
        self.loadshed.append(loadshed_power)
        self.marginalprice.append(marginalprice)
        '''
        # self.storageGeneratorsIdx.append(idx_generatorsWithStorage)

    def getAverageBranchFlows(self,timeMaxMin=None):
        '''
        Average flow on branches over a given time period
        
        Parameters
        ----------
        timeMaxMin (list) (default = None)
            [min, max] - lower and upper time interval
            
        Returns
        =======
        List with values for each branch:
        [flow from 1 to 2, flow from 2 to 1, average absolute flow]
        '''
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]

        #branchflow = self.db.getResultBranchFlowAll(timeMaxMin)
        avgflow = self.db.getResultBranchFlowsMean(timeMaxMin)
        #np.mean(branchflow,axis=1)
        return avgflow


    def getNodalPrices(self,node,timeMaxMin=None):
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]

        prices = self.db.getResultNodalPrice(node,timeMaxMin)
        # use asarray to convert None to nan
        prices = np.asarray(prices,dtype=float)
        return prices


    def getAverageNodalPrices(self,timeMaxMin=None):
        '''
        Average nodal price over a given time period
        
        Parameters
        ----------
        timeMaxMin (list) (default = None)
            [min, max] - lower and upper time interval
            
        Returns
        =======
        1-dim Array of nodal prices (one per node)
        '''
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]

        avgprices = self.db.getResultNodalPricesMean(timeMaxMin)
        # use asarray to convert None to nan
        avgprices = np.asarray(avgprices,dtype=float)
        return avgprices

    def getAreaPrices(self,area,timeMaxMin=None):
        '''
        Weighted average nodal price timeseries for given area
        '''
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]

        #area_nodes = [n._i for n in self.grid.node if n.area==area]
        loads = self.grid.getConsumersPerArea()[area]
        node_weight = [0]*len(self.grid.node.name)
        for ld in loads:
            the_node = self.grid.consumer.node[ld]
            the_load = self.grid.consumer.load[ld]
            node_indx = self.grid.node.name.index(the_node)
            node_weight[node_indx] += the_load
            
        sumWght = sum(node_weight)
        node_weight = [a/sumWght for a in node_weight]

        #print("Weights:")        
        #print(node_weight)
        prices = self.db.getResultAreaPrices(node_weight,timeMaxMin)

        return prices
       
    def getAreaPricesAverage(self,areas=None,timeMaxMin=None):
        '''
        Time average of weighted average nodal price per area 
        '''
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]
        if areas is None:
            areas = self.grid.getAllAreas()

        avg_nodalprices = self.getAverageNodalPrices(timeMaxMin)
        all_loads = self.grid.getConsumersPerArea()
        avg_areaprice = {}
        
        for area in areas:
            nodes_in_area = [i for i,n in enumerate(self.grid.node.area) 
                                if n==area]
            node_weight = [0]*len(self.grid.node.name)
            if area in all_loads:
                loads = all_loads[area]
                for ld in loads:
                    the_node = self.grid.consumer.node[ld]
                    the_load = self.grid.consumer.load[ld]
                    node_indx = self.grid.node.name.index(the_node)
                    node_weight[node_indx] += the_load                
                sumWght = sum(node_weight)
                node_weight = [a/sumWght for a in node_weight]                
    
                prices = [node_weight[i]*avg_nodalprices[i] 
                            for i in nodes_in_area]
            else:
                #flat weight if there are no loads in area
                prices = [avg_nodalprices[i]  for i in nodes_in_area]
            avg_areaprice[area] = sum(prices)

        return avg_areaprice


    def getLoadheddingInArea(self,area,timeMaxMin=None):
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]

        loadshed = self.db.getResultLoadheddingInArea(area,timeMaxMin)
        # use asarray to convert None to nan
        loadshed = np.asarray(loadshed,dtype=float)
        return loadshed

    def getLoadsheddingPerNode(self,timeMaxMin=None):
        '''get loadshedding sum per node'''
        timeMaxMin = [self.timerange[0],self.timerange[-1]+1]

        loadshed_per_node = self.db.getResultLoadheddingSum(timeMaxMin)
        return loadshed_per_node
        
    def getLoadheddingSums(self,timeMaxMin=None):
        '''get loadshedding sum per area'''
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]

        loadshed_per_node = self.db.getResultLoadheddingSum(timeMaxMin)
        areas = self.grid.node.area
        allareas = self.grid.getAllAreas()
        loadshed_sum = dict()
        for a in allareas:
            loadshed_sum[a] = sum([loadshed_per_node[i] 
                for i in range(len(areas)) if areas[i]==a])
            
        #loadshed_sum = np.asarray(loadshed_sum,dtype=float)
        return loadshed_sum


    def getAverageEnergyBalance(self,timeMaxMin=None):
        '''
        Average energy balance (generation minus demand) over a time period
        
        timeMaxMin (list) (default = None)
            [min, max] - lower and upper time interval
            
        Returns
        =======
        1-dim Array of nodal prices (one per node)
        '''
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]

        branchflows = self.db.getResultBranchFlowsMean(timeMaxMin)
        if self.grid.dcbranch.numBranches() > 0:
            branchflowsDc = self.db.getResultBranchFlowsMean(timeMaxMin,
                                                             ac=False)
        br_from = self.grid.branch.node_fromIdx(self.grid.node)
        br_to = self.grid.branch.node_toIdx(self.grid.node)
        dcbr_from = self.grid.dcbranch.node_fromIdx(self.grid.node)
        dcbr_to = self.grid.dcbranch.node_toIdx(self.grid.node)
        energybalance = []
        for n in range(len(self.grid.node.name)):
            idx_from = [ i for i,x in enumerate(br_from) if x==n]
            idx_to = [ i for i,x in enumerate(br_to) if x==n]
            dc_idx_from = [ i for i,x in enumerate(dcbr_from) if x==n]
            dc_idx_to = [ i for i,x in enumerate(dcbr_to) if x==n]
            energybalance.append(
                sum([branchflows[0][i]-branchflows[1][i] for i in idx_from])
                -sum([branchflows[0][j]-branchflows[1][j] for j in idx_to])
                +sum([branchflowsDc[0][i]-branchflowsDc[1][i] for i in dc_idx_from])
                -sum([branchflowsDc[0][j]-branchflowsDc[1][j] for j in dc_idx_to]) )
            
        # use asarray to convert None to nan
        energybalance = np.asarray(energybalance,dtype=float)
        return energybalance
           

    def getAverageBranchSensitivity(self,timeMaxMin=None):
        '''
        Average branch capacity sensitivity over a given time period
        
        Parameters
        ----------
        timeMaxMin (list) (default = None)
            [min, max] - lower and upper time interval
            
        Returns
        =======
        1-dim Array of sensitivities (one per branch)
        '''
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]

        avgsense = self.db.getResultBranchSensMean(timeMaxMin)
        # use asarray to convert None to nan
        avgsense = np.asarray(avgsense,dtype=float)
        return avgsense
    
    def getAverageUtilisation(self,timeMaxMin=None):
        '''
        Average branch utilisation over a given time period

        Parameters
        ----------
        timeMaxMin (list) (default = None)
            [min, max] - lower and upper time interval
            
        Returns
        =======
        1-dim Array of branch utilisation (power flow/capacity)
        '''
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]

        cap =self.grid.branch.capacity
        avgflow = self.getAverageBranchFlows(timeMaxMin)[2]
        utilisation = [avgflow[i] / cap[i] for i in range(len(cap))] 
        utilisation = np.asarray(utilisation)
        return utilisation
            
    def getSystemCost(self,timeMaxMin=None):
        '''
        Description
        Calculates system cost for energy produced by using generator fuel cost. 
        
        Parameters
        ----------
        timeMaxMin (list) (default = None)
            [min, max] - lower and upper time interval
        
        Returns
        =======
        array of tuples of total cost of energy per area for all areas
        [(area, costs), ...]
        '''
        
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]

        systemcost = []
        # for each area
        for area in self.grid.getAllAreas():
            areacost = 0
            # for each generator
            for gen in self.db.getGridGeneratorFromArea(area):
                    # sum generator output and multiply by fuel cost
                    for power in self.db.getResultGeneratorPower(gen[0],timeMaxMin):
                        areacost +=  power * self.grid.generator.fuelcost[gen[0]]
            systemcost.append(tuple([area, areacost]))
        return systemcost
        
        
    def getSystemCostFast(self,timeMaxMin=None):
        '''
        Description
        Calculates system cost for energy produced by using generator fuel cost. 
        
        Parameters
        ----------
        timeMaxMin (list) (default = None)
            [min, max] - lower and upper time interval
        
        Returns
        =======
        array of dictionary of cost of generation sorted per area
        '''
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]

        generation_per_gen = self.db.getResultGeneratorPowerSum(timeMaxMin)
        fuelcost_per_gen = self.grid.generator.fuelcost
        areas_per_gen = [self.grid.node.area[self.grid.node.name.index(n)] 
                    for n in self.grid.generator.node]
                
        allareas = self.grid.getAllAreas()
        generationcost = dict()
        for a in allareas:
            generationcost[a] = sum([generation_per_gen[i]*fuelcost_per_gen[i] 
                for i in range(len(areas_per_gen)) if areas_per_gen[i]==a])

        return generationcost

    def getGeneratorOutputSumPerArea(self,timeMaxMin=None):
        '''
        Description
        Sums up generation per area. 
        
        Parameters
        ----------
        timeMaxMin (list) (default = None)
            [min, max] - lower and upper time interval
        
        Returns
        =======
        array of dictionary of generation sorted per area
        '''
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]

        generation_per_gen = self.db.getResultGeneratorPowerSum(timeMaxMin)
        areas_per_gen = [self.grid.node.area[self.grid.node.name.index(n)] 
                    for n in self.grid.generator.node]
                
        allareas = self.grid.getAllAreas()
        generation = dict()
        for a in allareas:
            generation[a] = sum([generation_per_gen[i] 
                for i in range(len(areas_per_gen)) if areas_per_gen[i]==a])

        return generation

    def getGeneratorSpilledSums(self,timeMaxMin=None):
        '''Get sum of spilled inflow for all generators
        
        Parameters
        ----------
        timeMaxMin (list) (default = None)
            [min, max] - lower and upper time interval
        '''
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]
        v = self.db.getResultGeneratorSpilledSums(timeMaxMin)
        return v
        
        
    def getGeneratorSpilled(self,generatorindx,timeMaxMin=None):
        '''Get spilled inflow time series for given generator
        
        Parameters
        ----------
        generatorindx (int)
            index ofgenerator
        timeMaxMin (list) (default = None)
            [min, max] - lower and upper time interval
        '''
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]
        v = self.db.getResultGeneratorSpilled(generatorindx,timeMaxMin)
        return v

    def getGeneratorStorageAll(self,timestep):
        '''Get stored energy for all storage generators at given time
        
        Parameters
        ----------
        timestep : int
            timestep when storage is requested
        '''
        v = self.db.getResultStorageFillingAll(timestep)
        
        return v
        
    def getGeneratorStorageValues(self,timestep):
        '''Get value of stored energy for given time
        
        Parameters
        ----------
        timestep : int
            when to compute value
            
        Returns
        -------
        list of int
            Value of stored energy for all storage generators
        
        The method uses the storage value absolute level (basecost) per 
        generator to compute total storage value
        '''
        storage_energy = self.getGeneratorStorageAll(timestep)
        storage_values = self.grid.generator.storagevalue_abs
        indx_storage_generators = self.grid.getIdxGeneratorsWithStorage()
        storval = [storage_energy[i]*storage_values[v]
                    for i,v in enumerate(indx_storage_generators)]
        return storval 
        
        
    def _node2area(self, nodeName):
        '''Returns the area of a spacified node''' 
        #Is handy when you need to access more information about the node, 
        #but only the node name is avaiable. (which is the case in the generator file)
        try:
            nodeIndex = self.grid.node.name.index(nodeName)
            return self.grid.node.area[nodeIndex]
        except:
            return
            
    def _getAreaTypeProduction(self, area, generatorType, timeMaxMin):
        '''
        Returns total production for specified area nd generator type
        '''
        
        print("Looking for generators of type " + str(generatorType) + ", in " + str(area))
        print("Number of generator to run through: " + str(self.grid.generator.numGenerators()))
        totalProduction = 0
        
        
        for genNumber in range(0, self.grid.generator.numGenerators()):
            genNode = self.grid.generator.node[genNumber]
            genType = self.grid.generator.gentype[genNumber]
            genArea = self._node2area(genNode)
            #print str(genNumber) + ", " + genName + ", " + genNode + ", " + genType + ", " + genArea
            if (genType == generatorType) and (genArea == area):
                #print "\tGenerator is of right type and area. Adding production"                
                genProd = sum(self.db.getResultGeneratorPower(genNumber, 
                                                              timeMaxMin))
                totalProduction += genProd
                #print "\tGenerator production = " + str(genProd)
        return totalProduction
        
    def getAllGeneratorProductionOBSOLETE(self, timeMaxMin=None):
        '''Returns all production [MWh] for all generators'''
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]

        totGenNumbers = self.grid.generator.numGenerators()
        totalProduction = 0
        for genNumber in range(0, totGenNumbers):
            genProd = sum(self.db.getResultGeneratorPower(genNumber, 
                                                          timeMaxMin))
            print(str(genProd))
            totalProduction += genProd
            print("Progression: " + str(genNumber+1) + " of " 
                    + str(totGenNumbers))
        return totalProduction
    
    def _productionOverview(self, areas, types, timeMaxMin, 
                           TimeUnitCorrectionFactor):
        '''
        Returns a matrix with sum of generator production per area and type
        
        This function is manly used as the calculation part of the 
        writeProductionOverview Contains just numbers (production[MWH] for 
        each type(columns) and area(rows)), not headers
        '''
        
        numAreas = len(areas)
        numTypes = len(types)
        resultMatrix = np.zeros((numAreas, numTypes))
        for areaIndex in range(0, numAreas):
            for typeIndex in range(0, numTypes):
                prod = self._getAreaTypeProduction(areas[areaIndex], types[typeIndex], timeMaxMin)
                print("Total produced " + types[typeIndex] + " energy for " 
                        + areas[areaIndex] + " equals: " + str(prod))
                resultMatrix[areaIndex][typeIndex] = prod*TimeUnitCorrectionFactor
        return resultMatrix 
        

    def writeProductionOverview(self, areas, types, filename=None, 
                                timeMaxMin=None, TimeUnitCorrectionFactor=1):
        '''
        Export production overview to CSV file
        
        Write a .csv overview of the production[MWh] in timespan 'timeMaxMin' 
        with the different areas and types as headers.
        The vectors 'areas' and 'types' becomes headers (column- and row 
        headers), but the different elements
        of 'types' and 'areas' are also the key words in the search function
        'getAreaTypeProduction'.
        The vectors 'areas' and 'types' can be of any length. 
		'''

        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1] + 1]

            
        corner = "Countries"
        numAreas = len(areas)
        numTypes = len(types)        
        prodMat = self._productionOverview(areas, types, timeMaxMin, TimeUnitCorrectionFactor)
        if filename is not None:
            with open(filename, "wb") as f:
                writer = csv.writer(f)
                types.insert(0, corner)
                writer.writerow(types)
                for i in range(0,numAreas):
                    row = [areas[i]]
                    for j in range(0, numTypes):
                        row.append(str(prodMat[i][j]))
                    writer.writerow(row)        
        else:
            title=""
            for j in types:
                title = title + "\t" + j
            print("Area" + title)
            for i in range(0,numAreas):
                print(areas[i] + '\t%s' % '\t'.join(map(str,prodMat[i])))
                
    def getAverageInterareaBranchFlow(self, filename=None, timeMaxMin=None):
        ''' Calculate average flow in each direction and total flow for 
        inter-area branches. Requires sqlite version newer than 3.6
       
        Parameters
        ----------
        filename : string, optional
            if a filename is given then the information is stored to file.
            else the information is printed to console
        timeMaxMin : list with two integer values, or None, optional
            time interval for the calculation [start,end]
            
        Returns
        -------
        List with values for each inter-area branch:
        [flow from 1 to 2, flow from 2 to 1, average absolute flow]
        '''
        
        # Version control of database module. Must be 3.7.x or newer
        major = int(list(self.db.sqlite_version)[0])
        minor = int(list(self.db.sqlite_version)[2])
        version = major + minor / 10.0
        # print version
        if version < 3.7 :
            print('current SQLite version: ', self.db.sqlite_version)
            print('getAverageInterareaBranchFlow() requires 3.7.x or newer')
            return
            
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1] + 1]
    
        results = self.db.getAverageInterareaBranchFlow(timeMaxMin)
        
        if filename is not None:
            headers = ('branch','fromArea','toArea','average negative flow',
                       'average positive flow','average flow')
            with open(filename, "wb") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                for row in results:
                    writer.writerow(row) 
        else:
            for x in results:
                print(x)
            
        return results

    def getAverageImportExport(self,area,timeMaxMin=None):
        '''Return average import and export for a specified area'''
        
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1] + 1]
        ia =  self.getAverageInterareaBranchFlow(timeMaxMin=timeMaxMin)
        
        # export: A->B pos flow + A<-B neg flow
        sum_export = (sum([b[4] for b in ia if b[1]==area])
                        -sum([b[3] for b in ia if b[2]==area]) )
        # import: A->B neg flow + A<-B pos flow
        sum_import = (-sum([b[3] for b in ia if b[2]==area])
                        +sum([b[4] for b in ia if b[2]==area ]) )
        return dict(exp=sum_export,imp=sum_import)
        

    def getNetImport(self,area,timeMaxMin=None):        
        '''Return time series for net import for a specified area'''
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1] + 1]
            
        # find the associated branches
        br = self.grid.getInterAreaBranches(area_to=area,acdc='ac')
        br_p = br['branches_pos']
        br_n = br['branches_neg']
        dcbr = self.grid.getInterAreaBranches(area_to=area,acdc='dc')
        dcbr_p = dcbr['branches_pos']
        dcbr_n = dcbr['branches_neg']
        
        # AC branches
        ie =  self.db.getBranchesSumFlow(branches_pos=br_p,branches_neg=br_n,
                                         timeMaxMin=timeMaxMin,
                                         acdc='ac')
        # DC branches
        dcie =  self.db.getBranchesSumFlow(branches_pos=dcbr_p,
                                             branches_neg=dcbr_n,
                                             timeMaxMin=timeMaxMin,
                                             acdc='dc')
                                         
        if ie['pos'] and ie['neg']:
            res_ac = [a-b for a,b in zip(ie['pos'],ie['neg'])]
        elif ie['pos']:
            res_ac = ie['pos']
        elif ie['neg']:
            res_ac = [-a for a in ie['neg']]
        else:
            res_ac = [0]*(timeMaxMin[-1]-timeMaxMin[0]+1)   

        if dcie['pos'] and dcie['neg']:
            res_dc = [a-b for a,b in zip(dcie['pos'],dcie['neg'])]
        elif dcie['pos']:
            res_dc = dcie['pos']
        elif dcie['neg']:
            res_dc = [-a for a in dcie['neg']]
        else:
            res_dc = [0]*(timeMaxMin[-1]-timeMaxMin[0]+1)   
            
        res = [a+b for a,b in zip(res_ac,res_dc)]
        return res
        
              
    def plotNodalPrice(self,nodeIndx,timeMaxMin=None,showTitle=True):
        '''Show nodal price in single node
        
        Parameters
        ----------
        nodeIndx (int)
            index of node to plot from
        timeMaxMin (list) (default = None)
            [min, max] - lower and upper time interval
        '''

        # TODO allow for input to be multiple nodes
        # TODO plot storage price for storage in the same node?
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]
        timerange = range(timeMaxMin[0],timeMaxMin[-1]) 

        if nodeIndx  in self.db.getGridNodeIndices():
            nodalprice = self.db.getResultNodalPrice(
                nodeIndx,timeMaxMin)
            plt.figure()
            plt.plot(timerange,nodalprice)
            if showTitle:            
                plt.title("Nodal price for node %d"
                    %(nodeIndx))
            plt.show()
        else:
            print("Node not found")
        return
        
    def plotAreaPrice(self,areas,timeMaxMin=None,showTitle=True):
        '''Show area price(s)
        
        Parameters
        ----------
        areas (list)
            list of areas to show
        timeMaxMin (list) (default = None)
            [min, max] - lower and upper time interval
        '''

        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]
        timerange = range(timeMaxMin[0],timeMaxMin[-1]) 

        plt.figure()
        for a in areas:
            areaprice = self.getAreaPrices(a,timeMaxMin)
            plt.plot(timerange,areaprice,label=a)
            if showTitle:
                plt.title("Area price")
        
        plt.legend()
        plt.show()
        return
        
    def plotStorageFilling(self,generatorIndx,timeMaxMin=None,showTitle=True):
        '''Show storage filling level (MWh) for generators with storage
        
        Parameters
        ----------
        generatorIndx (int)
            index of generator to plot from
        timeMaxMin (list) (default = None)
            [min, max] - lower and upper time interval
        '''

        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]
        timerange = range(timeMaxMin[0],timeMaxMin[-1]) 

        if generatorIndx  in self.storage_idx_generators:
            storagefilling = self.db.getResultStorageFilling(
                generatorIndx,timeMaxMin)
            plt.figure()
            plt.plot(timerange,storagefilling)
            if showTitle:
                plt.title("Storage filling level for generator %d"
                    %(generatorIndx))
            plt.show()
        else:
            print("These are the generators with storage:")
            print(self.storage_idx_generators)
        return
        
    
    def plotGeneratorOutput(self,generator_index,timeMaxMin=None,
                            relativestorage=True,showTitle=True):
        '''Show output of a generator
        
        Parameters
        ----------
        generator_index (int)
            index of generator for which to make the plot
        timeMaxMin [int,int] (default=None)
            time interval for the plot [start,end]
        relativestorage (default=True)
            use filling fraction as y axis label for storage
        '''
        # TODO allow for input to be multiple generators
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]
        timerange = range(timeMaxMin[0],timeMaxMin[-1])
            
        generatoroutput = self.db.getResultGeneratorPower(
            generator_index,timeMaxMin)
        #powerinflow = db.getInflow(timeMaxMin,storageindx)

        ax1.plot(timerange,generatoroutput,'-r',label="output")
        ax1.set_ylim(ymin=0)


        # Power inflow (if generator has nonzero inflow factor)
        if self.grid.generator.inflow_factor[generator_index] > 0:
            profile = self.grid.generator.inflow_profile[generator_index]
            ax1.plot(timerange,[self.grid.inflowProfiles[profile][t-self.timerange[0]]
                *self.grid.generator.inflow_factor[generator_index]
                *self.grid.generator.prodMax[generator_index] 
                for t in timerange],'-b', label="inflow")

        # Power pumped (if generator has nonzero pumping capacity)
        if self.grid.generator.pump_cap[generator_index] > 0:
            pump_output = self.db.getResultPumpPower(
                generator_index,timeMaxMin)
            ax1.plot(timerange,pump_output,'-c', label="pumping")
        
        # Storage filling level (if generator has storage)
        ax2=None        
        if generator_index in self.storage_idx_generators:
            storagefilling = self.db.getResultStorageFilling(
                generator_index,timeMaxMin)
            if relativestorage:
                cap = self.grid.generator.storage[generator_index]
                storagefilling = [x/cap for x in storagefilling]
            ax2 = plt.twinx() #separate y axis
            ax2.plot(timerange,storagefilling,'-g', label='storage')
            ax2.legend(loc="upper right")
            ax2.set_ylim(ymin=0)
                     
        lgd=ax1.legend(loc="upper left")
        if ax2 is not None:
            ax2.add_artist(lgd)
            ax1.legend=None
        nodeidx = self.grid.node.name.index(
            self.grid.generator.node[generator_index])
        if showTitle:
            plt.title("Generator %d (%s) at node %d (%s)" 
                % (generator_index,
                   self.grid.generator.gentype[generator_index],
                   nodeidx, 
                   self.grid.generator.node[generator_index]))
        plt.show()
        return

    def plotDemandAtLoad(self,consumer_index,timeMaxMin=None,
                            relativestorage=True,showTitle=True):
        '''Make a time-series plot of consumption of a specified load
        
        Parameters
        ----------
        consumer_index (int)
            index of consumer for which to make the plot
        timeMaxMin [int,int] (default=None)
            time interval for the plot [start,end]
        relativestorage (default=True)
            use filling fraction as y axis label for storage
        '''
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]
        timerange = range(timeMaxMin[0],timeMaxMin[-1])

        # Fixed load 
        profile = self.grid.consumer.load_profile[consumer_index]
        ax1.plot(timerange,[self.grid.demandProfiles[profile][t-self.timerange[0]]
            *self.grid.consumer.load[consumer_index]
            *(1 - self.grid.consumer.flex_fraction[consumer_index]) 
            for t in timerange],'-b', label="fixed load")

        # Flexible load  (if consumer has nonzero flexible load)
        ax2=None
        if self.grid.consumer.flex_fraction[consumer_index] > 0:
            flexload_power = self.db.getResultFlexloadPower(
                consumer_index,timeMaxMin)
            ax1.plot(timerange,flexload_power,'-c', label="flexible load")
        
            # Storage filling level
            storagefilling = self.db.getResultFlexloadStorageFilling(
                consumer_index,timeMaxMin)
            if relativestorage:
                cap = self.grid.consumer.getFlexibleLoadStorageCapacity(
                            consumer_index)
                storagefilling = [x/cap for x in storagefilling]
            ax2 = plt.twinx() #separate y axis
            ax2.plot(timerange,storagefilling,'-g', label='storage')
            ax2.legend(loc="upper right")
            
        lgd=ax1.legend(loc="upper left")
        if ax2 is not None:
            ax2.add_artist(lgd)
            ax1.legend=None
        nodeidx = self.grid.node.name.index(
            self.grid.consumer.node[consumer_index])
        if showTitle:
            plt.title("Consumer %d at node %d (%s)" 
                % (consumer_index, nodeidx, 
                   self.grid.consumer.node[consumer_index]))
        plt.show()
        return


    def plotStoragePerArea(self,area,absolute=False,timeMaxMin=None,
                           showTitle=True):
        '''Show generation storage accumulated per area 
        
        Parameters
        ----------
        area (str)
        absolute (bool)(default=False)
            plot storage value in absolute or relative to maximum
        timeMaxMin (list) (default = None)
            [min, max] - lower and upper time interval'''
        
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]
        timerange = range(timeMaxMin[0],timeMaxMin[-1])

        plt.figure()
        generators = self.grid.getGeneratorsPerAreaAndType()
        cap = self.grid.generator.storage
        for gentype in generators[area].keys():
            idxGen = generators[area][gentype]
            idx_storage = [
                [i,v] for i,v in enumerate(self.storage_idx_generators) 
                if v in idxGen]
            # idx_storage is now a list of index pairs for generators with 
            #    storage in the given area
            # the first value is index in generator list
            # the second value is index in storage list (not used)
                
            if len(idx_storage) > 0:
                mystor = [sum([sum(
                    self.db.getResultStorageFilling(idx_storage[i][1],[t,t+1]))
                    for i in range(len(idx_storage))])
                    for t in timerange]
                mycap = sum( [ cap[idx_storage[i][1]]
                            for i in range(len(idx_storage))])
                
                if absolute:
                    sumStorAreaType = mystor
                else:
                    sumStorAreaType = [mystor[i]/mycap for i in range(len(mystor))]
                plt.plot(timerange,sumStorAreaType,label=gentype)
            
        # plt.legend(generators[area].keys() , loc="upper right")
        plt.legend(loc="upper right")
        if showTitle:
            plt.title("Total storage level in %s"%(area))
        plt.show()

        return
        
        
    def plotGenerationPerArea(self,area,timeMaxMin=None,fill=True,
                              reversed_order=False,net_import=True,
                              loadshed=True,showTitle=True):
        '''Show generation per area 
        
        Parameters
        ----------
        area (str)
        timeMaxMin (list) (default = None)
            [min, max] - lower and upper time interval
        fill (Boolean) - whether use filled plot
        reversed_order - whether to reverse order of generator types
        net_import - whether to include net import in graph
        loadshed - whether to include unmet demand
        '''
        
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]
        timerange = range(timeMaxMin[0],timeMaxMin[-1])
        fillfrom=[0]*len(timerange)
        count = 0
        plt.figure()
        ax = plt.subplot(111)
        generators = self.grid.getGeneratorsPerAreaAndType()
        gentypes_ordered = self._gentypes_ordered_by_fuelcost()
        if reversed_order:
            gentypes_ordered.reverse()
        numCurves = len(gentypes_ordered)+1
        colours = cm.gist_rainbow(np.linspace(0, 1, numCurves))
        for gentype in gentypes_ordered:
            if gentype in generators[area]:
                idxGen = generators[area][gentype]
                sumGenAreaType = self.db.getResultGeneratorPower(
                    idxGen,timeMaxMin)
                if fill:
                    aggregated = [x+y for x,y in zip(sumGenAreaType,fillfrom)]
                    ax.fill_between(timerange,y1=aggregated,
                                     y2=fillfrom,
                                     facecolor=colours[count])
                    #add this plot to get the legend right
                    ax.plot([],[],color=colours[count],linewidth=10,
                            label=gentype)
                    fillfrom = aggregated
                else:
                    ax.plot(timerange,sumGenAreaType,
                             color=colours[count],label=gentype)
            else:
                # in order to get the legend right
                ax.plot([],[],color=colours[count],label=gentype) 
            count=count+1
        if net_import:
            netimport = self.getNetImport(area,timeMaxMin)
            agg = [x+y for x,y in zip(netimport,fillfrom)]
            ax.plot(timerange,agg,
                    linestyle=':',linewidth=2,color='black',
                    label='net import')
        if loadshed:
            loadshed = self.getLoadheddingInArea(area,timeMaxMin)
            label = 'Load shed'
            col = 'dimgray'
            if fill:
                aggregated = [x+y for x,y in zip(loadshed,fillfrom)]
                #ax.fill_between(timerange,y1=aggregated,y2=fillfrom,
                #                facecolor=col)
                #ax.plot([],[],color=col,linewidth=10,
                #        label=label)
                ax.plot(timerange,aggregated,linestyle='--',
                        color=col,label=label)
                fillfrom = aggregated
            else:
                ax.plot(timerange,loadshed,linestyle='--',
                        color=col,label=label)
                                
            
        #plt.legend()
        handles, labels = ax.get_legend_handles_labels()
        handles.reverse()
        labels.reverse()
        plt.legend(handles, labels, loc=2,
                   bbox_to_anchor=(1.05,1), borderaxespad=0.0)

        if fill:
            plt.ylim(ymin=0)
            
        if showTitle:
            plt.title("Generation in %s"%(area))
        plt.show()
        return


    def plotDemandPerArea(self,areas,timeMaxMin=None,showTitle=True):
        '''Show demand in area(s) 
        
        Parameters
        ----------
        areas (list?)
            list of areas to be plotted
        timeMaxMin (list) (default = None)
            [min, max] - lower and upper time interval
        '''
        
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]
        timerange = range(timeMaxMin[0],timeMaxMin[-1])

        plt.figure()
        consumer = self.grid.consumer
        if type(areas) is str:
            areas = [areas]
        for co in areas:
            dem=[0]*len(self.timerange)
            flexdemand = [0]*len(self.timerange)
            consumers = self.grid.getConsumersPerArea()[co]
            for i in consumers:
                ref_profile = consumer.load_profile[i]
                # accumulate demand for all consumers in this area:
                dem = [dem[t-self.timerange[0]] + consumer.load[i] 
                    * (1 - consumer.flex_fraction[i])
                    * self.grid.demandProfiles[ref_profile][t-self.timerange[0]]
                    for t in timerange]
                flexdemand_i = self.db.getResultFlexloadPower(i,timeMaxMin)
                if len(flexdemand_i)>0:
                    flexdemand = [sum(x) for x in zip(flexdemand,flexdemand_i)]
            sumdemand = [sum(x) for x in zip(dem,flexdemand)]
            p, = plt.plot(timerange,sumdemand,label=co)
            # Fixed demand in dotted lines
            plt.plot(timerange,dem,'--',color=p.get_color())
            
        plt.legend(loc="upper right")
        if showTitle:
            plt.title("Power demand")
        plt.show()
        return

    
    def plotStorageValues(self,genindx, timeMaxMin=None,showTitle=True):
        '''Plot storage values (marginal prices) for generators with storage
        
        Parameters
        ----------
        genindx (int)
            index of generator for which to make the plot
        timeMaxMin [int,int] (default=None)
            time interval for the plot [start,end]
        '''

        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]
        timerange = range(timeMaxMin[0],timeMaxMin[-1])

        if genindx in self.storage_idx_generators:
            nodeidx = self.grid.node.name.index(
                self.grid.generator.node[genindx])
            storagevalue = self.db.getResultStorageValue(genindx,timeMaxMin)
            nodalprice = self.db.getResultNodalPrice(nodeidx,timeMaxMin)
            pumpprice = [x - self.grid.generator.pump_deadband[genindx]
                         for x in storagevalue]
            plt.figure()
            p, = plt.plot(timerange,storagevalue,label='storage value')
            if genindx in self.pump_idx_generators:
                pumpprice = [x - self.grid.generator.pump_deadband[genindx]
                             for x in storagevalue]
                plt.plot(timerange,pumpprice,'--',color=p.get_color(),
                         label='pump threshold')
            plt.plot(timerange,nodalprice,label='nodal price')
            plt.legend()
            if showTitle:
                plt.title("Storage value  for generator %d (%s) in %s"
                    % (genindx,
                       self.grid.generator.gentype[genindx],
                   self.grid.generator.node[genindx]))
            plt.show()
        else:
            print("These are the generators with storage:")
            print(self.storage_idx_generators)
        return
        
            
    def plotFlexibleLoadStorageValues(self,consumerindx, timeMaxMin=None,
                                      showTitle=True):
        '''Plot storage valuesfor flexible loads
        
        Parameters
        ----------
        consumerindx : int
            index of consumer for which to make the plot
        timeMaxMin : list, [int,int]
            time interval for the plot [start,end], or None for entire range
        '''

        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]
        timerange = range(timeMaxMin[0],timeMaxMin[-1])

        if consumerindx in self.flex_idx_consumers:
            nodeidx = self.grid.node.name.index(
                self.grid.consumer.node[consumerindx])
            storagevalue = self.db.getResultFlexloadStorageValue(
                                consumerindx,timeMaxMin)
            nodalprice = self.db.getResultNodalPrice(nodeidx,timeMaxMin)
            plt.figure()
            plt.plot(timerange,storagevalue)
            plt.plot(timerange,nodalprice)
            plt.legend(['storage value','nodal price'])
            if showTitle:
                plt.title("Storage value  for consumer %d at %s"
                    % (consumerindx,
                       self.grid.consumer.node[consumerindx]))
            plt.show()
        else:
            print("These are the consumers with flexible load:")
            print(self.flex_idx_consumers)
        return


        
    def plotMapGrid(self,nodetype='',branchtype='',dcbranchtype='',
                    show_node_labels=False,latlon=None,timeMaxMin=None,
                    dotsize=40, filter_node=None, filter_branch=None,
                    draw_par_mer=False,showTitle=True):
        '''
        Plot results to map
        
        Parameters
        ----------
        nodetype : string
            "", "area", "nodalprice", "energybalance", "loadshedding"
        branchtype : string
            "", "capacity", "area", "utilisation", "flow", "sensitivity"
        dcbranchtype : string
            ""
        show_node_labels : boolean
            whether to show node names (true/false)
        dotsize : integer
            set dot size for each plotted node
        latlon: list of four floats
            map area [lat_min, lon_min, lat_max, lon_max]
        filter_node : list of two floats
            [min,max] - lower and upper cutoff for node value
        filter_branch : list of two floats
            [min,max] - lower and upper cutoff for branch value
        draw_par_mer : boolean
            whether to draw parallels and meridians on map    
        '''
        
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]

        fig = plt.figure()
        data = self.grid
        #res = self
        
        if latlon is None:
            lat_max =  max(data.node.lat)+1
            lat_min =  min(data.node.lat)-1
            lon_max =  max(data.node.lon)+1
            lon_min =  min(data.node.lon)-1
        else:
            lat_min = latlon[0]
            lon_min = latlon[1]
            lat_max = latlon[2]
            lon_max = latlon[3]
        
        # Use the average latitude as latitude of true scale
        lat_truescale = np.mean(data.node.lat)
                
        m = Basemap(resolution='l',projection='merc',\
                      lat_ts=lat_truescale, \
                      llcrnrlon=lon_min, llcrnrlat=lat_min,\
                      urcrnrlon=lon_max ,urcrnrlat=lat_max, \
                      anchor='W')
         
        # Draw coastlines, meridians and parallels.
        m.drawcoastlines()
        m.drawcountries(zorder=0)
        m.fillcontinents(color='coral',lake_color='aqua',zorder=0)
        m.drawmapboundary(fill_color='aqua')
        
        if draw_par_mer:
            m.drawparallels(np.arange(_myround(lat_min,10,'floor'),
                _myround(lat_max,10,'ceil'),10),
                labels=[1,1,0,0])

            m.drawmeridians(np.arange(_myround(lon_min,10,'floor'),
                _myround(lon_max,10,'ceil'),10),
                labels=[0,0,0,1])
        
        num_branches = self.grid.branch.numBranches()
        num_dcbranches = self.grid.dcbranch.numBranches()
        num_nodes = self.grid.node.numNodes()


        # AC Branches
        
        lwidths = [2]*num_branches
        branch_plot_colorbar = True
        if branchtype=='area':
            areas = data.node.area
            allareas = data.getAllAreas()
            branch_value = [-1]*num_branches
            for i in range(num_branches):
                node_indx_from = data.node.name.index(data.branch.node_from[i])
                node_indx_to = data.node.name.index(data.branch.node_to[i])
                area_from = areas[node_indx_from]
                area_to = areas[node_indx_to]
                if area_from == area_to:
                    branch_value[i] = allareas.index(area_from)
                branch_value = np.asarray(branch_value)
            branch_colormap = plt.get_cmap('hsv')
            branch_colormap.set_under('k')
            filter_branch = [0,len(allareas)]
            branch_label = 'Branch area'
            branch_plot_colorbar = False
        elif branchtype=='utilisation':
            utilisation = self.getAverageUtilisation(timeMaxMin)
            branch_value = utilisation
            branch_colormap = plt.get_cmap('hot')
            branch_label = 'Branch utilisation'
        elif branchtype=='capacity':
            cap = self.grid.branch.capacity
            branch_value = np.asarray(cap)
            maxcap = np.nanmax(branch_value)
            branch_colormap = plt.get_cmap('hot')
            branch_label = 'Branch capacity'
            if filter_branch is None:
                # need an upper limit to avoid crash due to inf capacity
                filter_branch = [0,np.round(maxcap,-2)+100]
        elif branchtype=='flow':
            avgflow = self.getAverageBranchFlows(timeMaxMin)[2]
            branch_value = np.asarray(avgflow)
            branch_colormap = plt.get_cmap('hot')
            branch_label = 'Branch flow'
        elif branchtype=='sensitivity':
            branch_value = np.zeros(num_branches)
            avgsense = self.getAverageBranchSensitivity(timeMaxMin)
            branch_value[self.idxConstrainedBranchCapacity] = -avgsense
            branch_colormap = plt.get_cmap('hot')
            branch_label = 'Branch sensitivity'
            # These sensitivities are mostly negative 
            # (reduced cost by increasing branch capacity)
            #minsense = np.nanmin(avgsense)
            #maxsense = np.nanmax(avgsense)
        else:
            branch_value = np.asarray([0.5]*num_branches)
            branch_colormap = cm.gray
            branch_plot_colorbar = False
        
        idx_from = data.branch.node_fromIdx(data.node)
        idx_to = data.branch.node_toIdx(data.node)
        branch_lat1 = [data.node.lat[i] for i in idx_from]        
        branch_lon1 = [data.node.lon[i] for i in idx_from]        
        branch_lat2 = [data.node.lat[i] for i in idx_to]        
        branch_lon2 = [data.node.lon[i] for i in idx_to]

        x1, y1 = m(branch_lon1,branch_lat1)
        x2, y2 = m(branch_lon2,branch_lat2)

        ls = [[(x1[i],y1[i]),(x2[i],y2[i])] for i in range(len(x1))]
        line_segments_ac = mpl.collections.LineCollection(
                ls, linewidths=lwidths,cmap=branch_colormap)
    
        if filter_branch is not None:
            line_segments_ac.set_clim(filter_branch)
        line_segments_ac.set_array(branch_value)
        ax=plt.axes()    
        ax.add_collection(line_segments_ac)
      

        # DC Branches
        idx_from = data.dcbranch.node_fromIdx(data.node)
        idx_to = data.dcbranch.node_toIdx(data.node)
        branch_lat1 = [data.node.lat[i] for i in idx_from]        
        branch_lon1 = [data.node.lon[i] for i in idx_from]        
        branch_lat2 = [data.node.lat[i] for i in idx_to]        
        branch_lon2 = [data.node.lon[i] for i in idx_to]

        x1, y1 = m(branch_lon1,branch_lat1)
        x2, y2 = m(branch_lon2,branch_lat2)
        ls = [[(x1[i],y1[i]),(x2[i],y2[i])] for i in range(len(x1))]
        line_segments_dc = mpl.collections.LineCollection(
                ls, linewidths=2,colors='blue')
    
        ax.add_collection(line_segments_dc)
        
        # Nodes
        node_plot_colorbar = True
        if nodetype=='area':
            areas = data.node.area
            allareas = data.getAllAreas()
            #colours_co = cm.prism(np.linspace(0, 1, len(allareas)))
            node_label = 'Node area'
            node_value = [allareas.index(c) for c in areas]
            node_colormap = cm.prism
            node_plot_colorbar = False
        elif nodetype=='nodalprice':
            avgprice = self.getAverageNodalPrices(timeMaxMin)
            node_label = 'Nodal price'
            node_value = avgprice
            node_colormap = cm.jet
        elif nodetype=='energybalance':
            avg_energybalance = self.getAverageEnergyBalance(timeMaxMin)
            node_label = 'Nodal energy balance'
            node_value = avg_energybalance
            node_colormap = cm.hot
        elif nodetype=='loadshedding':
            node_value = self.getLoadsheddingPerNode(timeMaxMin)
            node_label = 'Loadshedding'
            node_colormap = cm.hot
        else:
            node_value = 'dimgray'
            node_colormap = cm.jet
            node_label = ''
            node_plot_colorbar  = False
        

        x, y = m(data.node.lon,data.node.lat)
        sc=m.scatter(x,y,marker='o',c=node_value, cmap=node_colormap,
                     zorder=2,s=dotsize)           
        #sc.cmap.set_under('dimgray')
        #sc.cmap.set_over('dimgray')
        if filter_node is not None:
            sc.set_clim(filter_node[0], filter_node[1])
            
            # #TODO: Er dette nødvendig lenger, Harald?
            # #nodes with NAN nodal price plotted in gray:
            # for i in range(len(avgprice)):
                # if np.isnan(avgprice[i]):
                    # m.scatter(x[i],y[i],c='dimgray',
                              # zorder=2,s=dotsize)
        
        
        #NEW Colorbar for nodes
        # m. or plt.?
        if node_plot_colorbar:
            axcb2 = plt.colorbar(sc)
            axcb2.set_label(node_label)

        #NEW Colorbar for branch capacity
        if branch_plot_colorbar:
            axcb = plt.colorbar(line_segments_ac)
            axcb.set_label(branch_label)


        # Show names of nodes
        if show_node_labels:
            labels = data.node.name
            x1,x2,y1,y2 = plt.axis()
            offset_x = (x2-x1)/50
            for label, xpt, ypt in zip(labels, x, y):
                if xpt > x1 and xpt < x2 and ypt > y1 and ypt < y2:
                    plt.text(xpt+offset_x, ypt, label)
        
        if showTitle:
            plt.title('Nodes (%s) and branches (%s)' %(nodetype,branchtype))
        plt.show()
                
        return
        # End plotGridMap
 
    def plotEnergyMix(self,areas=None,timeMaxMin=None,relative=False,
                      showTitle=True):
        '''
        Plot generation mix for specified areas as stacked bars
        
        Parameters
        ----------
        areas : list of sting
            Which areas to include, default=None means include all
        timeMaxMin : list of two integers
            Time range, [min,max]
        relative : boolean
            Whether to plot absolute (false) or relative (true) values
        '''        
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]
        #timerange = range(timeMaxMin[0],timeMaxMin[-1])
        #fillfrom=[0]*len(timerange)
        #count = 0
        print("Getting energy output from all generators...")
        gen_output=self.db.getResultGeneratorPowerSum(timeMaxMin)
        print("Sorting and plotting...")
        all_generators = self.grid.getGeneratorsPerAreaAndType()
        if areas is None:
            areas = all_generators.keys()
        #gentypes_ordered = self._gentypes_ordered_by_fuelcost(area)
        #gentypes = self.grid.getAllGeneratorTypes()
        gentypes = self._gentypes_ordered_by_fuelcost()
        if relative:
            prodsum={}
            for ar in areas:
                flatlist = [v for sublist in all_generators[ar].values() 
                            for v in sublist]
                prodsum[ar] = sum([gen_output[i] for i in flatlist])
                                
        plt.figure()
        ax = plt.subplot(111)
        width = 0.8
        previous = [0]*len(areas)
        numCurves = len(gentypes)+1
        colours = cm.hsv(np.linspace(0, 1, numCurves))
        count=0
        ind = range(len(areas))
        for typ in gentypes:
            A=[]
            for ar in areas:
                if typ in all_generators[ar]:
                    prod = sum([gen_output[i] 
                                    for i in all_generators[ar][typ]])
                    if relative:
                        prod = prod/prodsum[ar]
                    A.append(prod)
                else:
                    A.append(0)
                
            plt.bar(ind,A, width,label=typ,
                    bottom=previous,color=colours[count])
            previous = [previous[i]+A[i] for i in range(len(A))]
            count = count+1
        plt.legend()
        handles, labels = ax.get_legend_handles_labels()
        handles.reverse()
        labels.reverse()
        plt.legend(handles, labels, loc=2,
                   bbox_to_anchor=(1.05,1), borderaxespad=0.0)
        plt.xticks(np.arange(len(areas))+width/2., tuple(areas) )
        if showTitle:
            plt.title("Energy mix")
        plt.show()
        

    def plotEnergySpilled(self,areas=None,gentypes=None,
                          timeMaxMin=None,relative=False,showTitle=True):
        '''
        spilled energy for specified areas, as stacked bars
        
        Parameters
        ----------
        areas : list of strings
            which areas to include, default=None means include all
        gentypes : list of strings
            which generator types to include, default=None means include all
        timeMaxMin : list of integers
            time range [min,max] 
        relative : boolean
            whether to plot absolute (false) or relative (true) values
        '''        
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]

        gen_spilled=self.db.getResultGeneratorSpilledSums(timeMaxMin)

        all_generators = self.grid.getGeneratorsPerAreaAndType()
        if areas is None:
            areas = all_generators.keys()
        if gentypes is None:
            gentypes = self._gentypes_ordered_by_fuelcost()
        if relative:
            prodsum={}
            for ar in areas:
                flatlist = [v for sublist in all_generators[ar].values() 
                            for v in sublist]
                prodsum[ar] = sum([gen_spilled[i] for i in flatlist])
                                
        plt.figure()
        ax = plt.subplot(111)
        width = 0.8
        previous = [0]*len(areas)
        numCurves = len(gentypes)+1
        colours = cm.hsv(np.linspace(0, 1, numCurves))
        count=0
        ind = range(len(areas))
        for typ in gentypes:
            A=[]
            for ar in areas:
                if typ in all_generators[ar]:
                    prod = sum([gen_spilled[i] 
                                    for i in all_generators[ar][typ]])
                    if relative:
                        prod = prod/prodsum[ar]
                    A.append(prod)
                else:
                    A.append(0)
                
            plt.bar(ind,A, width,label=typ,
                    bottom=previous,color=colours[count])
            previous = [previous[i]+A[i] for i in range(len(A))]
            count = count+1
        plt.legend()
        handles, labels = ax.get_legend_handles_labels()
        handles.reverse()
        labels.reverse()
        plt.legend(handles, labels, loc=2,
                   bbox_to_anchor=(1.05,1), borderaxespad=0.0)
        plt.xticks(np.arange(len(areas))+width/2., tuple(areas) )
        if showTitle:
            plt.title("Energy spilled")
        plt.show()
        
    
    def plotTimeseriesColour(self,areas,value='nodalprice'):
        '''
        Plot timeseries values with days on x-axis and hour of day on y-axis
               
        
        Parameters
        ----------
        areas : list of strings
            which areas to include, default=None means include all
        value : 'nodalprice' (default), 
                'demand', 
                'gen_%<type1>%<type2>.. (where type=gentype)
            which times series value to plot
        '''        
        
        #print("Analysing...")
        p={}
        pm={}
        stepsperday = 24/self.grid.timeDelta
        numdays = len(self.grid.timerange)/stepsperday
        for a in areas:
            if value=='nodalprice':
                p[a] = self.getAreaPrices(area=a)
            elif value=='demand':
                #TODO: This is not generally correct. Should use
                ## weighted average for all loads in area
                p[a] = self.grid.demandProfiles['load_'+a]
            elif value[:3]=='gen':
                # value is now on form "gen_MA_hydro"
                strval = value.split('%')
                gens = self.grid.getGeneratorsPerAreaAndType()
                #genindx = gens[a][strval[1]]
                genindx = [i for s in strval[1:] for i in gens[a][s]]
                timerange=[self.timerange[0],self.timerange[-1]+1]
                p[a] = self.db.getResultGeneratorPower(genindx,timerange)
            pm[a]=np.reshape(p[a],(numdays,stepsperday)).T
        
        #print("Plotting...")
        vmin=min([min(p[a]) for a in areas])
        vmax=max([max(p[a]) for a in areas])
        num_areas = len(areas)
        fig,axes = plt.subplots(nrows=num_areas,ncols=1,figsize=(20,1.5*len(areas)))
        
        for n in range(num_areas):
            ax=plt.subplot(num_areas,1,n+1)
            ax.set_title(areas[n],x=-0.04,y=0.5,
                         verticalalignment='center',
                         horizontalalignment='right')
            im=plt.imshow(pm[areas[n]],vmin=vmin,vmax=vmax)
        
        fig.subplots_adjust(right=0.90)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        #fig.colorbar(im,cax=cbar_ax)
        plt.colorbar(cax=cbar_ax)
        plt.show()
        
        
        
    def _gentypes_ordered_by_fuelcost(self):
        '''Return a list of generator types ordered by their mean fuel cost'''
        generators = self.grid.getGeneratorsPerType()
        gentypes = generators.keys()
        fuelcosts = []
        for k in gentypes:
            gen_this_type = generators[k]
            fuelcosts.append(np.mean([self.grid.generator.fuelcost[i] 
                                     for i in gen_this_type]) )
        sorted_list = [x for (y,x) in 
                       sorted(zip(fuelcosts,gentypes))]    
        return sorted_list
        
        
        
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
