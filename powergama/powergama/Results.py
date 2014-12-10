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
       
    def getLoadheddingInArea(self,area,timeMaxMin=None):
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]

        loadshed = self.db.getResultLoadheddingInArea(area,timeMaxMin)
        # use asarray to convert None to nan
        loadshed = np.asarray(loadshed,dtype=float)
        return loadshed


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
        branchflowsDc = self.db.getResultBranchFlowsMean(timeMaxMin,ac=False)
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
              
    def plotNodalPrice(self,nodeIndx,timeMaxMin=None):
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
            plt.title("Nodal price for node %d"
                %(nodeIndx))
            plt.show()
        else:
            print("Node not found")
        return
        
        
    def plotStorageFilling(self,generatorIndx,timeMaxMin=None):
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
            plt.title("Storage filling level for generator %d"
                %(generatorIndx))
            plt.show()
        else:
            print("These are the generators with storage:")
            print(self.storage_idx_generators)
        return
        
    
    def plotGeneratorOutput(self,generator_index,timeMaxMin=None,
                            relativestorage=True):
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
        if generator_index in self.storage_idx_generators:
            storagefilling = self.db.getResultStorageFilling(
                generator_index,timeMaxMin)
            if relativestorage:
                cap = self.grid.generator.storage[generator_index]
                storagefilling = [x/cap for x in storagefilling]
            ax2 = plt.twinx() #separate y axis
            ax2.plot(timerange,storagefilling,'-g', label='storage')
            ax2.legend(loc="upper right")
                     
        ax1.legend(loc="upper left")
        nodeidx = self.grid.node.name.index(
            self.grid.generator.node[generator_index])
        plt.title("Generator %d (%s) at node %d (%s)" 
            % (generator_index,
               self.grid.generator.gentype[generator_index],
               nodeidx, 
               self.grid.generator.node[generator_index]))
        plt.show()
        return

    def plotDemandAtLoad(self,consumer_index,timeMaxMin=None,
                            relativestorage=True):
        '''Show demand by a load
        
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
            
        ax1.legend(loc="upper left")
        nodeidx = self.grid.node.name.index(
            self.grid.generator.node[consumer_index])
        plt.title("Consumer %d at node %d (%s)" 
            % (consumer_index, nodeidx, 
               self.grid.consumer.node[consumer_index]))
        plt.show()
        return


    def plotStoragePerArea(self,area,absolute=False,timeMaxMin=None):
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
        plt.title("Total storage level in %s"%(area))
        plt.show()

        return
        
    
    def gentypes_ordered_by_fuelcost(self,area):
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
        
        
    def plotGenerationPerArea(self,area,timeMaxMin=None,fill=False,
                              reversed_order=False):
        '''Show generation per area 
        
        Parameters
        ----------
        area (str)
        timeMaxMin (list) (default = None)
            [min, max] - lower and upper time interval
        fill (Boolean) - whether use filled plot
        '''
        
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]
        timerange = range(timeMaxMin[0],timeMaxMin[-1])
        fillfrom=[0]*len(timerange)
        count = 0
        plt.figure()
        ax = plt.subplot(111)
        generators = self.grid.getGeneratorsPerAreaAndType()
        gentypes_ordered = self.gentypes_ordered_by_fuelcost(area)
        if reversed_order:
            gentypes_ordered.reverse()
        numCurves = len(gentypes_ordered)+1
        colours = cm.gist_rainbow(np.linspace(0, 1, numCurves))
        for gentype in gentypes_ordered:
            if generators[area].has_key(gentype):
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
        #plt.legend()
        handles, labels = ax.get_legend_handles_labels()
        handles.reverse()
        labels.reverse()
        plt.legend(handles, labels, loc="upper right")
                   #fancybox=True, framealpha=0.5)
        plt.title("Generation in %s"%(area))
        plt.show()
        return colours


    def plotDemandPerArea(self,areas,timeMaxMin=None):
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
            plt.plot(timerange,dem,'--',color=p.get_color())
            
        plt.legend(loc="upper right")
        plt.title("Power demand")
        plt.show()
        return

    
    def plotStorageValues(self,genindx, timeMaxMin=None):
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
            plt.title("Storage value  for generator %d (%s) in %s"
                % (genindx,
               self.grid.generator.gentype[genindx],
               self.grid.generator.node[genindx]))
            plt.show()
        else:
            print("These are the generators with storage:")
            print(self.storage_idx_generators)
        return
        
            
    def plotFlexibleLoadStorageValues(self,consumerindx, timeMaxMin=None):
        '''Plot storage valuesfor flexible loads
        
        Parameters
        ----------
        consumerindx (int)
            index of consumer for which to make the plot
        timeMaxMin [int,int] (default=None)
            time interval for the plot [start,end]
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
                    dotsize=40, filter_price=None, draw_par_mer=True):
        '''
        Plot results to map
        
        Parameters
        ----------
        nodetype (str) (default = "")
            "", "area", "nodalprice", "energybalance"
        branchtype (str) (default = "")
            "", "area", "utilisation", "flow", "sensitivity"
        dcbranchtype (str) (default = "")
            ""
        show_node_labels (bool) (default=False)
            show node names (true/false)
        dotsize (int) (default=40)
            set dot size for each plotted node
        latlon (list) (default=None)
            map area [lat_min, lon_min, lat_max, lon_max]
        filter_price (list) (default=None)
            [min,max] - lower and upper cutof for price range
        draw_par_mer (bool) (default=True)
            draw parallels and meridians on map    
        '''
        
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]

        fig = plt.figure()
        data = self.grid
        res = self
        
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
        
        

        if branchtype=='area':
            areas = data.node.area
            allareas = data.getAllAreas()
            colours_b = cm.prism(np.linspace(0, 1, len(allareas)))
        elif branchtype=='utilisation' or branchtype=='flow':
            numBranchCategories = 11
            colours_b = cm.hot(np.linspace(0, 1, numBranchCategories))
            avgflow = self.getAverageBranchFlows(timeMaxMin)[2]
            utilisation = self.getAverageUtilisation(timeMaxMin)
        elif branchtype=='sensitivity':
            numBranchCategories = 11
            colours_b = cm.hot(np.linspace(0, 1, numBranchCategories))
            avgsense = self.getAverageBranchSensitivity(timeMaxMin)
            # These sensitivities are mostly negative 
            # (reduced cost by increasing branch capacity)
            minsense = np.nanmin(avgsense)
            maxsense = np.nanmax(avgsense)
        
        
        # Plot AC branches (as great circles)
        idx_from = data.branch.node_fromIdx(data.node)
        idx_to = data.branch.node_toIdx(data.node)
        branch_lat1 = [data.node.lat[i] for i in idx_from]        
        branch_lon1 = [data.node.lon[i] for i in idx_from]        
        branch_lat2 = [data.node.lat[i] for i in idx_to]        
        branch_lon2 = [data.node.lon[i] for i in idx_to]
        for j in range(len(branch_lat1)):
            if branchtype=='area':
                if areas[idx_from[j]] == areas[idx_to[j]]:
                    col = colours_b[allareas.index(areas[idx_from[j]])]
                    lwidth = 1
                else:
                    col = 'black'
                    lwidth = 4
            elif branchtype=='utilisation':
                cap = res.grid.branch.capacity[j]
                #print "utilisation cat=",category
                if cap == np.inf:
                    col = 'grey'
                    lwidth = 1
                else:
                    category = math.floor(utilisation[j]*(numBranchCategories-1))
                    col = colours_b[category]
                    lwidth = 2
            elif branchtype=='flow':
                maxflow = max(avgflow)
                minflow = min(avgflow)
                category = math.floor(
                    avgflow[j]/maxflow*(numBranchCategories-1))
                col = colours_b[category]
                lwidth = 1
                if category*2 > numBranchCategories:
                    lwidth = 2
            elif branchtype=='sensitivity':
                if j in res.idxConstrainedBranchCapacity:
                    idx = res.idxConstrainedBranchCapacity.index(j)
                    #idx = j
                    if not  np.isnan(avgsense[idx]) and minsense!=0:
                        category = math.floor(
                            avgsense[idx]/minsense*(numBranchCategories-1))
                        #print "sense cat=",category
                        col = colours_b[category]
                        lwidth = 2
                    else:
                        #NAN sensitivity (not returned by solver)
                        #Or all sensitivities are zero
                        col='grey'
                        lwidth = 2
                else:
                    col = 'grey'
                    lwidth = 1
            else:
                lwidth = 1
                col = 'black'
                 
            m.drawgreatcircle(branch_lon1[j],branch_lat1[j],\
                              branch_lon2[j],branch_lat2[j],\
                              linewidth=lwidth,color=col,zorder=1)

        # Plot DC branches
        idx_from = data.dcbranch.node_fromIdx(data.node)
        idx_to = data.dcbranch.node_toIdx(data.node)
        branch_lat1 = [data.node.lat[i] for i in idx_from]        
        branch_lon1 = [data.node.lon[i] for i in idx_from]        
        branch_lat2 = [data.node.lat[i] for i in idx_to]        
        branch_lon2 = [data.node.lon[i] for i in idx_to]
        for j in range(len(branch_lat1)):
            if dcbranchtype=='area':
                if areas[idx_from[j]] == areas[idx_to[j]]:
                    col = colours_b[allareas.index(areas[idx_from[j]])]
                    lwidth = 1
                else:
                    col = 'blue'
                    lwidth = 2
            else:
                col = 'blue'
                lwidth = 2
            m.drawgreatcircle(branch_lon1[j],branch_lat1[j],\
                              branch_lon2[j],branch_lat2[j],\
                              linewidth=lwidth,color=col,zorder=1)
            
        # Plot nodes

        if nodetype=='area':
            areas = data.node.area
            allareas = data.getAllAreas()
            colours_co = cm.prism(np.linspace(0, 1, len(allareas)))
        elif nodetype=='nodalprice':
            avgprice = self.getAverageNodalPrices(timeMaxMin)
        elif nodetype=='energybalance':
            avg_energybalance = self.getAverageEnergyBalance(timeMaxMin)
        

        x, y = m(data.node.lon,data.node.lat)
        if nodetype == 'area':
            for co in range(len(allareas)):
                co_nodes = [i for i, v in enumerate(data.node.area) 
                            if v==allareas[co]]
                co_x = [x[i] for i in co_nodes]
                co_y = [y[i] for i in co_nodes]
                col = colours_co[co]
                m.scatter(co_x,co_y,marker='o',color=col, 
                          zorder=2,s=dotsize)
        elif nodetype == 'nodalprice':
            if filter_price != None:
                for index, price in enumerate(avgprice):
                    if (price > filter_price[1]):
                        avgprice[index] = filter_price[1]
                    elif (price < filter_price[0]):
                        avgprice[index] = filter_price[0]
            s=m.scatter(x,y,marker='o',c=avgprice, cmap=cm.jet, 
                        zorder=2,s=dotsize)
            
            # #TODO: Er dette nødvendig lenger, Harald?
            # #nodes with NAN nodal price plotted in gray:
            # for i in range(len(avgprice)):
                # if np.isnan(avgprice[i]):
                    # m.scatter(x[i],y[i],c='dimgray',
                              # zorder=2,s=dotsize)
        elif nodetype == 'energybalance':
            if filter_price != None:
                for index, price in enumerate(avg_energybalance):
                    if (price > filter_price[1]):
                        avg_energybalance[index] = filter_price[1]
                    elif (price < filter_price[0]):
                        avg_energybalance[index] = filter_price[0]
            m.scatter(x,y,marker='o',c=avg_energybalance, cmap=cm.hot, 
                        zorder=2,s=dotsize)
            
        else:
            col='dimgray'
            m.scatter(x,y,marker='o',color=col, 
                      zorder=2,s=dotsize)
            #cb=m.colorbar()
        
        
        # Show names of nodes
        if show_node_labels:
            labels = data.node.name
            x1,x2,y1,y2 = plt.axis()
            for label, xpt, ypt in zip(labels, x, y):
                if xpt > x1 and xpt < x2 and ypt > y1 and ypt < y2:
                    plt.text(xpt, ypt, label)
        
        plt.title('Nodes (%s) and branches (%s)' %(nodetype,branchtype))
        plt.show()
        
        #Adding legends and colorbars
        if branchtype == '':
            ax_cb_node_offset = 0.15
        else:
            ax_cb_node_offset = 0
        #Colorbar for nodal price
        if nodetype == 'nodalprice':
            ax_cb_node = fig.add_axes((0.70+ax_cb_node_offset, 0.125,0.03,0.75))
            colormap = plt.get_cmap('jet')
            norm = mpl.colors.Normalize(min(avgprice), max(avgprice))
            colorbar_node = mpl.colorbar.ColorbarBase(ax_cb_node, cmap=colormap, norm=norm, orientation='vertical')
            colorbar_node.set_label('Nodal price')
        if nodetype == 'energybalance':
            ax_cb_node = fig.add_axes((0.70+ax_cb_node_offset, 0.125,0.03,0.75))
            colormap = plt.get_cmap('hot')
            norm = mpl.colors.Normalize(
                #-1*np.std(avg_energybalance), 1*np.std(avg_energybalance))
                min(avg_energybalance), max(avg_energybalance))
            colorbar_node = mpl.colorbar.ColorbarBase(
                ax_cb_node, cmap=colormap, norm=norm, orientation='vertical')
            colorbar_node.set_label('Energy balance')
        #Legend for nodal areas
        elif nodetype == 'area':
            patches = []
            p_labels = []
            for country in range(len(allareas)):
                patches.append(mpl.patches.Patch(color=colours_co[country]))
                p_labels.append(allareas[country])
            fig.legend(patches, p_labels, bbox_to_anchor=(0.75+ax_cb_node_offset,0.15,0.03,0.75), \
                        title='AREA', handlelength=0.7,handletextpad=0.4, frameon=False)
        #Legend for branch area
        if branchtype == 'area':
            patches = []
            p_labels = []
            for country in range(len(allareas)):
                patches.append(mpl.patches.Patch(color=colours_b[country]))
                p_labels.append(allareas[country])
            patches.append(mpl.patches.Patch(color='black'))
            p_labels.append('INT')
            fig.legend(patches, p_labels, bbox_to_anchor=(0.9,0.15,0.03,0.75), \
                        title='BRANCH', handlelength=0.7,handletextpad=0.4, frameon=False)
        #Colorbar for branch utilisation    
        elif branchtype == 'utilisation':
            ax_cb_branch = fig.add_axes((0.85, 0.125, 0.03, 0.75))
            colormap = plt.get_cmap('hot')
            bounds = np.linspace(0,1,numBranchCategories)
            norm = mpl.colors.BoundaryNorm(bounds,256)
            colorbar_branch = mpl.colorbar.ColorbarBase(
                ax_cb_branch, cmap=colormap, norm=norm, boundaries=bounds,
                spacing='uniform', orientation='vertical')
            colorbar_branch.set_label('Branch utilisation')
        #Colorbar for branch flow
        elif branchtype == 'flow':
            ax_cb_branch = fig.add_axes((0.85, 0.125, 0.03, 0.75))
            colormap = plt.get_cmap('hot')
            #norm = mpl.colors.Normalize(minflow,maxflow)
            bounds = np.linspace(minflow,maxflow,numBranchCategories)
            norm = mpl.colors.BoundaryNorm(bounds,256)
            colorbar_branch = mpl.colorbar.ColorbarBase(
                ax_cb_branch, cmap=colormap, norm=norm, boundaries=bounds,
                spacing='uniform', orientation='vertical')
            colorbar_branch.set_label('Branch flow')
        #Colorbar for branch sensitivity
        elif branchtype == 'sensitivity':
            ax_cb_branch = fig.add_axes((0.85, 0.125, 0.03, 0.75))
            colormap = plt.get_cmap('hot')
            #norm = mpl.colors.Normalize(0,abs(minsense))
            bounds = np.linspace(0,abs(minsense),numBranchCategories)
            norm = mpl.colors.BoundaryNorm(bounds,256)
            colorbar_branch = mpl.colorbar.ColorbarBase(
                ax_cb_branch, cmap=colormap, norm=norm, boundaries=bounds,
                spacing='uniform', orientation='vertical')
            colorbar_branch.set_label('Branch sensitivity')
        
        return
        # End plotGridMap
 

    def node2area(self, nodeName):
        '''name of a single node as input and return the index of the node.''' 
        #Is handy when you need to access more information about the node, 
        #but only the node name is avaiable. (which is the case in the generator file)
        try:
            nodeIndex = self.grid.node.name.index(nodeName)
            return self.grid.node.area[nodeIndex]
        except:
            return
            
    def getAreaTypeProduction(self, area, generatorType, timeMaxMin):
        #Returns total production [MWh] in the timerange 'timeMaxMin' for
        #all generators of 'generatorType' in 'area'
        
        print("Looking for generators of type " + str(generatorType) + ", in " + str(area))
        print("Number of generator to run through: " + str(self.grid.generator.numGenerators()))
        totalProduction = 0
        
        
        for genNumber in range(0, self.grid.generator.numGenerators()):
            genName = self.grid.generator.desc[genNumber]
            genNode = self.grid.generator.node[genNumber]
            genType = self.grid.generator.gentype[genNumber]
            genArea = self.node2area(genNode)
            #print str(genNumber) + ", " + genName + ", " + genNode + ", " + genType + ", " + genArea
            if (genType == generatorType) and (genArea == area):
                #print "\tGenerator is of right type and area. Adding production"                
                genProd = sum(self.db.getResultGeneratorPower(genNumber, 
                                                              timeMaxMin))
                totalProduction += genProd
                #print "\tGenerator production = " + str(genProd)
        return totalProduction
        
    def getAllGeneratorProduction(self, timeMaxMin):
        #Returns all production [MWh] for all generators in the timerange 'timeMaxMin'
        
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
    
    def productionOverview(self, areas, types, timeMaxMin, 
                           TimeUnitCorrectionFactor):
        #Return a matrix (numpy matrix, remember to include numpy) with productionOverview
        #This function is manly used as the calculation part of the writeProductionOverview
        #Contains just numbers (production[MWH] for each type(columns) and area(rows)), not headers
        
        numAreas = len(areas)
        numTypes = len(types)
        resultMatrix = np.zeros((numAreas, numTypes))
        for areaIndex in range(0, numAreas):
            for typeIndex in range(0, numTypes):
                prod = self.getAreaTypeProduction(areas[areaIndex], types[typeIndex], timeMaxMin)
                print("Total produced " + types[typeIndex] + " energy for " 
                        + areas[areaIndex] + " equals: " + str(prod))
                resultMatrix[areaIndex][typeIndex] = prod*TimeUnitCorrectionFactor
        return resultMatrix 
        

    def writeProductionOverview(self, areas, types, filename=None, 
                                timeMaxMin=None, TimeUnitCorrectionFactor=1):
        #Write an .csv overview of the production[MWh] in timespan 'timeMaxMin' with the different areas and types as headers.
        #The vectors 'areas' and 'types' becomes headers (column- and row headers), but the different elements
        #of 'types' and 'areas' are also the key words in the search function 'getAreaTypeProduction'.
        #The vectors 'areas' and 'types' can be of any length. 

        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1] + 1]

            
        corner = "Countries"
        numAreas = len(areas)
        numTypes = len(types)        
        prodMat = self.productionOverview(areas, types, timeMaxMin, TimeUnitCorrectionFactor)
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
        filename (str) (default=None)
            if a filename is given then the information is stored to file.
            else the information is printed to console
        timeMaxMin [int,int] (default=None)
            time interval for the calculation [start,end]
            
        Returns
        =======
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
