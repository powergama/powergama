# -*- coding: utf-8 -*-
'''
Module containing the PowerGAMA Results class
'''

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from mpl_toolkits.basemap import Basemap
import math
import powergama.database as db

class Results(object):
    '''
    Class for storing and analysing/presenting results from PowerGAMA
    '''    
    
    
    def __init__(self,grid,databasefile):
        '''
        Create a PowerGAMA Results object
        
        Parameters
        ----------
        grid
            GridData object reference
            
        '''
        self.grid = grid
        self.timerange = grid.timerange
        self.storage_idx_generators = grid.getIdxGeneratorsWithStorage()
        self.idxConstrainedBranchCapacity \
            = grid.getIdxBranchesWithFlowConstraints()
        
        self.db = db.Database(databasefile)
        self.db.createTables(grid)

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
        
    def addResultsFromTimestep(self,timestep,objective_function,generator_power,
                               branch_power,dcbranch_power,node_angle,
                               sensitivity_branch_capacity,
                               sensitivity_dcbranch_capacity,
                               sensitivity_node_power,
                               storage,
                               inflow_spilled,
                               loadshed_power,
                               marginalprice):
        '''Store results from optimal power flow for a new timestep'''
        
        # Store results in sqlite database on disk (to avoid memory problems)
        self.db.appendResults(
            timestep = timestep,
            objective_function = objective_function,
            generator_power = generator_power,
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
            idx_storagegen = self.storage_idx_generators,
            idx_branchsens = self.idxConstrainedBranchCapacity)
       
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

    def getAverageNodalPrices(self,timeMaxMin=None):
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]

        avgprices = self.db.getResultNodalPricesMean(timeMaxMin)
        # use asarray to convert None to nan
        avgprices = np.asarray(avgprices,dtype=float)
        return avgprices
       
    def getAverageBranchSensitivity(self,timeMaxMin=None):
        '''
        Average branch capacity sensitivity over a given time period
        
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
        
        Returns
        =======
        1-dim Array of branch utilisation (power flow/capacity)
        '''
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]

        cap =self.grid.branch.capacity
        avgflow = self.getAverageBranchFlows(timeMaxMin)[2]
        utilisation = [avgflow[i] / cap[i] for i in xrange(len(cap))] 
        utilisation = np.asarray(utilisation)
        return utilisation
        
        
    def plotStorageFilling(self,generatorIndx,timeMaxMin=None):
        '''Show storage filling level (MWh) for generators with storage'''

        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]
        timerange = xrange(timeMaxMin[0],timeMaxMin[-1]) 

        if generatorIndx  in self.storage_idx_generators:
            storagefilling = self.db.getResultStorageFilling(
                generatorIndx,timeMaxMin)
            plt.figure()
            plt.plot(timerange,storagefilling)
            plt.title("Storage filling level for generator %d"
                %(generatorIndx))
            plt.show()
        else:
            print "These are the generators with storage:"
            print self.storage_idx_generators
        return
        
    
    def plotGeneratorOutput(self,generator_index,timeMaxMin=None):
        '''Show output of a generator'''
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
        
        # Storage filling level (if generator has storage)
        if generator_index in self.storage_idx_generators:
            storagefilling = self.db.getResultStorageFilling(
                generator_index,timeMaxMin)
            ax2 = plt.twinx() #separate y axis
            ax2.plot(timerange,storagefilling,'-g', label='storage')
            ax2.legend(loc="upper right")
                     
        ax1.legend(loc="upper left")
        plt.title("Generator %d (%s) at node %s" 
            % (generator_index,
               self.grid.generator.gentype[generator_index],
               self.grid.generator.node[generator_index]))
        plt.show()
        return


    def plotStoragePerArea(self,area,absolute=False,timeMaxMin=None):
        '''Show generation storage accumulated per area '''
        
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
                    for i in xrange(len(idx_storage))])
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
        
        
    def plotGenerationPerArea(self,area,timeMaxMin=None):
        '''Show generation per area '''

        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]
        timerange = range(timeMaxMin[0],timeMaxMin[-1])
        
        plt.figure()
        generators = self.grid.getGeneratorsPerAreaAndType()
        for gentype in generators[area].keys():
            idxGen = generators[area][gentype]
            sumGenAreaType = self.db.getResultGeneratorPower(
                idxGen,timeMaxMin)
            plt.plot(timerange,sumGenAreaType)
            
        plt.legend(generators[area].keys() , loc="upper right")
        plt.title("Generation in %s"%(area))
        plt.show()
        return


    def plotDemandPerArea(self,areas,timeMaxMin=None):
        '''Show demand in area(s) '''
        
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]
        timerange = range(timeMaxMin[0],timeMaxMin[-1])

        plt.figure()
        consumer = self.grid.consumer
        if type(areas) is str:
            areas = [areas]
        for co in areas:
            dem=[0]*len(self.timerange)
            consumers = self.grid.getConsumersPerArea()[co]
            for i in consumers:
                ref_profile = consumer.load_profile[i]
                # accumulate demand for all consumers in this area:
                dem = [dem[t-self.timerange[0]] + consumer.load[i] 
                    * self.grid.demandProfiles[ref_profile][t-self.timerange[0]] 
                    for t in timerange]
            plt.plot(timerange,dem)
            
        plt.legend(areas , loc="upper right")
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
            plt.figure()
            plt.plot(timerange,storagevalue)
            plt.plot(timerange,nodalprice)
            plt.legend(['storage value','nodal price'])
            plt.title("Storage value  for generator %d (%s) in %s"
                % (genindx,
               self.grid.generator.gentype[genindx],
               self.grid.generator.node[genindx]))
            plt.show()
        else:
            print "These are the generators with storage:"
            print self.storage_idx_generators
        return
        
            


        
    def plotMapGrid(self,nodetype='',branchtype='',dcbranchtype='',
                    show_node_labels=False,latlon=None,timeMaxMin=None,
                    dotsize=40):
        '''
        Plot results to map
        
        Parameters
        ----------
        nodetype (str) (default = "")
            "", "area", "nodalprice"
        branchtype (str) (default = "")
            "", "area", "utilisation", "flow", "sensitivity"
        dcbranchtype (str) (default = "")
            ""
        show_node_labels (bool) (default=False)
            show node names (true/false)
        latlon (list) (default=None)
            map area [lat_min, lon_min, lat_max, lon_max]
        '''
        
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0],self.timerange[-1]+1]

        plt.figure()
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
                      urcrnrlon=lon_max ,urcrnrlat=lat_max,
                      )
         
        # Draw coastlines, meridians and parallels.
        m.drawcoastlines()
        m.drawcountries(zorder=0)
        m.fillcontinents(color='coral',lake_color='aqua',zorder=0)
        m.drawmapboundary(fill_color='aqua')
        
        m.drawparallels(np.arange(_myround(lat_min,10,'floor'),
                                  _myround(lat_max,10,'ceil'),10),
                        labels=[1,1,0,0])

        m.drawmeridians(np.arange(_myround(lon_min,10,'floor'),
                                  _myround(lon_max,10,'ceil'),10),
                        labels=[0,0,0,1])
        
        
        if nodetype=='area':
            areas = data.node.area
            allareas = data.getAllAreas()
            colours_co = cm.prism(np.linspace(0, 1, len(allareas)))
        elif nodetype=='nodalprice':
            avgprice = self.getAverageNodalPrices(timeMaxMin)
        

        if branchtype=='area':
            areas = data.node.area
            allareas = data.getAllAreas()
            colours_b = cm.prism(np.linspace(0, 1, len(allareas)))
        elif branchtype=='utilisation' or branchtype=='flow':
            numBranchCategories = 11
            colours_b = cm.jet(np.linspace(0, 1, numBranchCategories))
            avgflow = self.getAverageBranchFlows(timeMaxMin)[2]
            #branchflow = self.db.getResultBranchFlowAll(timeMaxMin)
            #avgflow = np.sqrt(np.average(np.asarray(
            #    branchflow,dtype=float)**2,axis=1)) #rms
            #cap =res.grid.branch.capacity
            #[avgflow[i] / cap[i] for i in xrange(len(cap))] 
            utilisation = self.getAverageUtilisation(timeMaxMin)
            # element-by-element
            #print ("Branch utilisation: max=%g, min=%g" 
            #    %(max(utilisation),min(utilisation)) )
        elif branchtype=='sensitivity':
            numBranchCategories = 11
            colours_b = cm.jet(np.linspace(0, 1, numBranchCategories))
            #branchsens = self.db.getResultBranchSensAll(timeMaxMin)
            #avgsense = np.sqrt(np.average(np.asarray(
            #    branchsens,dtype=float)**2,axis=1)) #rms 
            avgsense = self.getAverageBranchSensitivity(timeMaxMin)
            maxsense = np.nanmax(avgsense)
            #print ("Branch capacity senitivity: max=%g, min=%g" 
            #    %(np.nanmax(avgsense),np.nanmin(avgsense)) )
        
        
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
                category = math.floor(utilisation[j]*(numBranchCategories-1))
                #print "utilisation cat=",category
                col = colours_b[category]
                lwidth = 2
                if cap[j]== np.inf:
                    col = 'grey'
                    lwidth = 1
            elif branchtype=='flow':
                maxflow = max(avgflow)
                category = math.floor(
                    avgflow[j]/maxflow*(numBranchCategories-1))
                col = colours_b[category]
                lwidth = 1
                if category*2 > numBranchCategories:
                    lwidth = 2
            elif branchtype=='sensitivity':
                if j in res.idxConstrainedBranchCapacity:
                    #idx = res.idxConstrainedBranchCapacity.index(j)
                    idx = j
                    if not  np.isnan(avgsense[idx]) and maxsense>0:
                        category = math.floor(
                            avgsense[idx]/maxsense*(numBranchCategories-1))
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
                    col = 'white'
                    lwidth = 2
            else:
                col = 'white'
                lwidth = 2
            m.drawgreatcircle(branch_lon1[j],branch_lat1[j],\
                              branch_lon2[j],branch_lat2[j],\
                              linewidth=lwidth,color=col,zorder=1)
            
        # Plot nodes
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
            s=m.scatter(x,y,marker='o',c=avgprice, cmap=cm.jet, 
                        zorder=2,s=dotsize)
            cb=m.colorbar(s)
            cb.set_label('Nodal price')
            #nodes with NAN nodal price plotted in gray:
            for i in xrange(len(avgprice)):
                if np.isnan(avgprice[i]):
                    m.scatter(x[i],y[i],c='dimgray',
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

        return
        # End plotGridMap
 

       
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
