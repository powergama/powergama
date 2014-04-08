# -*- coding: utf-8 -*-
'''
Module containing the PowerGAMA Results class
'''

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from mpl_toolkits.basemap import Basemap
import math

class Results(object):
    '''
    Class for storing and analysing/presenting results from PowerGAMA
    '''    
    
    
    def __init__(self,grid):
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
        self.idxConstrainedBranchCapacity \
            = grid.getIdxBranchesWithFlowConstraints()
        
    def addResultsFromTimestep(self,objective_function,generator_power,
                               branch_power,node_angle,
                               sensitivity_branch_capacity,
                               sensitivity_dcbranch_capacity,
                               sensitivity_node_power,
                               storage,
                               inflow_spilled,
                               loadshed_power,
                               marginalprice):
        '''Store results from optimal power flow for a new timestep'''
                                   
        self.objectiveFunctionValue.append(objective_function)
        self.generatorOutput.append(generator_power)
        self.branchFlow.append(branch_power)
        self.nodeAngle.append(node_angle)
        self.sensitivityBranchCapacity.append(sensitivity_branch_capacity)
        self.sensitivityDcBranchCapacity.append(sensitivity_dcbranch_capacity)
        self.sensitivityNodePower.append(sensitivity_node_power)
        #self.demandPower.append(demand_power)
        self.storage.append(storage)
        self.inflowSpilled.append(inflow_spilled)
        self.loadshed.append(loadshed_power)
        self.marginalprice.append(marginalprice)
        
        # self.storageGeneratorsIdx.append(idx_generatorsWithStorage)


    
    def plotStorageFilling(self):
        '''Show storage filling level (MWh) for generators with storage'''
        fig = plt.figure()
        if len(self.storage_idx_generators) > 0:
            plt.plot(self.timerange,self.storage)
        else:
            print "There are no generators with storage in this system"
        plt.show()
        return
        
    
    def plotGeneratorOutput(self,generator_index):
        '''Show output of a generator'''
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(self.timerange,
                 [self.generatorOutput[t][generator_index] 
                 for t in self.timerange],'-r',
                 label="output")

        if self.grid.generator.inflow_factor[generator_index] > 0:
            profile = self.grid.generator.inflow_profile[generator_index]
            ax1.plot([self.grid.inflowProfiles[profile][t]
                *self.grid.generator.inflow_factor[generator_index]
                *self.grid.generator.prodMax[generator_index] 
                for t in self.timerange],'-b', label="inflow")
        
        if generator_index in self.storage_idx_generators:
            stor_idx = self.storage_idx_generators.index(generator_index)
            ax2 = plt.twinx() #separate y axis
            ax2.plot([self.storage[t][stor_idx] for t in self.timerange],
                     '-g', label='storage')
            ax2.legend(loc="upper right")
                     
        ax1.legend(loc="upper left")
        plt.show()
        return


    def plotStoragePerArea(self,area,absolute=False):
        '''Show generation per area '''
        
        fig = plt.figure()
        generators = self.grid.getGeneratorsPerAreaAndType()
        cap = self.grid.generator.storage
        for gentype in generators[area].keys():
            idxGen = generators[area][gentype]
            idx_storage = [
                [i,v] for i,v in enumerate(self.storage_idx_generators) 
                if v in idxGen]
            # idx_storage is now a list of index pairs.
            # the first value is index in generator list
            # the second value is index in storage list
                
            if len(idx_storage) > 0:
                #stor = map(list, zip(*self.storage)) #transpose
                
                mystor = [sum([self.storage[t][idx_storage[i][0]]
                            for i in range(len(idx_storage))])
                            for t in self.timerange]
                mycap = sum( [ cap[idx_storage[i][1]]
                            for i in range(len(idx_storage))])
                
                if absolute:
                    sumStorAreaType = mystor
                else:
                    sumStorAreaType = [mystor[i]/mycap for i in range(len(mystor))]
                plt.plot(self.timerange,sumStorAreaType,label=gentype)
            
        # plt.legend(generators[area].keys() , loc="upper right")
        plt.legend(loc="upper right")
        plt.title("Storage in %s"%(area))
        plt.show()

        return
        
        
    def plotGenerationPerArea(self,area):
        '''Show generation per area '''
        fig = plt.figure()
        generators = self.grid.getGeneratorsPerAreaAndType()
        for gentype in generators[area].keys():
            idxGen = generators[area][gentype]
            gen = zip(*self.generatorOutput) #transpose
            mygen = [gen[i] for i in idxGen]
            sumGenAreaType = [sum(x) for x in zip(*mygen) ]
            plt.plot(self.timerange,sumGenAreaType)
            
        plt.legend(generators[area].keys() , loc="upper right")
        plt.title("Generation in %s"%(area))
        plt.show()
        return


    def plotDemandPerArea(self,areas):
        '''Show demand per area(s) '''
        
        fig = plt.figure()
        consumer = self.grid.consumer
        if type(areas) is str:
            areas = [areas]
        for co in areas:
            dem=[0]*len(self.timerange)
            consumers = self.grid.getConsumersPerArea()[co]
            for i in consumers:
                ref_profile = consumer.load_profile[i]
                # accumulate demand for all consumers in this area:
                dem = [dem[t] + consumer.load[i] 
                    * self.grid.demandProfiles[ref_profile][t] 
                    for t in self.timerange]
            plt.plot(self.timerange,dem)
            
        plt.legend(areas , loc="upper right")
        plt.title("Power demand")
        plt.show()
        return

    
    def plotMarginalPrice(self,generator_index=None):
        '''Show marginal prices for generators with storage'''
        fig = plt.figure()
        if generator_index is None:
            plt.plot(self.timerange,self.marginalprice)
            plt.legend(
                [self.grid.generator.node[i] 
                    for i in self.storage_idx_generators], 
                loc="upper right")
        else:
            # show a single storage generator
            storageidx = self.storage_idx_generators.index(generator_index)
            nodeidx = self.grid.node.name.index(
                self.grid.generator.node[generator_index])
            plt.plot(self.timerange,
                     [self.marginalprice[h][storageidx] 
                         for h in self.timerange])
            plt.plot(self.timerange,
                     [self.sensitivityNodePower[h][nodeidx] 
                         for h in self.timerange])
            plt.legend(['storage value','nodal price'])
        plt.show()
        return
        
            
        
    def plotMapGrid_old(self):
        '''Show a map with nodes and branches'''
        data=self.grid
        
        fig = plt.figure()
        lat_max =  max(self.grid.node.lat)+1
        lat_min =  min(self.grid.node.lat)-1
        lon_max =  max(self.grid.node.lon)+1
        lon_min =  min(self.grid.node.lon)-1
        
        # Use stereographic projection
#        m = Basemap(width=3000000,height=2000000,\
#                      resolution='l',projection='stere',\
#                      lat_ts=50,lat_0=61,lon_0=15.)
        m = Basemap(resolution='l',projection='stere',\
                      lat_ts=50,lat_0=61,lon_0=15.0, \
                      llcrnrlon=lon_min, llcrnrlat=lat_min,\
                      urcrnrlon=lon_max ,urcrnrlat=lat_max,
                      )
 
        # Draw coastlines, meridians and parallels.
        m.drawcoastlines()
        m.drawcountries(zorder=0)
        m.fillcontinents(color='coral',lake_color='aqua',zorder=0)
        m.drawmapboundary(fill_color='aqua')
        m.drawparallels(np.arange(self._myround(lat_min,10,'floor'),
                                  self._myround(lat_max,10,'ceil'),10),
                        labels=[1,1,0,0])

        m.drawmeridians(np.arange(self._myround(lon_min,10,'floor'),
                                  self._myround(lon_max,10,'ceil'),10),
                        labels=[0,0,0,1])
        
        # Plot branches (as great circles)
        branch_lat1 = [data.node.lat[i] 
            for i in data.branch.node_fromIdx(data.node)]        
        branch_lon1 = [data.node.lon[i] 
            for i in data.branch.node_fromIdx(data.node)]        
        branch_lat2 = [data.node.lat[i] 
            for i in data.branch.node_toIdx(data.node)]        
        branch_lon2 = [data.node.lon[i] 
            for i in data.branch.node_toIdx(data.node)]

        for j in range(len(branch_lat1)):
            m.drawgreatcircle(branch_lon1[j],branch_lat1[j],\
                              branch_lon2[j],branch_lat2[j],\
                              linewidth=2,color='b')
            
        # Plot nodes
        x, y = m(data.node.lon,data.node.lat)
        m.scatter(x,y,marker='o',color='b')
        labels = data.node.name
        for label, xpt, ypt in zip(labels, x, y):
            plt.text(xpt, ypt, label)

        plt.title('Nodes and branches')
        plt.show()
        return



        
    def plotMapGrid(self,nodetype='',branchtype='',dcbranchtype='',
                    show_node_labels=False,latlon=None):
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
            # Note that sometimes the solver may return sensitivity = None
            # such values are converted to NAN in the asarray method
            avgprice = np.sqrt(
                np.average(np.asarray(res.sensitivityNodePower,dtype=float)**2,
                           axis=0)) #rms
            print ("Nodal prices: max=%g, min=%g" 
                %(max(avgprice),min(avgprice)) )
        

        if branchtype=='area':
            areas = data.node.area
            allareas = data.getAllAreas()
            colours_b = cm.prism(np.linspace(0, 1, len(allareas)))
        elif branchtype=='utilisation' or branchtype=='flow':
            numBranchCategories = 11
            colours_b = cm.jet(np.linspace(0, 1, numBranchCategories))
            #colours_ut[0]=(0.1, 0.1, 0.1, 1.0) #rgba
            avgflow = np.sqrt(np.average(np.asarray(res.branchFlow)**2,
                                         axis=0)) #rms
            cap = res.grid.branch.capacity
            utilisation = avgflow / cap # element-by-element
            print ("Branch utilisation: max=%g, min=%g" 
                %(max(utilisation),min(utilisation)) )
        elif branchtype=='sensitivity':
            numBranchCategories = 11
            colours_b = cm.jet(np.linspace(0, 1, numBranchCategories))
            avgsense = np.sqrt(
                np.average(
                np.asarray(res.sensitivityBranchCapacity,dtype=float)**2,
                axis=0)) #rms 
            print ("Branch capacity senitivity: max=%g, min=%g" 
                %(max(avgsense),min(avgsense)) )
        
        
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
                maxsense = max(avgsense)
                if j in res.idxConstrainedBranchCapacity:
                    idx = res.idxConstrainedBranchCapacity.index(j)
                    if not  np.isnan(avgsense[idx]):
                        category = math.floor(
                            avgsense[idx]/maxsense*(numBranchCategories-1))
                        #print "sense cat=",category
                        col = colours_b[category]
                        lwidth = 2
                    else:
                        #NAN sensitivity (not returned by solver)
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
                m.scatter(co_x,co_y,marker='o',color=col, zorder=2)
        elif nodetype == 'nodalprice':
            s=m.scatter(x,y,marker='o',c=avgprice, cmap=cm.jet, zorder=2)
            cb=m.colorbar(s)
            cb.set_label('Nodal price')
            #nodes with NAN nodal price plotted in gray:
            for i in xrange(len(avgprice)):
                if np.isnan(avgprice[i]):
                    m.scatter(x[i],y[i],c='dimgray',zorder=2)
            
        else:
            col='dimgray'
            m.scatter(x,y,marker='o',color=col, zorder=2)
            #cb=m.colorbar()
        
        
        # Show names of nodes
        if show_node_labels:
            labels = data.node.name
            x1,x2,y1,y2 = plt.axis()
            for label, xpt, ypt in zip(labels, x, y):
                if xpt > x1 and xpt < x2 and ypt > y1 and ypt < y2:
                    plt.text(xpt, ypt, label)
        
        plt.title('Nodes and branches')
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
