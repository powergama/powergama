# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 16:19:07 2013

@author: hsven
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap

class Results():
    '''
    Class for storing and analysing/presenting results from PowerGAMA
    '''    
    
    
    def __init__(self,grid,timerange):
        self.grid = grid
        self.timerange = timerange
        self.storage_idx_generators = grid.getIdxGeneratorsWithStorage()
        self.objectiveFunctionValue=[]    
        self.generatorOutput=[]
        self.branchFlow=[]
        self.nodeAngle=[]
        self.sensitivityBranchCapacity=[]
        self.sensitivityNodePower=[]
        self.storage=[]
        self.marginalprice=[]
        self.inflowSpilled=[]
        self.loadshed=[]
        self.idxConstrainedBranchCapacity = grid.getIdxBranchesWithFlowConstraints()
        
    def addResultsFromTimestep(self,objective_function,generator_power,
                               branch_power,node_angle,sensitivity_branch_capacity,
                               sensitivity_node_power,
                               storage,inflow_spilled,loadshed_power,marginalprice):
        '''Store results from optimal power flow for a new timestep'''
                                   
        self.objectiveFunctionValue.append(objective_function)
        self.generatorOutput.append(generator_power)
        self.branchFlow.append(branch_power)
        self.nodeAngle.append(node_angle)
        self.sensitivityBranchCapacity.append(sensitivity_branch_capacity)
        self.sensitivityNodePower.append(sensitivity_node_power)
        #self.demandPower.append(demand_power)
        self.storage.append(storage)
        self.inflowSpilled.append(inflow_spilled)
        self.loadshed.append(loadshed_power)
        self.marginalprice.append(marginalprice)
        
        # self.storageGeneratorsIdx.append(idx_generatorsWithStorage)
    
    def plotStorageFilling(self):
        '''Show storage filling level (MWh) for generators with storage'''
        if len(self.storage_idx_generators) > 0:
            plt.plot(self.timerange,self.storage)
        else:
            print "There are no generators with storage in this system"
        return
        
    
    def plotGeneratorOutput(self):
        '''Show power output of all generators'''
        plt.plot(self.timerange,self.generatorOutput)
        plt.legend(self.grid.generator.node , loc="upper right")
        return

    def plotGenerationPerArea(self,area):
        '''Show generation per area '''
        generators = self.grid.getGeneratorsPerAreaAndType()
        for gentype in generators[area].keys():
            idxGen = generators[area][gentype]
            gen = zip(*self.generatorOutput) #transpose
            mygen = [gen[i] for i in idxGen]
            sumGenAreaType = [sum(x) for x in zip(*mygen) ]
            plt.plot(self.timerange,sumGenAreaType)
            
        plt.legend(generators[area].keys() , loc="upper right")
        plt.title("Generation in %s"%(area))

        return

    def plotDemandPerArea(self,areas):
        '''Show demand per area(s) '''
        
        consumer = self.grid.consumer
        if type(areas) is str:
            areas = [areas]
        for co in areas:
            dem=[0]*len(self.timerange)
            consumers = self.grid.getConsumersPerArea()[co]
            for i in consumers:
                ref_profile = consumer.load_profile[i]
                # accumulate demand for all consumers in this area:
                dem = [dem[t] + consumer.load[i] * self.grid.demandProfiles[ref_profile][t] for t in self.timerange]
            plt.plot(self.timerange,dem)
            
        plt.legend(areas , loc="upper right")
        plt.title("Power demand")

        return
    
    def plotMarginalPrice(self):
        '''Show marginal price at each node'''
        plt.plot(self.timerange,self.marginalprice)
        plt.legend(self.grid.generator.node , loc="upper right")
        return
        
    def plotMapGrid(self):
        '''Show a map with nodes and branches'''
        data=self.grid
        
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
        m.drawparallels(np.arange(10,70,10),labels=[1,1,0,0])
        m.drawmeridians(np.arange(-30,50,10),labels=[0,0,0,1])
        
        # Plot branches (as great circles)
        branch_lat1 = [data.node.lat[i] for i in data.branch.node_fromIdx(data.node)]        
        branch_lon1 = [data.node.lon[i] for i in data.branch.node_fromIdx(data.node)]        
        branch_lat2 = [data.node.lat[i] for i in data.branch.node_toIdx(data.node)]        
        branch_lon2 = [data.node.lon[i] for i in data.branch.node_toIdx(data.node)]
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
        
