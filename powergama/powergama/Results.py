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
        
    def plotMapGrid_old(self):
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



        
    def plotMapGrid(self,nodetype='',branchtype='',\
                    show_node_labels=False,latlon=None):
        '''
        Plot results to map
        
        Parameters
        ----------
        nodetype (str) (default = "")
            "area", "nodalprice"
        branchtype (str) (default = "")
            "area", "utilisation", "flow", "sensitivity"
        show_node_labels (bool) (default=False)
            show node names (true/false)
        latlon (list) (default=None)
            map area [lat_min, lon_min, lat_max, lon_max]
        '''
        
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
        m.fillcontinents(color='coral',lake_color='white',zorder=0)
        m.drawmapboundary(fill_color='white')
        
        #TODO: remove hard-coding
        m.drawparallels(np.arange(10,70,10),labels=[1,1,0,0])
        m.drawmeridians(np.arange(-30,50,10),labels=[0,0,0,1])
        
        
        if nodetype=='area':
            areas = data.node.area
            allareas = data.getAllAreas()
            colours_co = cm.prism(np.linspace(0, 1, len(allareas)))
        elif nodetype=='nodalprice':
            avgprice = np.sqrt(np.average(np.asarray(res.sensitivityNodePower)**2,axis=0)) #rms
            print "Nodal prices: max=%g, min=%g" %(max(avgprice),min(avgprice))
        

        if branchtype=='area':
            areas = data.node.area
            allareas = data.getAllAreas()
            colours_b = cm.prism(np.linspace(0, 1, len(allareas)))
        elif branchtype=='utilisation' or branchtype=='flow':
            numBranchCategories = 11
            colours_b = cm.jet(np.linspace(0, 1, numBranchCategories))
            #colours_ut[0]=(0.1, 0.1, 0.1, 1.0) #rgba
            avgflow = np.sqrt(np.average(np.asarray(res.branchFlow)**2,axis=0)) #rms
            cap = res.grid.branch.capacity
            utilisation = avgflow / cap # element-by-element
            print "Branch utilisation: max=%g, min=%g" %(max(utilisation),min(utilisation))
        elif branchtype=='sensitivity':
            numBranchCategories = 11
            colours_b = cm.jet(np.linspace(0, 1, numBranchCategories))
            avgsense = np.sqrt(np.average(np.asarray(res.sensitivityBranchCapacity)**2,axis=0)) #rms 
            print "Branch capacity senitivity: max=%g, min=%g" %(max(avgsense),min(avgsense))
        
        
        # Plot branches (as great circles)
        idx_from = data.branch.node_fromIdx(data.node)
        idx_to = data.branch.node_toIdx(data.node)
        branch_lat1 = [data.node.lat[i] for i in idx_from]        
        branch_lon1 = [data.node.lon[i] for i in idx_from]        
        branch_lat2 = [data.node.lat[i] for i in idx_to]        
        branch_lon2 = [data.node.lon[i] for i in idx_to]
        for j in range(len(branch_lat1)):
            if branchtype=='area':
                if areas[idx_from[j]] == areas[idx_to[j]]:
                    col = colours_co[allareas.index(areas[idx_from[j]])]
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
                category = math.floor(avgflow[j]/maxflow*(numBranchCategories-1))
                col = colours_b[category]
                lwidth = 1
                if category*2 > numBranchCategories:
                    lwidth = 2
            elif branchtype=='sensitivity':
                maxsense = max(avgsense)
                if j in res.idxConstrainedBranchCapacity:
                    idx = res.idxConstrainedBranchCapacity.index(j)
                    category = math.floor(avgsense[idx]/maxsense*(numBranchCategories-1))
                    #print "sense cat=",category
                    col = colours_b[category]
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

            
        # Plot nodes
        x, y = m(data.node.lon,data.node.lat)
        if nodetype == 'area':
            for co in range(len(allareas)):
                co_nodes = [i for i, v in enumerate(data.node.area) if v==allareas[co]]
                co_x = [x[i] for i in co_nodes]
                co_y = [y[i] for i in co_nodes]
                col = colours_co[co]
                m.scatter(co_x,co_y,marker='o',color=col, zorder=2)
        elif nodetype == 'nodalprice':
            s=m.scatter(x,y,marker='o',c=avgprice, cmap=cm.jet, zorder=2)
            cb=m.colorbar(s)
            cb.set_label('Nodal price')
        else:
            col='dimgray'
            m.scatter(x,y,marker='o',color=col, zorder=2)
            #cb=m.colorbar()
        
        
        # Show names of nodes
        if show_node_labels:
            labels = data.node.name
            for label, xpt, ypt in zip(labels, x, y):
                plt.text(xpt, ypt, label)
        
        plt.title('Nodes and branches')
        plt.show()

        return
        # End plotGridMap