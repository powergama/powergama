# -*- coding: utf-8 -*-
'''
Module containing PowerGAMA GridData class and sub-classes

Grid data and time-dependent profiles
'''

import pandas as pd
import numpy
from scipy.sparse import csr_matrix as sparse
import math                                 # Used in myround
import matplotlib.pyplot as plt             
import matplotlib as mpl                    
from mpl_toolkits.basemap import Basemap    
from collections import OrderedDict         #plotRelativeLoadDistribution



##=============================================================================


    
class GridData(object):
    '''
    Class for grid data storage and import
    '''        
    
    # Headers and default values in input files:
    # default=None: column _must_ be present in input file
    _keys_node = {'id':None, 'area':None, 'lat':None,'lon':None}
    _keys_branch = {'node_from':None,'node_to':None,
                    'reactance':None,'capacity':None}
    _keys_dcbranch = {'node_from':None, 'node_to':None, 'capacity':None}
    _keys_generator = {'type':None,'desc':'', 'node':None,
                       'pmax':None,'pmin':None,'fuelcost':None,
                       'inflow_fac':None,'inflow_ref':None,
                       'storage_cap':0,'storage_price':0,'storage_ini':0,
                       'storval_filling_ref':'','storval_time_ref':'',
                       'pump_cap':0,'pump_efficiency':0,'pump_deadband':0}
    _keys_consumer = {'node':None, 'demand_avg':None,'demand_ref':None,
                      'flex_fraction':0, 'flex_on_off':0, 'flex_basevalue':0,
                      'flex_storage':0, 'flex_storval_filling':'',
                      'flex_storval_time':'', 'flex_storagelevel_init':0.5}

    # Required fields for investment analysis input data    
    keys_sipdata = {
        'node': ['id', 'lat', 'lon','offshore','type',
                 'existing','cost_scaling'],
        'branch': ['node_from','node_to','capacity',
                   'expand','distance','cost_scaling',
                   'type'],
        'generator': ['node','pmax','pmin',
                      'fuelcost','fuelcost_ref','pavg',
                      'inflow_fac','inflow_ref'],
        'consumer': ['node', 'demand_avg', 'demand_ref']
        }
        

    def __init__(self):
        '''
        Create GridData object with data and methods for import and 
        processing of PowerGAMA grid data            
        '''
        self.node = None
        self.branch = None
        self.dcbranch = None
        self.generator = None
        self.consumer = None
        #self.inflowProfiles = None
        #self.demandProfiles = None
        self.profiles = None
        self.storagevalue_filling = None
        self.storagevalue_time = None
        self.timeDelta = None
        self.timerange = None
        
        self.CSV_SEPARATOR = None

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
            
    def readGridData(self,nodes,ac_branches,dc_branches,generators,consumers):
        '''Read grid data from files into data variables'''
        
        self.node = pd.read_csv(nodes)
        self.node.set_index('id',inplace=True,append=True)
        self.branch = pd.read_csv(ac_branches)
        if not dc_branches is None:
            self.dcbranch = pd.read_csv(dc_branches)
        else:
            self.dcbranch = pd.DataFrame(columns=self._keys_dcbranch.keys())
        self.generator = pd.read_csv(generators)
        self.consumer = pd.read_csv(consumers)

        for k,v in self._keys_node.items():
            if v==None and not k in self.node:
                raise Exception("Node input file must contain %s" %k)
        for k,v in self._keys_branch.items():
            if v==None and not k in self.branch:
                raise Exception("Branch input file must contain %s" %k)
        for k,v in self._keys_dcbranch.items():
            if v==None and not k in self.dcbranch:
                raise Exception("DC branch input file must contain %s" %k)
        for k,v in self._keys_generator.items():
            if v==None and not k in self.generator:
                raise Exception("Generator input file must contain %s" %k)
        for k,v in self._keys_consumer.items():
            if v==None and not k in self.consumer:
                raise Exception("Consumer input file must contain %s" %k)
        

        self._checkGridData()
        self._addDefaultColumns()
        self._fillEmptyCells()
        
    def readSipData(self,nodes,branches,generators,consumers):
        '''Read grid data for investment analysis from files (PowerGIM)

        This is used with the grid investment module (PowerGIM)   
        
        time-series data may be used for 
        consumer demand
        generator inflow (e.g. solar and wind)
        generator fuelcost (e.g. one generator with fuelcost = power price)
        '''
        self.node = pd.read_csv(nodes,
                                usecols=self.keys_sipdata['node'],
                                dtype={'id':str})
        self.node.set_index('id',inplace=True)
        self.node['id']=self.node.index
        self.branch = pd.read_csv(branches,
                                  usecols=self.keys_sipdata['branch'],
                                  dtype={'node_from':str,'node_to':str})
        # dcbranch variable only needed for powergama.plotMapGrid
        self.dcbranch = pd.DataFrame()
        self.generator = pd.read_csv(generators,
                                     usecols=self.keys_sipdata['generator'],
                                     dtype={'node':str})
        self.consumer = pd.read_csv(consumers,
                                    usecols=self.keys_sipdata['consumer'],
                                    dtype={'node':str})
        
        # treat node id as string (even if numeric)                            
        #self.node['id'] = self.node['id'].astype(str)
        #self.generator['node'] = self.generator['node'].astype(str)
        #self.branch['node_from'] = self.branch['node_from'].astype(str)
        #self.branch['node_to'] = self.branch['node_to'].astype(str)
        #self.consumer['node'] = self.consumer['node'].astype(str)
    
        self._checkGridData()



    def _fillEmptyCells(self):
        '''Use default data where none is given'''
        #generators:
        for col,val in self._keys_generator.items():
            if val != None:
                self.generator[col] = self.generator[col].fillna(
                    self._keys_generator[col])
        #consumers:
        for col,val in self._keys_consumer.items():
            if val != None:
                self.consumer[col] = self.consumer[col].fillna(
                    self._keys_consumer[col])
        
        
    def _addDefaultColumns(self):
        '''insert optional columns with default values when none
        are provided in input files'''
        for k in self._keys_generator:
            if k not in self.generator.keys():
                self.generator[k] = self._keys_generator[k]
        for k in self._keys_consumer:
            if k not in self.consumer.keys():
                self.consumer[k] = self._keys_consumer[k]
           
        
    def _checkGridData(self):
        '''Check consistency of grid data'''
                    
        #generator nodes
        for g in self.generator['node']:
            if not g in self.node['id'].values:
                raise Exception("Generator node does not exist: %s" %g)
        #consumer nodes
        for c in self.consumer['node']:
            if not c in self.node['id'].values:
                raise Exception("Consumer node does not exist: %s" %c)
                

    def _readProfileFromFile(self,filename,timerange):          
        profiles = pd.read_csv(filename,sep=self.CSV_SEPARATOR)
        profiles = profiles.ix[timerange]
        profiles.index = range(len(timerange))
        return profiles

    def _readStoragevaluesFromFile(self,filename):  
        profiles = pd.read_csv(filename,sep=self.CSV_SEPARATOR)
        return profiles
        
        
    def readProfileData(self,filename,timerange,
                        storagevalue_filling=None,
                        storagevalue_time=None,
                        timedelta=1.0):
        """Read profile (timeseries) into numpy arrays"""
        
        #self.inflowProfiles = self._readProfileFromFile(inflow,timerange)
        #self.demandProfiles = self._readProfileFromFile(demand,timerange)
        self.profiles = self._readProfileFromFile(filename,timerange)
        self.timerange = timerange
        self.timeDelta = timedelta
        
        '''
        Storage values have both time dependence and filling level dependence
       
       The dependence is on filling level (0-100%), is given as an array
        with 101 elements
        '''
        if not storagevalue_filling is None:
            self.storagevalue_time = self._readProfileFromFile(
                storagevalue_time,timerange)
            self.storagevalue_filling = self._readStoragevaluesFromFile(
                storagevalue_filling)        
        return  
    
    
    def _sample_kmeans(self, X, samplesize):
        """
        K-means is a centroid based algorithm. It clusters datapoints
        into N centroids. The importance/weight of each time series is 
        curretly only based on its absoulute values.
        
        Input:
        Weighted time series wrt max capacity.
        
        Output:
        New clustered time series with e.g. N=200 time steps
        (OR time step index to closest original?)
        """
        import time
        import numpy as np
        import pandas as pd
        from sklearn.cluster import KMeans
        
        k_means = KMeans(init='k-means++', n_clusters=samplesize, n_init=10)
        t0 = time.time()
        k_means.fit(X)
        t_batch = time.time() - t0
        print("\n Calculation time for kmeans: ", t_batch, " seconds \n")
        k_means_labels = k_means.labels_    # which cluster nr it belongs to
        k_means_cluster_centers = pd.DataFrame(k_means.cluster_centers_)
        k_means_cluster_centers.columns = X.columns.copy()
        k_means_labels_unique, X_idx = np.unique(k_means_labels, return_index=True)        

        return k_means_cluster_centers, X_idx
        
    def _sample_mmatching(self, X, samplesize):
        """
        The idea is to make e.g. 10000 randomsample-sets of size=samplesize 
        from the originial datasat X. 
        
        Choose the sampleset with the lowest objective:
        MINIMIZE [(meanSample - meanX)^2 + (stdvSample - stdvX)^2...]
        
        in terms of stitistical measures
        
        TODO: finish this algorithm
        """
        
        return
    
    def _sample_meanshift(self, X, samplesize):
        """
        It is a centroid-based algorithm, which works by updating candidates 
        for centroids to be the mean of the points within a given region. 
        These candidates are then filtered in a post-processing stage to 
        eliminate near-duplicates to form the final set of centroids.
        
        TODO: finish this algorithm
        """
        import numpy as np
        from sklearn.cluster import MeanShift, estimate_bandwidth
    
        # The following bandwidth can be automatically detected using
        bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=samplesize)
        
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(X)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        
        print("number of estimated clusters : %d" % n_clusters_)
        
        return cluster_centers
    
    def _sample_latinhypercube(self, X, samplesize):
        """
        lhs(n, [samples, criterion, iterations])
    
        n:          an integer that designates the number of factors (required)
        samples:    an integer that designates the number of sample points to generate 
                    for each factor (default: n) 
        criterion:  a string that tells lhs how to sample the points (default: None,  
                    which simply randomizes the points within the intervals):
                    “center” or “c”: center the points within the sampling intervals
                    “maximin” or “m”: maximize the minimum distance between points, but place 
                              the point in a randomized location within its interval
                    “centermaximin” or “cm”: same as “maximin”, but centered within the intervals
                    “correlation” or “corr”: minimize the maximum correlation coefficient
                    
        TODO: finish this algorithm. May be necessary to fit Kernel Density and not normal distr?
        """
        from pyDOE import lhs
        from scipy.stats.distributions import norm
        X_rows = X.shape[0]; X_cols = X.shape[1]
        X_mean = X.mean(); X_std = X.std()
        lhX = lhs( X_cols , samples=samplesize , criterion='center' )
        kernelDensity=False
        if kernelDensity:
            # Fit data with kernel density
            from sklearn.neighbors.kde import KernelDensity
            kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
            kde.score_samples(X)
            kde_sample = kde.sample(samplesize)     # random sampling (TODO: fit to latin hypercube sample)
        else:
            # Fit data with normal distribution
            for i in range(X_rows):
                lhX[:,i] = norm(loc=X_mean[i] , scale=X_std[i]).ppf(lhX[:,i])
        return lhX


    def sampleProfileData(self, sampling_method, samplesize ):
        """
        Collect time series in absoulute values/magnitude and use
        a proper sampling/clustering technique.
        
        kmeans, mmatching, latinhypercube
        
        Returns:
        grid_data.timerange (or grid_data.profiles?)
        """
        
        # Multiply time series for load and VRES with their respective 
        # maximum capacities in order to get the correct clusters/samples
        
        X = self.profiles.copy()
        for k,row in self.generator.iterrows():
            pmax = row['pmax']
            ref = row['inflow_ref']
            X[ref] = self.profiles[ref] * pmax
        
        for k,row in self.consumer.iterrows():
            pmax = row['demand_avg']
            ref = row['demand_ref']
            X[ref] = self.profiles[ref] * pmax
    
        
        # Sampling and clustering methods.         
        if sampling_method == "kmeans":
            print("Using k-means...")
            centroids, X_idx = self._sample_kmeans(X, samplesize)
            self.timerange = list(X_idx)    # OBS: more or less random
            timedelta = 1.0
            # convert back to relative values
            for k,row in self.generator.iterrows():
                pmax = row['pmax']
                ref = row['inflow_ref']
                if pmax == 0:
                    centroids[ref] = 0
                else:
                    centroids[ref] = centroids[ref] / pmax
            for k,row in self.consumer.iterrows():
                pmax = row['demand_avg']
                ref = row['demand_ref']
                if pmax == 0:
                    centroids[ref] = 0
                else:
                    centroids[ref] = centroids[ref] / pmax
            self.profiles = centroids
            self.profiles['const'] = 1
            #self.readProfileData(filename = "data/profiles.csv",
            #                 timerange = timerange,
            #                 timedelta = timedelta)
        elif sampling_method == "mmatching":
            print("Using moment matching...")
        elif sampling_method == "meanshift":
            print("Using Mean-Shift...")
            X_sample = self._sample_meanshift(X, samplesize)
        elif sampling_method == "latinhyper":
            print("Using Latin-Hypercube...")
            X_sample = self._sample_latinhypercube(X, samplesize)
        else:
            print("Using random sampling (consider changing sampling method!)...")
            import random
            random.seed('irpwind')
            timerange = random.sample(range(8760),samplesize)
            timedelta = 1.0
            self.readProfileData(filename = "data/profiles.csv",
                             timerange = timerange,
                             timedelta = timedelta)
                             
        return

    def writeGridDataToFiles(self,prefix):
        '''
        Save data to new input files
        ''' 

        file_nodes = prefix+"nodes.csv"
        file_branches = prefix+"branches.csv"
        file_consumers = prefix+"consumers.csv"     
        file_generators = prefix+"generators.csv"       
        file_hvdc = prefix+"hvdc.csv"       

        print('TODO: Not implemented using pandas yet')
        # OLD CODE:
        self.node.to_csv(file_nodes,sep=self.CSV_SEPARATOR)
        self.node.writeToFile(file_nodes)
        self.branch.writeToFile(file_branches)
        self.consumer.writeToFile(file_consumers)
        self.generator.writeToFile(file_generators)
        self.dcbranch.writeToFile(file_hvdc)
        
        return
        
        
    def numConsumers(self):
        return self.consumer.shape[0]
    
    def numGenerators(self):
        return self.generator.shape[0]
        
    def numNodes(self):
        return self.node.shape[0]
    
    def numBranches(self):
        return self.branch.shape[0]
    
    def numDcBranches(self):
        if not self.dcbranch is None:
            return self.dcbranch.shape[0]
        else:
            return 0

    def branchFromNodeIdx(self):
        """get node indices for branch FROM node"""
        #return [self.node[self.node['id']==self.branch['node_from'][k]]
        #        .index.tolist()[0] for k in range(self.numBranches())]
        return [self.node[self.node['id']==b['node_from']]['id'].tolist()[0]
                for i,b in self.branch.iterrows()]

    def branchToNodeIdx(self):
        """get node indices for branch FROM node"""
        return [self.node[self.node['id']==self.branch['node_to'][k]] 
                .index.tolist()[0] for k in self.branch.index.tolist()]
    
    def dcBranchFromNodeIdx(self):
        """get node indices for dc branch FROM node"""
        return [self.node[self.node['id']==self.dcbranch['node_from'][k]]
                .index.tolist()[0] for k in self.dcbranch.index.tolist()]

    def dcBranchToNodeIdx(self):
        """get node indices for dc branch FROM node"""
        return [self.node[self.node['id']==self.dcbranch['node_to'][k]] 
                .index.tolist()[0] for k in self.dcbranch.index.tolist()]
    
    
    def getGeneratorsAtNode(self,nodeIdx):
        """Indices of all generators attached to a particular node"""
        indices = [i for i, x in enumerate(self.generator['node']) 
                    if x == self.node['id'][nodeIdx]]
        return indices
        
    def getGeneratorsWithPumpAtNode(self,nodeIdx):
        """Indices of all pumps attached to a particular node"""
        indices = [i for i, x in enumerate(self.generator['node']) 
                    if x == self.node['id'][nodeIdx]
                    and self.generator['pump_cap'][i]>0]
        return indices
        
    def getLoadsAtNode(self,nodeIdx):
        """Indices of all loads (consumers) attached to a particular node"""
        indices = [i for i, x in enumerate(self.consumer['node']) 
                    if x == self.node['id'][nodeIdx]]
        return indices

    def getLoadsFlexibleAtNode(self,nodeIdx):
        """Indices of all flexible nodes attached to a particular node"""
        indices = [i for i, x in enumerate(self.consumer['node']) 
                    if x == self.node['id'][nodeIdx]
                    and self.consumer['flex_fraction'][i]>0
                    and self.consumer['demand_avg'][i]>0]
        return indices
        
    def getIdxConsumersWithFlexibleLoad(self):
        """Indices of all consumers with flexible load"""
        idx = [i for i,v in enumerate(self.consumer['flex_fraction']) 
            if v>0 and v<numpy.inf and self.consumer['demand_avg'][i]>0]
        return idx
        
    def getFlexibleLoadStorageCapacity(self,consumer_indx):
        ''' flexible load storage capacity in MWh'''
        cap = (self.consumer['demand_avg'][consumer_indx] 
                * self.consumer['flex_fraction'][consumer_indx] 
                * self.consumer['flex_storage'][consumer_indx] )
        return cap
   

    def getDcBranchesAtNode(self,nodeIdx,direction):
        """Indices of all DC branches attached to a particular node"""
        if direction=='from':
            indices = [i for i, x in enumerate(self.dcbranch['node_from']) 
            if x == self.node['id'][nodeIdx]]
        elif direction=='to':
            indices = [i for i, x in enumerate(self.dcbranch['node_to']) 
            if x == self.node['id'][nodeIdx]]
        else:
            raise Exception("Unknown direction in GridData.getDcBranchesAtNode")
        return indices


    def getDcBranches(self):
        '''
        Returns a list with DC branches in the format
        [index,from area,to area]
        '''
        hvdcBranches = []
        for idx in self.dcbranch.index.tolist():
            fromNodeIdx = self.node['id'].index(self.dcbranch['node_from'][idx])
            toNodeIdx = self.node.name.index(self.dcbranch['node_to'][idx])
            areaFrom = self.node['area'][fromNodeIdx]
            areaTo = self.node['area'][toNodeIdx]
            hvdcBranches.append([idx,areaFrom,areaTo])
        return hvdcBranches    

    
    def getIdxNodesWithLoad(self):
        """Indices of nodes that have load (consumer) attached to them"""        
        # Get index of node associated with all consumer        
        indices = numpy.asarray(self.consumer.nodeIdx(self.node))
        # Return indices only once (unique values)
        indices = numpy.unique(indices)
        return indices
        
        
    def getIdxGeneratorsWithStorage(self):
        """Indices of all generators with nonzero and non-infinite storage"""
        idx = [i for i,v in enumerate(self.generator['storage_cap']) 
            if v>0 and v<numpy.inf]
        return idx
        
    def getIdxGeneratorsWithNonzeroInflow(self):
        """Indices of all generators with nonzero inflow"""
        idx = [i for i,v in enumerate(self.generator['inflow_fac']) 
            if v>0]
        return idx

    def getIdxGeneratorsWithPumping(self):
        """Indices of all generators with pumping capacity"""
        idx = [i for i,v in enumerate(self.generator['pump_cap']) 
            if v>0 and v<numpy.inf]
        return idx
        
    def getIdxBranchesWithFlowConstraints(self):
        '''Indices of branches with less than infinite branch capacity'''
        idx = [i for i,v in enumerate(self.branch['capacity']) if v<numpy.inf]
        return idx
        
    def getIdxDcBranchesWithFlowConstraints(self):
        '''Indices of DC branches with less than infinite branch capacity'''
        if self.dcbranch is None:
            idx = []
        else:
            idx = [i for i,v in enumerate(self.dcbranch['capacity']) 
                        if v<numpy.inf]
        return idx


    def _susceptancePu(self,baseOhm):
        return [-1/self.branch['reactance'][i]*baseOhm 
                for i in self.branch.index.tolist()]

    def computePowerFlowMatrices(self,baseZ):
        """
        Compute and return dc power flow matrices B' and DA
                
        Returns sparse matrices (csr - compressed sparse row matrix)              
        """
        # node-branch incidence matrix
        # element b,n is  1 if branch b starts at node n
        #                -1 if branch b ends at node n
        num_nodes = self.numNodes()
        num_branches = self.numBranches()
        
        fromIdx = self.branchFromNodeIdx()
        toIdx = self.branchToNodeIdx()
        data = numpy.r_[numpy.ones(num_branches),-numpy.ones(num_branches)]
        row = numpy.r_[range(num_branches),range(num_branches)]
        col = numpy.r_[fromIdx, toIdx]
        A_incidence_matrix = sparse( (data, (row,col)),(num_branches,num_nodes))
        
        # Diagonal matrix
        b = numpy.asarray(self._susceptancePu(baseZ))
        D = sparse(numpy.eye(num_branches)*b*(-1))
        DA = D*A_incidence_matrix
        
        # Bprime matrix
        ## build Bf such that Bf * Va is the vector of real branch powers injected
        ## at each branch's "from" bus
        Bf = sparse((numpy.r_[b, -b],(row, numpy.r_[fromIdx, toIdx])))
        Bprime = A_incidence_matrix.T * Bf
        
        return Bprime, DA

    
    def getAllAreas(self):
        '''Return list of areas included in the grid model'''
        areas = self.node['area']
        allareas = []
        for co in areas:
            if co not in allareas:
                allareas.append(co)
        return allareas
        
    def getAllGeneratorTypes(self):
        '''Return list of generator types included in the grid model'''
        gentypes = self.generator['gentype']
        alltypes = []
        for ge in gentypes:
            if ge not in alltypes:
                alltypes.append(ge)
        return alltypes
    
    def getConsumerAreas(self):
        """List of areas for each consumer"""
        areas = [self.node['area'][self.node['id']==n].tolist()[0]
                    for n in self.consumer['node']]
        return areas
    
    def getGeneratorAreas(self):
        """List of areas for each generator"""
        areas = [self.node['area'][self.node['id']==n].tolist()[0]
                    for n in self.generator['node']]
        return areas
    
    def getConsumersPerArea(self):
        '''Returns dictionary with indices of loads within each area'''
        consumers = {}
        consumer_areas = self.getConsumerAreas()
        for idx_load in self.consumer.index.tolist():
            area_name = consumer_areas[idx_load]
            if area_name in consumers:
                consumers[area_name].append(idx_load)
            else:
                consumers[area_name] =  [idx_load]   
        return consumers
   
    def getGeneratorsPerAreaAndType(self): 
        '''Returns dictionary with indices of generators within each area'''
        generators = {}
        generator_areas = self.getGeneratorAreas()
        for idx_gen in self.generator.index.tolist():
            gtype = self.generator['type'][idx_gen]
            area_name = generator_areas[idx_gen]
            if area_name in generators:
                if gtype in generators[area_name]:
                    generators[area_name][gtype].append(idx_gen)
                else:
                    generators[area_name][gtype] = [idx_gen]
            else:
                generators[area_name] = {gtype:[idx_gen]}
        return generators

    def getGeneratorsPerType(self): 
        '''Returns dictionary with indices of generators per type'''
        generators = {}
        for idx_gen in self.generator.index.tolist():
            gtype = self.generator['type'][idx_gen]
            if gtype in generators:
                generators[gtype].append(idx_gen)
            else:
                generators[gtype] = [idx_gen]
        return generators


    def getGeneratorsWithPumpByArea(self):
        '''
        Returns dictionary with indices of generators with pumps within
        each area
        '''
        generators = {}
        for pumpIdx,cap in enumerate(self.generator['pump_cap']):
            if cap>0 and cap<numpy.inf:
                nodeName = self.generator['node'][pumpIdx]
                nodeIdx = self.node['id'].index(nodeName)
                areaName = self.node['area'][nodeIdx]
                if areaName in generators:
                    generators[areaName].append(pumpIdx)
                else:
                    generators[areaName] = [pumpIdx]
        return generators


    def getInterAreaBranches(self,area_from=None,area_to=None,acdc='ac'):
        '''
        Get indices of branches from and/or to specified area(s)
        
        area_from = area from. Use None (default) to leave unspecifie
        area_to= area to. Use None (default) to leave unspecified
        acdc = 'ac' (default) for ac branches, 'dc' for dc branches        
        '''
        
        if area_from is None and area_to is None:
            raise Exception("Either from area or to area (or both) has"
                            +"to be specified)")
                            
        # indices of from and to nodes of all branches:
        if acdc=='ac':
            br_from_nodes = self.branchFromNodeIdx()
            br_to_nodes = self.branchToNodeIdx()
        elif acdc=='dc':
            br_from_nodes = self.dcBranchFromNodeIdx()
            br_to_nodes = self.dcBranchToNodeIdx()
        else:
            raise Exception('Branch type must be "ac" or "dc"')
        
        
        br_from_area = [self.node.area[i] for i in br_from_nodes]
        br_to_area = [self.node.area[i] for i in br_to_nodes]
        
        # indices of all inter-area branches (from area != to area)        
        br_is_interarea = [i for i in range(len(br_from_area)) 
                                if br_from_area[i] != br_to_area[i]]
        
        # branches connected to area_from
        fromArea_branches_pos = [i for i in br_is_interarea
                                 if br_from_area[i]==area_from]
        fromArea_branches_neg = [i for i in br_is_interarea
                                 if br_to_area[i]==area_from]

        # branches connected to area_to
        toArea_branches_pos = [i for i in br_is_interarea
                                 if br_to_area[i]==area_to]
        toArea_branches_neg = [i for i in br_is_interarea
                                 if br_from_area[i]==area_to]

        if area_from is None:
            # Only to node has been specified
            branches_pos = toArea_branches_pos
            branches_neg = toArea_branches_neg
        elif area_to is None:
            # Only from node has been specified
            branches_pos = fromArea_branches_pos
            branches_neg = fromArea_branches_neg
        else:
            # Both to and from area has been specified
            branches_pos = [b for b in fromArea_branches_pos 
                                    if b in toArea_branches_neg ]
            branches_neg = [b for b in fromArea_branches_neg 
                                    if b in toArea_branches_pos ]
        return dict(branches_pos=branches_pos,
                    branches_neg=branches_neg)   

    def branchDistances(self,R=6373.0):
        '''computes branch distance from node coordinates, resuls in km
        
        Uses haversine formula
        
        Parameters
        ----------
        R : radius of the Earth
        '''
        
        # approximate radius of earth in km
        n_from = self.branchFromNodeIdx()
        n_to = self.branchToNodeIdx()
        distance = []
        for i,br in self.branch.iterrows():
            n_from= br['node_from']
            n_to=br['node_to']
            lat1 = math.radians(self.node.ix[n_from]['lat'])
            lon1 = math.radians(self.node.ix[n_from]['lon'])
            lat2 = math.radians(self.node.ix[n_to]['lat'])
            lon2 = math.radians(self.node.ix[n_to]['lon'])
        
            dlon = lon2 - lon1
            dlat = lat2 - lat1
        
            a = (math.sin(dlat/2)**2 
                + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2 )
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            #atan2 better than asin: c = 2 * math.asin(math.sqrt(a))
            distance.append(R * c)
        return distance

    def plotRelativeLoadDistribution(self,show_node_labels=False,latlon=None,
                                     dotsize=40,draw_par_mer=False,
                                     colours=True,showTitle=True):
        '''
        Plots the relative input load distribution.
        
        Parameters
        ----------
        show_node_labels : boolean
            whether to show node names (true/false)
        latlon : list of four floats
            map area [lat_min, lon_min, lat_max, lon_max]
        draw_par_mer : boolean
            whether to draw parallels and meridians on map
        colours : boolean
            whether to draw the map in colours or black and white
            
        '''
        
        if latlon is None:
            lat_max =  max(self.node.lat)+1
            lat_min =  min(self.node.lat)-1
            lon_max =  max(self.node.lon)+1
            lon_min =  min(self.node.lon)-1
        else:
            lat_min = latlon[0]
            lon_min = latlon[1]
            lat_max = latlon[2]
            lon_max = latlon[3]
        
        # Use the average latitude as latitude of true scale
        lat_truescale = numpy.mean(self.node.lat)
                
        m = Basemap(resolution='l',projection='merc',\
                      lat_ts=lat_truescale, \
                      llcrnrlon=lon_min, llcrnrlat=lat_min,\
                      urcrnrlon=lon_max ,urcrnrlat=lat_max, \
                      anchor='W')
        
        # Draw coastlines, meridians and parallels.
        m.drawcoastlines()
        m.drawcountries(zorder=0)
        if colours:
            m.fillcontinents(color='coral',lake_color='aqua',zorder=0)
            m.drawmapboundary(fill_color='aqua')
        else:
            m.fillcontinents(zorder=0)
            m.drawmapboundary()
        
        if draw_par_mer:
            m.drawparallels(numpy.arange(_myround(lat_min,10,'floor'),
                _myround(lat_max,10,'ceil'),10),
                labels=[1,1,0,0])

            m.drawmeridians(numpy.arange(_myround(lon_min,10,'floor'),
                _myround(lon_max,10,'ceil'),10),
                labels=[0,0,0,1])
        
        #AC Branches
        idx_from = self.branch.node_fromIdx(self.node)
        idx_to = self.branch.node_toIdx(self.node)
        branch_lat1 = [self.node.lat[i] for i in idx_from]        
        branch_lon1 = [self.node.lon[i] for i in idx_from]        
        branch_lat2 = [self.node.lat[i] for i in idx_to]        
        branch_lon2 = [self.node.lon[i] for i in idx_to]
        
        x1, y1 = m(branch_lon1,branch_lat1)
        x2, y2 = m(branch_lon2,branch_lat2)
        
        ls = [[(x1[i],y1[i]),(x2[i],y2[i])] for i in range(len(x1))]
        line_segments_ac = mpl.collections.LineCollection(
                ls, linewidths=2, colors='k')
        
        ax=plt.axes()    
        ax.add_collection(line_segments_ac)
        
        #DC Branches
        idx_from = self.dcbranch.node_fromIdx(self.node)
        idx_to = self.dcbranch.node_toIdx(self.node)
        branch_lat1 = [self.node.lat[i] for i in idx_from]        
        branch_lon1 = [self.node.lon[i] for i in idx_from]        
        branch_lat2 = [self.node.lat[i] for i in idx_to]        
        branch_lon2 = [self.node.lon[i] for i in idx_to]
        
        x1, y1 = m(branch_lon1,branch_lat1)
        x2, y2 = m(branch_lon2,branch_lat2)
        ls = [[(x1[i],y1[i]),(x2[i],y2[i])] for i in range(len(x1))]
        if colours:
            dcColour = 'b'
        else:
            dcColour = 'k'
        line_segments_dc = mpl.collections.LineCollection(
                ls, linewidths=2, colors=dcColour)

        ax.add_collection(line_segments_dc)
        
        #Loads
        x, y = m(self.node.lon,self.node.lat)
        loadByNode = OrderedDict([(k,0) for k in self.node.name])
        consNodes = self.consumer.node
        consValues = self.consumer.load
        
        for idx in range(len(consNodes)):
            loadByNode[consNodes[idx]] += consValues[idx]
        
        avgLoad = numpy.mean(consValues)
        relativeLoads = [dotsize*(l[1]/avgLoad) for l in loadByNode.items()]        
        
        if colours:
            loadColour = 'b'
        else:
            loadColour = 'w'
        m.scatter(x, y, marker='o', c=loadColour, zorder=2, s=relativeLoads)
        
        # Show names of nodes
        if show_node_labels:
            labels = self.node.name
            x1,x2,y1,y2 = plt.axis()
            offset_x = (x2-x1)/50
            for label, xpt, ypt in zip(labels, x, y):
                if xpt > x1 and xpt < x2 and ypt > y1 and ypt < y2:
                    plt.text(xpt+offset_x, ypt, label)
        
        if showTitle:
            plt.title('Load distribution')
            

    def plotRelativeGenerationCapacity(self,tech,show_node_labels=False,latlon=None,
                                     dotsize=40,draw_par_mer=False,
                                     colours=True,showTitle=True):
        '''
        Plots the relative input generation capacity.
        
        Parameters
        ----------
        tech : string
            production technology to be plotted
        show_node_labels : boolean
            whether to show node names (true/false)
        latlon : list of four floats
            map area [lat_min, lon_min, lat_max, lon_max]
        draw_par_mer : boolean
            whether to draw parallels and meridians on map
        colours : boolean
            whether to draw the map in colours or black and white
            
        '''
        
        
        genTypes = self.getAllGeneratorTypes()
        if tech not in genTypes:
            raise Exception('No generators classified as ' + tech + '.\n'
                            'Generator classifications: ' + 
                            str(genTypes)[1:-1])
        
        if latlon is None:
            lat_max =  max(self.node.lat)+1
            lat_min =  min(self.node.lat)-1
            lon_max =  max(self.node.lon)+1
            lon_min =  min(self.node.lon)-1
        else:
            lat_min = latlon[0]
            lon_min = latlon[1]
            lat_max = latlon[2]
            lon_max = latlon[3]
        
        # Use the average latitude as latitude of true scale
        lat_truescale = numpy.mean(self.node.lat)
                
        m = Basemap(resolution='l',projection='merc',\
                      lat_ts=lat_truescale, \
                      llcrnrlon=lon_min, llcrnrlat=lat_min,\
                      urcrnrlon=lon_max ,urcrnrlat=lat_max, \
                      anchor='W')
        
        # Draw coastlines, meridians and parallels.
        m.drawcoastlines()
        m.drawcountries(zorder=0)
        if colours:
            m.fillcontinents(color='coral',lake_color='aqua',zorder=0)
            m.drawmapboundary(fill_color='aqua')
        else:
            m.fillcontinents(zorder=0)
            m.drawmapboundary()
        
        if draw_par_mer:
            m.drawparallels(numpy.arange(_myround(lat_min,10,'floor'),
                _myround(lat_max,10,'ceil'),10),
                labels=[1,1,0,0])

            m.drawmeridians(numpy.arange(_myround(lon_min,10,'floor'),
                _myround(lon_max,10,'ceil'),10),
                labels=[0,0,0,1])        
        
        num_branches = self.branch.numBranches()
        lwidths = [2]*num_branches
        
        #AC Branches
        idx_from = self.branch.node_fromIdx(self.node)
        idx_to = self.branch.node_toIdx(self.node)
        branch_lat1 = [self.node.lat[i] for i in idx_from]        
        branch_lon1 = [self.node.lon[i] for i in idx_from]        
        branch_lat2 = [self.node.lat[i] for i in idx_to]        
        branch_lon2 = [self.node.lon[i] for i in idx_to]
        
        x1, y1 = m(branch_lon1,branch_lat1)
        x2, y2 = m(branch_lon2,branch_lat2)
        
        ls = [[(x1[i],y1[i]),(x2[i],y2[i])] for i in range(len(x1))]
        line_segments_ac = mpl.collections.LineCollection(
                ls, linewidths=lwidths, colors='k')
        
        ax=plt.axes()    
        ax.add_collection(line_segments_ac)
        
        #DC Branches
        idx_from = self.dcbranch.node_fromIdx(self.node)
        idx_to = self.dcbranch.node_toIdx(self.node)
        branch_lat1 = [self.node.lat[i] for i in idx_from]        
        branch_lon1 = [self.node.lon[i] for i in idx_from]        
        branch_lat2 = [self.node.lat[i] for i in idx_to]        
        branch_lon2 = [self.node.lon[i] for i in idx_to]
        
        x1, y1 = m(branch_lon1,branch_lat1)
        x2, y2 = m(branch_lon2,branch_lat2)
        ls = [[(x1[i],y1[i]),(x2[i],y2[i])] for i in range(len(x1))]
        if colours:
            dcColour = 'b'
        else:
            dcColour = 'k'
        line_segments_dc = mpl.collections.LineCollection(
                ls, linewidths=2, colors=dcColour)

        ax.add_collection(line_segments_dc)
        
        #Generators
        x, y = m(self.node.lon,self.node.lat)
        generators = self.getGeneratorsPerType()[tech]
        genNode = self.generator.node
        genCapacity = self.generator.prodMax
        capByNode = OrderedDict([(k,0) for k in self.node.name])
        totCap = 0
        
        for gen in generators:
            capByNode[genNode[gen]] += genCapacity[gen]
            totCap += genCapacity[gen]
        
        
        avgCap = totCap/len(generators)
        relativeCap = [dotsize*(g[1]/avgCap) for g in capByNode.items()]        
        
        if colours:
            loadColour = 'b'
        else:
            loadColour = 'w'
        m.scatter(x, y, marker='o', c=loadColour, zorder=2, s=relativeCap)
        
        # Show names of nodes
        if show_node_labels:
            labels = self.node.name
            x1,x2,y1,y2 = plt.axis()
            offset_x = (x2-x1)/50
            for label, xpt, ypt in zip(labels, x, y):
                if xpt > x1 and xpt < x2 and ypt > y1 and ypt < y2:
                    plt.text(xpt+offset_x, ypt, label)
        
        if showTitle:
            plt.title(tech + ' capacity distribution') 
            
            
    def plotGenerationScatter(self,area,tech=[],annotations=True):
        '''
        Scatter plot of generation capacity and correlation of inflow
        with load.
        
        Parameters
        ----------
        area : string
            area to plot
        tech : list of strings
            production technologies to plot. Empty list = all
        annotations: boolean
            whether to plot annotations
        '''
        
        if len(tech) > 0:
            genTypes = self.getAllGeneratorTypes()
            for gt in tech:
                if gt not in genTypes:
                    raise Exception('No generators classified as ' + gt + 
                                    '.\n Generator classifications: ' + 
                                    str(genTypes)[1:-1])
        else:
            tech = self.getAllGeneratorTypes()
        
        genByType = self.getGeneratorsPerAreaAndType()[area]
        detailedGen = {}
        for gt in genByType.keys():
            if gt in tech:
                detailedGen[gt] = {}
                genIdx = genByType[gt]
                for i in genIdx:
                    capacity = self.generator.prodMax[i]
                    infProfile = self.generator.inflow_profile[i]
                    if infProfile in detailedGen[gt]:
                        detailedGen[gt][infProfile] += capacity
                    else:
                        detailedGen[gt][infProfile] = capacity
        
        cons = self.consumer.load_profile[self.getConsumersPerArea()[area][0]]
        demandProfile = self.demandProfiles[cons]
        x, y, label, size, colCoor, tickLabel = [], [], [], [], [], []
        mainCount = 0
        for gt in detailedGen.keys():
            tickLabel.append(gt)
            subCount = 0
            for infProf in detailedGen[gt].keys():
                if infProf == 'const':
                    y.append(1)
                else:
                    y.append(numpy.corrcoef(self.inflowProfiles[infProf],
                                            demandProfile)[0,1])
                x.append(mainCount)
                label.append(infProf)
                size.append(detailedGen[gt][infProf])
                subCount += 1
            mainCount += 1
        
        avgSize = numpy.mean(size)
        size = [300*s/avgSize for s in size]
        colCoor = [c/mainCount for c in x]
        colours = mpl.cm.hsv(colCoor)
        plt.scatter(x,y,s=size,c=colours,alpha=0.5)
        if annotations:        
            for label, x, y in zip(label, x, y):
                plt.annotate(label, xy = (x, y), xytext = (20, 0),
                             textcoords = 'offset points', ha = 'left', 
                             va = 'bottom',
                             arrowprops = dict(arrowstyle = '-', 
                                               connectionstyle = 'arc3,rad=0'))
                                          
        plt.ylabel('Correlation coefficient of inflow to load')
        plt.xlim(xmin = -0.5, xmax = mainCount + 0.5)
        plt.xticks(range(mainCount + 1), tickLabel)
        
