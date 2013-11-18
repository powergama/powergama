# -*- coding: utf-8 -*-
"""
Created on Fri Sep 06 11:43:58 2013

@author: Harald G Svendsen

Grid data and time-dependent profiles
"""

import csv
import numpy
from scipy.sparse import csr_matrix as sparse


def parseId(num):
    '''parse ID string/integer and return a string'''    
    
    # This method is used when reading input data in order to not interpret 
    # an integer node id o e.g. 100 as "100.0", but always as "100"
    try:
        d = int(num)
    except ValueError:
        d=num
    return str(d)
    

class _Nodes:
    name = []
    area = []
    lat = []
    lon = []
    
    def __init__(self):
        pass
    
    def readFromFile(self,filename):
        with open(filename,'rb') as csvfile:
            datareader = csv.DictReader(csvfile,delimiter=',',quoting=csv.QUOTE_NONNUMERIC)         
            for row in datareader:
                self.name.append(parseId(row["id"]))
                self.area.append(row["area"])
                self.lat.append(row["lat"])
                self.lon.append(row["lon"])
        return
        
    def writeToFile(self,filename):
        print "Saving node data to file",filename
        
        headers = ["id","area","lat","lon"]
        with open(filename,'wb') as csvfile:
            datawriter = csv.DictWriter(csvfile, delimiter=',',fieldnames=headers,\
                            quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
                        
            datawriter.writerow(dict((fn,fn) for fn in headers))
            for i in range(self.numNodes()):
                datarow = {"id":self.name[i],"area":self.area[i],\
                        "lat":self.lat[i],"lon":self.lon[i]}
                datawriter.writerow(datarow)
        return
        
    def numNodes(self):
        return len(self.name)
    
    
class _Branches:
    node_from = []
    node_to = []
    reactance = []
    _susceptance = []
    capacity = []
    
    def __init__(self):
        pass
    
    def readFromFile(self,filename):
        with open(filename,'rb') as csvfile:
            datareader = csv.DictReader(csvfile,delimiter=',',quoting=csv.QUOTE_NONNUMERIC)           
            for row in datareader:
                self.node_from.append(parseId(row["from"]))
                self.node_to.append(parseId(row["to"]))
                # Using float() to make sure it's not treated as an integer                
                self.reactance.append(float(row["reactance"])) 
                self._susceptance.append(-1.0/row["reactance"]) #redundant, but useful
                self.capacity.append(float(row["capacity"]))
#                if row["capacity"]=="Inf" or row["capacity"]=="inf":
#                    self.capacity.append(inf)
#                else:
#                    self.capacity.append(row["capacity"])
        return
    
    def writeToFile(self,filename):
        print "Saving branch data to file",filename
        
        headers = ["from","to","reactance","capacity"]
        with open(filename,'wb') as csvfile:
            datawriter = csv.DictWriter(csvfile, delimiter=',',\
                                quotechar='"', quoting=csv.QUOTE_NONNUMERIC,\
                                fieldnames=headers)
            datawriter.writerow(dict((fn,fn) for fn in headers))
            for i in range(self.numBranches()):
                datarow = {"from":self.node_from[i],"to":self.node_to[i],\
                        "reactance":self.reactance[i],"capacity":self.capacity[i]}
                datawriter.writerow(datarow)
        return

    def numBranches(self):
        return len(self.node_from)
        
    def node_fromIdx(self,nodes):
        return [nodes.name.index(self.node_from[k]) for k in range(self.numBranches())]
        #return nodes.name.index(fromnode in self.node_from)

    def node_toIdx(self,nodes):
        return [nodes.name.index(self.node_to[k]) for k in range(self.numBranches())]
    
    def getSusceptancePu(self,baseOhm):
        return [self._susceptance[i]*baseOhm for i in range(self.numBranches())]

class _Generators:
    node = []
    prodMax = []
    prodMin = []
    marginalcost = []
    storage = []
    storagevalue_type = []
    storagelevel_init = []
    inflow_factor = [] 
    inflow_profile = []
    desc = []
    gentype = []
    #idxHasProfile = []
    
    def __init__(self):
        pass
    
    def numGenerators(self):
        return len(self.node)

    #def nodeIdx(self,nodes):
    #    return [nodes.name.index(self.node[k]) for k in range(self.numGenerators())]

    def readFromFile(self,filename):
        with open(filename,'rb') as csvfile:
            datareader = csv.DictReader(csvfile,delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
            for row in datareader:
                #print(row)
                self.node.append(parseId(row["node"]))
                self.prodMax.append(row["pmax"])
                self.prodMin.append(row["pmin"])
                self.marginalcost.append(row["basecost"])
                self.storage.append(float(row["storage_cap"]))
                self.storagevalue_type.append(parseId(row["storagevalue_ref"]))
                self.storagelevel_init.append(row["storage_ini"])
                self.inflow_factor.append(row["inflow_fac"])
                self.inflow_profile.append(parseId(row["inflow_ref"]))
                self.gentype.append(row["type"])
                self.desc.append(row["desc"])
                
                #if inflow is set to zero, then use Pmax instead
                # (unlimited amount of fuel)
                #if self.inflow[-1]==0:
                #    self.inflow[-1] = self.prodMax[-1]
        self.idxHasProfile = [i for i, j in enumerate(self.inflow_factor) if j != 0]      
        return
        
    def writeToFile(self,filename):
        print "Saving generator data to file",filename
        
        headers = ["desc","type","node",\
                    "pmax","pmin","basecost",\
                    "inflow_fac","inflow_ref",\
                    "storage_cap","storage_ini","storagevalue_ref"]
        with open(filename,'wb') as csvfile:
            datawriter = csv.writer(csvfile, delimiter=',',\
                                quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
            datawriter.writerow(headers)
            for i in range(self.numGenerators()):
                datarow = [\
                    self.desc[i], self.gentype[i], self.node[i],\
                    self.prodMax[i], self.prodMin[i], self.marginalcost[i],\
                    self.inflow_factor[i], self.inflow_profile[i],\
                    self.storage[i], self.storagelevel_init[i],self.storagevalue_type[i] \
                    ]
                datawriter.writerow(datarow)
        return
        
    
        

class _Consumers:
    node = []
    load = []
    load_profile = []
    
    def __init__(self):
        pass
    
    def readFromFile(self,filename):
        with open(filename,'rb') as csvfile:
           datareader = csv.DictReader(csvfile,delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
           for row in datareader:
               self.node.append(parseId(row["node"]))
               self.load.append(row["demand_avg"])
               self.load_profile.append(parseId(row["demand_ref"]))
        return

    def writeToFile(self,filename):
        print "Saving consumer data to file",filename
        
        headers = ["node","demand_avg","demand_ref"]
        with open(filename,'wb') as csvfile:
            datawriter = csv.writer(csvfile, delimiter=',',\
                                quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
            datawriter.writerow(headers)
            for i in range(self.numConsumers()):
                datarow = [self.node[i], self.load[i],self.load_profile[i]]
                datawriter.writerow(datarow)
        return
        
    def numConsumers(self):
        return len(self.node)

    #def nodeIdx(self,nodes):
    #    # return the list of node indices (rather than names) for all consumers
    #    return [nodes.name.index(self.node[k]) for k in range(self.numConsumers())]

    def getDemand(self,timeIdx):
        # return average demand for now.
        return self.load






##=============================================================================


    
class GridData:
    '''
    Class for grid data storage and import
    '''        
        
    node = _Nodes()
    branch = _Branches()
    generator = _Generators()
    consumer = _Consumers()
    inflowProfiles = None
    demandProfiles = None
    storagevalue = None
    numTimesteps = None
    timeDelta = None

    def __init__(self):
        pass

    def readGridData(self,nodes,branches,generators,consumers):
        '''Read grid data from files into data variables'''
        self.node.readFromFile(nodes)
        self.branch.readFromFile(branches)
        self.generator.readFromFile(generators)
        self.consumer.readFromFile(consumers)


    def _readProfileFromFile(self,filename,timerange):          
        profiles={}      
        with open(filename,'rb') as csvfile:
            #values = numpy.loadtxt(csvfile,delimiter=",",skiprows=1)
            
            datareader = csv.DictReader(csvfile,delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
            fieldnames = datareader.fieldnames
            profiles= {fn:[] for fn in fieldnames}
            rowNum=0
            for row in datareader:
                if rowNum in timerange:
                    for fn in fieldnames:
                        profiles[fn].append(row[fn])
                rowNum = rowNum+1
        return profiles
        # keep only values within given time range:
        #values = values[timerange,:]
        #return values


    def _readStoragevaluesFromFile(self,filename):          
        with open(filename,'rb') as csvfile:
            #values = numpy.loadtxt(csvfile,delimiter=",",skiprows=1)
            datareader = csv.DictReader(csvfile,delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
            fieldnames = datareader.fieldnames
            profiles= {fn:[] for fn in fieldnames}
            for row in datareader:
                for fn in fieldnames:
                    profiles[fn].append(row[fn])
        return profiles
        #return values
        
        
    def readProfileData(self,inflow,demand,storagevalues,timerange,timedelta=1.0):
        """Read profile (timeseries) into numpy arrays"""
        
        self.inflowProfiles = self._readProfileFromFile(inflow,timerange)
        self.demandProfiles = self._readProfileFromFile(demand,timerange)
        self.numTimesteps = len(timerange)
        self.timeDelta = timedelta
        
        '''
        Storage values have no time dependence
        Instead, the dependence is on filling level (0-100%), i.e. an array
        with 101 elements
        '''
        self.storagevalue = self._readStoragevaluesFromFile(storagevalues)
        
        return    

    def writeGridDataToFiles(self,prefix):
        '''
        Save data to new input files
        ''' 

        file_nodes = prefix+"nodes.csv"
        file_branches = prefix+"branches.csv"
        file_consumers = prefix+"consumers.csv"     
        file_generators = prefix+"generators.csv"       

        self.node.writeToFile(file_nodes)
        self.branch.writeToFile(file_branches)
        self.consumer.writeToFile(file_consumers)
        self.generator.writeToFile(file_generators)
        
        return
    
    def getGeneratorsAtNode(self,nodeIdx):
        """Indices of all generators attached to a particular node"""
        #indices = [i for i, x in enumerate(self.generator.nodeIdx(self.node)) if x == nodeIdx]
        indices = [i for i, x in enumerate(self.generator.node) if x == self.node.name[nodeIdx]]
        return indices
        
        
    def getLoadsAtNode(self,nodeIdx):
        """Indices of all loads (consumers) attached to a particular node"""
        indices = [i for i, x in enumerate(self.consumer.node) if x == self.node.name[nodeIdx]]
        return indices


    def getHvdcAtNode(self,nodeIdx,direction):
        """Indices of all HVDC branche attached to a particular node"""
        #indices = [i for i, x in enumerate(self.hvdc.nodeIdx) if x == nodeIdx]
        #return indices
        print("not implemented!")
        return None
	
	
    def getIdxNodesWithLoad(self):
        """Indices of nodes that have load (consumer) attached to them"""        
        # Get index of node associated with all consumer        
        indices = numpy.asarray(self.consumer.nodeIdx(self.node))
        # Return indices only once (unique values)
        indices = numpy.unique(indices)
        return indices
        
        
    def getIdxGeneratorsWithStorage(self):
        """Indices of all generators with nonzero and non-infinite storage"""
        idx = [i for i,v in enumerate(self.generator.storage) if v>0 and v<numpy.inf]
        return idx
        #nonzeros = numpy.nonzero(self.generator.storage)[0]
        #return nonzeros.tolist()
        
    def getIdxBranchesWithFlowConstraints(self):
        '''Indices of branches with less than infinite branch capacity'''
        idx = [i for i,v in enumerate(self.branch.capacity) if v<numpy.inf]
        return idx
        

    def computePowerFlowMatrices(self,baseZ):
        """
        Compute and return dc power flow matrices B' and DA
                
        Returns sparse matrices (csr - compressed sparse row matrix)              
        """
        # node-branch incidence matrix
        # element b,n is  1 if branch b starts at node n
        #                -1 if branch b ends at node n
        num_nodes = self.node.numNodes()
        num_branches = self.branch.numBranches()
        
        fromIdx = self.branch.node_fromIdx(self.node)
        toIdx = self.branch.node_toIdx(self.node)
        data = numpy.r_[numpy.ones(num_branches),-numpy.ones(num_branches)]
        row = numpy.r_[range(num_branches),range(num_branches)]
        col = numpy.r_[fromIdx, toIdx]
        A_incidence_matrix = sparse( (data, (row,col)),(num_branches,num_nodes))
        
        # Diagonal matrix
        b = numpy.asarray(self.branch.getSusceptancePu(baseZ))
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
        areas = self.node.area
        allareas = []
        for co in areas:
            if co not in allareas:
                allareas.append(co)
        return allareas
        
    def getAllGeneratorTypes(self):
        '''Return list of generator types included in the grid model'''
        gentypes = self.generator.gentype
        alltypes = []
        for ge in gentypes:
            if ge not in alltypes:
                alltypes.append(ge)
        return alltypes
        
    def getConsumersPerArea(self):
        '''Returns dictionary with indices of loads within each area'''
        consumers = {}
        for idx_load in range(self.consumer.numConsumers()):
            node_name = self.consumer.node[idx_load]
            node_idx = self.node.name.index(node_name)
            area_name = self.node.area[node_idx]
            if consumers.has_key(area_name):
                consumers[area_name].append(idx_load)
            else:
                consumers[area_name] =  [idx_load]   
        return consumers
   
    def getGeneratorsPerAreaAndType(self): 
        '''Returns dictionary with indices of generators within each area'''
        generators = {}
        for idx_gen in range(self.generator.numGenerators()):
            gtype = self.generator.gentype[idx_gen]
            node_name = self.generator.node[idx_gen]
            node_idx = self.node.name.index(node_name)
            area_name = self.node.area[node_idx]
            if generators.has_key(area_name):
                if generators[area_name].has_key(gtype):
                    generators[area_name][gtype].append(idx_gen)
                else:
                    generators[area_name][gtype] = [idx_gen]
            else:
                generators[area_name] = {gtype:[idx_gen]}
        return generators

  
#data.writeGridDataToFiles("test")
  