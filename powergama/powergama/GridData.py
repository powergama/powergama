# -*- coding: utf-8 -*-
'''
Module containing PowerGAMA GridData class and sub-classes

Grid data and time-dependent profiles
'''

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

def parseNum(num,default=None):
    '''parse number and return a float'''
    if default is None:
        return float(num)
    elif num=='':
        return default
    else:
        return float(num)


#_QUOTINGTYPE=csv.QUOTE_NONNUMERIC
_QUOTINGTYPE=csv.QUOTE_MINIMAL

class _Nodes(object):
    '''Private class for grid model nodes'''
    
    def __init__(self):
        self.name = []
        self.area = []
        self.lat = []
        self.lon = []
    
    def readFromFile(self,filename):
        with open(filename,'rb') as csvfile:
            datareader = csv.DictReader(csvfile,delimiter=',',quoting=_QUOTINGTYPE)         
            for row in datareader:
                self.name.append(parseId(row["id"]))
                self.area.append(parseId(row["area"]))
                self.lat.append(parseNum(row["lat"]))
                self.lon.append(parseNum(row["lon"]))
        return
        
    def writeToFile(self,filename):
        print "Saving node data to file",filename
        
        headers = ["id","area","lat","lon"]
        with open(filename,'wb') as csvfile:
            datawriter = csv.DictWriter(csvfile, delimiter=',',fieldnames=headers,\
                            quotechar='"', quoting=_QUOTINGTYPE)
                        
            datawriter.writerow(dict((fn,fn) for fn in headers))
            for i in range(self.numNodes()):
                datarow = {"id":self.name[i],"area":self.area[i],\
                        "lat":self.lat[i],"lon":self.lon[i]}
                datawriter.writerow(datarow)
        return
        
    def numNodes(self):
        return len(self.name)
    
    
class _Branches(object):
    '''Private class for grid model branches'''
    
    def __init__(self):
        self.node_from = []
        self.node_to = []
        self.reactance = []
        self.capacity = []
        self._susceptance = []
    
    def readFromFile(self,filename):
        with open(filename,'rb') as csvfile:
            datareader = csv.DictReader(csvfile,delimiter=',',quoting=_QUOTINGTYPE)           
            for row in datareader:
                self.node_from.append(parseId(row["from"]))
                self.node_to.append(parseId(row["to"]))
                self.reactance.append(parseNum(row["reactance"])) 
                self._susceptance.append(-1.0/parseNum(row["reactance"])) #redundant, but useful
                self.capacity.append(parseNum(row["capacity"]))
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
                                quotechar='"', quoting=_QUOTINGTYPE,\
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



class _DcBranches(object):
    '''Private class for grid model HVDC branches'''
    
    def __init__(self):
        self.node_from = []
        self.node_to = []
        self.capacity = []
    
    def readFromFile(self,filename):
        with open(filename,'rb') as csvfile:
            datareader = csv.DictReader(csvfile,delimiter=',',quoting=_QUOTINGTYPE)           
            for row in datareader:
                self.node_from.append(parseId(row["from"]))
                self.node_to.append(parseId(row["to"]))
                # Using float() to make sure it's not treated as an integer                
                self.capacity.append(parseNum(row["capacity"]))
        return
    
    def writeToFile(self,filename):
        print "Saving DC branch data to file",filename
        
        headers = ["from","to","capacity"]
        with open(filename,'wb') as csvfile:
            datawriter = csv.DictWriter(csvfile, delimiter=',',\
                                quotechar='"', quoting=_QUOTINGTYPE,\
                                fieldnames=headers)
            datawriter.writerow(dict((fn,fn) for fn in headers))
            for i in range(self.numBranches()):
                datarow = {"from":self.node_from[i],"to":self.node_to[i],\
                        "capacity":self.capacity[i]}
                datawriter.writerow(datarow)
        return

    def numBranches(self):
        return len(self.node_from)
        
    def node_fromIdx(self,nodes):
        return [nodes.name.index(self.node_from[k]) for k in range(self.numBranches())]

    def node_toIdx(self,nodes):
        return [nodes.name.index(self.node_to[k]) for k in range(self.numBranches())]
    



class _Generators(object):
    '''Private class for grid model generators'''
    
    
    def __init__(self):
        self.node = []
        self.prodMax = []
        self.prodMin = []
        self.marginalcost = []
        self.storage = []
        self.storagevalue_profile_filling = []
        self.storagevalue_profile_time = []
        self.storagelevel_init = []
        self.inflow_factor = [] 
        self.inflow_profile = []
        self.desc = []
        self.gentype = []
        self.pump_cap = []
        self.pump_efficiency = []
        self.pump_deadband = []
    
    def numGenerators(self):
        return len(self.node)

    #def nodeIdx(self,nodes):
    #    return [nodes.name.index(self.node[k]) for k in range(self.numGenerators())]

    def readFromFile(self,filename):
        with open(filename,'rb') as csvfile:
            datareader = csv.DictReader(csvfile,delimiter=',',
                                        quoting=_QUOTINGTYPE)
            for row in datareader:
                #print(row)
                self.node.append(parseId(row["node"]))
                self.prodMax.append(parseNum(row["pmax"]))
                self.prodMin.append(parseNum(row["pmin"]))
                self.marginalcost.append(parseNum(row["basecost"]))
                self.storage.append(parseNum(row["storage_cap"]))
                self.storagelevel_init.append(parseNum(row["storage_ini"]))
                self.storagevalue_profile_filling.append(
                    parseId(row["storval_filling_ref"]))
                self.storagevalue_profile_time.append(
                    parseId(row["storval_time_ref"]))
                self.inflow_factor.append(parseNum(row["inflow_fac"]))
                self.inflow_profile.append(parseId(row["inflow_ref"]))
                self.gentype.append(parseId(row["type"]))
                self.desc.append(parseId(row["desc"]))
                # Pumping data is optional, so check if it is present in the
                # input files
                if "pump_cap" in row.keys():
                    self.pump_cap.append(parseNum(row["pump_cap"],default=0))
                    self.pump_efficiency.append(
                        parseNum(row["pump_efficiency"],default=0))
                    self.pump_deadband.append(
                        parseNum(row["pump_deadband"],default=0))
                else:
                    # default values are zero
                    self.pump_cap.append(0)
                    self.pump_efficiency.append(0)
                    self.pump_deadband.append(0)
                    
                
        self.idxHasProfile = [i for i, j in enumerate(self.inflow_factor) if j != 0]      
        return
        
    def writeToFile(self,filename):
        print "Saving generator data to file",filename
        
        headers = ["desc","type","node",
                    "pmax","pmin","basecost",
                    "inflow_fac","inflow_ref",
                    "storage_cap","storage_ini",
                    "storval_filling_ref",
                    "storval_time_ref",
                    "pump_cap","pump_efficiency","pump_deadband"]
        with open(filename,'wb') as csvfile:
            datawriter = csv.writer(csvfile, delimiter=',',\
                                quotechar='"', quoting=_QUOTINGTYPE)
            datawriter.writerow(headers)
            for i in range(self.numGenerators()):
                datarow = [
                    self.desc[i], self.gentype[i], self.node[i],
                    self.prodMax[i], self.prodMin[i], self.marginalcost[i],
                    self.inflow_factor[i], self.inflow_profile[i],
                    self.storage[i], 
                    self.storagelevel_init[i],
                    self.storagevalue_profile_filling[i],
                    self.storagevalue_profile_time[i],
                    self.pump_cap[i],
                    self.pump_efficiency[i],
                    self.pump_deadband[i]
                    ]
                datawriter.writerow(datarow)
        return
        
    
        

class _Consumers(object):
    '''Private class for consumers'''
    
    
    def __init__(self):
        self.node = []
        self.load = []
        self.load_profile = []
    
    def readFromFile(self,filename):
        with open(filename,'rb') as csvfile:
           datareader = csv.DictReader(csvfile,delimiter=',',
                                       quoting=_QUOTINGTYPE)
           for row in datareader:
               self.node.append(parseId(row["node"]))
               self.load.append(parseNum(row["demand_avg"]))
               self.load_profile.append(parseId(row["demand_ref"]))
        return

    def writeToFile(self,filename):
        print "Saving consumer data to file",filename
        
        headers = ["node","demand_avg","demand_ref"]
        with open(filename,'wb') as csvfile:
            datawriter = csv.writer(csvfile, delimiter=',',\
                                quotechar='"', quoting=_QUOTINGTYPE)
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


    
class GridData(object):
    '''
    Class for grid data storage and import
    '''        
        

    def __init__(self):
        '''
        Create GridData object with data and methods for import and 
        processing of PowerGAMA grid data            
        '''
        self.node = _Nodes()
        self.branch = _Branches()
        self.dcbranch = _DcBranches()
        self.generator = _Generators()
        self.consumer = _Consumers()
        self.inflowProfiles = None
        self.demandProfiles = None
        self.storagevalue_filling = None
        self.storagevalue_time = None
        self.timeDelta = None
        self.timerange = None


    def readGridData(self,nodes,ac_branches,dc_branches,generators,consumers):
        '''Read grid data from files into data variables'''
        
        self.node.readFromFile(nodes)
        self.branch.readFromFile(ac_branches)
        if not dc_branches is None:
            self.dcbranch.readFromFile(dc_branches)
        self.generator.readFromFile(generators)
        self.consumer.readFromFile(consumers)
        self._checkGridData()


    def _checkGridData(self):
        '''Check consistency of grid data'''
        #generator nodes
        for g in self.generator.node:
            if not g in self.node.name:
                raise Exception("Generator node does not exist: %s" %g)
        #consumer nodes
        for c in self.consumer.node:
            if not c in self.node.name:
                raise Exception("Consumer node does not exist: %s" %c)
                

    def _readProfileFromFile(self,filename,timerange):          
        profiles={}      
        with open(filename,'rb') as csvfile:
            #values = numpy.loadtxt(csvfile,delimiter=",",skiprows=1)
            
            datareader = csv.DictReader(csvfile,delimiter=',',
                                        quoting=_QUOTINGTYPE)
            fieldnames = datareader.fieldnames
            profiles= {fn:[] for fn in fieldnames}
            rowNum=0
            for row in datareader:
                if rowNum in timerange:
                    for fn in fieldnames:
                        profiles[fn].append(parseNum(row[fn]))
                rowNum = rowNum+1
        return profiles
        # keep only values within given time range:
        #values = values[timerange,:]
        #return values


    def _readStoragevaluesFromFile(self,filename):          
        with open(filename,'rb') as csvfile:
            #values = numpy.loadtxt(csvfile,delimiter=",",skiprows=1)
            datareader = csv.DictReader(csvfile,delimiter=',',
                                        quoting=_QUOTINGTYPE)
            fieldnames = datareader.fieldnames
            profiles= {fn:[] for fn in fieldnames}
            for row in datareader:
                for fn in fieldnames:
                    profiles[fn].append(parseNum(row[fn]))
        return profiles
        #return values
        
        
    def readProfileData(self,inflow,demand,storagevalue_filling,
                        storagevalue_time,timerange,timedelta=1.0):
        """Read profile (timeseries) into numpy arrays"""
        
        self.inflowProfiles = self._readProfileFromFile(inflow,timerange)
        self.demandProfiles = self._readProfileFromFile(demand,timerange)
        self.timerange = timerange
        self.timeDelta = timedelta
        
        '''
        Storage values have both time dependence and filling level dependence
       
       The dependence is on filling level (0-100%), is given as an array
        with 101 elements
        '''
        self.storagevalue_time = self._readProfileFromFile(
            storagevalue_time,timerange)
        self.storagevalue_filling = self._readStoragevaluesFromFile(
            storagevalue_filling)
        
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


    def getDcBranchesAtNode(self,nodeIdx,direction):
        """Indices of all DC branches attached to a particular node"""
        if direction=='from':
            indices = [i for i, x in enumerate(self.dcbranch.node_from) if x == self.node.name[nodeIdx]]
        elif direction=='to':
            indices = [i for i, x in enumerate(self.dcbranch.node_to) if x == self.node.name[nodeIdx]]
        else:
            raise Exception("Unknown direction in GridData.getDcBranchesAtNode")
        return indices
	
	
    def getIdxNodesWithLoad(self):
        """Indices of nodes that have load (consumer) attached to them"""        
        # Get index of node associated with all consumer        
        indices = numpy.asarray(self.consumer.nodeIdx(self.node))
        # Return indices only once (unique values)
        indices = numpy.unique(indices)
        return indices
        
        
    def getIdxGeneratorsWithStorage(self):
        """Indices of all generators with nonzero and non-infinite storage"""
        idx = [i for i,v in enumerate(self.generator.storage) 
            if v>0 and v<numpy.inf]
        return idx
        #nonzeros = numpy.nonzero(self.generator.storage)[0]
        #return nonzeros.tolist()
        
    def getIdxGeneratorsWithPumping(self):
        """Indices of all generators with pumping capacity"""
        idx = [i for i,v in enumerate(self.generator.pump_cap) 
            if v>0 and v<numpy.inf]
        return idx
        
    def getIdxBranchesWithFlowConstraints(self):
        '''Indices of branches with less than infinite branch capacity'''
        idx = [i for i,v in enumerate(self.branch.capacity) if v<numpy.inf]
        return idx
        
    def getIdxDcBranchesWithFlowConstraints(self):
        '''Indices of DC branches with less than infinite branch capacity'''
        idx = [i for i,v in enumerate(self.dcbranch.capacity) if v<numpy.inf]
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
  