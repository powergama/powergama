# -*- coding: utf-8 -*-
'''
Module containing PowerGAMA GridData class and sub-classes

Grid data and time-dependent profiles
'''

import csv
import sys
import numpy
from scipy.sparse import csr_matrix as sparse

def openfile(file,rw=''):
    '''open file in a manner compatible with both Python 2 and Python 3'''
    if sys.version_info >= (3,0,0):
        f = open(file, rw, newline='', encoding="utf-8")
    else:
        f = open(file, rw+'b')
    return f


def parseId(num):
    '''parse ID string/integer and return a string'''    
    
    # This method is used when reading input data in order to not interpret 
    # an integer node id o e.g. 100 as "100.0", but always as "100"
    if num is None:
        return ''
    else:
        try:
            d = int(num)
        except ValueError:
            d=num
        return str(d)

def parseNum(num,default=None):
    '''parse number and return a float'''
    if default is None:
        return float(num)
    elif num=='' or num is None:
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
        with openfile(filename,'r') as csvfile:
            datareader = csv.DictReader(csvfile,delimiter=',',quoting=_QUOTINGTYPE)         
            for row in datareader:
                self.name.append(parseId(row["id"]))
                self.area.append(parseId(row["area"]))
                self.lat.append(parseNum(row["lat"]))
                self.lon.append(parseNum(row["lon"]))
        return
        
    def writeToFile(self,filename):
        print ("Saving node data to file "+str(filename))
        
        headers = ["id","area","lat","lon"]
        with openfile(filename,'w') as csvfile:
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
        with openfile(filename,'r') as csvfile:
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
        print("Saving branch data to file "+str(filename))
        
        headers = ["from","to","reactance","capacity"]
        with openfile(filename,'w') as csvfile:
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
        with openfile(filename,'r') as csvfile:
            datareader = csv.DictReader(csvfile,delimiter=',',quoting=_QUOTINGTYPE)           
            for row in datareader:
                self.node_from.append(parseId(row["from"]))
                self.node_to.append(parseId(row["to"]))
                self.capacity.append(parseNum(row["capacity"]))
        return
    
    def writeToFile(self,filename):
        print("Saving DC branch data to file "+str(filename))
        
        headers = ["from","to","capacity"]
        with openfile(filename,'w') as csvfile:
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
        self.fuelcost = []
        self.storage = []
        self.storagevalue_abs= []
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
        with openfile(filename,'r') as csvfile:
            datareader = csv.DictReader(csvfile,delimiter=',',
                                        quoting=_QUOTINGTYPE)
            for row in datareader:
                #print(row)
                self.gentype.append(parseId(row["type"]))
                self.desc.append(parseId(row["desc"]))
                self.node.append(parseId(row["node"]))
                self.prodMax.append(parseNum(row["pmax"]))
                self.prodMin.append(parseNum(row["pmin"]))
                self.fuelcost.append(parseNum(row["fuelcost"]))
                self.inflow_factor.append(parseNum(row["inflow_fac"]))
                self.inflow_profile.append(parseId(row["inflow_ref"]))
                self.storage.append(
                    parseNum(row["storage_cap"],default=0))
                self.storagevalue_abs.append(
                    parseNum(row["storage_price"],default=0))
                self.storagelevel_init.append(
                    parseNum(row["storage_ini"],default=0))
                self.storagevalue_profile_filling.append(
                    parseId(row["storval_filling_ref"]))
                self.storagevalue_profile_time.append(
                    parseId(row["storval_time_ref"]))
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
        print("Saving generator data to file "+str(filename))
        
        headers = ["desc","type","node",
                    "pmax","pmin",
                    "fuelcost",
                    "inflow_fac","inflow_ref",
                    "storage_cap","storage_price",
                    "storage_ini",
                    "storval_filling_ref",
                    "storval_time_ref",
                    "pump_cap","pump_efficiency","pump_deadband"]
        with openfile(filename,'w') as csvfile:
            datawriter = csv.writer(csvfile, delimiter=',',\
                                quotechar='"', quoting=_QUOTINGTYPE)
            datawriter.writerow(headers)
            for i in range(self.numGenerators()):
                datarow = [
                    self.desc[i], self.gentype[i], self.node[i],
                    self.prodMax[i], self.prodMin[i], 
                    self.fuelcost[i],self.inflow_factor[i], 
                    self.inflow_profile[i]]
                if self.storage[i]>0:
                    datarow = datarow +[
                        self.storage[i], 
                        self.storagevalue_abs[i],
                        self.storagelevel_init[i],
                        self.storagevalue_profile_filling[i],
                        self.storagevalue_profile_time[i] ]
                    if self.pump_cap[i]>0:
                        datarow = datarow + [
                            self.pump_cap[i],
                            self.pump_efficiency[i],
                            self.pump_deadband[i] ]
                    
                datawriter.writerow(datarow)
        return
        
    
        

class _Consumers(object):
    '''Private class for consumers'''
    
    
    def __init__(self):
        self.node = []
        self.load = []
        self.load_profile = []
        self.flex_fraction = []
        self.flex_on_off =[]
        self.flex_basevalue = []
        self.flex_storage = []
        self.flex_storagevalue_profile_filling = []
        self.flex_storagevalue_profile_time = []
        self.flex_storagelevel_init = []
    
    def readFromFile(self,filename):
        with openfile(filename,'r') as csvfile:
           datareader = csv.DictReader(csvfile,delimiter=',',
                                       quoting=_QUOTINGTYPE)
           for row in datareader:
               self.node.append(parseId(row["node"]))
               self.load.append(parseNum(row["demand_avg"]))
               self.load_profile.append(parseId(row["demand_ref"]))
               if "flex_fraction" in row.keys():
                   self.flex_fraction.append(
                       parseNum(row["flex_fraction"],default=0))
                   self.flex_on_off.append(
                       parseNum(row["flex_on_off"],default=0))
                   self.flex_basevalue.append(
                       parseNum(row["flex_basevalue"],default=0))
                   self.flex_storage.append(
                       parseNum(row["flex_storage"],default=0))
                   self.flex_storagevalue_profile_filling.append(
                       parseId(row["flex_storval_filling"]))
                   self.flex_storagevalue_profile_time.append(
                       parseId(row["flex_storval_time"]))
               else:
                   # default values are zero
                   self.flex_fraction.append(0)
                   self.flex_on_off.append(0)
                   self.flex_basevalue.append(0)
                   self.flex_storage.append(0)
                   self.flex_storagevalue_profile_filling.append(0)
                   self.flex_storagevalue_profile_time.append(0)
                   
           #Hard-coded initial filling level of storage equal to 50%
           print("OBS: Initial flexible storage filling set to 0.5")        
           self.flex_storagelevel_init = [0.5]*len(self.flex_fraction)
        return

    def writeToFile(self,filename):
        print("Saving consumer data to file "+str(filename))
        
        headers = ["node","demand_avg","demand_ref",
                   "flex_fraction","flex_on_off",
                   "flex_basevalue","flex_storage",
                   "flex_storval_filling"]
        with openfile(filename,'w') as csvfile:
            datawriter = csv.writer(csvfile, delimiter=',',\
                                quotechar='"', quoting=_QUOTINGTYPE)
            datawriter.writerow(headers)
            for i in range(self.numConsumers()):
                datarow = [self.node[i], self.load[i],self.load_profile[i]]
                if self.flex_fraction[i]>0:
                    datarow = datarow + [
                           self.flex_fraction[i],
                           self.flex_on_off[i],
                           self.flex_basevalue[i],
                           self.flex_storage[i],
                           self.flex_storagevalue_profile_filling[i] 
                           ] 
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

    def getFlexibleLoadStorageCapacity(self,indx):
        ''' flexible load storage capacity in MWh'''
        cap = (self.load[indx] * self.flex_fraction[indx] 
                * self.flex_storage[indx] )
        return cap





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
        with openfile(filename,'r') as csvfile:
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
        with openfile(filename,'r') as csvfile:
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
        file_hvdc = prefix+"hvdc.csv"       

        self.node.writeToFile(file_nodes)
        self.branch.writeToFile(file_branches)
        self.consumer.writeToFile(file_consumers)
        self.generator.writeToFile(file_generators)
        self.dcbranch.writeToFile(file_hvdc)
        
        return
    
    def getGeneratorsAtNode(self,nodeIdx):
        """Indices of all generators attached to a particular node"""
        indices = [i for i, x in enumerate(self.generator.node) 
                    if x == self.node.name[nodeIdx]]
        return indices
        
    def getGeneratorsWithPumpAtNode(self,nodeIdx):
        """Indices of all pumps attached to a particular node"""
        indices = [i for i, x in enumerate(self.generator.node) 
                    if x == self.node.name[nodeIdx]
                    and self.generator.pump_cap[i]>0]
        return indices
        
    def getLoadsAtNode(self,nodeIdx):
        """Indices of all loads (consumers) attached to a particular node"""
        indices = [i for i, x in enumerate(self.consumer.node) 
                    if x == self.node.name[nodeIdx]]
        return indices

    def getLoadsFlexibleAtNode(self,nodeIdx):
        """Indices of all flexible nodes attached to a particular node"""
        indices = [i for i, x in enumerate(self.consumer.node) 
                    if x == self.node.name[nodeIdx]
                    and self.consumer.flex_fraction[i]>0
                    and self.consumer.load[i]>0]
        return indices
        
    def getIdxConsumersWithFlexibleLoad(self):
        """Indices of all consumers with flexible load"""
        idx = [i for i,v in enumerate(self.consumer.flex_fraction) 
            if v>0 and v<numpy.inf and self.consumer.load[i]>0]
        return idx
        

    def getDcBranchesAtNode(self,nodeIdx,direction):
        """Indices of all DC branches attached to a particular node"""
        if direction=='from':
            indices = [i for i, x in enumerate(self.dcbranch.node_from) 
            if x == self.node.name[nodeIdx]]
        elif direction=='to':
            indices = [i for i, x in enumerate(self.dcbranch.node_to) 
            if x == self.node.name[nodeIdx]]
        else:
            raise Exception("Unknown direction in GridData.getDcBranchesAtNode")
        return indices


    def getDcBranches(self):
        '''
        Returns a list with DC branches in the format
        [index,from area,to area]
        '''
        hvdcBranches = []
        for idx in range(len(self.dcbranch.capacity)):
            fromNodeIdx = self.node.name.index(self.dcbranch.node_from[idx])
            toNodeIdx = self.node.name.index(self.dcbranch.node_to[idx])
            areaFrom = self.node.area[fromNodeIdx]
            areaTo = self.node.area[toNodeIdx]
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
        idx = [i for i,v in enumerate(self.generator.storage) 
            if v>0 and v<numpy.inf]
        return idx
        
    def getIdxGeneratorsWithNonzeroInflow(self):
        """Indices of all generators with nonzero inflow"""
        idx = [i for i,v in enumerate(self.generator.inflow_factor) 
            if v>0]
        return idx

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
            if area_name in consumers:
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
        for idx_gen in range(self.generator.numGenerators()):
            gtype = self.generator.gentype[idx_gen]
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
        for pumpIdx,cap in enumerate(self.generator.pump_cap):
            if cap>0 and cap<numpy.inf:
                nodeName = self.generator.node[pumpIdx]
                nodeIdx = self.node.name.index(nodeName)
                areaName = self.node.area[nodeIdx]
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
            br_from_nodes = self.branch.node_fromIdx(self.node)
            br_to_nodes = self.branch.node_toIdx(self.node)
        elif acdc=='dc':
            br_from_nodes = self.dcbranch.node_fromIdx(self.node)
            br_to_nodes = self.dcbranch.node_toIdx(self.node)
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
    
#data.writeGridDataToFiles("test")
  