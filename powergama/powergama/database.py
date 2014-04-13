# -*- coding: utf-8 -*-
'''
Module dealing with database IO
'''

import sqlite3 as db
import os

class Database(object):
    '''
    Class for storing results from PowerGAMA in sqlite databse
    '''    

    def __init__(self,filename):
        self.filename = filename


    def createTables(self,data):
        """
        Create database for PowerGAMA results
        """    
        
        # convert from lists to tuple of tuples
        nodes = tuple((
            i,
            data.node.name[i],
            data.node.area[i],
            data.node.lat[i],
            data.node.lon[i]
            ) for i in xrange(len(data.node.name)))
        generators = tuple((
            i,            
            data.generator.node[i],
            data.generator.gentype[i],
            ) for i in xrange(len(data.generator.node)))
        
        if os.path.isfile(self.filename):
            #Must use a new file
            raise IOError('Cannot append existing file. Choose new file name.')
        con = db.connect(self.filename)
        with con:        
            cur = con.cursor()    
            cur.execute("CREATE TABLE Grid_Nodes(indx INT, id TEXT, area TEXT,"
                        +"lat DOUBLE, lon DOUBLE)")
            cur.executemany("INSERT INTO Grid_Nodes VALUES(?,?,?,?,?)",
                            nodes)
            cur.execute("CREATE TABLE Grid_Generators(indx INT, node TEXT,"
                        +"type TEXT)")
            cur.executemany("INSERT INTO Grid_Generators VALUES(?,?,?)",
                            generators)
    
            cur.execute("CREATE TABLE Res_ObjFunc(timestep INT, value DOUBLE)")
            cur.execute("CREATE TABLE Res_Branches(timestep INT, indx INT,"
                        +"flow DOUBLE)")
            cur.execute("CREATE TABLE Res_BranchesSens(timestep INT, indx INT,"
                        +"cap_sensitivity DOUBLE)")
            cur.execute("CREATE TABLE Res_Dcbranches(timestep INT, indx INT,"
                        +"flow DOUBLE, cap_sensitivity DOUBLE)")
            cur.execute("CREATE TABLE Res_Nodes(timestep INT, indx INT,"
                        +"angle DOUBLE, nodalprice DOUBLE, loadshed DOUBLE)")
            cur.execute("CREATE TABLE Res_Generators(timestep INT, indx INT,"
                        +"output DOUBLE, inflow_spilled DOUBLE)")
            cur.execute("CREATE TABLE Res_Storage(timestep INT, indx INT,"
                        +"storage DOUBLE, marginalprice DOUBLE)")
    
    
        return nodes
    
    def appendResults(self,timestep,objective_function,generator_power,
                       branch_flow,dcbranch_flow,node_angle,
                       sensitivity_branch_capacity,
                       sensitivity_dcbranch_capacity,
                       sensitivity_node_power,
                       storage,
                       inflow_spilled,
                       loadshed_power,
                       marginalprice):
        '''
        Store results from a given timestep to the database
    
        Parameters
        ----------
        timestep (int) = timestep number
        objective_function (float) = value of objective function
        generator_power (list of floats) = power output of generators
        branch_power (list of floats) = power flow on branches
        node_angle (list of floats) = phase angle (relative to node 0) at nodes
        sensitivity_branch_capacity (list of floats) = sensitivity to branch capacity
        sensitivity_dcbranch_capacity (list of floats) = sensitivty to DC branch capacity
        sensitivity_node_power (list of floats) = sensitivity to node power (nodal price)
        storage (list of floats) = storage filling level of generators
        inflow_spilled (list of floats) = spilled power inflow of generators
        loadshed_power (list of floats) = unmet power demand at nodes
        marginalprice (list of floats) = price of generators with storage
        '''
        
        con = db.connect(self.filename)
        with con:        
            cur = con.cursor()    
            cur.execute("INSERT INTO Res_ObjFunc VALUES(?,?)",(timestep,objective_function))
            cur.executemany("INSERT INTO Res_Nodes VALUES(?,?,?,?,?)",
                    tuple((timestep,i,node_angle[i],
                          sensitivity_node_power[i],loadshed_power[i]) 
                    for i in xrange(len(sensitivity_node_power))))
            cur.executemany("INSERT INTO Res_Branches VALUES(?,?,?)",
                    tuple((timestep,i,branch_flow[i]) 
                    for i in xrange(len(branch_flow))))
            cur.executemany("INSERT INTO Res_BranchesSens VALUES(?,?,?)",
                    tuple((timestep,i,sensitivity_branch_capacity[i]) 
                    for i in xrange(len(sensitivity_branch_capacity))))
            cur.executemany("INSERT INTO Res_Dcbranches VALUES(?,?,?,?)",
                    tuple((timestep,i,dcbranch_flow[i],
                          sensitivity_dcbranch_capacity[i]) 
                    for i in xrange(len(dcbranch_flow))))
            cur.executemany("INSERT INTO Res_Generators VALUES(?,?,?,?)",
                    tuple((timestep,i,generator_power[i],inflow_spilled[i]) 
                    for i in xrange(len(generator_power))))
            cur.executemany("INSERT INTO Res_Storage VALUES(?,?,?,?)",
                    tuple((timestep,i,storage[i],marginalprice[i]) 
                    for i in xrange(len(storage))))
      
    
      
    def getStorageFilling(self,storageindx,timeMaxMin):
        '''Get storage filling level for storage generators'''
        con = db.connect(self.filename)
        with con:        
            #con.row_factory = db.Row
            cur = con.cursor()
            cur.execute("SELECT storage FROM Res_Storage "
                +"WHERE timestep>=? AND timestep<? AND indx=?"
                +" ORDER BY timestep",
                (timeMaxMin[0],timeMaxMin[-1],storageindx))
            rows = cur.fetchall()
            values = [row[0] for row in rows]        
        return values


    def getNodalPrice(self,nodeindx,timeMaxMin):
        '''Get nodal price at specified node'''
        con = db.connect(self.filename)
        with con:        
            cur = con.cursor()
            cur.execute("SELECT nodalprice FROM Res_Nodes "
                +"WHERE timestep>=? AND timestep<? AND indx=?"
                +" ORDER BY timestep",
                (timeMaxMin[0],timeMaxMin[-1],nodeindx))
            rows = cur.fetchall()
            values = [row[0] for row in rows]        
        return values

   
    def getStorageValue(self,storageindx,timeMaxMin):
        '''Get storage value for storage generators'''
        con = db.connect(self.filename)
        with con:        
            cur = con.cursor()
            cur.execute("SELECT marginalprice FROM Res_Storage "
                +"WHERE timestep>=? AND timestep<? AND indx=?"
                +" ORDER BY timestep",
                (timeMaxMin[0],timeMaxMin[-1],storageindx))
            rows = cur.fetchall()
            values = [row[0] for row in rows]        
        return values

    
    def getGeneratorPower(self,generatorindx,timeMaxMin):
        '''Get storage filling level for storage generators'''
        
        if not isinstance(generatorindx,list): 
            generatorindx = [generatorindx]
        if len(generatorindx)==0:
            return None
        con = db.connect(self.filename)
        with con:        
            cur = con.cursor()
            cur.execute("SELECT timestep,SUM(output) FROM Res_Generators "
                +"WHERE timestep>=? AND timestep<? AND indx IN ("
                +"".join(["?," for i in xrange(len(generatorindx)-1)])+"?"                
                +")"
                +" GROUP BY timestep ORDER BY timestep",
                (timeMaxMin[0],timeMaxMin[-1])+tuple(generatorindx))
            rows = cur.fetchall()
            output = [row[1] for row in rows]        
        return output
 
 
    def getGeneratorPowerInArea(area,timeMaxMin):
        '''Get accumulated generation per type in given area'''
        con = db.connect(self.filename)
        with con:        
            cur = con.cursor()
            cur.execute("SELECT output FROM Res_Generators "
                +"WHERE timestep>=? AND timestep<? AND indx IN "
                +"(SELECT indx FROM Data_Generators WHERE node IN "
                +" (SELECT id FROM Data_Nodes WHERE area IN (?)))"
                +" ORDER BY timestep",
                (timeMaxMin[0],timeMaxMin[-1],area))
            rows = cur.fetchall()
            output = [row[0] for row in rows]        
        return output
        