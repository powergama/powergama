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
        self.sqlite_version = db.sqlite_version

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
        br_from = data.branch.node_fromIdx(data.node)
        br_to = data.branch.node_toIdx(data.node)        
        branches = tuple((
            i,
            br_from[i],
            br_to[i],
            data.branch.capacity[i],
            data.branch.reactance[i]
            ) for i in xrange(len(data.branch.capacity)))
        
        if os.path.isfile(self.filename):
            #delete existing file
            print("OBS: Deleting existing SQLite file ""%s"""%self.filename )
            os.remove(self.filename)
            #Must use a new file
            #raise IOError('Cannot append existing file. Choose new file name.')
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
            cur.execute("CREATE TABLE Grid_Branches(indx INT, fromIndx INT,"
                        +"toIndx INT, capacity DOUBLE, reactance DOUBLE)")
            cur.executemany("INSERT INTO Grid_Branches VALUES(?,?,?,?,?)",
                            branches)
    
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
                       marginalprice,
                       idx_storagegen,
                       idx_branchsens):
        '''
        Store results from a given timestep to the database
    
        Parameters
        ----------
        timestep (int)
            timestep number
        objective_function (float)
            value of objective function
        generator_power (list of floats)
            power output of generators
        branch_power (list of floats)
            power flow on branches
        node_angle (list of floats)
            phase angle (relative to node 0) at nodes
        sensitivity_branch_capacity (list of floats)
            sensitivity to branch capacity
        sensitivity_dcbranch_capacity (list of floats)
            sensitivty to DC branch capacity
        sensitivity_node_power (list of floats)
            sensitivity to node power (nodal price)
        storage
            storage filling level of generators
        inflow_spilled (list of floats)
            spilled power inflow of generators
        loadshed_power (list of floats)
            unmet power demand at nodes
        marginalprice
            price of generators with storage
        idx_storagegen
            index in generator list of generators with storage
        idx_branchsens
            index in branch list of branches with limited capacity
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
                    tuple((timestep,idx_branchsens[i],
                           sensitivity_branch_capacity[i]) 
                    for i in xrange(len(sensitivity_branch_capacity))))
            cur.executemany("INSERT INTO Res_Dcbranches VALUES(?,?,?,?)",
                    tuple((timestep,i,dcbranch_flow[i],
                          sensitivity_dcbranch_capacity[i]) 
                    for i in xrange(len(dcbranch_flow))))
            cur.executemany("INSERT INTO Res_Generators VALUES(?,?,?,?)",
                    tuple((timestep,i,generator_power[i],inflow_spilled[i]) 
                    for i in xrange(len(generator_power))))
            cur.executemany("INSERT INTO Res_Storage VALUES(?,?,?,?)",
                    tuple((timestep,idx_storagegen[i],
                           storage[i],marginalprice[i]) 
                    for i in xrange(len(storage))))
      

########## Get grid data
    
    def getGridNodeIndices(self):
        '''Get node indices as a list'''
        con = db.connect(self.filename)
        with con:        
            cur = con.cursor()
            cur.execute("SELECT indx FROM Grid_Nodes ")
            rows = cur.fetchall()
            values = [row[0] for row in rows]        
        return values
       
    def getGridBranches(self):
        '''Get branch indices as a list'''
        con = db.connect(self.filename)
        with con:        
            #con.row_factory = db.Row
            cur = con.cursor()
            cur.execute("SELECT indx,fromIndx,toIndx,capacity,reactance "
                +" FROM Grid_Branches ")
            rows = cur.fetchall()
            values = {
                'indx':[row[0] for row in rows],
                'fromIndx':[row[1] for row in rows],
                'toIndx':[row[2] for row in rows],
                'capacity':[row[3] for row in rows],
                'reactance':[row[4] for row in rows]
                }      
        return values

    def getGridInterareaBranches(self):
        '''
        Get indices of branches between different areas as a list
        
        Returns
        =======
        
        (indice, fromArea, toArea)
        '''
        con = db.connect(self.filename)
        con.text_factory = str
        with con:
            cur = con.cursor()
            cur.execute("SELECT b.indx, fromNode.area, toNode.area"
                +" FROM Grid_Branches b"
                +" INNER JOIN Grid_Nodes fromNode ON b.fromIndx = fromNode.indx"
                +" INNER JOIN Grid_Nodes toNode ON b.toIndx = toNode.indx"
                +" WHERE fromNode.area != toNode.area")
            output = cur.fetchall()
        return output


########## Get result data
          
### Node results

    def getResultNodalPrice(self,nodeindx,timeMaxMin):
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
        
    def getResultNodalPricesAll(self,timeMaxMin):
        '''
        Get nodal price at all nodes (list of tuples)
        
        Returns
        =======
        List of tuples with values:
        (timestep, node index, nodal price)
        
        '''
        con = db.connect(self.filename)
        with con:        
            cur = con.cursor()
            cur.execute("SELECT timestep,indx,nodalprice FROM Res_Nodes"
                +" WHERE timestep>=? AND timestep<?"
                +" ORDER BY indx,timestep",
                (timeMaxMin[0],timeMaxMin[-1]))
            rows = cur.fetchall()
        return rows

    def getResultNodalPricesMean(self,timeMaxMin):
        '''Get average nodal price at all nodes'''
        con = db.connect(self.filename)
        with con:        
            cur = con.cursor()
            cur.execute("SELECT indx,AVG(nodalprice) FROM Res_Nodes"
                +" WHERE timestep>=? AND timestep<?"
                +" GROUP BY indx ORDER BY indx",
                (timeMaxMin[0],timeMaxMin[-1]))
            rows = cur.fetchall()
        values = [row[1] for row in rows]        
        return values

### Branch results

    def getResultBranchFlow(self,branchindx,timeMaxMin):
        '''Get branch flow at specified branch'''
        con = db.connect(self.filename)
        with con:        
            cur = con.cursor()
            cur.execute("SELECT flow FROM Res_Branches "
                +"WHERE timestep>=? AND timestep<? AND indx=?"
                +" ORDER BY timestep",
                (timeMaxMin[0],timeMaxMin[-1],branchindx))
            rows = cur.fetchall()
        values = [row[0] for row in rows]        
        return values

    def getResultBranchFlowAll(self,timeMaxMin):
        '''
        Get branch flow at all branches (list of tuples)
        
        Returns
        =======
        List of tuples with values:
        (timestep, branch index, flow)
        
        '''
        con = db.connect(self.filename)
        with con:        
            cur = con.cursor()
            cur.execute("SELECT timestep,indx,flow FROM Res_Branches"
                +" WHERE timestep>=? AND timestep<?"
                +" ORDER BY indx,timestep",
                (timeMaxMin[0],timeMaxMin[-1]))
            rows = cur.fetchall()
        return rows

    def getResultBranchFlowsMean(self,timeMaxMin):
        '''
        Get average branch flow on branches in both direction
        
        Returns
        =======
        List with values for each branch:
        [average flow 1->2, average flow 2->1, average absolute flow]
        
        '''
        con = db.connect(self.filename)
        with con:        
            cur = con.cursor()
            cur.execute("SELECT indx,TOTAL(flow) FROM Res_Branches"
                +" WHERE timestep>=? AND timestep<? AND flow>=0"
                +" GROUP BY indx ORDER BY indx",
                (timeMaxMin[0],timeMaxMin[-1]))
            rows1 = cur.fetchall()
            cur.execute("SELECT indx,TOTAL(flow) FROM Res_Branches"
                +" WHERE timestep>=? AND timestep<? AND flow<0"
                +" GROUP BY indx ORDER BY indx",
                (timeMaxMin[0],timeMaxMin[-1]))
            rows2 = cur.fetchall()
            cur.execute("SELECT MAX(indx) FROM Res_Branches")
            numBranches = 1 + cur.fetchone()[0]
            #Calculate average flow for each direction
            numTimeSteps = timeMaxMin[-1] - timeMaxMin[0]
            rows1 = [(index, tot_flow / numTimeSteps) for (index, tot_flow) in rows1]
            rows2 = [(index, tot_flow / numTimeSteps) for (index, tot_flow) in rows2]
        # The length of rows1 and rows2 may be less than the number
        # of branches if the flow is always in one direction
        values_pos = [0]*numBranches
        values_neg = [0]*numBranches
        values_abs = [0]*numBranches
        i1=0
        i2=0
        for i in xrange(numBranches):
            if i1<len(rows1) and rows1[i1][0] == i:
                values_pos[i] = rows1[i1][1]
                i1 = i1+1
            if i2<len(rows2) and rows2[i2][0] == i:
                values_neg[i] = abs(rows2[i2][1])
                i2 = i2+1
            values_abs[i]=values_pos[i]+values_neg[i]
        values = [values_pos,values_neg,values_abs]
        return values

    def getResultBranchSens(self,branchindx,timeMaxMin):
        '''Get branch capacity sensitivity at specified branch'''
        con = db.connect(self.filename)
        with con:        
            cur = con.cursor()
            cur.execute("SELECT cap_sensitivity FROM Res_BranchesSens "
                +"WHERE timestep>=? AND timestep<? AND indx=?"
                +" ORDER BY timestep",
                (timeMaxMin[0],timeMaxMin[-1],branchindx))
            rows = cur.fetchall()
        values = [row[0] for row in rows]        
        return values
        
    def getResultBranchSensAll(self,timeMaxMin):
        '''Get branch capacity sensitivity at all branches'''
        branches = self.getGridBranches()        
        branchlist = branches['indx']     
        sens = []        
        for branch in  branchlist:
            this_sens = self.getResultBranchSens(branch,timeMaxMin)
            sens.append(this_sens if this_sens !=[] 
                else [None]*(timeMaxMin[1]-timeMaxMin[0]))
        return sens

    def getResultBranchSensMean(self,timeMaxMin):
        '''Get average sensitivity of all  branches'''
        con = db.connect(self.filename)
        with con:        
            cur = con.cursor()
            cur.execute("SELECT indx,AVG(cap_sensitivity) FROM Res_BranchesSens"
                +" WHERE timestep>=? AND timestep<?"
                +" GROUP BY indx ORDER BY indx",
                (timeMaxMin[0],timeMaxMin[-1]))
            rows = cur.fetchall()
        values = [row[1] for row in rows]        
        return values
       
    def getAverageInterareaBranchFlow(self, timeMaxMin):
        '''
        Get average negative flow, positive flow and total flow of branches between different areas
        
        Returns
        =======
        List of tuples for inter-area branches with following values:
        (indices, fromArea, toArea, average negative flow, average positive flow, average flow)
        '''
        
        con = db.connect(self.filename)
        con.text_factory = db.OptimizedUnicode
        db.enable_callback_tracebacks(True)
        with con:
            cur = con.cursor()
            cur.execute("SELECT b.indx, fromNode.area, toNode.area"
                +" FROM Grid_Branches b"
                +" INNER JOIN Grid_Nodes fromNode ON b.fromIndx = fromNode.indx"
                +" INNER JOIN Grid_Nodes toNode ON b.toIndx = toNode.indx"
                +" WHERE fromNode.area != toNode.area")
            branches = cur.fetchall()
            
            #fetch flows
                        
            cur.execute("SELECT res.indx, TOTAL(res.flow)"
                +" FROM Res_Branches res, Grid_Branches b"
                +" INNER JOIN Grid_Nodes fromNode ON b.fromIndx = fromNode.indx"
                +" INNER JOIN Grid_Nodes toNode ON b.toIndx = toNode.indx"
                +" WHERE fromNode.area != toNode.area AND res.indx = b.indx AND timestep>=? AND timestep<? AND res.flow<=0"
                +" GROUP BY res.indx",
                (timeMaxMin[0],timeMaxMin[-1]))
            flow_negative = cur.fetchall()
            
            cur.execute("SELECT res.indx, TOTAL(res.flow)"
                +" FROM Res_Branches res, Grid_Branches b"
                +" INNER JOIN Grid_Nodes fromNode ON b.fromIndx = fromNode.indx"
                +" INNER JOIN Grid_Nodes toNode ON b.toIndx = toNode.indx"
                +" WHERE fromNode.area != toNode.area AND res.indx = b.indx AND timestep>=? AND timestep<? AND res.flow>=0"
                +" GROUP BY res.indx",
                (timeMaxMin[0],timeMaxMin[-1]))
            flow_positive = cur.fetchall()
            
            cur.execute("SELECT res.indx, TOTAL(res.flow)"
                +" FROM Res_Branches res, Grid_Branches b"
                +" INNER JOIN Grid_Nodes fromNode ON b.fromIndx = fromNode.indx"
                +" INNER JOIN Grid_Nodes toNode ON b.toIndx = toNode.indx"
                +" WHERE fromNode.area != toNode.area AND res.indx = b.indx AND timestep>=? AND timestep<?"
                +" GROUP BY res.indx",
                (timeMaxMin[0],timeMaxMin[-1]))
            flow_total = cur.fetchall()
            
            #calculate average flow
            
            numTimeSteps = timeMaxMin[-1] - timeMaxMin[0]
            flow_negative = [(index, flow/numTimeSteps) for (index, flow) in flow_negative]
            flow_positive = [(index, flow/numTimeSteps) for (index, flow) in flow_positive]
            flow_total = [(index, flow/numTimeSteps) for (index, flow) in flow_total]
            
            #Sort results
            # The length of flow lists may be less than the number
            # of branches if the flow is always in one direction
            values=[]
            
            #for all inter-area branches
            for index, indice in enumerate([x for (x,y,z) in branches]):
                #find negative flow
                try :
                    temp_ind = [y[0] for y in flow_negative].index(indice)
                    neg = (flow_negative[temp_ind][1],)
                except ValueError:
                    neg = (0,)
                #find positive flow
                try :
                    temp_ind = [y[0] for y in flow_positive].index(indice)
                    pos = (flow_positive[temp_ind][1],)
                except ValueError:
                    pos = (0,)
                #find total flow
                tot = (flow_total[index][1],)

                values.append(branches[index] + neg + pos + tot)
        return values
        
        
### Generator results
        
    def getResultStorageFilling(self,genindx,timeMaxMin):
        '''Get storage filling level for storage generators'''
        con = db.connect(self.filename)
        with con:        
            #con.row_factory = db.Row
            cur = con.cursor()
            cur.execute("SELECT storage FROM Res_Storage "
                +"WHERE timestep>=? AND timestep<? AND indx=?"
                +" ORDER BY timestep",
                (timeMaxMin[0],timeMaxMin[-1],genindx))
            rows = cur.fetchall()
            values = [row[0] for row in rows]        
        return values

    def getResultStorageValue(self,storageindx,timeMaxMin):
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

    
    def getResultGeneratorPower(self,generatorindx,timeMaxMin):
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
 
 
    def getResultGeneratorPowerInArea(self,area,timeMaxMin):
        '''Get accumulated generation per type in given area'''
        con = db.connect(self.filename)
        with con:        
            cur = con.cursor()
            cur.execute("SELECT output FROM Res_Generators "
                +"WHERE timestep>=? AND timestep<? AND indx IN "
                +"(SELECT indx FROM Grid_Generators WHERE node IN "
                +" (SELECT id FROM Grid_Nodes WHERE area IN (?)))"
                +" ORDER BY timestep",
                (timeMaxMin[0],timeMaxMin[-1],area))
            rows = cur.fetchall()
            output = [row[0] for row in rows]        
        return output
           