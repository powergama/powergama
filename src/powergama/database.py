"""
Module dealing with database IO
"""

import os
import sqlite3 as db

import pandas as pd


class DatabaseBaseClass(object):
    """
    Stripped-down version of class for storing results from PowerGAMA in sqlite databse
    """

    SQLITE_MAX_VARIABLE_NUMBER = 990

    def __init__(self, filename):
        self.filename = os.path.abspath(filename)
        self.sqlite_version = db.sqlite_version
        self.timestep_str = "timestep INT"
        self.timestep_qs = "?"

    def createTables(self, data):
        """
        Create database for PowerGAMA results
        """
        num_nodes = data.numNodes()
        num_branches = data.numBranches()
        num_generators = data.numGenerators()
        # convert from lists to tuple of tuples
        nodes = tuple(
            (i, data.node["id"][i], data.node["area"][i], 1.0 * data.node["lat"][i], 1.0 * data.node["lon"][i])
            for i in range(num_nodes)
        )
        generators = tuple(
            (
                i,
                data.generator["node"][i],
                data.generator["type"][i],
            )
            for i in range(num_generators)
        )
        br_from = data.branchFromNodeIdx()
        br_to = data.branchToNodeIdx()
        branches = tuple(
            (
                i,
                int(br_from[i]),
                int(br_to[i]),
                1.0 * data.branch["capacity"][i],
                1.0 * data.branch["reactance"][i],
                1.0 * data.branch["resistance"][i],
            )
            for i in range(num_branches)
        )

        if os.path.isfile(self.filename):
            # delete existing file
            print('Replacing existing SQLite file "{}"'.format(self.filename))
            os.remove(self.filename)
            # Must use a new file
            # raise IOError('Cannot append existing file. Choose new file name.')
        con = db.connect(self.filename)
        with con:
            cur = con.cursor()
            cur.execute("CREATE TABLE Grid_Nodes(indx INT, id TEXT, area TEXT," + "lat DOUBLE, lon DOUBLE)")
            cur.executemany("INSERT INTO Grid_Nodes VALUES(?,?,?,?,?)", nodes)
            cur.execute("CREATE TABLE Grid_Generators(indx INT, node TEXT," + "type TEXT)")
            cur.executemany("INSERT INTO Grid_Generators VALUES(?,?,?)", generators)
            cur.execute(
                "CREATE TABLE Grid_Branches(indx INT, fromIndx INT,"
                + "toIndx INT, capacity DOUBLE, reactance DOUBLE,"
                + "resistance DOUBLE)"
            )
            cur.executemany("INSERT INTO Grid_Branches VALUES(?,?,?,?,?,?)", branches)

            cur.execute(f"CREATE TABLE Res_ObjFunc({self.timestep_str}, value DOUBLE)")
            cur.execute(f"CREATE TABLE Res_Branches({self.timestep_str}, indx INT," + "flow DOUBLE, loss DOUBLE)")
            cur.execute(f"CREATE TABLE Res_BranchesSens({self.timestep_str}, indx INT," + "cap_sensitivity DOUBLE)")
            cur.execute(
                f"CREATE TABLE Res_DcBranches({self.timestep_str}, indx INT,"
                + "flow DOUBLE, cap_sensitivity DOUBLE,loss DOUBLE)"
            )
            cur.execute(
                f"CREATE TABLE Res_Nodes({self.timestep_str}, indx INT,"
                + "angle DOUBLE, nodalprice DOUBLE, loadshed DOUBLE)"
            )
            cur.execute(
                f"CREATE TABLE Res_Generators({self.timestep_str}, indx INT," + "output DOUBLE, inflow_spilled DOUBLE)"
            )
            cur.execute(
                f"CREATE TABLE Res_Storage({self.timestep_str}, indx INT," + "storage DOUBLE, marginalprice DOUBLE)"
            )
            cur.execute(f"CREATE TABLE Res_Pumping({self.timestep_str}, indx INT," + "output DOUBLE)")
            cur.execute(
                f"CREATE TABLE Res_FlexibleLoad({self.timestep_str}, indx INT,"
                + "demand DOUBLE, storage DOUBLE, value DOUBLE)"
            )

        return nodes

    def getTimerange(self):
        """
        Get the timesteps
        """
        con = db.connect(self.filename)
        with con:
            cur = con.cursor()
            cur.execute("SELECT timestep FROM Res_ObjFunc")
            rows = cur.fetchall()
            values = [row[0] for row in rows]
        return values

    def timestep_tuple(self, timestep, fault_start):
        # Base version only uses the timestep
        return (timestep,)

    def appendResults(
        self,
        timestep,
        objective_function,
        generator_power,
        generator_pumped,
        branch_flow,
        dcbranch_flow,
        node_angle,
        sensitivity_branch_capacity,
        sensitivity_dcbranch_capacity,
        sensitivity_node_power,
        storage,
        inflow_spilled,
        loadshed_power,
        marginalprice,
        flexload_power,
        flexload_storage,
        flexload_storagevalue,
        idx_storagegen,
        idx_branchsens,
        idx_pumpgen,
        idx_flexload,
        branch_ac_losses,
        branch_dc_losses,
        fault_start=None,
    ):
        """
        Store results from a given timestep to the database

        Parameters
        ----------
        timestep (int)
            timestep number
        objective_function (float)
            value of objective function
        generator_power (list of floats)
            power output of generators
        generator_pumped (list of floats)
            pumped power for generators
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
        flexload_power (list of floats)
            flexible load power consumption
        flexload_storage
            storage filling level of flexible load
        flexload_storagevalue
            storage value in flexible load energy storage
        idx_storagegen
            index in generator list of generators with storage
        idx_branchsens
            index in branch list of branches with limited capacity
        idx_pumpgen
            index in generator list of generators with pumping
        idx_flexload
            index in consumer list of flexible loads
        branch_ac_losses : list
            ac branch losses
        branch_dc_losses : list
            dc branch losses
        fault_start  (int)
            extra identifier for when simulations are run for several timesteps
            from multiple starting timepoints
        """

        timestep_tuple = self.timestep_tuple(timestep, fault_start)

        con = db.connect(self.filename)
        with con:
            cur = con.cursor()
            cur.execute(f"INSERT INTO Res_ObjFunc VALUES({self.timestep_qs},?)", timestep_tuple + (objective_function,))
            cur.executemany(
                f"INSERT INTO Res_Nodes VALUES({self.timestep_qs},?,?,?,?)",
                tuple(
                    timestep_tuple + (i, node_angle[i], sensitivity_node_power[i], loadshed_power[i])
                    for i in range(len(sensitivity_node_power))
                ),
            )
            cur.executemany(
                f"INSERT INTO Res_Branches VALUES({self.timestep_qs},?,?,?)",
                tuple(timestep_tuple + (i, branch_flow[i], branch_ac_losses[i]) for i in range(len(branch_flow))),
            )
            cur.executemany(
                f"INSERT INTO Res_BranchesSens VALUES({self.timestep_qs},?,?)",
                tuple(
                    timestep_tuple + (idx_branchsens[i], sensitivity_branch_capacity[i])
                    for i in range(len(sensitivity_branch_capacity))
                ),
            )
            cur.executemany(
                f"INSERT INTO Res_DcBranches VALUES({self.timestep_qs},?,?,?,?)",
                tuple(
                    timestep_tuple + (i, dcbranch_flow[i], sensitivity_dcbranch_capacity[i], branch_dc_losses[i])
                    for i in range(len(dcbranch_flow))
                ),
            )
            cur.executemany(
                f"INSERT INTO Res_Generators VALUES({self.timestep_qs},?,?,?)",
                tuple(timestep_tuple + (i, generator_power[i], inflow_spilled[i]) for i in range(len(generator_power))),
            )
            cur.executemany(
                f"INSERT INTO Res_Storage VALUES({self.timestep_qs},?,?,?)",
                tuple(timestep_tuple + (idx_storagegen[i], storage[i], marginalprice[i]) for i in range(len(storage))),
            )
            cur.executemany(
                f"INSERT INTO Res_Pumping VALUES({self.timestep_qs},?,?)",
                tuple(
                    timestep_tuple
                    + (
                        idx_pumpgen[i],
                        generator_pumped[i],
                    )
                    for i in range(len(generator_pumped))
                ),
            )
            cur.executemany(
                f"INSERT INTO Res_FlexibleLoad VALUES({self.timestep_qs},?,?,?,?)",
                tuple(
                    timestep_tuple + (idx_flexload[i], flexload_power[i], flexload_storage[i], flexload_storagevalue[i])
                    for i in range(len(flexload_power))
                ),
            )


class Database(DatabaseBaseClass):
    """
    Class for storing results from PowerGAMA in sqlite databse
    """

    def getGridNodeIndices(self):
        """Get node indices as a list"""
        con = db.connect(self.filename)
        with con:
            cur = con.cursor()
            cur.execute("SELECT indx FROM Grid_Nodes ")
            rows = cur.fetchall()
            values = [row[0] for row in rows]
        return values

    def getGridBranches(self):
        """Get branch indices as a list"""
        con = db.connect(self.filename)
        with con:
            # con.row_factory = db.Row
            cur = con.cursor()
            cur.execute("SELECT indx,fromIndx,toIndx,capacity,reactance " + " FROM Grid_Branches ")
            rows = cur.fetchall()
            values = {
                "indx": [row[0] for row in rows],
                "fromIndx": [row[1] for row in rows],
                "toIndx": [row[2] for row in rows],
                "capacity": [row[3] for row in rows],
                "reactance": [row[4] for row in rows],
            }
        return values

    def getGridInterareaBranches(self):
        """
        Get indices of branches between different areas as a list

        Returns
        =======

        (indice, fromArea, toArea)
        """
        con = db.connect(self.filename)
        with con:
            cur = con.cursor()
            cur.execute(
                "SELECT b.indx, fromNode.area, toNode.area"
                + " FROM Grid_Branches b"
                + " INNER JOIN Grid_Nodes fromNode ON b.fromIndx = fromNode.indx"
                + " INNER JOIN Grid_Nodes toNode ON b.toIndx = toNode.indx"
                + " WHERE fromNode.area != toNode.area"
            )
            output = cur.fetchall()
        return output

    def getGridGeneratorFromArea(self, area):
        """
        Get indices of generators  in given area as a list

        Returns
        =======

        (indice)
        """
        con = db.connect(self.filename)
        with con:
            cur = con.cursor()
            cur.execute(
                "SELECT g.indx FROM Res_Generators g"
                " INNER JOIN Grid_Generators gg ON g.indx = gg.indx"
                " INNER JOIN Grid_Nodes gn ON gg.node = gn.id"
                " WHERE gn.area=?"
                " GROUP BY g.indx",
                (area),
            )
            output = cur.fetchall()
        return output

    def getResultNodalPrice(self, nodeindx, timeMaxMin):
        """Get nodal price at specified node"""
        con = db.connect(self.filename)
        with con:
            cur = con.cursor()
            cur.execute(
                "SELECT nodalprice FROM Res_Nodes " "WHERE timestep>=? AND timestep<? AND indx=?" " ORDER BY timestep",
                (timeMaxMin[0], timeMaxMin[-1], nodeindx),
            )
            rows = cur.fetchall()
        values = [row[0] for row in rows]
        return values

    def getResultNodalPricesAll(self, timeMaxMin):
        """
        Get nodal price at all nodes (list of tuples)

        Returns
        =======
        List of tuples with values:
        (timestep, node index, nodal price)

        """
        con = db.connect(self.filename)
        with con:
            cur = con.cursor()
            cur.execute(
                "SELECT timestep,indx,nodalprice FROM Res_Nodes"
                " WHERE timestep>=? AND timestep<?"
                " ORDER BY indx,timestep",
                (timeMaxMin[0], timeMaxMin[-1]),
            )
            rows = cur.fetchall()
        return rows

    def getResultNodalPricesMean(self, timeMaxMin):
        """Get average nodal price at all nodes"""
        con = db.connect(self.filename)
        with con:
            cur = con.cursor()
            cur.execute(
                "SELECT indx,AVG(nodalprice) FROM Res_Nodes"
                " WHERE timestep>=? AND timestep<?"
                " GROUP BY indx ORDER BY indx",
                (timeMaxMin[0], timeMaxMin[-1]),
            )
            rows = cur.fetchall()
        values = [row[1] for row in rows]
        return values

    def getResultAreaPrices(self, node_weight, timeMaxMin):
        """Get area price timeseries

        node_weight = list of weights for each node
        """

        # print("weight sum = "+str(sum(node_weight)))
        con = db.connect(self.filename)
        with con:
            cur = con.cursor()

            # Temporary table to store weights
            cur.execute("CREATE TABLE IF NOT EXISTS weights" + " (node INT, weight DOUBLE)")
            val = tuple((i, w) for i, w in enumerate(node_weight))
            cur.executemany("INSERT INTO weights VALUES(?,?)", val)

            cur.execute(
                "SELECT n.timestep,SUM(n.nodalprice * w.weight)"
                " FROM Res_Nodes n INNER JOIN weights w ON n.indx==w.node"
                " WHERE n.timestep>=? AND n.timestep<?"
                " GROUP BY n.timestep ORDER BY n.timestep",
                (timeMaxMin[0], timeMaxMin[-1]),
            )
            rows = cur.fetchall()
            cur.execute("DROP TABLE weights")
        values = [row[1] for row in rows]
        return values

    def getResultBranchFlow(self, branchindx, timeMaxMin, ac=True):
        """Get branch flow at specified branch"""
        con = db.connect(self.filename)
        with con:
            cur = con.cursor()
            if ac:
                table = "Res_Branches"
            else:
                table = "Res_DcBranches"
            cur.execute(
                f"SELECT flow FROM {table} WHERE timestep>=? AND timestep<? AND indx=? ORDER BY timestep",  # nosec B608 - is safe even if bandit says no
                (timeMaxMin[0], timeMaxMin[-1], branchindx),
            )
            rows = cur.fetchall()
        values = [row[0] for row in rows]
        return values

    def getResultBranchFlowAll(self, timeMaxMin, acdc="ac"):
        """
        Get branch flow at all branches (list of tuples)

        Returns
        =======
        List of tuples with values:
        (timestep, branch index, flow)

        """
        valid_tables = {"ac": "Res_Branches", "dc": "Res_DcBranches"}
        if acdc not in valid_tables:
            raise Exception('branch type must be "ac" or "dc"')
        table = valid_tables[acdc]

        con = db.connect(self.filename)
        with con:
            cur = con.cursor()
            cur.execute(
                f"SELECT timestep,indx,flow FROM {table} WHERE timestep>=? AND timestep<? ORDER BY indx,timestep",  # nosec B608 - is safe even if bandit says no
                (timeMaxMin[0], timeMaxMin[-1]),
            )
            rows = cur.fetchall()
        return rows

    def getResultBranchFlowsMean(self, timeMaxMin, ac=True):
        """
        Get average branch flow on branches in both direction

        Parameters
        ----------
        timeMaxMin (list of two elements) - time interval
        ac (bool) - ac (true) or dc (false) branches

        Returns
        =======
        List with values for each branch:
        [average flow 1->2, average flow 2->1, average absolute flow]

        """
        if ac:
            table = "Res_Branches"
        else:
            table = "Res_DcBranches"

        con = db.connect(self.filename)
        with con:
            cur = con.cursor()
            cur.execute(
                f"""SELECT indx,TOTAL(flow) FROM {table}
                WHERE timestep>=? AND timestep<? AND flow>=0
                GROUP BY indx ORDER BY indx""",  # nosec B608 - is safe even if bandit says no
                (timeMaxMin[0], timeMaxMin[-1]),
            )
            rows1 = cur.fetchall()
            cur.execute(
                f"""SELECT indx,TOTAL(flow) FROM {table}
                WHERE timestep>=? AND timestep<? AND flow<0
                GROUP BY indx ORDER BY indx""",  # nosec B608 - is safe even if bandit says no
                (timeMaxMin[0], timeMaxMin[-1]),
            )
            rows2 = cur.fetchall()
            cur.execute(f"SELECT MAX(indx) FROM {table}")  # nosec B608 - is safe even if bandit says no
            numBranches = 1 + cur.fetchone()[0]
            # Calculate average flow for each direction
            numTimeSteps = timeMaxMin[-1] - timeMaxMin[0]
            rows1 = [(index, tot_flow / numTimeSteps) for (index, tot_flow) in rows1]
            rows2 = [(index, tot_flow / numTimeSteps) for (index, tot_flow) in rows2]
        # The length of rows1 and rows2 may be less than the number
        # of branches if the flow is always in one direction
        values_pos = [0] * numBranches
        values_neg = [0] * numBranches
        values_abs = [0] * numBranches
        i1 = 0
        i2 = 0
        for i in range(numBranches):
            if i1 < len(rows1) and rows1[i1][0] == i:
                values_pos[i] = rows1[i1][1]
                i1 = i1 + 1
            if i2 < len(rows2) and rows2[i2][0] == i:
                values_neg[i] = abs(rows2[i2][1])
                i2 = i2 + 1
            values_abs[i] = values_pos[i] + values_neg[i]
        values = [values_pos, values_neg, values_abs]
        return values

    def getResultBranchSens(self, branchindx, timeMaxMin, acdc="ac"):
        """Get branch capacity sensitivity at specified branch"""
        valid_tables = {"ac": "Res_BranchesSens", "dc": "Res_DcBranches"}
        if acdc not in valid_tables:
            raise Exception('branch type must be "ac" or "dc"')
        branch_table = valid_tables[acdc]
        con = db.connect(self.filename)
        with con:
            cur = con.cursor()
            cur.execute(
                f"SELECT cap_sensitivity FROM {branch_table} WHERE timestep>=? AND timestep<? AND indx=? ORDER BY timestep",  # nosec B608 - is safe even if bandit says no
                (timeMaxMin[0], timeMaxMin[-1], branchindx),
            )
            rows = cur.fetchall()
        values = [row[0] for row in rows]
        return values

    def getResultBranchSensAll(self, timeMaxMin):
        """Get branch capacity sensitivity at all branches"""
        branches = self.getGridBranches()
        branchlist = branches["indx"]
        sens = []
        for branch in branchlist:
            this_sens = self.getResultBranchSens(branch, timeMaxMin)
            sens.append(this_sens if this_sens != [] else [None] * (timeMaxMin[1] - timeMaxMin[0]))
        return sens

    def getResultBranchSensMean(self, timeMaxMin, acdc="ac"):
        """Get average sensitivity of all  branches
        acdc = 'ac' or 'dc'
        """
        valid_tables = {"ac": "Res_BranchesSens", "dc": "Res_DcBranches"}
        if acdc not in valid_tables:
            raise Exception('branch type must be "ac" or "dc"')
        branch_table = valid_tables[acdc]
        con = db.connect(self.filename)
        with con:
            cur = con.cursor()
            cur.execute(
                f"""SELECT indx,AVG(cap_sensitivity)
                FROM {branch_table}
                WHERE timestep>=? AND timestep<?
                GROUP BY indx ORDER BY indx""",  # nosec B608 - is safe even if bandit says no
                (timeMaxMin[0], timeMaxMin[-1]),
            )
            rows = cur.fetchall()
        values = [row[1] for row in rows]
        return values

    def getAverageInterareaBranchFlow(self, timeMaxMin):
        """
        Get average negative flow, positive flow and total flow of branches
        between different areas

        Returns
        =======
        List of tuples for inter-area branches with following values:
        (indices, fromArea, toArea, average negative flow, average positive
        flow, average flow)
        """

        con = db.connect(self.filename)
        db.enable_callback_tracebacks(True)
        with con:
            cur = con.cursor()
            cur.execute(
                "SELECT b.indx, fromNode.area, toNode.area"
                " FROM Grid_Branches b"
                " INNER JOIN Grid_Nodes fromNode ON b.fromIndx = fromNode.indx"
                " INNER JOIN Grid_Nodes toNode ON b.toIndx = toNode.indx"
                " WHERE fromNode.area != toNode.area"
            )
            branches = cur.fetchall()

            # fetch flows
            cur.execute(
                "SELECT res.indx, TOTAL(res.flow)"
                " FROM Res_Branches res, Grid_Branches b"
                " INNER JOIN Grid_Nodes fromNode ON b.fromIndx = fromNode.indx"
                " INNER JOIN Grid_Nodes toNode ON b.toIndx = toNode.indx"
                " WHERE fromNode.area != toNode.area"
                " AND res.indx = b.indx AND timestep>=?"
                " AND timestep<? AND res.flow<=0"
                " GROUP BY res.indx",
                (timeMaxMin[0], timeMaxMin[-1]),
            )
            flow_negative = cur.fetchall()

            cur.execute(
                "SELECT res.indx, TOTAL(res.flow)"
                " FROM Res_Branches res, Grid_Branches b"
                " INNER JOIN Grid_Nodes fromNode ON b.fromIndx = fromNode.indx"
                " INNER JOIN Grid_Nodes toNode ON b.toIndx = toNode.indx"
                " WHERE fromNode.area != toNode.area"
                " AND res.indx = b.indx AND timestep>=? AND timestep<?"
                " AND res.flow>=0"
                " GROUP BY res.indx",
                (timeMaxMin[0], timeMaxMin[-1]),
            )
            flow_positive = cur.fetchall()

            cur.execute(
                "SELECT res.indx, TOTAL(res.flow)"
                " FROM Res_Branches res, Grid_Branches b"
                " INNER JOIN Grid_Nodes fromNode ON b.fromIndx = fromNode.indx"
                " INNER JOIN Grid_Nodes toNode ON b.toIndx = toNode.indx"
                " WHERE fromNode.area != toNode.area"
                " AND res.indx = b.indx AND timestep>=? AND timestep<?"
                " GROUP BY res.indx",
                (timeMaxMin[0], timeMaxMin[-1]),
            )
            flow_total = cur.fetchall()

            # calculate average flow

            numTimeSteps = timeMaxMin[-1] - timeMaxMin[0]
            flow_negative = [(index, flow / numTimeSteps) for (index, flow) in flow_negative]
            flow_positive = [(index, flow / numTimeSteps) for (index, flow) in flow_positive]
            flow_total = [(index, flow / numTimeSteps) for (index, flow) in flow_total]

            # Sort results
            # The length of flow lists may be less than the number
            # of branches if the flow is always in one direction
            values = []

            # for all inter-area branches
            for index, indice in enumerate([x for (x, y, z) in branches]):
                # find negative flow
                try:
                    temp_ind = [y[0] for y in flow_negative].index(indice)
                    neg = (flow_negative[temp_ind][1],)
                except ValueError:
                    neg = (0,)
                # find positive flow
                try:
                    temp_ind = [y[0] for y in flow_positive].index(indice)
                    pos = (flow_positive[temp_ind][1],)
                except ValueError:
                    pos = (0,)
                # find total flow
                tot = (flow_total[index][1],)

                values.append(branches[index] + neg + pos + tot)
        return values

    def getBranchesSumFlow(self, branches_pos, branches_neg, timeMaxMin, acdc):
        """
        Return time series for aggregated flow along specified branches

        branches_pos = indices of branches with positive flow direction
        branches_neg = indices of branches with negative flow direction
        timeMaxMin = [start, end]
        acdc = 'ac' or 'dc'

        Note: This function can be used to get net import, but not separate
        import/export values (requires summing positive and negative values
        separately)

        """
        values_pos = []
        values_neg = []
        valid_tables = {"ac": "Res_Branches", "dc": "Res_DcBranches"}
        if acdc not in valid_tables:
            raise Exception('branch type must be "ac" or "dc"')
        branch_table = valid_tables[acdc]

        con = db.connect(self.filename)
        with con:
            if branches_pos:
                cur = con.cursor()
                str_bind = ",".join("?" for _ in branches_pos)
                cur.execute(
                    f"""
                    SELECT SUM(flow) FROM {branch_table}
                    WHERE timestep>=? AND timestep<? AND indx IN ({str_bind})
                    GROUP BY timestep ORDER BY timestep""",  # nosec B608 - is safe even if bandit says no
                    (timeMaxMin[0], timeMaxMin[-1]) + tuple(branches_pos),
                )
                rows = cur.fetchall()
                values_pos = [row[0] for row in rows]
            if branches_neg:
                cur = con.cursor()
                str_bind = ",".join("?" for _ in branches_neg)
                cur.execute(
                    f"""
                    SELECT SUM(flow) FROM {branch_table}
                    WHERE timestep>=? AND timestep<? AND indx IN ({str_bind})
                    GROUP BY timestep ORDER BY timestep""",  # nosec B608 - is safe even if bandit says no
                    (timeMaxMin[0], timeMaxMin[-1]) + tuple(branches_neg),
                )
                rows = cur.fetchall()
                values_neg = [row[0] for row in rows]
            values = dict(pos=values_pos, neg=values_neg)

        return values

    def getResultPumpPower(self, genindx, timeMaxMin):
        """Get pumping for generators with pumping"""
        con = db.connect(self.filename)
        with con:
            # con.row_factory = db.Row
            cur = con.cursor()
            cur.execute(
                "SELECT output FROM Res_Pumping WHERE timestep>=? AND timestep<? AND indx=? ORDER BY timestep",
                (timeMaxMin[0], timeMaxMin[-1], genindx),
            )
            rows = cur.fetchall()
            values = [row[0] for row in rows]
        return values

    def getResultPumpPowerMultiple(self, genindx, timeMaxMin, negative=True):
        """Get pumping for generators with pumping"""

        con = db.connect(self.filename)
        with con:
            # con.row_factory = db.Row
            cur = con.cursor()
            str_bind = ",".join("?" for _ in genindx)
            cur.execute(
                f"""SELECT SUM(output) FROM Res_Pumping
                WHERE timestep>=? AND timestep<? AND indx IN ({str_bind})
                GROUP BY timestep ORDER BY timestep""",  # nosec B608 - is safe even if bandit says no
                (timeMaxMin[0], timeMaxMin[-1]) + tuple(genindx),
            )
            rows = cur.fetchall()
            if negative:
                values = [-row[0] for row in rows]
            else:
                values = [row[0] for row in rows]
        return values

    def getResultStorageFilling(self, genindx, timeMaxMin):
        """Get storage filling level for storage generators"""
        con = db.connect(self.filename)
        with con:
            # con.row_factory = db.Row
            cur = con.cursor()
            cur.execute(
                "SELECT storage FROM Res_Storage WHERE timestep>=? AND timestep<? AND indx=? ORDER BY timestep",
                (timeMaxMin[0], timeMaxMin[-1], genindx),
            )
            rows = cur.fetchall()
            values = [row[0] for row in rows]
        return values

    def getResultStorageFillingAll(self, timestep):
        """Get storage filling level for all storage generators"""
        con = db.connect(self.filename)
        with con:
            cur = con.cursor()
            cur.execute("SELECT indx,storage FROM Res_Storage WHERE timestep=?  ORDER BY indx", (timestep,))
            rows = cur.fetchall()
            values = [row[1] for row in rows]
        return values

    def getResultStorageFillingMultiple(self, genindx, timeMaxMin, capacity=None):
        """Get storage filling level for multiple storage generators"""
        con = db.connect(self.filename)
        with con:
            # con.row_factory = db.Row
            str_bind = ",".join("?" for _ in genindx)
            cur = con.cursor()
            cur.execute(
                f"""SELECT SUM(storage) FROM Res_Storage
                WHERE timestep>=? AND timestep<? AND indx IN ({str_bind})
                GROUP BY timestep ORDER BY timestep""",  # nosec B608 - is safe even if bandit says no
                (timeMaxMin[0], timeMaxMin[-1]) + tuple(genindx),
            )
            rows = cur.fetchall()
            if capacity:
                values = [row[0] / capacity for row in rows]
            else:
                values = [row[0] for row in rows]
        return values

    def getResultStorageValue(self, storageindx, timeMaxMin):
        """Get storage value for storage generators"""
        con = db.connect(self.filename)
        with con:
            cur = con.cursor()
            cur.execute(
                "SELECT marginalprice FROM Res_Storage "
                " WHERE timestep>=? AND timestep<? AND indx=?"
                " ORDER BY timestep",
                (timeMaxMin[0], timeMaxMin[-1], storageindx),
            )
            rows = cur.fetchall()
            values = [row[0] for row in rows]
        return values

    def getResultStorageValueMultiple(self, storageindx, timeMaxMin):
        """Get average storage value (marginal price) for multiple
        storage generators"""

        con = db.connect(self.filename)
        with con:
            str_bind = ",".join("?" for _ in storageindx)
            cur = con.cursor()
            cur.execute(
                f"""SELECT AVG(marginalprice) FROM Res_Storage
                WHERE timestep>=? AND timestep<? AND indx IN ({str_bind})
                GROUP BY timestep ORDER BY timestep""",  # nosec B608 - is safe even if bandit says no
                (timeMaxMin[0], timeMaxMin[-1]) + tuple(storageindx),
            )
            rows = cur.fetchall()
            values = [row[0] for row in rows]
        return values

    def getResultGeneratorSpilledSums(self, timeMaxMin):
        """Get sum of spilled power for all generator"""

        con = db.connect(self.filename)
        with con:
            cur = con.cursor()
            cur.execute(
                "SELECT indx,SUM(inflow_spilled) "
                " FROM Res_Generators "
                " WHERE timestep>=? AND timestep<?"
                " GROUP BY indx ORDER BY indx",
                (timeMaxMin[0], timeMaxMin[-1]),
            )
            rows = cur.fetchall()
            output = [row[1] for row in rows]
        return output

    def getResultGeneratorSpilled(self, generatorindx, timeMaxMin):
        """Get spilled power time series for specified generator"""

        if not isinstance(generatorindx, list):
            generatorindx = [generatorindx]
        if len(generatorindx) == 0:
            return None
        con = db.connect(self.filename)
        with con:
            cur = con.cursor()
            str_bind = ",".join("?" for _ in generatorindx)
            cur.execute(
                f"""SELECT timestep,SUM(inflow_spilled)
                FROM Res_Generators
                WHERE timestep>=? AND timestep<? AND indx IN ({str_bind})
                GROUP BY timestep ORDER BY timestep""",  # nosec B608 - is safe even if bandit says no
                (timeMaxMin[0], timeMaxMin[-1]) + tuple(generatorindx),
            )
            rows = cur.fetchall()
            output = [row[1] for row in rows]
        return output

    def getResultGeneratorPower(self, generatorindx, timeMaxMin):
        """Get power output time series for specified generator"""

        if not isinstance(generatorindx, list):
            generatorindx = [generatorindx]
        if len(generatorindx) == 0:
            return None

        # If there are many generators we need to worry about the max variable number in sqlite
        maxnum = self.SQLITE_MAX_VARIABLE_NUMBER
        indx_chunks = [generatorindx[i : i + maxnum] for i in range(0, len(generatorindx), maxnum)]
        con = db.connect(self.filename)
        with con:
            cur = con.cursor()
            # get results for chunks of generators:
            output_chunk = []
            for indx in indx_chunks:
                str_bind = ",".join("?" for _ in indx)
                cur.execute(
                    f"""SELECT timestep,SUM(output) FROM Res_Generators
                    WHERE timestep>=? AND timestep<? AND indx IN ({str_bind})
                    GROUP BY timestep ORDER BY timestep""",  # nosec B608 - is safe even if bandit says no
                    (timeMaxMin[0], timeMaxMin[-1]) + tuple(indx),
                )
                rows = cur.fetchall()
                output_chunk.append([row[1] for row in rows])
            # sum chunks for each timestep:
            output = [sum(i) for i in zip(*output_chunk)]
        return output

    def getResultPumpingSum(self, timeMaxMin, variable="output"):
        """Sum of pumping  per generator"""
        con = db.connect(self.filename)
        with con:
            # cur = con.cursor()
            query = (
                "SELECT indx,SUM(?) FROM Res_Pumping "
                " WHERE timestep>=? AND timestep<?"
                " GROUP BY indx"
                " ORDER BY indx",
                (variable, timeMaxMin[0], timeMaxMin[-1]),
            )
            df = pd.read_sql_query(query, con)
        return df

    def getResultGeneratorPowerSum(self, timeMaxMin):
        """Sum of generator power output per generator"""
        con = db.connect(self.filename)
        with con:
            cur = con.cursor()
            cur.execute(
                "SELECT indx,SUM(output) FROM Res_Generators "
                " WHERE timestep>=? AND timestep<?"
                " GROUP BY indx"
                " ORDER BY indx",
                (timeMaxMin[0], timeMaxMin[-1]),
            )
            rows = cur.fetchall()
            values = [row[1] for row in rows]
        return values

    def getResultGeneratorPowerInArea(self, area, timeMaxMin):
        """Get accumulated generation per type in given area"""
        con = db.connect(self.filename)
        with con:
            cur = con.cursor()
            cur.execute(
                "SELECT output FROM Res_Generators "
                " WHERE timestep>=? AND timestep<? AND indx IN "
                " (SELECT indx FROM Grid_Generators WHERE node IN "
                " (SELECT id FROM Grid_Nodes WHERE area IN (?)))"
                " ORDER BY timestep",
                (timeMaxMin[0], timeMaxMin[-1], area),
            )
            rows = cur.fetchall()
            output = [row[0] for row in rows]
        return output

    def getResultFlexloadPower(self, consumerindx, timeMaxMin):
        """Get flexible load for consumer with flexible load"""
        con = db.connect(self.filename)
        with con:
            # con.row_factory = db.Row
            cur = con.cursor()
            cur.execute(
                "SELECT demand FROM Res_FlexibleLoad "
                " WHERE timestep>=? AND timestep<? AND indx=?"
                " ORDER BY timestep",
                (timeMaxMin[0], timeMaxMin[-1], consumerindx),
            )
            rows = cur.fetchall()
            values = [row[0] for row in rows]
        return values

    def getResultFlexloadStorageFilling(self, consumerindx, timeMaxMin):
        """Get storage filling level for flexible loads"""
        con = db.connect(self.filename)
        with con:
            # con.row_factory = db.Row
            cur = con.cursor()
            cur.execute(
                "SELECT storage FROM Res_FlexibleLoad "
                " WHERE timestep>=? AND timestep<? AND indx=?"
                " ORDER BY timestep",
                (timeMaxMin[0], timeMaxMin[-1], consumerindx),
            )
            rows = cur.fetchall()
            values = [row[0] for row in rows]
        return values

    def getResultFlexloadStorageValue(self, consumerindx, timeMaxMin):
        """Get storage value for flexible loads"""
        con = db.connect(self.filename)
        with con:
            cur = con.cursor()
            cur.execute(
                "SELECT value FROM Res_FlexibleLoad "
                " WHERE timestep>=? AND timestep<? AND indx=?"
                " ORDER BY timestep",
                (timeMaxMin[0], timeMaxMin[-1], consumerindx),
            )
            rows = cur.fetchall()
            values = [row[0] for row in rows]
        return values

    def getResultLoadheddingInArea(self, area, timeMaxMin):
        """Aggregated loadshedding timeseries for specified area"""
        con = db.connect(self.filename)
        with con:
            cur = con.cursor()
            cur.execute(
                "SELECT SUM(loadshed) FROM Res_Nodes "
                " WHERE timestep>=? AND timestep<? AND indx IN "
                " (SELECT indx FROM Grid_Nodes WHERE area IN (?))"
                " GROUP BY timestep"
                " ORDER BY timestep",
                (timeMaxMin[0], timeMaxMin[-1], area),
            )
            rows = cur.fetchall()
            values = [row[0] for row in rows]
        return values

    def getResultLoadheddingSum(self, timeMaxMin):
        """Sum of loadshedding timeseries per node"""
        con = db.connect(self.filename)
        with con:
            cur = con.cursor()
            cur.execute(
                "SELECT indx,SUM(loadshed) FROM Res_Nodes "
                " WHERE timestep>=? AND timestep<?"
                " GROUP BY indx"
                " ORDER BY indx",
                (timeMaxMin[0], timeMaxMin[-1]),
            )
            rows = cur.fetchall()
            values = [row[1] for row in rows]
        return values

    def getResultBranchLossesSum(self, timeMaxMin, acdc="ac"):
        """Sum of losses for each time-step time step"""
        sqlTable = "Res_Branches"
        if acdc == "dc":
            sqlTable = "Res_DcBranches"
        con = db.connect(self.filename)
        with con:
            cur = con.cursor()
            cur.execute(
                "SELECT indx,SUM(loss) FROM ?" " WHERE timestep>=? AND timestep<?" " GROUP BY timestep",
                (sqlTable, timeMaxMin[0], timeMaxMin[-1]),
            )
            rows = cur.fetchall()
            values = [row[1] for row in rows]
        return values

    def getResultBranches(self, timeMaxMin, br_indx=None, acdc="ac"):
        """Branch results for each time-step

        Parameters
        ==========
        timeMaxMin : [start,end]
            tuple with time window start <= t < end
        br_indx : list (optional)
            list of branches to consider, None=include all
        """
        valid_tables = {"ac": "Res_Branches", "dc": "Res_DcBranches"}
        if acdc not in valid_tables:
            raise Exception('branch type must be "ac" or "dc"')
        table = valid_tables[acdc]
        con = db.connect(self.filename)
        with con:
            if br_indx is None:
                query = f"SELECT * FROM {table} WHERE timestep>=? AND timestep<? GROUP BY indx,timestep"  # nosec B608 - is safe even if bandit says no
                df = pd.read_sql_query(query, con, params=(timeMaxMin[0], timeMaxMin[-1]))
            elif len(br_indx) == 0:
                # Empty dataframe
                df = pd.DataFrame()
            else:
                str_bind = ",".join("?" for _ in br_indx)
                query = f"""SELECT * FROM {table}
                    WHERE timestep>=? AND timestep<?  AND indx IN ({str_bind})
                    GROUP BY indx,timestep"""  # nosec B608 - is safe even if bandit says no
                df = pd.read_sql_query(query, con, params=(timeMaxMin[0], timeMaxMin[-1]) + tuple(br_indx))
        return df
