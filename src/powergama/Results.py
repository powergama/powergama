# -*- coding: utf-8 -*-
"""
Module containing the PowerGAMA Results class
"""

import csv
import math

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import powergama.database as db


class ResultsBaseClass(object):
    """
    Class for storing and analysing/presenting results from PowerGAMA
    Base version without plotting or results extraction

    Parameters
    ----------
    grid : GridData
        PowerGAMA GridData object
    databasefile : string
        name of sqlite3 file for storage of results
    replace : boolean
        whether to replace existing sqlite file (default=true).
        replace=false is useful to analyse previously
        generated results
    """

    def __init__(self, grid, databasefile, replace=True, sip=False):
        """
        Create a PowerGAMA Results object



        """
        self.grid = grid
        self.timerange = grid.timerange
        if sip:
            return
        self.storage_idx_generators = grid.getIdxGeneratorsWithStorage()
        self.pump_idx_generators = grid.getIdxGeneratorsWithPumping()
        self.flex_idx_consumers = grid.getIdxConsumersWithFlexibleLoad()
        self.idxConstrainedBranchCapacity = grid.getIdxBranchesWithFlowConstraints()

        self._init_database(databasefile)
        if replace:
            self.db.createTables(grid)
        else:
            # check that the length of the specified timerange matches the
            # database
            timerange_db = self.db.getTimerange()
            if timerange_db != list(self.timerange):
                print("Database time range = [%d,%d]\n" % (timerange_db[0], timerange_db[-1]))
                raise Exception("Database time range mismatch")

        """
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
        """

    def _init_database(self, databasefile):
        self.db = db.Database(databasefile)

    def addResultsFromTimestep(
        self,
        timestep,
        objective_function,
        generator_power,
        generator_pumped,
        branch_power,
        dcbranch_power,
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
        branch_ac_losses=None,
        branch_dc_losses=None,
        fault_start=None,
    ):
        """Store results from optimal power flow for a new timestep

        timestep : int
            timestamp of results
        objective_function : float
            value of objective function
        generator_power : list
            generator output (list of same length and order as generators)
        generator_pumped : list
            position according to grid.getIdxGeneratorsWithPumping()
        branch_power : list
            AC branch power flow (list of same length and order as branches)
        dcbranch_power : list
            DC branch power flow (list of same length and order as dcbranches)
        node_angles : list
            voltage angles (list of same length and order as nodes)
        sensitivity_branch_capacity : list
            dual, position according to grid.getIdxBranchesWithFlowConstraints()
        sensitivity_dcbranch_capacity : list
            dual, capacity (list of same length and order as dcbranches)
        sensitivity_node_power : list
            dual, node demand(list of same length and order as nodes)
        storage : list
            position accordint to grid.getIdxGeneratorsWithStorage()
        inflow_spilled : list
            spileld power (list of same length and order as generators)
        loadshed_power : list
            same length and order as nodes
        marginalprice : list
            position according to grid.getIdxGeneratorsWithStorage()
        flexload_power : list
            position according to grid.getIdxConsumersWithFlexibleLoad()
        flexload_storage : list
            position according to grid.getIdxConsumersWithFlexibleLoad()
        flexload_storagevalue : list
            position according to grid.getIdxConsumersWithFlexibleLoad()
        branch_ac_losses : list
            ac branch losses
        branch_dc_losses : list
            dc branch losses
        fault_start  (int)
            extra identifier for when simulations are run for several timesteps
            from multiple starting timepoints
        """
        # Use zero if no branch power losses given:
        if branch_ac_losses is None:
            branch_ac_losses = [0] * len(branch_power)
        if branch_dc_losses is None:
            branch_dc_losses = [0] * len(dcbranch_power)
        # Store results in sqlite database on disk (to avoid memory problems)
        self.db.appendResults(
            timestep=timestep,
            objective_function=objective_function,
            generator_power=generator_power,
            generator_pumped=generator_pumped,
            branch_flow=branch_power,
            dcbranch_flow=dcbranch_power,
            node_angle=node_angle,
            sensitivity_branch_capacity=sensitivity_branch_capacity,
            sensitivity_dcbranch_capacity=sensitivity_dcbranch_capacity,
            sensitivity_node_power=sensitivity_node_power,
            storage=storage,
            inflow_spilled=inflow_spilled,
            loadshed_power=loadshed_power,
            marginalprice=marginalprice,
            flexload_power=flexload_power,
            flexload_storage=flexload_storage,
            flexload_storagevalue=flexload_storagevalue,
            idx_storagegen=self.storage_idx_generators,
            idx_branchsens=self.idxConstrainedBranchCapacity,
            idx_pumpgen=self.pump_idx_generators,
            idx_flexload=self.flex_idx_consumers,
            branch_ac_losses=branch_ac_losses,
            branch_dc_losses=branch_dc_losses,
            fault_start=fault_start,
        )

        """
        self.objectiveFunctionValue.append(objective_function)
        self.generatorOutput.append(generator_power)
        self.branchFlow.append(branch_power)
        self.nodeAngle.append(node_angle)
        self.sensitivityBranchCapacity.append(sensitivity_branch_capacity)
        self.sensitivityDcBranchCapacity.append(sensitivity_dcbranch_capacity)
        self.sensitivityNodePower.append(sensitivity_node_power)
        self.storage.append(storage)
        self.inflowSpilled.append(inflow_spilled)
        self.loadshed.append(loadshed_power)
        self.marginalprice.append(marginalprice)
        """
        # self.storageGeneratorsIdx.append(idx_generatorsWithStorage)


class Results(ResultsBaseClass):
    """
    Class for storing and analysing/presenting results from PowerGAMA

    Parameters
    ----------
    grid : GridData
        PowerGAMA GridData object
    databasefile : string
        name of sqlite3 file for storage of results
    replace : boolean
        whether to replace existing sqlite file (default=true).
        replace=false is useful to analyse previously
        generated results
    """

    def getAverageBranchFlows(self, timeMaxMin=None, branchtype="ac"):
        """
        Average flow on branches over a given time period

        Parameters
        ==========
        timeMaxMin : list (default = None)
            [min, max] - lower and upper time interval
        branchtype : string
            'ac' (default) or 'dc'

        Returns
        =======
        List with values for each branch:
        [flow from 1 to 2, flow from 2 to 1, average absolute flow]
        """
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]

        # branchflow = self.db.getResultBranchFlowAll(timeMaxMin)
        if branchtype == "ac":
            ac = True
        elif branchtype == "dc":
            ac = False
        else:
            raise Exception('Branch type must be "ac" or "dc"')

        avgflow = self.db.getResultBranchFlowsMean(timeMaxMin, ac)
        # np.mean(branchflow,axis=1)
        return avgflow

    def getNodalPrices(self, node, timeMaxMin=None):
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]

        prices = self.db.getResultNodalPrice(node, timeMaxMin)
        # use asarray to convert None to nan
        prices = np.asarray(prices, dtype=float)
        return prices

    def getAverageNodalPrices(self, timeMaxMin=None):
        """
        Average nodal price over a given time period

        Parameters
        ----------
        timeMaxMin (list) (default = None)
            [min, max] - lower and upper time interval

        Returns
        =======
        1-dim Array of nodal prices (one per node)
        """
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]

        avgprices = self.db.getResultNodalPricesMean(timeMaxMin)
        # use asarray to convert None to nan
        avgprices = np.asarray(avgprices, dtype=float)
        return avgprices

    def getAreaPrices(self, area, timeMaxMin=None):
        """
        Weighted average nodal price timeseries for given area
        """
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]

        # area_nodes = [n._i for n in self.grid.node if n.area==area]
        loads = self.grid.getConsumersPerArea()[area]
        node_weight = [0] * len(self.grid.node["id"])
        for ld in loads:
            the_node = self.grid.consumer["node"][ld]
            the_load = self.grid.consumer["demand_avg"][ld]
            node_indx = self.grid.node["id"].tolist().index(the_node)
            node_weight[node_indx] += the_load

        sumWght = sum(node_weight)
        node_weight = [a / sumWght for a in node_weight]

        # print("Weights:")
        # print(node_weight)
        prices = self.db.getResultAreaPrices(node_weight, timeMaxMin)

        return prices

    def getAreaPricesAverage(self, areas=None, timeMaxMin=None):
        """
        Time average of weighted average nodal price per area
        """
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]
        if areas is None:
            areas = self.grid.getAllAreas()

        avg_nodalprices = self.getAverageNodalPrices(timeMaxMin)
        all_loads = self.grid.getConsumersPerArea()
        avg_areaprice = {}

        for area in areas:
            nodes_in_area = [i for i, n in enumerate(self.grid.node.area) if n == area]
            node_weight = [0] * len(self.grid.node.id)
            if area in all_loads:
                loads = all_loads[area]
                for ld in loads:
                    the_node = self.grid.consumer.node[ld]
                    the_load = self.grid.consumer.demand_avg[ld]
                    node_indx = self.grid.node.id.tolist().index(the_node)
                    node_weight[node_indx] += the_load
                sumWght = sum(node_weight)
                node_weight = [a / sumWght for a in node_weight]

                prices = [node_weight[i] * avg_nodalprices[i] for i in nodes_in_area]
            else:
                # flat weight if there are no loads in area
                prices = [avg_nodalprices[i] for i in nodes_in_area]
            avg_areaprice[area] = sum(prices)

        return avg_areaprice

    def getLoadheddingInArea(self, area, timeMaxMin=None):
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]

        loadshed = self.db.getResultLoadheddingInArea(area, timeMaxMin)
        # use asarray to convert None to nan
        loadshed = np.asarray(loadshed, dtype=float)
        return loadshed

    def getLoadsheddingPerNode(self, timeMaxMin=None):
        """get loadshedding sum per node"""
        timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]

        loadshed_per_node = self.db.getResultLoadheddingSum(timeMaxMin)
        return loadshed_per_node

    def getLoadheddingSums(self, timeMaxMin=None):
        """get loadshedding sum per area"""
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]

        loadshed_per_node = self.db.getResultLoadheddingSum(timeMaxMin)
        areas = self.grid.node.area
        allareas = self.grid.getAllAreas()
        loadshed_sum = dict()
        for a in allareas:
            loadshed_sum[a] = sum([loadshed_per_node[i] for i in range(len(areas)) if areas[i] == a])

        # loadshed_sum = np.asarray(loadshed_sum,dtype=float)
        return loadshed_sum

    def getAverageEnergyBalance(self, timeMaxMin=None):
        """
        Average energy balance (generation minus demand) over a time period

        timeMaxMin (list) (default = None)
            [min, max] - lower and upper time interval

        Returns
        =======
        1-dim Array of nodal prices (one per node)
        """
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]

        branchflows = self.db.getResultBranchFlowsMean(timeMaxMin)
        if self.grid.numDcBranches() > 0:
            branchflowsDc = self.db.getResultBranchFlowsMean(timeMaxMin, ac=False)
        br_from = self.grid.branchFromNodeIdx()
        br_to = self.grid.branchToNodeIdx()
        dcbr_from = self.grid.dcBranchFromNodeIdx()
        dcbr_to = self.grid.dcBranchToNodeIdx()
        energybalance = []
        for n in range(len(self.grid.node["id"])):
            idx_from = [i for i, x in enumerate(br_from) if x == n]
            idx_to = [i for i, x in enumerate(br_to) if x == n]
            dc_idx_from = [i for i, x in enumerate(dcbr_from) if x == n]
            dc_idx_to = [i for i, x in enumerate(dcbr_to) if x == n]
            energybalance.append(
                sum([branchflows[0][i] - branchflows[1][i] for i in idx_from])
                - sum([branchflows[0][j] - branchflows[1][j] for j in idx_to])
                + sum([branchflowsDc[0][i] - branchflowsDc[1][i] for i in dc_idx_from])
                - sum([branchflowsDc[0][j] - branchflowsDc[1][j] for j in dc_idx_to])
            )

        # use asarray to convert None to nan
        energybalance = np.asarray(energybalance, dtype=float)
        return energybalance

    def getAverageBranchSensitivity(self, timeMaxMin=None, branchtype="ac"):
        """
        Average branch capacity sensitivity over a given time period

        Parameters
        ----------
        timeMaxMin (list) (default = None)
            [min, max] - lower and upper time interval
        branchtype : str
            ac or dc branch type

        Returns
        =======
        1-dim Array of sensitivities (one per branch)
        """
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]

        avgsense = self.db.getResultBranchSensMean(timeMaxMin, branchtype)
        # use asarray to convert None to nan
        avgsense = np.asarray(avgsense, dtype=float)
        return avgsense

    def getAverageUtilisation(self, timeMaxMin=None, branchtype="ac"):
        """
        Average branch utilisation over a given time period

        Parameters
        ----------
        timeMaxMin :  (list) (default = None)
            [min, max] - lower and upper time interval
        branchtype : str
            ac or dc branch type

        Returns
        =======
        1-dim Array of branch utilisation (power flow/capacity)
        """
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]
        if branchtype == "ac":
            cap = self.grid.branch.capacity
        elif branchtype == "dc":
            cap = self.grid.dcbranch.capacity
        avgflow = self.getAverageBranchFlows(timeMaxMin, branchtype)[2]
        utilisation = [avgflow[i] / cap.iloc[i] for i in range(len(cap))]
        utilisation = np.asarray(utilisation)
        return utilisation

    def getSystemCostOBSOLETE(self, timeMaxMin=None):
        """
        Calculates system cost for energy produced by using generator fuel cost.

        Parameters
        ----------
        timeMaxMin (list) (default = None)
            [min, max] - lower and upper time interval

        Returns
        =======
        array of tuples of total cost of energy per area for all areas
        [(area, costs), ...]
        """

        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]

        systemcost = []
        # for each area
        for area in self.grid.getAllAreas():
            areacost = 0
            # for each generator
            for gen in self.db.getGridGeneratorFromArea(area):
                # sum generator output and multiply by fuel cost
                for power in self.db.getResultGeneratorPower(gen[0], timeMaxMin):
                    areacost += power * self.grid.generator.fuelcost[gen[0]]
            systemcost.append(tuple([area, areacost]))
        return systemcost

    def getSystemCost(self, timeMaxMin=None):
        """
        Calculates system cost for energy produced by using generator fuel cost.

        Parameters
        ----------
        timeMaxMin (list) (default = None)
            [min, max] - lower and upper time interval

        Returns
        -------
        array of dictionary of cost of generation sorted per area
        """
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]

        generation_per_gen = self.db.getResultGeneratorPowerSum(timeMaxMin)
        fuelcost_per_gen = self.grid.generator["fuelcost"]
        areas_per_gen = [
            self.grid.node["area"][self.grid.node["id"] == n].tolist()[0] for n in self.grid.generator["node"]
        ]

        allareas = self.grid.getAllAreas()
        generationcost = dict()
        for a in allareas:
            generationcost[a] = sum(
                [
                    generation_per_gen[i] * fuelcost_per_gen[i]
                    for i in range(len(areas_per_gen))
                    if areas_per_gen[i] == a
                ]
            )

        return generationcost

    def getGeneratorOutputSumPerArea(self, timeMaxMin=None):
        """
        Description
        Sums up generation per area.

        Parameters
        ----------
        timeMaxMin (list) (default = None)
            [min, max] - lower and upper time interval

        Returns
        =======
        array of dictionary of generation sorted per area
        """
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]

        generation_per_gen = self.db.getResultGeneratorPowerSum(timeMaxMin)
        areas_per_gen = [self.grid.node.area[self.grid.node.id == n].tolist()[0] for n in self.grid.generator.node]

        allareas = self.grid.getAllAreas()
        generation = dict()
        for a in allareas:
            generation[a] = sum([generation_per_gen[i] for i in range(len(areas_per_gen)) if areas_per_gen[i] == a])

        return generation

    def getGeneratorSpilledSums(self, timeMaxMin=None):
        """Get sum of spilled inflow for all generators

        Parameters
        ----------
        timeMaxMin (list) (default = None)
            [min, max] - lower and upper time interval
        """
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]
        v = self.db.getResultGeneratorSpilledSums(timeMaxMin)
        return v

    def getGeneratorSpilled(self, generatorindx, timeMaxMin=None):
        """Get spilled inflow time series for given generator

        Parameters
        ----------
        generatorindx (int)
            index ofgenerator
        timeMaxMin (list) (default = None)
            [min, max] - lower and upper time interval
        """
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]
        v = self.db.getResultGeneratorSpilled(generatorindx, timeMaxMin)
        return v

    def getGeneratorStorageAll(self, timestep):
        """Get stored energy for all storage generators at given time

        Parameters
        ----------
        timestep : int
            timestep when storage is requested
        """
        v = self.db.getResultStorageFillingAll(timestep)

        return v

    def getGeneratorStorageValues(self, timestep):
        """Get value of stored energy for given time

        Parameters
        ----------
        timestep : int
            when to compute value

        Returns
        -------
        list of int
            Value of stored energy for all storage generators

        The method uses the storage value absolute level (basecost) per
        generator to compute total storage value
        """
        storage_energy = self.getGeneratorStorageAll(timestep)
        storage_values = self.grid.generator.storage_price
        indx_storage_generators = self.grid.getIdxGeneratorsWithStorage()
        storval = [storage_energy[i] * storage_values[v] for i, v in enumerate(indx_storage_generators)]
        return storval

    def _node2area(self, nodeName):
        """Returns the area of a spacified node"""
        # Is handy when you need to access more information about the node,
        # but only the node name is avaiable. (which is the case in the generator file)
        area = self.grid.node.loc[self.grid.node["id"] == nodeName, "area"].iloc[0]
        return area

    def _getAreaTypeProduction(self, area, generatorType, timeMaxMin):
        """
        Returns total production for specified area nd generator type
        """

        print("Looking for generators of type " + str(generatorType) + ", in " + str(area))
        print("Number of generator to run through: " + str(self.grid.generator.numGenerators()))
        totalProduction = 0

        for genNumber in range(0, self.grid.generator.numGenerators()):
            genNode = self.grid.generator.node[genNumber]
            genType = self.grid.generator.type[genNumber]
            genArea = self._node2area(genNode)
            # print str(genNumber) + ", " + genName + ", " + genNode + ", " + genType + ", " + genArea
            if (genType == generatorType) and (genArea == area):
                # print "\tGenerator is of right type and area. Adding production"
                genProd = sum(self.db.getResultGeneratorPower(genNumber, timeMaxMin))
                totalProduction += genProd
                # print "\tGenerator production = " + str(genProd)
        return totalProduction

    def getAllGeneratorProductionOBSOLETE(self, timeMaxMin=None):
        """Returns all production [MWh] for all generators"""
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]

        totGenNumbers = self.grid.generator.numGenerators()
        totalProduction = 0
        for genNumber in range(0, totGenNumbers):
            genProd = sum(self.db.getResultGeneratorPower(genNumber, timeMaxMin))
            print(str(genProd))
            totalProduction += genProd
            print("Progression: " + str(genNumber + 1) + " of " + str(totGenNumbers))
        return totalProduction

    def _productionOverview(self, areas, types, timeMaxMin, TimeUnitCorrectionFactor):
        """
        Returns a matrix with sum of generator production per area and type

        This function is manly used as the calculation part of the
        writeProductionOverview Contains just numbers (production[MWH] for
        each type(columns) and area(rows)), not headers
        """

        numAreas = len(areas)
        numTypes = len(types)
        resultMatrix = np.zeros((numAreas, numTypes))
        for areaIndex in range(0, numAreas):
            for typeIndex in range(0, numTypes):
                prod = self._getAreaTypeProduction(areas[areaIndex], types[typeIndex], timeMaxMin)
                print(
                    "Total produced " + types[typeIndex] + " energy for " + areas[areaIndex] + " equals: " + str(prod)
                )
                resultMatrix[areaIndex][typeIndex] = prod * TimeUnitCorrectionFactor
        return resultMatrix

    def writeProductionOverview(self, areas, types, filename=None, timeMaxMin=None, TimeUnitCorrectionFactor=1):
        """
        Export production overview to CSV file

        Write a .csv overview of the production[MWh] in timespan 'timeMaxMin'
        with the different areas and types as headers.
        The vectors 'areas' and 'types' becomes headers (column- and row
        headers), but the different elements
        of 'types' and 'areas' are also the key words in the search function
        'getAreaTypeProduction'.
        The vectors 'areas' and 'types' can be of any length.
        """

        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]

        corner = "Countries"
        numAreas = len(areas)
        numTypes = len(types)
        prodMat = self._productionOverview(areas, types, timeMaxMin, TimeUnitCorrectionFactor)
        if filename is not None:
            with open(filename, "wb") as f:
                writer = csv.writer(f)
                types.insert(0, corner)
                writer.writerow(types)
                for i in range(0, numAreas):
                    row = [areas[i]]
                    for j in range(0, numTypes):
                        row.append(str(prodMat[i][j]))
                    writer.writerow(row)
        else:
            title = ""
            for j in types:
                title = title + "\t" + j
            print("Area" + title)
            for i in range(0, numAreas):
                print(areas[i] + "\t%s" % "\t".join(map(str, prodMat[i])))

    def getAverageInterareaBranchFlow(self, filename=None, timeMaxMin=None):
        """Calculate average flow in each direction and total flow for
        inter-area branches. Requires sqlite version newer than 3.6

        Parameters
        ----------
        filename : string, optional
            if a filename is given then the information is stored to file.
        timeMaxMin : list with two integer values, or None, optional
            time interval for the calculation [start,end]

        Returns
        -------
        List with values for each inter-area branch:
        [flow from 1 to 2, flow from 2 to 1, average absolute flow]
        """

        #        # Version control of database module. Must be 3.7.x or newer
        #        major = int(list(self.db.sqlite_version)[0])
        #        minor = int(list(self.db.sqlite_version)[1])
        #        version = major + minor / 10.0
        #        # print version
        #        if ((major < 4) and (minor < 7)):
        #            print('current SQLite version: {} ({})'
        #                  .format(self.db.sqlite_version,version))
        #            print('getAverageInterareaBranchFlow() requires 3.7.x or newer')
        #            return

        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]

        try:
            results = self.db.getAverageInterareaBranchFlow(timeMaxMin)
        except Exception as err:
            print("Error occured. Maybe because you are using sqlite<3.7")
            raise (err)

        if filename is not None:
            headers = ("branch", "fromArea", "toArea", "average negative flow", "average positive flow", "average flow")
            with open(filename, "wb") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                for row in results:
                    writer.writerow(row)
        # else:
        #    for x in results:
        #        print(x)

        return results

    def getAverageImportExport(self, area, timeMaxMin=None):
        """Return average import and export for a specified area"""

        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]
        ia = self.getAverageInterareaBranchFlow(timeMaxMin=timeMaxMin)

        # export: A->B pos flow + A<-B neg flow
        sum_export = sum([b[4] for b in ia if b[1] == area]) - sum([b[3] for b in ia if b[2] == area])
        # import: A->B neg flow + A<-B pos flow
        sum_import = -sum([b[3] for b in ia if b[2] == area]) + sum([b[4] for b in ia if b[2] == area])
        return dict(exp=sum_export, imp=sum_import)

    def getEnergyBalanceInArea(
        self,
        area,
        spillageGen,
        resolution="H",
        fileName=None,
        timeMaxMin=None,
        start_date="2014-01-01",
    ):
        """
        Print time series of energy balance in an area, including
        production, spillage, load shedding, storage, pump consumption
        and imports

        Parameters
        ----------
        area : string
            area code
        spillageGen : list
            generator types for which to show spillage (renewables)
        resolution : string
            resolution of output, see pandas:resample
        fileName : string (default=None)
            name of file to export results
        timeMaxMin : list
            time range to consider
        start_date : date string
            date when time series start

        """
        # Eirik/Arne

        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]

        # data resolution in whole seconds (usually, timeDelta=1.0)
        resolutionS = int(self.grid.timeDelta * 3600)

        prod = pd.DataFrame()
        genTypes = self.grid.getAllGeneratorTypes()
        generators = self.grid.getGeneratorsPerAreaAndType()[area]
        pumpIdx = self.grid.getGeneratorsWithPumpByArea()
        if len(pumpIdx) > 0:
            pumpIdx = pumpIdx[area]
        storageGen = self.grid.getIdxGeneratorsWithStorage()
        areaGen = [item for sublist in list(generators.values()) for item in sublist]
        matches = [x for x in areaGen if x in storageGen]
        for gt in genTypes:
            if gt in generators:
                prod[gt] = self.db.getResultGeneratorPower(generators[gt], timeMaxMin)
                if gt in spillageGen:
                    prod[gt + " spilled"] = self.db.getResultGeneratorSpilled(generators[gt], timeMaxMin)
        prod["load shedding"] = self.getLoadheddingInArea(area, timeMaxMin)
        storage = self.db.getResultStorageFillingMultiple(matches, timeMaxMin, capacity=False)
        if storage:
            prod["storage"] = storage
        if len(pumpIdx) > 0:
            prod["pumped"] = self.db.getResultPumpPowerMultiple(pumpIdx, timeMaxMin, negative=True)
        prod["net import"] = self.getNetImport(area, timeMaxMin)
        prod.index = pd.date_range(start_date, periods=timeMaxMin[-1] - timeMaxMin[0], freq="{}s".format(resolutionS))
        if resolution != "H":
            prod = prod.resample(resolution, how="sum")
        if fileName:
            prod.to_csv(fileName)
        else:
            return prod

    def getStorageFillingInAreas(self, areas, generator_type, relative_storage=True, timeMaxMin=None):
        """
        Gets time-series with aggregated storage filling for specified area(s)
        for a specific generator type.

        Parameters
        ----------
        areas : list
            list of area codes (e.g. ['DE','FR'])
        generator_type : string
            generator type string (e.g. 'hydro')
        relative_storage : boolean
            show relative (True) or absolute (False) storage
        timeMaxMin : list
            time range to consider (e.g. [0,8760])
        """
        # Eirik/Arne

        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]
        storageGen = self.grid.getIdxGeneratorsWithStorage()
        storageTypes = self.grid.generator.type
        nodeNames = self.grid.generator.node
        nodeAreas = self.grid.node.area
        storCapacities = self.grid.generator.storage_cap
        generators = []
        capacity = 0
        for gen in storageGen:
            area = nodeAreas[self.grid.node.id.tolist().index(nodeNames[gen])]
            if area in areas and storageTypes[gen] == generator_type:
                generators.append(gen)
                if relative_storage:
                    capacity += storCapacities[gen]
            filling = self.db.getResultStorageFillingMultiple(generators, timeMaxMin, capacity)
        return filling

    def getNetImport(self, area, timeMaxMin=None):
        """Return time series for net import for a specified area"""
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]

        # find the associated branches
        br = self.grid.getInterAreaBranches(area_to=area, acdc="ac")
        br_p = br["branches_pos"]
        br_n = br["branches_neg"]
        dcbr = self.grid.getInterAreaBranches(area_to=area, acdc="dc")
        dcbr_p = dcbr["branches_pos"]
        dcbr_n = dcbr["branches_neg"]

        # AC branches
        ie = self.db.getBranchesSumFlow(branches_pos=br_p, branches_neg=br_n, timeMaxMin=timeMaxMin, acdc="ac")
        # DC branches
        dcie = self.db.getBranchesSumFlow(branches_pos=dcbr_p, branches_neg=dcbr_n, timeMaxMin=timeMaxMin, acdc="dc")

        if ie["pos"] and ie["neg"]:
            res_ac = [a - b for a, b in zip(ie["pos"], ie["neg"])]
        elif ie["pos"]:
            res_ac = ie["pos"]
        elif ie["neg"]:
            res_ac = [-a for a in ie["neg"]]
        else:
            res_ac = [0] * (timeMaxMin[-1] - timeMaxMin[0])

        if dcie["pos"] and dcie["neg"]:
            res_dc = [a - b for a, b in zip(dcie["pos"], dcie["neg"])]
        elif dcie["pos"]:
            res_dc = dcie["pos"]
        elif dcie["neg"]:
            res_dc = [-a for a in dcie["neg"]]
        else:
            res_dc = [0] * (timeMaxMin[-1] - timeMaxMin[0])

        res = [a + b for a, b in zip(res_ac, res_dc)]
        return res

    def getImportExport(self, areas=None, timeMaxMin=None, acdc=["ac", "dc"]):
        """Return time series for import and export for a specified area"""
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]
        if areas is None:
            areas = self.grid.getAllAreas()

        df_importexport = pd.DataFrame(index=areas, columns=["import", "export"])
        for area in areas:
            print(area, end=",")
            # find the associated branches (pos = into area)
            # br = self.grid.getInterAreaBranches(area_to=area,acdc='ac')
            # br_p = br['branches_pos']
            # br_n = br['branches_neg']
            # dcbr = self.grid.getInterAreaBranches(area_to=area,acdc='dc')
            # dcbr_p = dcbr['branches_pos']
            # dcbr_n = dcbr['branches_neg']
            flow_in = 0
            flow_out = 0
            for acdc_type in acdc:
                br = self.grid.getInterAreaBranches(area_to=area, acdc=acdc_type)
                br_pos = self.db.getResultBranches(timeMaxMin, br_indx=br["branches_pos"], acdc=acdc_type)
                br_neg = self.db.getResultBranches(timeMaxMin, br_indx=br["branches_neg"], acdc=acdc_type)
                if br_pos.shape[0] > 0:
                    flow_in += br_pos[br_pos["flow"] > 0]["flow"].sum()
                    flow_out -= br_pos[br_pos["flow"] < 0]["flow"].sum()
                if br_neg.shape[0] > 0:
                    flow_in -= br_neg[br_neg["flow"] < 0]["flow"].sum()
                    flow_out += br_neg[br_neg["flow"] > 0]["flow"].sum()

            # ie =  self.db.getBranchesSumFlow(branches_pos=br_p,branches_neg=br_n,
            #                                 timeMaxMin=timeMaxMin,
            #                                 acdc='ac')
            # DC branches

            #            dcie =  self.db.getBranchesSumFlow(branches_pos=dcbr_p,
            #                                                 branches_neg=dcbr_n,
            #                                                 timeMaxMin=timeMaxMin,
            #                                                 acdc='dc')
            #            import_a = (sum(v for v in ie['pos'] if v>=0)
            #                         +sum(-v for v in ie['neg'] if v<0)
            #                         #+sum(v for v in dcie['pos'] if v>=0)
            #                         #+sum(-v for v in dcie['neg'] if v<0)
            #                         )
            #            export_a = (sum(-v for v in ie['pos'] if v<0)
            #                         +sum(v for v in ie['neg'] if v>=0)
            #                         #+sum(-v for v in dcie['pos'] if v<0)
            #                         #+sum(v for v in dcie['neg'] if v>=0)
            #                         )
            df_importexport.loc[area, "import"] = flow_in
            df_importexport.loc[area, "export"] = flow_out
        print()
        return df_importexport

    def getDemandPerArea(self, area, timeMaxMin=None):
        """Returns demand timeseries for given area, as dictionary with
        fields "fixed", "flex", and "sum"

        Parameters
        ----------
        area (string)
            area to get demand for
        timeMaxMin (list) (default = None)
            [min, max] - lower and upper time interval
        """

        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]
        timerange = range(timeMaxMin[0], timeMaxMin[-1])

        consumer = self.grid.consumer

        dem = [0] * len(self.timerange)
        flexdemand = [0] * len(self.timerange)
        consumers = self.grid.getConsumersPerArea()[area]
        for i in consumers:
            ref_profile = consumer.demand_ref[i]
            # accumulate demand for all consumers in this area:
            dem = [
                dem[t - self.timerange[0]]
                + consumer.demand_avg[i]
                * (1 - consumer.flex_fraction[i])
                * self.grid.profiles[ref_profile][t - self.timerange[0]]
                for t in timerange
            ]
            flexdemand_i = self.db.getResultFlexloadPower(i, timeMaxMin)
            if len(flexdemand_i) > 0:
                flexdemand = [sum(x) for x in zip(flexdemand, flexdemand_i)]
        sumdemand = [sum(x) for x in zip(dem, flexdemand)]

        return {"fixed": dem, "flex": flexdemand, "sum": sumdemand}

    def plotNodalPrice(self, nodeIndx, timeMaxMin=None, showTitle=True):
        """Show nodal price in single node

        Parameters
        ----------
        nodeIndx (int)
            index of node to plot from
        timeMaxMin (list) (default = None)
            [min, max] - lower and upper time interval
        """

        # TODO allow for input to be multiple nodes
        # TODO plot storage price for storage in the same node?
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]
        timerange = range(timeMaxMin[0], timeMaxMin[-1])

        if nodeIndx in self.db.getGridNodeIndices():
            nodalprice = self.db.getResultNodalPrice(nodeIndx, timeMaxMin)
            plt.figure()
            plt.plot(timerange, nodalprice)
            if showTitle:
                plt.title("Nodal price for node %d" % (nodeIndx))
            plt.show()
        else:
            print("Node not found")
        return

    def plotAreaPrice(self, areas, timeMaxMin=None, showTitle=True):
        """Show area price(s)

        Parameters
        ----------
        areas (list)
            list of areas to show
        timeMaxMin (list) (default = None)
            [min, max] - lower and upper time interval
        """

        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]
        timerange = range(timeMaxMin[0], timeMaxMin[-1])

        plt.figure()
        for a in areas:
            areaprice = self.getAreaPrices(a, timeMaxMin)
            plt.plot(timerange, areaprice, label=a)
            if showTitle:
                plt.title("Area price")

        plt.legend()
        plt.show()
        return

    def plotStorageFilling(self, generatorIndx, timeMaxMin=None, showTitle=True):
        """Show storage filling level (MWh) for generators with storage

        Parameters
        ----------
        generatorIndx (int)
            index of generator to plot from
        timeMaxMin (list) (default = None)
            [min, max] - lower and upper time interval
        """

        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]
        timerange = range(timeMaxMin[0], timeMaxMin[-1])

        if generatorIndx in self.storage_idx_generators:
            storagefilling = self.db.getResultStorageFilling(generatorIndx, timeMaxMin)
            plt.figure()
            plt.plot(timerange, storagefilling)
            if showTitle:
                plt.title("Storage filling level for generator %d" % (generatorIndx))
            plt.show()
        else:
            print("These are the generators with storage:")
            print(self.storage_idx_generators)
        return

    def plotGeneratorOutput(self, generator_index, timeMaxMin=None, relativestorage=True, showTitle=True):
        """Show output of a generator

        Parameters
        ----------
        generator_index (int)
            index of generator for which to make the plot
        timeMaxMin [int,int] (default=None)
            time interval for the plot [start,end]
        relativestorage (default=True)
            use filling fraction as y axis label for storage
        """
        # TODO allow for input to be multiple generators
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]
        timerange = range(timeMaxMin[0], timeMaxMin[-1])

        generatoroutput = self.db.getResultGeneratorPower(generator_index, timeMaxMin)
        # powerinflow = db.getInflow(timeMaxMin,storageindx)

        ax1.plot(timerange, generatoroutput, "-r", label="output")
        ax1.set_ylim(ymin=0)

        # Power inflow (if generator has nonzero inflow factor)
        if self.grid.generator["inflow_fac"][generator_index] > 0:
            profile = self.grid.generator["inflow_ref"][generator_index]
            ax1.plot(
                timerange,
                [
                    self.grid.profiles[profile][t - self.timerange[0]]
                    * self.grid.generator["inflow_fac"][generator_index]
                    * self.grid.generator["pmax"][generator_index]
                    for t in timerange
                ],
                "-.b",
                label="inflow",
            )

        # Power pumped (if generator has nonzero pumping capacity)
        if self.grid.generator["pump_cap"][generator_index] > 0:
            pump_output = self.db.getResultPumpPower(generator_index, timeMaxMin)
            ax1.plot(timerange, pump_output, ":c", label="pumping")

        # Storage filling level (if generator has storage)
        ax2 = None
        if generator_index in self.storage_idx_generators:
            storagefilling = self.db.getResultStorageFilling(generator_index, timeMaxMin)
            if relativestorage:
                cap = self.grid.generator["storage_cap"][generator_index]
                storagefilling = [x / cap for x in storagefilling]
            ax2 = plt.twinx()  # separate y axis
            ax2.plot(timerange, storagefilling, "--g", label="storage")
            ax2.legend(loc="lower right", bbox_to_anchor=(1, 1), borderaxespad=0.0, frameon=False)
            ax2.set_ylim(ymin=0)

        ax1.legend(loc="lower left", bbox_to_anchor=(0, 1), borderaxespad=0.0, frameon=False)
        if ax2 is not None:
            # TODO: Add comment - What whas the point of this??
            # ax2.add_artist(lgd)
            # ax1.legend=None
            pass
        nodeidx = self.grid.node["id"].tolist().index(self.grid.generator["node"][generator_index])
        if showTitle:
            plt.title(
                "Generator %d (%s) at node %d (%s)"
                % (
                    generator_index,
                    self.grid.generator["type"][generator_index],
                    nodeidx,
                    self.grid.generator["node"][generator_index],
                )
            )
        plt.show()
        return

    def plotDemandAtLoad(self, consumer_index, timeMaxMin=None, relativestorage=True, showTitle=True):
        """Make a time-series plot of consumption of a specified load

        Parameters
        ----------
        consumer_index (int)
            index of consumer for which to make the plot
        timeMaxMin [int,int] (default=None)
            time interval for the plot [start,end]
        relativestorage (default=True)
            use filling fraction as y axis label for storage
        """
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]
        timerange = range(timeMaxMin[0], timeMaxMin[-1])

        # Fixed load
        profile = self.grid.consumer["demand_ref"][consumer_index]
        ax1.plot(
            timerange,
            [
                self.grid.profiles[profile][t - self.timerange[0]]
                * self.grid.consumer["demand_avg"][consumer_index]
                * (1 - self.grid.consumer["flex_fraction"][consumer_index])
                for t in timerange
            ],
            "-r",
            label="fixed load",
        )

        # Flexible load  (if consumer has nonzero flexible load)
        ax2 = None
        if self.grid.consumer.flex_fraction[consumer_index] > 0:
            flexload_power = self.db.getResultFlexloadPower(consumer_index, timeMaxMin)
            ax1.plot(timerange, flexload_power, "-.b", label="flexible load")

            # Storage filling level
            storagefilling = self.db.getResultFlexloadStorageFilling(consumer_index, timeMaxMin)
            if relativestorage:
                cap = self.grid.getFlexibleLoadStorageCapacity(consumer_index)
                storagefilling = [x / cap for x in storagefilling]
            ax2 = plt.twinx()  # separate y axis
            ax2.plot(timerange, storagefilling, "--g", label="storage")
            ax2.legend(loc="lower right", bbox_to_anchor=(1, 1), borderaxespad=0.0, frameon=False)

        ax1.legend(loc="lower left", bbox_to_anchor=(0, 1), borderaxespad=0.0, frameon=False)
        # if ax2 is not None:
        #    ax2.add_artist(lgd)
        #    ax1.legend=None
        nodeidx = self.grid.node["id"].tolist().index(self.grid.consumer.node[consumer_index])
        if showTitle:
            plt.title(
                "Consumer %d at node %d (%s)" % (consumer_index, nodeidx, self.grid.consumer.node[consumer_index])
            )
        plt.show()
        return

    def plotStoragePerArea(self, area, absolute=False, timeMaxMin=None, showTitle=True):
        """Show generation storage accumulated per area

        Parameters
        ----------
        area (str)
        absolute (bool)(default=False)
            plot storage value in absolute or relative to maximum
        timeMaxMin (list) (default = None)
            [min, max] - lower and upper time interval"""

        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]
        timerange = range(timeMaxMin[0], timeMaxMin[-1])

        plt.figure()
        generators = self.grid.getGeneratorsPerAreaAndType()
        cap = self.grid.generator["storage_cap"]
        for gentype in generators[area].keys():
            idxGen = generators[area][gentype]
            idx_storage = [[i, v] for i, v in enumerate(self.storage_idx_generators) if v in idxGen]
            # idx_storage is now a list of index pairs for generators with
            #    storage in the given area
            # the first value is index in generator list
            # the second value is index in storage list (not used)

            if len(idx_storage) > 0:
                mystor = [
                    sum(
                        [
                            sum(self.db.getResultStorageFilling(idx_storage[i][1], [t, t + 1]))
                            for i in range(len(idx_storage))
                        ]
                    )
                    for t in timerange
                ]
                mycap = sum([cap[idx_storage[i][1]] for i in range(len(idx_storage))])

                if absolute:
                    sumStorAreaType = mystor
                else:
                    sumStorAreaType = [mystor[i] / mycap for i in range(len(mystor))]
                plt.plot(timerange, sumStorAreaType, label=gentype)

        # plt.legend(generators[area].keys() , loc="upper right")
        plt.legend(loc="upper right")
        if showTitle:
            plt.title("Total storage level in %s" % (area))
        plt.show()

        return

    def plotGenerationPerArea(
        self, area, timeMaxMin=None, fill=True, reversed_order=False, net_import=True, loadshed=True, showTitle=True
    ):
        """Show generation per area

        Parameters
        ----------
        area (str)
        timeMaxMin (list) (default = None)
            [min, max] - lower and upper time interval
        fill (Boolean) - whether use filled plot
        reversed_order - whether to reverse order of generator types
        net_import - whether to include net import in graph
        loadshed - whether to include unmet demand
        """

        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]
        timerange = range(timeMaxMin[0], timeMaxMin[-1])
        fillfrom = [0] * len(timerange)
        count = 0
        plt.figure()
        ax = plt.subplot(111)
        generators = self.grid.getGeneratorsPerAreaAndType()
        # gentypes_ordered = self._gentypes_ordered_by_fuelcost()
        gentypes_ordered = self.grid.getAllGeneratorTypes(sort="fuelcost")
        if reversed_order:
            gentypes_ordered.reverse()
        numCurves = len(gentypes_ordered) + 1
        # colours = cm.gist_rainbow(np.linspace(0, 1, numCurves))
        colours = cm.tab20([i % 20 for i in range(numCurves)])
        for gentype in gentypes_ordered:
            if gentype in generators[area]:
                idxGen = generators[area][gentype]
                sumGenAreaType = self.db.getResultGeneratorPower(idxGen, timeMaxMin)
                if fill:
                    aggregated = [x + y for x, y in zip(sumGenAreaType, fillfrom)]
                    ax.fill_between(timerange, y1=aggregated, y2=fillfrom, facecolor=colours[count])
                    # could add edgecolor="lightgrey"
                    # add this plot to get the legend right
                    ax.plot([], [], color=colours[count], linewidth=10, label=gentype)
                    fillfrom = aggregated
                else:
                    ax.plot(timerange, sumGenAreaType, color=colours[count], label=gentype)
            else:
                # in order to get the legend right
                ax.plot([], [], color=colours[count], label=gentype)
            count = count + 1
        if net_import:
            netimport = self.getNetImport(area, timeMaxMin)
            agg = [x + y for x, y in zip(netimport, fillfrom)]
            ax.plot(timerange, agg, linestyle=":", linewidth=2, color="black", label="net import")
        if loadshed:
            loadshed = self.getLoadheddingInArea(area, timeMaxMin)
            label = "Load shed"
            col = "dimgray"
            if fill:
                aggregated = [x + y for x, y in zip(loadshed, fillfrom)]
                # ax.fill_between(timerange,y1=aggregated,y2=fillfrom,
                #                facecolor=col)
                # ax.plot([],[],color=col,linewidth=10,
                #        label=label)
                ax.plot(timerange, aggregated, linestyle="--", color=col, label=label)
                fillfrom = aggregated
            else:
                ax.plot(timerange, loadshed, linestyle="--", color=col, label=label)

        # plt.legend()
        handles, labels = ax.get_legend_handles_labels()
        handles.reverse()
        labels.reverse()
        plt.legend(handles, labels, loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.0)

        if fill:
            plt.ylim(ymin=0)

        if showTitle:
            plt.title("Generation in %s" % (area))
        plt.show()
        return

    def plotDemandPerArea(self, areas, timeMaxMin=None, showTitle=True):
        """Show demand in area(s)

        Parameters
        ----------
        areas (list?)
            list of areas to be plotted
        timeMaxMin (list) (default = None)
            [min, max] - lower and upper time interval
        """

        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]
        timerange = range(timeMaxMin[0], timeMaxMin[-1])

        plt.figure()
        consumer = self.grid.consumer
        if type(areas) is str:
            areas = [areas]
        for co in areas:
            dem = [0] * len(self.timerange)
            flexdemand = [0] * len(self.timerange)
            consumers = self.grid.getConsumersPerArea()[co]
            for i in consumers:
                ref_profile = consumer["demand_ref"][i]
                # accumulate demand for all consumers in this area:
                dem = [
                    dem[t - self.timerange[0]]
                    + consumer["demand_avg"][i]
                    * (1 - consumer["flex_fraction"][i])
                    * self.grid.profiles[ref_profile][t - self.timerange[0]]
                    for t in timerange
                ]
                flexdemand_i = self.db.getResultFlexloadPower(i, timeMaxMin)
                if len(flexdemand_i) > 0:
                    flexdemand = [sum(x) for x in zip(flexdemand, flexdemand_i)]
            sumdemand = [sum(x) for x in zip(dem, flexdemand)]
            (p,) = plt.plot(timerange, sumdemand, label=co)
            # Fixed demand in dotted lines
            plt.plot(timerange, dem, "--", color=p.get_color())

        plt.legend(loc="upper right")
        if showTitle:
            plt.title("Power demand")
        plt.show()
        return

    def plotStorageValues(self, genindx, timeMaxMin=None, showTitle=True):
        """Plot storage values (marginal prices) for generators with storage

        Parameters
        ----------
        genindx (int)
            index of generator for which to make the plot
        timeMaxMin [int,int] (default=None)
            time interval for the plot [start,end]
        """

        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]
        timerange = range(timeMaxMin[0], timeMaxMin[-1])

        if genindx in self.storage_idx_generators:
            nodeidx = self.grid.node["id"].tolist().index(self.grid.generator.node[genindx])
            storagevalue = self.db.getResultStorageValue(genindx, timeMaxMin)
            nodalprice = self.db.getResultNodalPrice(nodeidx, timeMaxMin)
            pumpprice = [x - self.grid.generator.pump_deadband[genindx] for x in storagevalue]
            plt.figure()
            (p,) = plt.plot(timerange, storagevalue, "-b", label="storage value")
            if genindx in self.pump_idx_generators:
                pumpprice = [x - self.grid.generator.pump_deadband[genindx] for x in storagevalue]
                plt.plot(timerange, pumpprice, ":", color=p.get_color(), label="pump threshold")
            plt.plot(timerange, nodalprice, "--r", label="nodal price")
            plt.legend()
            plt.legend(loc="lower left", bbox_to_anchor=(0, 1), borderaxespad=0.0, ncol=4, frameon=False)

            if showTitle:
                plt.title(
                    "Storage value  for generator %d (%s) in %s"
                    % (genindx, self.grid.generator.type[genindx], self.grid.generator.node[genindx])
                )
            plt.show()
        else:
            print("These are the generators with storage:")
            print(self.storage_idx_generators)
        return

    def plotFlexibleLoadStorageValues(self, consumerindx, timeMaxMin=None, showTitle=True):
        """Plot storage valuesfor flexible loads

        Parameters
        ----------
        consumerindx : int
            index of consumer for which to make the plot
        timeMaxMin : list, [int,int]
            time interval for the plot [start,end], or None for entire range
        """

        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]
        timerange = range(timeMaxMin[0], timeMaxMin[-1])

        if consumerindx in self.flex_idx_consumers:
            nodeidx = self.grid.node.id.tolist().index(self.grid.consumer.node[consumerindx])
            storagevalue = self.db.getResultFlexloadStorageValue(consumerindx, timeMaxMin)
            nodalprice = self.db.getResultNodalPrice(nodeidx, timeMaxMin)
            plt.figure()
            plt.plot(timerange, storagevalue)
            plt.plot(timerange, nodalprice)
            plt.legend(["storage value", "nodal price"])
            if showTitle:
                plt.title(
                    "Storage value  for consumer %d at %s" % (consumerindx, self.grid.consumer.node[consumerindx])
                )
            plt.show()
        else:
            print("These are the consumers with flexible load:")
            print(self.flex_idx_consumers)
        return

    def plotMapGrid(
        self,
        nodetype=None,
        branchtype=None,
        dcbranchtype=None,
        show_node_labels=False,
        branch_style="c",
        latlon=None,
        timeMaxMin=None,
        dotsize=40,
        filter_node=None,
        filter_branch=None,
        draw_par_mer=False,
        showTitle=True,
        colors=True,
    ):
        """
        Plot results to map

        Parameters
        ----------
        nodetype : string
            "", "area", "nodalprice", "energybalance", "loadshedding"
        branchtype : string
            "", "capacity", "area", "utilisation", "flow", "sensitivity"
        dcbranchtype : string
            "", "capacity"
        show_node_labels : boolean
            whether to show node names (true/false)
        branch_style : string or list of strings (optional)
            How branch capacity and flow should be visualised.
            "c" = colour, "t" = thickness. The two options may be combined.
        dotsize : integer (optional)
            set dot size for each plotted node
        latlon: list of four floats (optional)
            map area [lat_min, lon_min, lat_max, lon_max]
        filter_node : list of two floats (optional)
            [min,max] - lower and upper cutoff for node value
        filter_branch : list of two floats
            [min,max] - lower and upper cutoff for branch value
        draw_par_mer : boolean
            whether to draw parallels and meridians on map
        showTitle : boolean
        colors : boolean
            Whether to use colours or not
        """

        # basemap is only used here, so to allow using powergama without
        # basemap installed, it is best to put import statement here.
        from mpl_toolkits.basemap import Basemap

        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]

        plt.figure()
        data = self.grid
        # res = self

        if latlon is None:
            lat_max = max(data.node.lat) + 1
            lat_min = min(data.node.lat) - 1
            lon_max = max(data.node.lon) + 1
            lon_min = min(data.node.lon) - 1
        else:
            lat_min = latlon[0]
            lon_min = latlon[1]
            lat_max = latlon[2]
            lon_max = latlon[3]

        # Use the average latitude as latitude of true scale
        lat_truescale = np.mean(data.node.lat)

        m = Basemap(
            resolution="l",
            projection="merc",
            lat_ts=lat_truescale,
            llcrnrlon=lon_min,
            llcrnrlat=lat_min,
            urcrnrlon=lon_max,
            urcrnrlat=lat_max,
            anchor="W",
        )

        # Draw coastlines, meridians and parallels.
        m.drawcoastlines()
        # m.drawcountries(zorder=0)

        if colors:
            m.fillcontinents(color="coral", lake_color="aqua", zorder=0)
            m.drawmapboundary(fill_color="aqua")
        else:
            m.fillcontinents(zorder=0)
            m.drawmapboundary()

        if draw_par_mer:
            m.drawparallels(
                np.arange(_myround(lat_min, 10, "floor"), _myround(lat_max, 10, "ceil"), 10), labels=[1, 1, 0, 0]
            )

            m.drawmeridians(
                np.arange(_myround(lon_min, 10, "floor"), _myround(lon_max, 10, "ceil"), 10), labels=[0, 0, 0, 1]
            )

        # AC Branches
        num_branches = data.branch.shape[0]

        lwidths = [2] * num_branches

        # default values:
        branch_value = np.asarray([0.5] * num_branches)
        branch_colormap = cm.gray
        branch_plot_colorbar = True

        if branchtype == "area":
            areas = data.node.area
            allareas = data.getAllAreas()
            branch_value = [-1] * num_branches
            nodes_from = data.branchFromNodeIdx()
            nodes_to = data.branchToNodeIdx()
            for i in range(num_branches):
                # node_indx_from = data.node.name.index(data.branch.node_from[i])
                # node_indx_to = data.node.name.index(data.branch.node_to[i])
                area_from = areas[nodes_from[i]]
                area_to = areas[nodes_to[i]]
                if area_from == area_to:
                    branch_value[i] = allareas.index(area_from)
                branch_value = np.asarray(branch_value)
            #            branch_colormap = plt.get_cmap('hsv')
            branch_colormap = cm.prism
            branch_colormap.set_under("k")
            filter_branch = [0, len(allareas)]
            branch_label = "Branch area"
            branch_plot_colorbar = False
        elif branchtype == "utilisation":
            utilisation = self.getAverageUtilisation(timeMaxMin)
            branch_value = utilisation
            branch_colormap = plt.get_cmap("hot")
            branch_label = "Branch utilisation"
        elif branchtype == "capacity":
            cap = data.branch.capacity
            branch_plot_colorbar = False
            branch_label = "Branch capacity"
            if "c" in branch_style:
                branch_value = np.asarray(cap)
                branch_colormap = plt.get_cmap("hot")
                branch_plot_colorbar = True
                if filter_branch is None:
                    # need an upper limit to avoid crash due to inf capacity
                    maxcap = np.nanmax(branch_value)
                    filter_branch = [0, np.round(maxcap, -2) + 100]
            if "t" in branch_style:
                avgcap = np.mean(cap)
                if avgcap == 0:
                    lwidths = [0 for f in cap]
                else:
                    lwidths = [2 * f / avgcap for f in cap]
        elif branchtype == "flow":
            avgflow = self.getAverageBranchFlows(timeMaxMin)[2]
            if "c" in branch_style:
                branch_value = np.asarray(avgflow)
                branch_colormap = plt.get_cmap("hot")
            else:
                branch_plot_colorbar = False
            if "t" in branch_style:
                avgavgflow = np.mean(avgflow)
                lwidths = [2 * f / avgavgflow for f in avgflow]
            branch_label = "Branch flow"
            # branch_value = np.asarray(avgflow)
            # branch_colormap = plt.get_cmap('hot')
        elif branchtype == "sensitivity":
            branch_value = np.zeros(num_branches)
            avgsense = self.getAverageBranchSensitivity(timeMaxMin)
            branch_value[self.idxConstrainedBranchCapacity] = -avgsense
            branch_colormap = plt.get_cmap("hot")
            branch_label = "Branch sensitivity"
            # These sensitivities are mostly negative
            # (reduced cost by increasing branch capacity)
            # minsense = np.nanmin(avgsense)
            # maxsense = np.nanmax(avgsense)
        else:
            branch_value = np.asarray([0.5] * num_branches)
            branch_colormap = cm.gray
            branch_plot_colorbar = False

        idx_from = data.branchFromNodeIdx()
        idx_to = data.branchToNodeIdx()
        branch_lat1 = [data.node.lat[i] for i in idx_from]
        branch_lon1 = [data.node.lon[i] for i in idx_from]
        branch_lat2 = [data.node.lat[i] for i in idx_to]
        branch_lon2 = [data.node.lon[i] for i in idx_to]

        x1, y1 = m(branch_lon1, branch_lat1)
        x2, y2 = m(branch_lon2, branch_lat2)

        ls = [[(x1[i], y1[i]), (x2[i], y2[i])] for i in range(len(x1))]
        # ls = [[(x1[i],y1[i]),(x2[i],y2[i])] for i in range(num_branches)]
        ax = plt.axes()
        if not lwidths:
            print("No new data")
        else:
            line_segments_ac = mpl.collections.LineCollection(ls, linewidths=lwidths, cmap=branch_colormap)

            if filter_branch is not None:
                line_segments_ac.set_clim(filter_branch)
            line_segments_ac.set_array(branch_value)
            ax.add_collection(line_segments_ac)

        # DC Branches
        lwidths = [2] * 1
        # lwidths = [2] * num_dcbranches
        if dcbranchtype is not None:
            branch_plot_colorbar = False
            branch_value = np.asarray([0.1] * num_branches)
            branch_colormap = cm.winter

        if dcbranchtype == "flow":
            avgTot = self.getAverageBranchFlows(timeMaxMin, False)[2]
            if "c" in branch_style:
                branch_value = np.asarray(avgTot)
                branch_colormap = plt.get_cmap("hot")
                branch_plot_colorbar = True
            if "t" in branch_style:
                avgavgflow = np.mean(avgTot)
                lwidths = [2 * f / avgavgflow for f in avgTot]
            branch_label = "Branch flow"
        elif dcbranchtype == "capacity":
            cap = [data.branch.capacity[i] for i in range(len(data.branch.capacity)) if data.branch.type[i] == "dc"]
            branch_value = np.asarray(cap)
            maxcap = np.nanmax(branch_value)
            branch_colormap = plt.get_cmap("hot")
            branch_label = "Branch capacity"
            if filter_branch is None:
                # need an upper limit to avoid crash due to inf capacity
                filter_branch = [0, np.round(maxcap, -2) + 100]
            if "t" in branch_style:
                avgcap = np.mean(cap)
                lwidths = [2 * f / avgcap for f in cap]

        idx_from = data.dcBranchFromNodeIdx()  # empty
        idx_to = data.dcBranchToNodeIdx()  # empty
        branch_lat1 = [data.node.lat[i] for i in idx_from]
        branch_lon1 = [data.node.lon[i] for i in idx_from]
        branch_lat2 = [data.node.lat[i] for i in idx_to]
        branch_lon2 = [data.node.lon[i] for i in idx_to]

        x1, y1 = m(branch_lon1, branch_lat1)
        x2, y2 = m(branch_lon2, branch_lat2)
        ls = [[(x1[i], y1[i]), (x2[i], y2[i])] for i in range(len(x1))]
        line_segments_dc = mpl.collections.LineCollection(ls, linewidths=lwidths, colors="black")
        # line_segments_dc = mpl.collections.LineCollection(
        #        ls, linewidths=2,colors='blue')
        if filter_branch is not None:
            line_segments_dc.set_clim(filter_branch)
        line_segments_dc.set_array(branch_value)
        ax.add_collection(line_segments_dc)

        # Nodes
        node_plot_colorbar = True
        if nodetype == "area":
            areas = data.node.area
            allareas = data.getAllAreas()
            # colours_co = cm.prism(np.linspace(0, 1, len(allareas)))
            node_label = "Node area"
            node_value = [allareas.index(c) for c in areas]
            node_colormap = cm.prism
            node_plot_colorbar = False
            # this is to get same colours as for branches:
            node_colormap.set_under("k")
            filter_node = [0, len(allareas)]
        elif nodetype == "nodalprice":
            avgprice = self.getAverageNodalPrices(timeMaxMin)
            node_label = "Nodal price"
            node_value = avgprice
            node_colormap = cm.jet
        elif nodetype == "lmp":
            node_value = self.getAverageNodalPrices(timeMaxMin)
            node_label = "Locational marginal price"
            node_colormap = cm.jet
        elif nodetype == "energybalance":
            avg_energybalance = self.getAverageEnergyBalance(timeMaxMin)
            node_label = "Nodal energy balance"
            node_value = avg_energybalance
            node_colormap = cm.hot
        elif nodetype == "loadshedding":
            node_value = self.getLoadsheddingPerNode(timeMaxMin)
            node_label = "Loadshedding"
            node_colormap = cm.hot
        else:
            node_value = "dimgray"
            node_colormap = cm.jet
            node_label = ""
            node_plot_colorbar = False

        x, y = m(data.node["lon"].tolist(), data.node["lat"].tolist())
        if nodetype == "lmp":
            sc = plt.hexbin(x, y, gridsize=20, C=node_value, cmap=node_colormap)
        else:
            sc = m.scatter(x, y, marker="o", c=node_value, cmap=node_colormap, zorder=2, s=dotsize)
        # sc.cmap.set_under('dimgray')
        # sc.cmap.set_over('dimgray')
        if filter_node is not None:
            sc.set_clim(filter_node[0], filter_node[1])

            # #TODO: Er dette nødvendig lenger, Harald?
            # #nodes with NAN nodal price plotted in gray:
            # for i in range(len(avgprice)):
            # if np.isnan(avgprice[i]):
            # m.scatter(x[i],y[i],c='dimgray',
            # zorder=2,s=dotsize)

        # NEW Colorbar for nodes
        # m. or plt.?
        if node_plot_colorbar:
            axcb2 = plt.colorbar(sc)
            axcb2.set_label(node_label)

        # NEW Colorbar for branch capacity
        if branch_plot_colorbar:
            axcb = plt.colorbar(line_segments_ac)
            axcb.set_label(branch_label)

        # Show names of nodes
        if show_node_labels:
            labels = data.node["id"]
            x1, x2, y1, y2 = plt.axis()
            offset_x = (x2 - x1) / 50
            for label, xpt, ypt in zip(labels, x, y):
                if xpt > x1 and xpt < x2 and ypt > y1 and ypt < y2:
                    plt.text(xpt + offset_x, ypt, label)

        if showTitle:
            plt.title("Nodes %s and branches %s" % (nodetype, branchtype))
        plt.show()

        return
        # End plotGridMap

    def getEnergyMix(self, timeMaxMin=None, relative=False, showTitle=True, variable="energy"):
        """
        Get energy, generation capacity or spilled energy per area per type

        Parameters
        ----------
        timeMaxMin : list of two integers
            Time range, [min,max]
        relative : boolean
            Whether to plot absolute (false) or relative (true) values
        variable : string ("energy","capacity","spilled")
            Which variable to plot (default is energy production)
        """

        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]

        if variable == "energy":
            print("Getting energy output from all generators...")
            gen_output = self.db.getResultGeneratorPowerSum(timeMaxMin)
        elif variable == "capacity":
            gen_output = self.grid.generator.pmax
        elif variable == "spilled":
            gen_output = self.db.getResultGeneratorSpilledSums(timeMaxMin)
        else:
            print("Variable not valid")
            return

        df = self.grid.generator[["node", "type", "pmax"]].merge(
            self.grid.node[["id", "area"]], how="left", left_on="node", right_on="id"
        )
        df["VALUE"] = gen_output
        dfplot = df[["area", "type", "VALUE"]].groupby(["area", "type"]).sum()["VALUE"].unstack()

        if relative:
            dfplot = dfplot.mul(1 / dfplot.sum(axis=1), axis=0)

        return dfplot

    def plotEnergyMix(
        self, areas=None, timeMaxMin=None, relative=False, showTitle=True, variable="energy", gentypes=None
    ):
        """
        Plot energy, generation capacity or spilled energy as stacked bars

        Parameters
        ----------
        areas : list of strings
            Which areas to include, default=None means include all
        timeMaxMin : list of two integers
            Time range, [min,max]
        relative : boolean
            Whether to plot absolute (false) or relative (true) values
        variable : string ("energy","capacity","spilled")
            Which variable to plot (default is energy production)
        gentypes : list
            List of generator types to include. None gives all.
        """

        if timeMaxMin is None:
            timeMaxMin = [self.timerange[0], self.timerange[-1] + 1]

        dfplot = self.getEnergyMix(timeMaxMin=timeMaxMin, relative=relative, variable=variable)

        titles = {"energy": "Energy mix", "capacity": "Capacity mix", "spilled": "Energy spilled"}
        title = ""
        if variable in titles:
            title = titles[variable]
        if areas is None:
            areas = dfplot.index
        if gentypes is None:
            gentypes = dfplot.columns
        dfplot.loc[areas, gentypes].plot(kind="bar", stacked=True)
        plt.legend()
        handles, labels = plt.gca().get_legend_handles_labels()
        handles.reverse()
        labels.reverse()
        plt.legend(handles, labels, loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.0)

        if showTitle:
            plt.title(title)
        plt.show()
        return dfplot

    def plotTimeseriesColour(self, areas, value="nodalprice", filter_values=None):
        """
        Plot timeseries values with days on x-axis and hour of day on y-axis


        Parameters
        ----------
        areas : list of strings
            which areas to include, default=None means include all
        value : 'nodalprice' (default),
                'demand',
                'gen%<type1>%<type2>.. (where type=gentype)
            which times series value to plot

        Example: res.plotTimeseriescolour(['ES'],value='gen%solar_csp%wind')
        """

        p = {}
        pm = {}
        stepsperday = int(24 / self.grid.timeDelta)
        numdays = int(len(self.grid.timerange) / stepsperday)
        for a in areas:
            if value == "nodalprice":
                p[a] = self.getAreaPrices(area=a)
            elif value == "demand":
                # TODO: This is not generally correct. Should use
                # weighted average for all loads in area
                p[a] = self.grid.profiles["load_" + a]
            elif value[:3] == "gen":
                # value is now on form "gen_MA_hydro"
                strval = value.split("%")
                gens = self.grid.getGeneratorsPerAreaAndType()
                # genindx = gens[a][strval[1]]
                genindx = [i for s in strval[1:] for i in gens[a][s]]
                timerange = [self.timerange[0], self.timerange[-1] + 1]
                p[a] = self.db.getResultGeneratorPower(genindx, timerange)
            pm[a] = np.reshape(p[a], (numdays, stepsperday)).T

        # print("Plotting...")
        if not filter_values:
            vmin = min([min(p[a]) for a in areas])
            vmax = max([max(p[a]) for a in areas])
        else:
            vmin = filter_values[0]
            vmax = filter_values[1]
        num_areas = len(areas)
        fig, axes = plt.subplots(
            nrows=num_areas,
            ncols=1,
            figsize=((min(max(6, (len(self.grid.timerange) / 100)), 20)), (max(5, 1.5 * len(areas)))),
        )

        for n in range(num_areas):
            ax = plt.subplot(num_areas, 1, n + 1)
            ax.set_title(areas[n], x=-0.04, y=0.5, verticalalignment="center", horizontalalignment="right")
            plt.imshow(pm[areas[n]], vmin=vmin, vmax=vmax)

        fig.subplots_adjust(right=0.90)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        # fig.colorbar(im,cax=cbar_ax)
        plt.colorbar(cax=cbar_ax)
        plt.show()

    def plotRelativeLoadDistribution(
        self, show_node_labels=False, latlon=None, dotsize=40, draw_par_mer=False, colours=True, showTitle=True
    ):
        """
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

        """
        # basemap is only used here, so to allow using powergama without
        # basemap installed, it is best to put import statement here.
        from collections import OrderedDict

        from mpl_toolkits.basemap import Basemap

        plt.figure()

        data = self.grid

        if latlon is None:
            lat_max = max(data.node.lat) + 1
            lat_min = min(data.node.lat) - 1
            lon_max = max(data.node.lon) + 1
            lon_min = min(data.node.lon) - 1
        else:
            lat_min = latlon[0]
            lon_min = latlon[1]
            lat_max = latlon[2]
            lon_max = latlon[3]

        # Use the average latitude as latitude of true scale
        lat_truescale = np.mean(data.node.lat)

        m = Basemap(
            resolution="l",
            projection="merc",
            lat_ts=lat_truescale,
            llcrnrlon=lon_min,
            llcrnrlat=lat_min,
            urcrnrlon=lon_max,
            urcrnrlat=lat_max,
            anchor="W",
        )

        # Draw coastlines, meridians and parallels.
        m.drawcoastlines()
        m.drawcountries(zorder=0)
        if colours:
            m.fillcontinents(color="coral", lake_color="aqua", zorder=0)
            m.drawmapboundary(fill_color="aqua")
        else:
            m.fillcontinents(zorder=0)
            m.drawmapboundary()

        if draw_par_mer:
            m.drawparallels(
                np.arange(_myround(lat_min, 10, "floor"), _myround(lat_max, 10, "ceil"), 10), labels=[1, 1, 0, 0]
            )

            m.drawmeridians(
                np.arange(_myround(lon_min, 10, "floor"), _myround(lon_max, 10, "ceil"), 10), labels=[0, 0, 0, 1]
            )

        # AC Branches
        idx_from = data.branchFromNodeIdx()
        idx_to = data.branchToNodeIdx()
        branch_lat1 = [data.node.lat[i] for i in idx_from]
        branch_lon1 = [data.node.lon[i] for i in idx_from]
        branch_lat2 = [data.node.lat[i] for i in idx_to]
        branch_lon2 = [data.node.lon[i] for i in idx_to]

        x1, y1 = m(branch_lon1, branch_lat1)
        x2, y2 = m(branch_lon2, branch_lat2)

        ls = [[(x1[i], y1[i]), (x2[i], y2[i])] for i in range(len(x1))]
        line_segments_ac = mpl.collections.LineCollection(ls, linewidths=2, colors="k")

        ax = plt.axes()
        ax.add_collection(line_segments_ac)

        # DC Branches
        idx_from = data.dcBranchFromNodeIdx()
        idx_to = data.dcBranchToNodeIdx()
        branch_lat1 = [data.node.lat[i] for i in idx_from]
        branch_lon1 = [data.node.lon[i] for i in idx_from]
        branch_lat2 = [data.node.lat[i] for i in idx_to]
        branch_lon2 = [data.node.lon[i] for i in idx_to]

        x1, y1 = m(branch_lon1, branch_lat1)
        x2, y2 = m(branch_lon2, branch_lat2)
        ls = [[(x1[i], y1[i]), (x2[i], y2[i])] for i in range(len(x1))]
        if colours:
            dcColour = "b"
        else:
            dcColour = "k"
        line_segments_dc = mpl.collections.LineCollection(ls, linewidths=2, colors=dcColour)

        ax.add_collection(line_segments_dc)

        # Loads
        x, y = m(data.node["lon"].tolist(), data.node["lat"].tolist())
        loadByNode = OrderedDict([(k, 0) for k in data.node["id"]])
        consNodes = data.consumer.node
        consValues = data.consumer.demand_avg

        for idx in range(len(consNodes)):
            loadByNode[consNodes[idx]] += consValues[idx]

        avgLoad = np.mean(consValues)
        relativeLoads = [dotsize * (myload[1] / avgLoad) for myload in loadByNode.items()]

        if colours:
            loadColour = "b"
        else:
            loadColour = "w"
        m.scatter(x, y, marker="o", c=loadColour, zorder=2, s=relativeLoads)

        # Show names of nodes
        if show_node_labels:
            labels = data.node["id"]
            x1, x2, y1, y2 = plt.axis()
            offset_x = (x2 - x1) / 50
            for label, xpt, ypt in zip(labels, x, y):
                if xpt > x1 and xpt < x2 and ypt > y1 and ypt < y2:
                    plt.text(xpt + offset_x, ypt, label)

        if showTitle:
            plt.title("Load distribution")

    def plotRelativeGenerationCapacity(
        self, tech, show_node_labels=False, latlon=None, dotsize=40, draw_par_mer=False, colours=True, showTitle=True
    ):
        """
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

        """

        # basemap is only used here, so to allow using powergama without
        # basemap installed, it is best to put import statement here.
        from collections import OrderedDict

        from mpl_toolkits.basemap import Basemap

        data = self.grid

        plt.figure()

        num_branches = data.branch.shape[0]

        genTypes = data.getAllGeneratorTypes()
        if tech not in genTypes:
            raise Exception(
                "No generators classified as " + tech + ".\n" "Generator classifications: " + str(genTypes)[1:-1]
            )

        if latlon is None:
            lat_max = max(data.node.lat) + 1
            lat_min = min(data.node.lat) - 1
            lon_max = max(data.node.lon) + 1
            lon_min = min(data.node.lon) - 1
        else:
            lat_min = latlon[0]
            lon_min = latlon[1]
            lat_max = latlon[2]
            lon_max = latlon[3]

        # Use the average latitude as latitude of true scale
        lat_truescale = np.mean(data.node.lat)

        m = Basemap(
            resolution="l",
            projection="merc",
            lat_ts=lat_truescale,
            llcrnrlon=lon_min,
            llcrnrlat=lat_min,
            urcrnrlon=lon_max,
            urcrnrlat=lat_max,
            anchor="W",
        )

        # Draw coastlines, meridians and parallels.
        m.drawcoastlines()
        m.drawcountries(zorder=0)
        if colours:
            m.fillcontinents(color="coral", lake_color="aqua", zorder=0)
            m.drawmapboundary(fill_color="aqua")
        else:
            m.fillcontinents(zorder=0)
            m.drawmapboundary()

        if draw_par_mer:
            m.drawparallels(
                np.arange(_myround(lat_min, 10, "floor"), _myround(lat_max, 10, "ceil"), 10), labels=[1, 1, 0, 0]
            )

            m.drawmeridians(
                np.arange(_myround(lon_min, 10, "floor"), _myround(lon_max, 10, "ceil"), 10), labels=[0, 0, 0, 1]
            )

        lwidths = [2] * num_branches

        # AC Branches
        idx_from = data.branchFromNodeIdx()
        idx_to = data.branchToNodeIdx()
        branch_lat1 = [data.node.lat[i] for i in idx_from]
        branch_lon1 = [data.node.lon[i] for i in idx_from]
        branch_lat2 = [data.node.lat[i] for i in idx_to]
        branch_lon2 = [data.node.lon[i] for i in idx_to]

        x1, y1 = m(branch_lon1, branch_lat1)
        x2, y2 = m(branch_lon2, branch_lat2)

        ls = [[(x1[i], y1[i]), (x2[i], y2[i])] for i in range(len(x1))]
        line_segments_ac = mpl.collections.LineCollection(ls, linewidths=lwidths, colors="k")

        ax = plt.axes()
        ax.add_collection(line_segments_ac)

        # DC Branches
        idx_from = data.dcBranchFromNodeIdx()
        idx_to = data.dcBranchToNodeIdx()
        branch_lat1 = [data.node.lat[i] for i in idx_from]
        branch_lon1 = [data.node.lon[i] for i in idx_from]
        branch_lat2 = [data.node.lat[i] for i in idx_to]
        branch_lon2 = [data.node.lon[i] for i in idx_to]

        x1, y1 = m(branch_lon1, branch_lat1)
        x2, y2 = m(branch_lon2, branch_lat2)
        ls = [[(x1[i], y1[i]), (x2[i], y2[i])] for i in range(len(x1))]
        if colours:
            dcColour = "b"
        else:
            dcColour = "k"
        line_segments_dc = mpl.collections.LineCollection(ls, linewidths=2, colors=dcColour)

        ax.add_collection(line_segments_dc)

        # Generators
        x, y = m(data.node["lon"].tolist(), data.node["lat"].tolist())
        generators = data.getGeneratorsPerType()[tech]
        genNode = data.generator.node
        genCapacity = data.generator.pmax
        capByNode = OrderedDict([(k, 0) for k in data.node.id])
        totCap = 0

        for gen in generators:
            capByNode[genNode[gen]] += genCapacity[gen]
            totCap += genCapacity[gen]

        avgCap = totCap / len(generators)
        relativeCap = [dotsize * (g[1] / avgCap) for g in capByNode.items()]

        if colours:
            loadColour = "b"
        else:
            loadColour = "w"
        m.scatter(x, y, marker="s", c=loadColour, zorder=2, s=relativeCap)

        # Show names of nodes
        if show_node_labels:
            labels = data.node.id
            x1, x2, y1, y2 = plt.axis()
            offset_x = (x2 - x1) / 50
            for label, xpt, ypt in zip(labels, x, y):
                if xpt > x1 and xpt < x2 and ypt > y1 and ypt < y2:
                    plt.text(xpt + offset_x, ypt, label)

        if showTitle:
            plt.title(tech + " capacity distribution")

    def plotGenerationScatter(self, area, tech=[], dotsize=300, annotations=True):
        """
        Scatter plot of generation capacity and correlation of inflow
        with load.

        Parameters
        ----------
        area : string
            area to plot
        tech : list of strings
            production technologies to plot. Empty list = all
        dotsize : integer
            adjust the size of scatterplots
        annotations: boolean
            whether to plot annotations
        """
        data = self.grid

        plt.figure()

        if len(tech) > 0:
            genTypes = data.getAllGeneratorTypes()
            for gt in tech:
                if gt not in genTypes:
                    raise Exception(
                        "No generators classified as " + gt + ".\n Generator classifications: " + str(genTypes)[1:-1]
                    )
        else:
            tech = data.getAllGeneratorTypes()

        genByType = data.getGeneratorsPerAreaAndType()[area]
        detailedGen = {}
        for gt in genByType.keys():
            if gt in tech:
                detailedGen[gt] = {}
                genIdx = genByType[gt]
                for i in genIdx:
                    capacity = data.generator.pmax[i]
                    infProfile = data.generator.inflow_ref[i]
                    if infProfile in detailedGen[gt]:
                        detailedGen[gt][infProfile] += capacity
                    else:
                        detailedGen[gt][infProfile] = capacity

        cons = data.consumer.demand_ref[data.getConsumersPerArea()[area][0]]
        demandProfile = data.profiles[cons]
        x, y, label, size, colCoor, tickLabel = [], [], [], [], [], []
        mainCount = 0
        for gt in detailedGen.keys():
            tickLabel.append(gt)
            subCount = 0
            for infProf in detailedGen[gt].keys():
                if infProf == "const":
                    y.append(1)
                else:
                    y.append(np.corrcoef(data.profiles[infProf], demandProfile)[0, 1])
                x.append(mainCount)
                label.append(infProf)
                size.append(detailedGen[gt][infProf])
                subCount += 1
            mainCount += 1

        avgSize = np.mean(size)
        size = [dotsize * s / avgSize for s in size]
        colCoor = [c / mainCount for c in x]
        colours = mpl.cm.hsv(colCoor)
        plt.scatter(x, y, s=size, c=colours, alpha=0.5)
        if annotations:
            for label, x, y in zip(label, x, y):
                plt.annotate(
                    label,
                    xy=(x, y),
                    xytext=(20, 0),
                    textcoords="offset points",
                    ha="left",
                    va="bottom",
                    arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0"),
                )

        plt.ylabel("Correlation coefficient of inflow to load")
        plt.xlim(xmin=-0.5, xmax=mainCount + 0.5)
        plt.xticks(range(mainCount + 1), tickLabel)


def _myround(x, base=1, method="round"):
    """Round to nearest multiple of base"""
    if method == "round":
        return int(base * round(float(x) / base))
    elif method == "floor":
        return int(base * math.floor(float(x) / base))
    elif method == "ceil":
        return int(base * math.ceil(float(x) / base))
    else:
        raise
