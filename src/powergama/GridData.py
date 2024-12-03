# -*- coding: utf-8 -*-
"""
Module containing PowerGAMA GridData class and sub-classes

Grid data and time-dependent profiles
"""

import math  # Used in myround
import warnings

import networkx as nx
import numpy
import pandas as pd
import scipy.sparse


class GridData(object):
    """
    Class for grid data storage and import
    """

    # Headers and default values in input files:
    # default=None: column _must_ be present in input file
    keys_powergama = {
        "node": {"id": None, "area": None, "lat": None, "lon": None},
        "branch": {"node_from": None, "node_to": None, "reactance": None, "capacity": None, "resistance": 0},
        "dcbranch": {"node_from": None, "node_to": None, "capacity": None, "resistance": 0},
        "generator": {
            "type": None,
            "desc": "",
            "node": None,
            "pmax": None,
            "pmin": None,
            "fuelcost": None,
            "inflow_fac": None,
            "inflow_ref": None,
            "storage_cap": 0,
            "storage_price": 0,
            "storage_ini": 0,
            "storval_filling_ref": "",
            "storval_time_ref": "",
            "pump_cap": 0,
            "pump_efficiency": 0,
            "pump_deadband": 0,
        },
        "consumer": {
            "node": None,
            "demand_avg": None,
            "demand_ref": None,
            "flex_fraction": 0,
            "flex_on_off": 0,
            "flex_basevalue": 0,
            "flex_storage": 0,
            "flex_storval_filling": "",
            "flex_storval_time": "",
            "flex_storagelevel_init": 0.5,
        },
    }

    # Required fields for investment analysis input data
    # Default value = -1 means that it should be computed by the program
    keys_sipdata = {
        "node": {
            "id": None,
            "area": None,
            "lat": None,
            "lon": None,
            "offshore": None,
            "type": None,
            "existing": None,
            "cost_scaling": None,
        },
        "branch": {
            "node_from": None,
            "node_to": None,
            "capacity": None,
            "capacity2": 0,
            "reactance": 0,
            "expand": None,
            "expand2": None,
            "max_newCap": -1,
            "distance": -1,
            "cost_scaling": None,
            "type": None,
        },
        "dcbranch": {"node_from": "", "node_to": "", "capacity": 0, "resistance": 0},
        "generator": {
            "type": None,
            "node": None,
            "desc": "",
            "pmax": None,
            "pmax2": 0,
            "pmin": None,
            "expand": None,
            "expand2": None,
            "p_maxNew": -1,
            "cost_scaling": 1,
            "fuelcost": None,
            "fuelcost_ref": None,
            "pavg": 0,
            "inflow_fac": None,
            "inflow_ref": None,
        },
        "consumer": {"node": None, "demand_avg": None, "emission_cap": -1, "demand_ref": None},
    }

    def __init__(self):
        """
        Create GridData object with data and methods for import and
        processing of PowerGAMA grid data
        """
        self.node = None
        self.branch = None
        self.dcbranch = None
        self.generator = None
        self.consumer = None
        # self.inflowProfiles = None
        # self.demandProfiles = None
        self.profiles = None
        self.storagevalue_filling = None
        self.storagevalue_time = None
        self.timeDelta = None
        self.timerange = None

        self.CSV_SEPARATOR = None  # automatically detect

    def _myround(x, base=1, method="round"):
        """Round to nearest multiple of base"""
        if method == "round":
            return int(base * round(float(x) / base))
        elif method == "floor":
            return int(base * math.floor(float(x) / base))
        elif method == "ceil":
            return int(base * math.ceil(float(x) / base))
        else:
            raise Exception("Rounding error")

    def readGridData(self, nodes, ac_branches, dc_branches, generators, consumers, remove_extra_columns=False):
        """Read grid data from files into data variables"""

        self.node = pd.read_csv(nodes, dtype={"id": str, "area": str})
        self.branch = pd.read_csv(ac_branches, dtype={"node_from": str, "node_to": str, "capacity": float})
        if dc_branches is not None:
            self.dcbranch = pd.read_csv(dc_branches, dtype={"node_from": str, "node_to": str, "capacity": float})
        else:
            self.dcbranch = pd.DataFrame(columns=self.keys_powergama["dcbranch"].keys())
        self.generator = pd.read_csv(generators, dtype={"node": str, "type": str})
        self.consumer = pd.read_csv(consumers, dtype={"node": str})

        self._checkGridDataFields(self.keys_powergama)

        # TODO: fix difference in node index between powergama and powergim
        # numerical index is needed in method computePowerFlowMatrices
        # (and maybe elsewhere). This does _not_ work with powergama:
        # self.node.set_index('id',inplace=True,append=False)
        # self.node['id']=self.node.index
        self._checkGridData()
        self._addDefaultColumns(keys=self.keys_powergama, remove_extra_columns=remove_extra_columns)
        self._fillEmptyCells(keys=self.keys_powergama)
        self._checkConsistency()

    def readSipData(self, nodes, branches, generators, consumers):
        """Read grid data for investment analysis from files (PowerGIM)

        This is used with the grid investment module (PowerGIM)

        time-series data may be used for
        consumer demand
        generator inflow (e.g. solar and wind)
        generator fuelcost (e.g. one generator with fuelcost = power price)
        """
        self.node = pd.read_csv(
            nodes,
            # usecols=self.keys_sipdata['node'],
            dtype={"id": str, "area": str},
        )
        # TODO use integer range index instead of id string, cf powergama
        self.node.set_index("id", inplace=True)
        self.node["id"] = self.node.index
        self.node.index.name = "index"
        self.branch = pd.read_csv(
            branches,
            # usecols=self.keys_sipdata['branch'],
            dtype={"node_from": str, "node_to": str},
        )
        # dcbranch variable only needed for powergama.plotMapGrid
        self.dcbranch = pd.DataFrame()
        self.generator = pd.read_csv(
            generators,
            # usecols=self.keys_sipdata['generator'],
            dtype={"node": str, "type": str},
        )
        self.consumer = pd.read_csv(
            consumers,
            # usecols=self.keys_sipdata['consumer'],
            dtype={"node": str},
        )

        self._checkGridDataFields(self.keys_sipdata)
        self._addDefaultColumns(keys=self.keys_sipdata)
        self._fillEmptyCells(keys=self.keys_sipdata)
        self._checkGridData()

    def _fillEmptyCells(self, keys):
        """Use default data where none is given"""
        # generators:
        for col, val in keys["generator"].items():
            if val is not None:
                self.generator[col] = self.generator[col].fillna(keys["generator"][col])
        # consumers:
        for col, val in keys["consumer"].items():
            if val is not None:
                self.consumer[col] = self.consumer[col].fillna(keys["consumer"][col])

        # branches:
        for col, val in keys["branch"].items():
            if val is not None:
                self.branch[col] = self.branch[col].fillna(keys["branch"][col])

    def _addDefaultColumns(self, keys, remove_extra_columns=False):
        """insert optional columns with default values when none
        are provided in input files"""
        for k in keys["generator"]:
            if k not in self.generator.keys():
                self.generator[k] = keys["generator"][k]
        for k in keys["consumer"]:
            if k not in self.consumer.keys():
                self.consumer[k] = keys["consumer"][k]
        for k in keys["branch"]:
            if k not in self.branch.keys():
                self.branch[k] = keys["branch"][k]
        for k in keys["dcbranch"]:
            if k not in self.dcbranch.keys():
                self.dcbranch[k] = keys["dcbranch"][k]

        # Discard extra columns (comments etc)
        if remove_extra_columns:
            self.node = self.node[list(keys["node"].keys())]
            self.branch = self.branch[list(keys["branch"].keys())]
            self.dcbranch = self.dcbranch[list(keys["dcbranch"].keys())]
            self.generator = self.generator[list(keys["generator"].keys())]
            self.consumer = self.consumer[list(keys["consumer"].keys())]

    def _checkGridDataFields(self, keys):
        """check if all required columns are present
        (ie. all columns with no default value)"""
        for k, v in keys["node"].items():
            if v is None and k not in self.node:
                raise Exception("Node input file must contain %s" % k)
        for k, v in keys["branch"].items():
            if v is None and k not in self.branch:
                raise Exception("Branch input file must contain %s" % k)
        for k, v in keys["dcbranch"].items():
            if v is None and k not in self.dcbranch:
                raise Exception("DC branch input file must contain %s" % k)
        for k, v in keys["generator"].items():
            if v is None and k not in self.generator:
                raise Exception("Generator input file must contain %s" % k)
        for k, v in keys["consumer"].items():
            if v is None and k not in self.consumer:
                raise Exception("Consumer input file must contain %s" % k)

    def _checkConsistency(self):
        """Check consistency between default and provided values.
        Currently only checks for pumping capacity and efficiency,
        but more tests could be added."""

        # generators
        for g_idx in range(len(self.generator)):
            # Check whether pump capacity and efficiency are consistent
            cur_gen = self.generator.iloc[g_idx]
            if not math.isclose(cur_gen.pump_cap, 0, abs_tol=1e-8) and math.isclose(
                cur_gen.pump_efficiency, 0, abs_tol=1e-8
            ):
                warn_message = (
                    "Warning, generator at node '%s' " % cur_gen.node
                    + "has very low pump efficiency (%s), " % cur_gen.pump_efficiency
                    + "but non-zero pump capacity (%s)." % cur_gen.pump_cap
                )
                warnings.warn(warn_message, UserWarning)

    def _checkGridData(self):
        """Check consistency of grid data"""

        # generator nodes
        for g in self.generator["node"]:
            if g not in self.node["id"].values:
                raise Exception("Generator node does not exist: '%s'" % g)

        # consumer nodes
        for c in self.consumer["node"]:
            if c not in self.node["id"].values:
                raise Exception("Consumer node does not exist: '%s'" % c)

        # branch nodes
        for c in self.branch["node_from"]:
            if c not in self.node["id"].values:
                raise Exception("Branch from node does not exist: '%s'" % c)
        for c in self.branch["node_to"]:
            if c not in self.node["id"].values:
                raise Exception("Branch to node does not exist: '%s'" % c)

        # dcbranch nodes
        for c in self.dcbranch["node_from"]:
            if c not in self.node["id"].values:
                raise Exception("DC Branch from node does not exist: '%s'" % c)
        for c in self.dcbranch["node_to"]:
            if c not in self.node["id"].values:
                raise Exception("DC Branch to node does not exist: '%s'" % c)

    def _readProfileFromFile(self, filename, timerange):
        profiles = pd.read_csv(filename, sep=self.CSV_SEPARATOR, engine="python")
        profiles = profiles.loc[timerange]
        profiles.index = range(len(timerange))
        return profiles

    def _readStoragevaluesFromFile(self, filename):
        profiles = pd.read_csv(filename, sep=self.CSV_SEPARATOR, engine="python")
        return profiles

    def readProfileData(self, filename, timerange, storagevalue_filling=None, storagevalue_time=None, timedelta=1.0):
        """Read profile (timeseries) into numpy arrays"""

        # self.inflowProfiles = self._readProfileFromFile(inflow,timerange)
        # self.demandProfiles = self._readProfileFromFile(demand,timerange)
        self.profiles = self._readProfileFromFile(filename, timerange)
        self.timerange = timerange
        self.timeDelta = timedelta

        """
        Storage values have both time dependence and filling level dependence

       The dependence is on filling level (0-100%), is given as an array
        with 101 elements
        """
        if storagevalue_filling is not None:
            self.storagevalue_time = self._readProfileFromFile(storagevalue_time, timerange)
            self.storagevalue_filling = self._readStoragevaluesFromFile(storagevalue_filling)
        return

    def writeGridDataToFiles(self, prefix):
        """
        Save data to new input files
        """

        file_nodes = prefix + "node.csv"
        file_branches = prefix + "branch.csv"
        file_consumers = prefix + "consumer.csv"
        file_generators = prefix + "generator.csv"
        file_dcbranch = prefix + "dcbranch.csv"

        sep = self.CSV_SEPARATOR
        if sep is None:
            sep = ","

        self.node.to_csv(file_nodes, sep=sep, index=False)
        self.branch.to_csv(file_branches, sep=sep, index=False)
        self.consumer.to_csv(file_consumers, sep=sep, index=False)
        self.generator.to_csv(file_generators, sep=sep, index=False)
        self.dcbranch.to_csv(file_dcbranch, sep=sep, index=False)
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
        if self.dcbranch is not None:
            return self.dcbranch.shape[0]
        else:
            return 0

    def branchFromNodeIdx(self):
        """get node indices for branch FROM node"""
        # return [self.node[self.node['id']==self.branch['node_from'][k]]
        #        .index.tolist()[0] for k in range(self.numBranches())]
        return [self.node[self.node["id"] == b["node_from"]].index.tolist()[0] for i, b in self.branch.iterrows()]

    def branchToNodeIdx(self):
        """get node indices for branch TO node"""
        return [
            self.node[self.node["id"] == self.branch["node_to"][k]].index.tolist()[0]
            for k in self.branch.index.tolist()
        ]

    def dcBranchFromNodeIdx(self):
        """get node indices for dc branch FROM node"""
        return [
            self.node[self.node["id"] == self.dcbranch["node_from"][k]].index.tolist()[0]
            for k in self.dcbranch.index.tolist()
        ]

    def dcBranchToNodeIdx(self):
        """get node indices for dc branch TO node"""
        return [
            self.node[self.node["id"] == self.dcbranch["node_to"][k]].index.tolist()[0]
            for k in self.dcbranch.index.tolist()
        ]

    def getGeneratorsAtNode(self, nodeIdx):
        """Indices of all generators attached to a particular node"""
        # indices = [i for i, x in enumerate(self.generator['node'])
        #            if x == self.node['id'][nodeIdx]]
        indices = self.generator["node"][self.generator.loc[:, "node"] == self.node["id"][nodeIdx]].index.tolist()
        return indices

    def getGeneratorsWithPumpAtNode(self, nodeIdx):
        """Indices of all pumps attached to a particular node"""
        # indices = [i for i, x in enumerate(self.generator['node'])
        #            if x == self.node['id'][nodeIdx]
        #            and self.generator['pump_cap'][i]>0]
        indices = self.generator["node"][
            (self.generator.loc[:, "node"] == self.node["id"][nodeIdx]) & (self.generator.loc[:, "pump_cap"] > 0)
        ].index.tolist()
        return indices

    def getLoadsAtNode(self, nodeIdx):
        """Indices of all loads (consumers) attached to a particular node"""
        # indices = [i for i, x in enumerate(self.consumer['node'])
        #            if x == self.node['id'][nodeIdx]]:
        # 25 times faster:
        indices = self.consumer["node"][self.consumer.loc[:, "node"] == self.node["id"][nodeIdx]].index.tolist()
        return indices

    def getLoadsFlexibleAtNode(self, nodeIdx):
        """Indices of all flexible nodes attached to a particular node"""
        # indices = [i for i, x in enumerate(self.consumer['node'])
        #            if x == self.node['id'][nodeIdx]
        #            and self.consumer['flex_fraction'][i]>0
        #            and self.consumer['demand_avg'][i]>0]
        # faster:
        indices = self.consumer["node"][
            (self.consumer.loc[:, "node"] == self.node["id"][nodeIdx])
            & (self.consumer.loc[:, "flex_fraction"] > 0)
            & (self.consumer.loc[:, "demand_avg"] > 0)
        ].index.tolist()
        return indices

    def getIdxConsumersWithFlexibleLoad(self):
        """Indices of all consumers with flexible load"""
        idx = [
            i
            for i, v in enumerate(self.consumer["flex_fraction"])
            if v > 0 and v < numpy.inf and self.consumer["demand_avg"][i] > 0
        ]
        return idx

    def getFlexibleLoadStorageCapacity(self, consumer_indx):
        """flexible load storage capacity in MWh"""
        cap = (
            self.consumer["demand_avg"][consumer_indx]
            * self.consumer["flex_fraction"][consumer_indx]
            * self.consumer["flex_storage"][consumer_indx]
        )
        return cap

    def getDcBranchesAtNode(self, nodeIdx, direction):
        """Indices of all DC branches attached to a particular node"""
        if direction == "from":
            indices = [i for i, x in enumerate(self.dcbranch["node_from"]) if x == self.node["id"][nodeIdx]]
        elif direction == "to":
            indices = [i for i, x in enumerate(self.dcbranch["node_to"]) if x == self.node["id"][nodeIdx]]
        else:
            raise Exception("Unknown direction in GridData.getDcBranchesAtNode")
        return indices

    def getDcBranches(self):
        """
        Returns a list with DC branches in the format
        [index,from area,to area]
        """
        hvdcBranches = []
        for idx in self.dcbranch.index.tolist():
            fromNodeIdx = self.node["id"].index(self.dcbranch["node_from"][idx])
            toNodeIdx = self.node.name.index(self.dcbranch["node_to"][idx])
            areaFrom = self.node["area"][fromNodeIdx]
            areaTo = self.node["area"][toNodeIdx]
            hvdcBranches.append([idx, areaFrom, areaTo])
        return hvdcBranches

    def getIdxNodesWithLoad(self):
        """Indices of nodes that have load (consumer) attached to them"""
        # Get index of node associated with all consumer
        # indices = numpy.asarray(self.consumer.nodeIdx(self.node))
        # Return indices only once (unique values)
        # indices = numpy.unique(indices)
        loadnodes = self.node[self.node["id"].isin(self.consumer["node"])]
        indices = numpy.asarray(loadnodes.index)
        return indices

    def getIdxGeneratorsWithStorage(self):
        """Indices of all generators with nonzero and non-infinite storage"""
        idx = [i for i, v in enumerate(self.generator["storage_cap"]) if v > 0 and v < numpy.inf]
        return idx

    def getIdxGeneratorsWithNonzeroInflow(self):
        """Indices of all generators with nonzero inflow"""
        idx = [i for i, v in enumerate(self.generator["inflow_fac"]) if v > 0]
        return idx

    def getIdxGeneratorsWithPumping(self):
        """Indices of all generators with pumping capacity"""
        idx = [i for i, v in enumerate(self.generator["pump_cap"]) if v > 0 and v < numpy.inf]
        return idx

    def getIdxBranchesWithFlowConstraints(self):
        """Indices of branches with less than infinite branch capacity"""
        idx = [i for i, v in enumerate(self.branch["capacity"]) if v < numpy.inf]
        return idx

    def getIdxDcBranchesWithFlowConstraints(self):
        """Indices of DC branches with less than infinite branch capacity"""
        if self.dcbranch is None:
            idx = []
        else:
            idx = [i for i, v in enumerate(self.dcbranch["capacity"]) if v < numpy.inf]
        return idx

    def getIdxBranchesWithLength(self):
        """Indices of branches with specified length"""
        idx = [i for i, v in enumerate(self.branch["length"]) if not math.isnan(v)]
        return idx

    def _susceptancePu(self, baseOhm=1):
        """If impedance is already given in pu, baseOhm should be 1
        If not, well... baseOhm depends on the voltage level, so need to know
        the nominal voltage at the bus to convert from ohm to pu.
        """
        if (self.branch["reactance"] == 0).any():
            raise Exception("Zero branch reactance is not allowed.")
        return [-1 / self.branch["reactance"][i] * baseOhm for i in self.branch.index.tolist()]

    def compute_power_flow_matrices(self, base_Z=1):
        """
        Compute and return dc power flow matrices B' and DA
        Parameters
        ==========
        base_Z : float (impedance should already be in pu.)
                base value for impedance
        Returns
        =======
        (coeff_B, coeff_DA) : scipy.sparse matrices
        """

        # df_branch = pd.DataFrame.from_dict(branches, orient="index")
        branches = self.branch
        susceptance = 1 / branches["reactance"] * base_Z
        nodes = list(self.node["id"])
        # node_ids = nodes
        edge_ids = []
        edges = []
        # for br_id, branch in branches.items():
        for br_id, branch in branches.iterrows():
            b = susceptance[br_id]
            edges.append((branch["node_from"], branch["node_to"], br_id, {"i": br_id, "b": b}))
            edge_ids.append(br_id)

        # MultiDiGraph to allow parallel lines
        G = nx.MultiDiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        A_incidence_matrix = -nx.incidence_matrix(G, oriented=True, nodelist=nodes, edgelist=edges).T
        A_incidence_matrix = scipy.sparse.csc_matrix(A_incidence_matrix)

        # Diagonal matrix
        D = scipy.sparse.diags(-susceptance, offsets=0)
        # Element-wise multiplication:
        DA = D * A_incidence_matrix

        # Bf constructed from incidence matrix with branch susceptance
        # used as weight (this is quite fast)
        Bf = -nx.incidence_matrix(G, oriented=True, nodelist=nodes, edgelist=edges, weight="b").T
        # See note above for this conversion
        Bf = scipy.sparse.csc_matrix(Bf)
        # Element-wise multiplication:
        Bbus = A_incidence_matrix.T * Bf
        return Bbus, DA

    def getAllAreas(self):
        """Return list of areas included in the grid model"""
        areas = self.node["area"]
        allareas = []
        for co in areas:
            if co not in allareas:
                allareas.append(co)
        return allareas

    def getAllGeneratorTypes(self, sort="fuelcost"):
        """Return list of generator types included in the grid model"""
        gentypes = self.generator.type
        alltypes = []
        if sort is None:
            gentypes = self.generator.type
            alltypes = []
            for ge in gentypes:
                if ge not in alltypes:
                    alltypes.append(ge)
            return alltypes
        elif sort == "fuelcost":
            generators = self.getGeneratorsPerType()
            avg = {ge_k: numpy.mean(self.generator.fuelcost[ge_v]) for ge_k, ge_v in generators.items()}
            sorted_list = [k for k in sorted(avg, key=avg.get, reverse=False)]

            #            gentypes = generators.keys()
            #            fuelcosts = []
            #            for ge in gentypes:
            #                gen_this_type = generators[ge]
            #                fuelcosts.append(numpy.mean([self.generator.fuelcost[i]
            #                                         for i in gen_this_type]) )
            #            sorted_list = [x for (y,x) in
            #                           sorted(zip(fuelcosts,gentypes))]
            return sorted_list
        else:
            raise Exception("sort must be None or 'fuelcost'")

    def getConsumerAreas(self):
        """List of areas for each consumer"""
        areas = [self.node.area[self.node["id"] == n].tolist()[0] for n in self.consumer["node"]]
        return areas

    def getGeneratorAreas(self):
        """List of areas for each generator"""
        areas = [self.node.area[self.node["id"] == n].tolist()[0] for n in self.generator.node]
        return areas

    def getConsumersPerArea(self):
        """Returns dictionary with indices of loads within each area"""
        consumers = {}
        consumer_areas = self.getConsumerAreas()
        for idx_load in self.consumer.index.tolist():
            area_name = consumer_areas[idx_load]
            if area_name in consumers:
                consumers[area_name].append(idx_load)
            else:
                consumers[area_name] = [idx_load]
        return consumers

    def getGeneratorsPerAreaAndType(self):
        """Returns dictionary with indices of generators within each area"""
        generators = {}
        generator_areas = self.getGeneratorAreas()
        for idx_gen in self.generator.index.tolist():
            gtype = self.generator.type[idx_gen]
            area_name = generator_areas[idx_gen]
            if area_name in generators:
                if gtype in generators[area_name]:
                    generators[area_name][gtype].append(idx_gen)
                else:
                    generators[area_name][gtype] = [idx_gen]
            else:
                generators[area_name] = {gtype: [idx_gen]}
        return generators

    def getGeneratorsPerType(self):
        """Returns dictionary with indices of generators per type"""
        generators = {}
        for idx_gen in self.generator.index.tolist():
            gtype = self.generator["type"][idx_gen]
            if gtype in generators:
                generators[gtype].append(idx_gen)
            else:
                generators[gtype] = [idx_gen]
        return generators

    def getGeneratorsWithPumpByArea(self):
        """
        Returns dictionary with indices of generators with pumps within
        each area
        """
        generators = {}
        for pumpIdx, gen in self.generator.iterrows():
            cap = gen["pump_cap"]
            if cap > 0 and cap < numpy.inf:
                nodeId = self.generator.loc[pumpIdx, "node"]
                nodeIdx = self.node[self.node["id"] == nodeId].index[0]
                areaName = self.node.loc[nodeIdx, "area"]
                if areaName in generators:
                    generators[areaName].append(pumpIdx)
                else:
                    generators[areaName] = [pumpIdx]
        return generators

    def getBranchAreas(self):
        br_witharea = pd.DataFrame()
        br_witharea["area_from"] = self.branch.merge(
            right=self.node[["id", "area"]], how="left", left_on="node_from", right_on="id"
        )["area"]
        br_witharea["area_to"] = self.branch.merge(
            right=self.node[["id", "area"]], how="left", left_on="node_to", right_on="id"
        )["area"]
        return br_witharea

    def getDcBranchAreas(self):
        br_witharea = self.dcbranch
        br_witharea["area_from"] = self.dcbranch.merge(
            right=self.node[["id", "area"]], how="left", left_on="node_from", right_on="id"
        )["area"]
        br_witharea["area_to"] = self.dcbranch.merge(
            right=self.node[["id", "area"]], how="left", left_on="node_to", right_on="id"
        )["area"]
        return br_witharea

    def getInterAreaBranches(self, area_from=None, area_to=None, acdc="ac"):
        """
        Get indices of branches from and/or to specified area(s)

        area_from = area from. Use None (default) to leave unspecifie
        area_to= area to. Use None (default) to leave unspecified
        acdc = 'ac' (default) for ac branches, 'dc' for dc branches
        """

        if area_from is None and area_to is None:
            raise Exception("Either from area or to area (or both) has" + "to be specified)")

        if acdc == "ac":
            br_witharea = self.getBranchAreas()
        elif acdc == "dc":
            br_witharea = self.getDcBranchAreas()

        if area_to is not None:
            mask_pos = (br_witharea["area_to"] == area_to) & (br_witharea["area_from"] != area_to)
            mask_neg = (br_witharea["area_from"] == area_to) & (br_witharea["area_to"] != area_to)
        if area_from is not None:
            mask_pos2 = (br_witharea["area_from"] == area_from) & (br_witharea["area_to"] != area_from)
            mask_neg2 = (br_witharea["area_to"] == area_from) & (br_witharea["area_from"] != area_from)
            if area_to is not None:
                mask_pos = mask_pos & mask_pos2
                mask_neg = mask_neg & mask_neg2
            else:
                mask_pos = mask_pos2
                mask_neg = mask_neg2

        branches_pos = list(br_witharea.index[mask_pos])
        branches_neg = list(br_witharea.index[mask_neg])

        #        # indices of from and to nodes of all branches:
        #        if acdc=='ac':
        #            br_from_nodes = self.branchFromNodeIdx()
        #            br_to_nodes = self.branchToNodeIdx()
        #        elif acdc=='dc':
        #            br_from_nodes = self.dcBranchFromNodeIdx()
        #            br_to_nodes = self.dcBranchToNodeIdx()
        #        else:
        #            raise Exception('Branch type must be "ac" or "dc"')
        #
        #
        #        br_from_area = [self.node.area[i] for i in br_from_nodes]
        #        br_to_area = [self.node.area[i] for i in br_to_nodes]
        #
        #        # indices of all inter-area branches (from area != to area)
        #        br_is_interarea = [i for i in range(len(br_from_area))
        #                                if br_from_area[i] != br_to_area[i]]
        #
        #        # branches connected to area_from
        #        fromArea_branches_pos = [i for i in br_is_interarea
        #                                 if br_from_area[i]==area_from]
        #        fromArea_branches_neg = [i for i in br_is_interarea
        #                                 if br_to_area[i]==area_from]
        #
        #        # branches connected to area_to
        #        toArea_branches_pos = [i for i in br_is_interarea
        #                                 if br_to_area[i]==area_to]
        #        toArea_branches_neg = [i for i in br_is_interarea
        #                                 if br_from_area[i]==area_to]
        #
        #        if area_from is None:
        #            # Only to node has been specified
        #            branches_pos = toArea_branches_pos
        #            branches_neg = toArea_branches_neg
        #        elif area_to is None:
        #            # Only from node has been specified
        #            branches_pos = fromArea_branches_pos
        #            branches_neg = fromArea_branches_neg
        #        else:
        #            # Both to and from area has been specified
        #            branches_pos = [b for b in fromArea_branches_pos
        #                                    if b in toArea_branches_neg ]
        #            branches_neg = [b for b in fromArea_branches_neg
        #                                    if b in toArea_branches_pos ]
        return dict(branches_pos=branches_pos, branches_neg=branches_neg)

    def branchDistances(self, R=6373.0):
        """computes branch distance from node coordinates, resuls in km

        Uses haversine formula

        Parameters
        ----------
        R : radius of the Earth
        """

        # approximate radius of earth in km
        n_from = self.branchFromNodeIdx()
        n_to = self.branchToNodeIdx()
        distance = []
        # get endpoint coordinates and convert to radians
        lats1 = self.node["lat"][n_from].apply(math.radians)
        lons1 = self.node["lon"][n_from].apply(math.radians)
        lats2 = self.node["lat"][n_to].apply(math.radians)
        lons2 = self.node["lon"][n_to].apply(math.radians)
        lats1.index = self.branch.index
        lons1.index = self.branch.index
        lats2.index = self.branch.index
        lons2.index = self.branch.index

        for b in self.branch.index:
            lat1 = lats1[b]
            lon1 = lons1[b]
            lat2 = lats2[b]
            lon2 = lons2[b]

            dlon = lon2 - lon1
            dlat = lat2 - lat1

            a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            # atan2 better than asin: c = 2 * math.asin(math.sqrt(a))
            distance.append(R * c)
        return distance

    def spreadNodeCoordinates(self, radius=0.01, inplace=False):
        """Spread nodes with identical coordinates in a small circle
        with radius r

        Parameters
        ----------
        radius : float
            radius in degrees for the size of the spread
        inplace : boolean
            if true, update GridData object

        Returns
        -------
        coords : array
            lat,lon pandas array for nodes
        """
        coords = self.node[["lat", "lon"]]
        dupl_coords = pd.DataFrame()
        dupl_coords["cumcount"] = coords.groupby(["lat", "lon"]).cumcount()
        dupl_coords["count"] = coords.groupby(["lat", "lon"])["lon"].transform("count")
        coords_new = coords.copy()
        for i, c in coords.iterrows():
            n_sum = dupl_coords.loc[i, "count"]
            if n_sum > 1:
                # there are more nodes with the same coordinates
                n = dupl_coords.loc[i, "cumcount"]
                theta = 2 * math.pi / n_sum
                coords_new.loc[i, "lat"] += radius * math.cos(n * theta)
                coords_new.loc[i, "lon"] += radius * math.sin(n * theta)
        if inplace:
            self.node[["lat", "lon"]] = coords_new
        return coords_new
