import powergama


def test_grid_data_getters(testcase_9bus_data: powergama.GridData):
    """
    Test functions for retrieving grid data info

    Parameters
    ----------
    data : powergama.GridData object
        object holding grid model

    """
    data = testcase_9bus_data

    # check if methods execute without error
    data.compute_power_flow_matrices()
    data.getAllAreas()
    data.getAllGeneratorTypes()
    data.getBranchAreas()
    data.getConsumerAreas()
    data.getConsumersPerArea()
    data.getDcBranchAreas()
    data.getDcBranchesAtNode(0, "from")
    data.getDcBranches()
    data.getFlexibleLoadStorageCapacity(0)
    data.getGeneratorAreas()
    data.getGeneratorAvailablePower(0, timestep=1)
    data.getGeneratorsAtNode(0)
    data.getGeneratorsPerAreaAndType()
    data.getGeneratorsWithPumpAtNode(0)
    data.getGeneratorsPerType()
    data.getGeneratorsWithPumpByArea()
    data.getIdxBranchesWithFlowConstraints()
    data.getIdxConsumersWithFlexibleLoad()
    data.getIdxDcBranchesWithFlowConstraints()
    data.getIdxGeneratorsWithNonzeroInflow()
    data.getIdxGeneratorsWithPumping()
    data.getIdxGeneratorsWithStorage()
    data.getIdxNodesWithLoad()
    data.getInterAreaBranches(area_from="AREA")
    data.getLoadsAtNode(0)
    data.getLoadsFlexibleAtNode(0)


def test_result_getters(testcase_9bus_data, testcase_9bus_res):
    """
    Test functions for retrieving results

    Parameters
    ----------
    data : powergama.GridData object
        object holding grid model
    res : powergama.Results object
        object holding simulation results
    """
    data = testcase_9bus_data
    res = testcase_9bus_res

    area = data.getAllAreas()[0]
    gentype = data.getAllGeneratorTypes(sort="fuelcost")[0]

    # check if methods execute without error
    res.getAreaPrices(area)
    res.getAreaPricesAverage()
    res.getAverageBranchFlows()
    res.getAverageBranchSensitivity()
    res.getAverageEnergyBalance()
    res.getAverageImportExport(area)
    res.getAverageInterareaBranchFlow()
    res.getAverageNodalPrices()
    res.getAverageUtilisation()
    res.getDemandPerArea(area)
    res.getEnergyBalanceInArea(area, spillageGen=[gentype])
    res.getGeneratorOutputSumPerArea()
    res.getGeneratorSpilled(0)
    res.getGeneratorSpilledSums()
    res.getGeneratorStorageAll(res.timerange[0])
    res.getGeneratorStorageValues(res.timerange[0])
    res.getLoadheddingInArea(area)
    res.getLoadheddingSums()
    res.getLoadsheddingPerNode()
    res.getNetImport(area)
    res.getNodalPrices(0)
    res.getStorageFillingInAreas([area], gentype)
    res.getSystemCost()
