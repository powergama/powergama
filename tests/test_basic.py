import matplotlib.pyplot as plt

import powergama.plots


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


def test_plots(testcase_9bus_data, testcase_9bus_res):
    """
    Test PowerGAMA plotting functions

    Parameters
    ----------
    data : powergama.GridData object
        object holding grid model
    res : powergama.Results object
        object holding simulation results
    """
    data = testcase_9bus_data
    res = testcase_9bus_res

    plt.switch_backend("Agg")

    area = data.getAllAreas()[0]

    # Try plots. Success if no errors are raised
    powergama.plots.plotMap(data, res, nodetype="nodeprice", branchtype="utilisation")

    res.plotNodalPrice(0)
    res.plotAreaPrice([area])
    indices = data.getIdxGeneratorsWithStorage()
    if indices:
        res.plotStorageFilling(indices[0])
        res.plotStorageValues(indices[0])
    res.plotGeneratorOutput(0)
    res.plotDemandAtLoad(0)
    res.plotStoragePerArea(area)
    res.plotGenerationPerArea(area)
    res.plotDemandPerArea([area])
    indices = data.getIdxConsumersWithFlexibleLoad()
    if indices:
        res.plotFlexibleLoadStorageValues(indices[0])
    res.plotEnergyMix([area])
    res.plotTimeseriesColour(areas=[area], value="nodalprice")

    # skip these - outdated
    # res.plotGenerationScatter(area)

    # res.plotMapGrid(nodetype='nodalprice',branchtype='sensitivity',
    #                dotsize=40,show_node_labels=False,filter_branch=[0,1])
    # res.plotRelativeLoadDistribution()
    # res.plotRelativeGenerationCapacity(tech=data.getAllGeneratorTypes()[0])
