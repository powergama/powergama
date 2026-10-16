import matplotlib.pyplot as plt

import powergama.plots as ppl
import powergama.plots2 as ppl2


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


def test_map_plot(testcase_9bus_data):
    """Plot using folium"""

    data = testcase_9bus_data
    ppl.plotMap(pg_data=data, pg_res=None, nodetype="area", branchtype="capacity", zoom_start=5)


def test_map_plot2(testcase_9bus_data):
    """Plot using geopandas"""

    data = testcase_9bus_data
    ppl2.plot_map2(pg_data=data, pg_res=None, nodetype="area", branchtype="capacity")

    plot_options = {
        "branch": {"width_col": ("capacity", 100, 500), "annotation": {"column": "capacity", "color": "blue"}},
        "dcbranch": {"width_col": ("capacity", 100, 500), "annotation": {"column": "capacity", "color": "blue"}},
    }
    ppl2.plot_map2(
        pg_data=data,
        pg_res=None,
        nodetype="area",
        branchtype="capacity",
        plot_options=plot_options,
        plot_gentypes="all",
    )
