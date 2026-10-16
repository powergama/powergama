import pytest


def test_result_methods(tmp_path, testcase_9bus_res):
    """Test functions for processing results"""

    res = testcase_9bus_res

    fname = tmp_path / "production_overview.csv"
    fname = None
    res.writeProductionOverview(areas=["AREA"], types=["wind", "csp"], filename=fname)

    res.getImportExport(areas=["AREA"], timeMaxMin=None, acdc=["ac", "dc"])

    res.plotGenerationScatter(area="AREA", tech=[], dotsize=300, annotations=True)

    res.plotGenerationPerArea(
        "AREA", timeMaxMin=None, fill=True, reversed_order=False, net_import=True, loadshed=True, showTitle=True
    )

    res.plotFlexibleLoadStorageValues(0, timeMaxMin=None, showTitle=True)

    res.plotTimeseriesColour(["AREA"], value="demand", filter_values=[0, 100])
    res.plotTimeseriesColour(["AREA"], value="gen%wind%csp", filter_values=[0, 100])


@pytest.mark.xfail(reason="Needs rewriting to avoid dependence on mpl_toolkits.basemap")
def test_plot_special_maps(testcase_9bus_res):
    testcase_9bus_res.plotRelativeLoadDistribution(
        show_node_labels=False, latlon=None, dotsize=40, draw_par_mer=False, colours=True, showTitle=True
    )


@pytest.mark.xfail(reason="Needs rewriting to avoid dependence on mpl_toolkits.basemap")
def test_plot_map2(testcase_9bus_res):
    testcase_9bus_res.plotRelativeGenerationCapacity(
        tech="wind", show_node_labels=False, latlon=None, dotsize=40, draw_par_mer=False, colours=True, showTitle=True
    )


@pytest.mark.xfail(reason="Needs rewriting to avoid dependence on mpl_toolkits.basemap")
def test_plot_map3(testcase_9bus_res):
    testcase_9bus_res.plotMapGrid()
