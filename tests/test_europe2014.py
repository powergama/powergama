import time
from pathlib import Path

import powergama
import powergama.plots
import powergama.scenarios

datapath = Path(__file__).parent / "test_data/data_europe2014/"


def test_integration_europe2014():
    timerange = range(0, 6)

    data = powergama.GridData()
    solver = "glpk"

    resultpath = ""
    rerun = True
    sqlfile = "example_europe2014.sqlite3"

    data.readGridData(
        nodes=datapath / "2014_nodes.csv",
        ac_branches=datapath / "2014_branches.csv",
        dc_branches=datapath / "2014_hvdc.csv",
        generators=datapath / "2014_generators.csv",
        consumers=datapath / "2014_consumers.csv",
    )
    data.readProfileData(
        filename=datapath / "profiles.csv",
        storagevalue_filling=datapath / "profiles_storval_filling.csv",
        storagevalue_time=datapath / "profiles_storval_time.csv",
        timerange=timerange,
        timedelta=1.0,
    )

    lp = powergama.LpProblem(data)
    if rerun:
        res = powergama.Results(data, resultpath + sqlfile, replace=True)
        start_time = time.time()
        lp.solve(res, solver=solver)
        end_time = time.time()
        print("\nExecution time = " + str(end_time - start_time) + "seconds")
    else:
        res = powergama.Results(data, resultpath + sqlfile, replace=False)

    # Total system cost
    # this does not seem entirely stable, so skip it for now
    # assert sum(res.getSystemCost(timeMaxMin=[4, 6]).values()) == pytest.approx(
    #    9984938.213366672
    #    # 9959497.753850404 # sometimes get this value instead
    # )

    # Average area price
    # this does not seem entirely stable, so skip it for now
    # assert sum(res.getAreaPricesAverage(timeMaxMin=[4, 6]).values()) / len(
    #    res.getAreaPricesAverage()
    # ) == pytest.approx(58.267524372667154)
    # # 58.27706628551738 # sometimes get this value instead

    # TODO: Add checks on results

    # SOME PLOTS:
    powergama.plots.plotMap(data, res, nodetype="nodalprice", branchtype="utilisation")
