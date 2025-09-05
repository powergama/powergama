from pathlib import Path

import powergama
import powergama.plots as ppl

datapath = Path(__file__).parent / "test_data/data_9bus"


def test_map_plot():
    # timerange = range(24 * 100, 24 * 101)
    data = powergama.GridData()

    data.readGridData(
        nodes=datapath / "9busmod_nodes.csv",
        ac_branches=datapath / "9busmod_branches.csv",
        dc_branches=None,
        generators=datapath / "9busmod_generators.csv",
        consumers=datapath / "9busmod_consumers.csv",
    )
    # data.readProfileData(
    #    filename=datapath / "9busmod_profiles.csv",
    #    storagevalue_filling=datapath / "9busmod_profiles_storval_filling.csv",
    #    storagevalue_time=datapath / "9busmod_profiles_storval_time.csv",
    #    timerange=timerange,
    #    timedelta=1.0,
    # )

    # lp = powergama.LpProblem(data)

    ppl.plotMap(pg_data=data, pg_res=None, nodetype="area", branchtype="capacity", zoom_start=5)
