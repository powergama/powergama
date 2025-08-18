"""
Integration test: IEEE 9 bus system
"""

from pathlib import Path

import powergama

datapath = Path(__file__).parent / "test_data/data_9bus"


def test_input():
    # single files - check that no errors are raised
    data = powergama.GridData()
    data.readGridData(
        nodes=datapath / "9busmod_nodes.csv",
        ac_branches=datapath / "9busmod_branches.csv",
        dc_branches=None,
        generators=datapath / "9busmod_generators.csv",
        consumers=datapath / "9busmod_consumers.csv",
    )

    # reading (and combining) tow files that specify nodes - check that no errors are raised
    data = powergama.GridData()
    data.readGridData(
        nodes=[datapath / "9busmod_nodes.csv", datapath / "9busmod_nodes.csv"],
        ac_branches=datapath / "9busmod_branches.csv",
        dc_branches=None,
        generators=datapath / "9busmod_generators.csv",
        consumers=datapath / "9busmod_consumers.csv",
    )

    # reading partial data. Some set to None - check that no errors are raised
    data = powergama.GridData()
    data.readGridData(
        nodes=[datapath / "9busmod_nodes.csv", datapath / "9busmod_nodes.csv"],
        ac_branches=datapath / "9busmod_branches.csv",
        dc_branches=None,
        generators=None,
        consumers=None,
    )
