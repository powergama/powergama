"""
Integration test: IEEE 9 bus system
"""

from pathlib import Path

import pandas.testing as pdt

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


def test_via_database(testcase_9bus_data, tmp_path):
    data = testcase_9bus_data

    # this is to create database file
    powergama.Results(data, databasefile=tmp_path / "test_sql_io.sqlite")

    # read back from database file
    res2 = powergama.Results.from_existing(databasefile=tmp_path / "test_sql_io.sqlite", timedelta=1.0)
    dat2 = res2.grid

    # Check that they are the same
    pdt.assert_frame_equal(data.node, dat2.node)
    pdt.assert_frame_equal(data.branch, dat2.branch)
    # pdt.assert_frame_equal(data.dcbranch, dat2.dcbranch) # empty, so no point comparing
    pdt.assert_frame_equal(data.generator, dat2.generator)
    pdt.assert_frame_equal(data.consumer, dat2.consumer)
    pdt.assert_frame_equal(data.profiles, dat2.profiles)
