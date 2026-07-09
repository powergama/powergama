"""
Integration test: IEEE 9 bus system
"""

from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

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


def test_input_parquet(tmp_path):
    pytest.importorskip("pyarrow", reason="Parquet test requires pyarrow")

    nodes_csv = datapath / "9busmod_nodes.csv"
    branches_csv = datapath / "9busmod_branches.csv"
    generators_csv = datapath / "9busmod_generators.csv"
    consumers_csv = datapath / "9busmod_consumers.csv"
    profiles_csv = datapath / "9busmod_profiles.csv"

    nodes_pq = tmp_path / "nodes.parquet"
    branches_pq = tmp_path / "branches.parquet"
    generators_pq = tmp_path / "generators.parquet"
    consumers_pq = tmp_path / "consumers.parquet"
    profiles_pq = tmp_path / "profiles.parquet"

    # Convert reference CSV files to parquet and ensure GridData reads them transparently.
    pd.read_csv(nodes_csv).to_parquet(nodes_pq, index=False)
    pd.read_csv(branches_csv).to_parquet(branches_pq, index=False)
    pd.read_csv(generators_csv).to_parquet(generators_pq, index=False)
    pd.read_csv(consumers_csv).to_parquet(consumers_pq, index=False)
    pd.read_csv(profiles_csv).to_parquet(profiles_pq, index=False)

    data = powergama.GridData()
    data.readGridData(
        nodes=nodes_pq,
        ac_branches=branches_pq,
        dc_branches=None,
        generators=generators_pq,
        consumers=consumers_pq,
    )
    data.readProfileData(
        filename=profiles_pq,
        timerange=range(24),
    )

    assert data.numNodes() > 0
    assert data.numBranches() > 0
    assert data.numGenerators() > 0
    assert data.numConsumers() > 0
    assert len(data.profiles) == 24
