"""Reading a run's dispatch as wide frames: Results.get_dispatch_frames and is_complete.

These are what a second run (such as the balancing run) reads from a first one, so the
frames must be exactly the stored values, on the stored timesteps, with a column for every
element of each kind.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import powergama

datapath = Path(__file__).parent / "test_data/two_area"

FRAME_SOURCES = {
    "generator_output": ("Res_Generators", "output"),
    "pumping": ("Res_Pumping", "output"),
    "flexload_demand": ("Res_FlexibleLoad", "demand"),
    "storage": ("Res_Storage", "storage"),
    "branch_flow": ("Res_Branches", "flow"),
    "dcbranch_flow": ("Res_DcBranches", "flow"),
}


def _two_area(timerange):
    data = powergama.GridData()
    data.readGridData(
        nodes=datapath / "nodes.csv",
        ac_branches=datapath / "branches.csv",
        dc_branches=datapath / "dcbranches.csv",
        generators=datapath / "generators.csv",
        consumers=datapath / "consumers.csv",
        inter_area_rules=datapath / "inter_area_rules.csv",
    )
    data.readProfileData(
        filename=datapath / "profiles.csv",
        storagevalue_filling=datapath / "profiles_storval_filling.csv",
        storagevalue_time=datapath / "profiles_storval_time.csv",
        timerange=timerange,
        timedelta=1.0,
    )
    return data


def _solve(data, dbfile):
    lp = powergama.LpProblem(data)
    res = powergama.Results(data, dbfile, replace=True)
    lp.solve(res, solver="appsi_highs", solve_args={})
    return res


@pytest.fixture(scope="module")
def res_0_24(tmp_path_factory):
    return _solve(_two_area(range(0, 24)), tmp_path_factory.mktemp("frames") / "res.sqlite")


@pytest.fixture(scope="module")
def res_24_48(tmp_path_factory):
    return _solve(_two_area(range(24, 48)), tmp_path_factory.mktemp("frames") / "res_offset.sqlite")


def _sql_pivot(res, table, column):
    with sqlite3.connect(res.db.filename) as con:
        df = pd.read_sql_query(f"SELECT timestep, indx, {column} FROM {table}", con)  # nosec B608
    return df.pivot(index="timestep", columns="indx", values=column)


def _expected_columns(grid):
    return {
        "generator_output": list(grid.generator.index),
        "pumping": grid.getIdxGeneratorsWithPumping(),
        "flexload_demand": grid.getIdxConsumersWithFlexibleLoad(),
        "storage": grid.getIdxGeneratorsWithStorage(),
        "branch_flow": list(grid.branch.index),
        "dcbranch_flow": list(grid.dcbranch.index),
    }


def test_frames_equal_raw_sql(res_0_24):
    frames = res_0_24.get_dispatch_frames()
    assert set(frames) == set(FRAME_SOURCES)
    for key, (table, column) in FRAME_SOURCES.items():
        expected = _sql_pivot(res_0_24, table, column)
        got = frames[key]
        assert list(got.index) == list(range(24)), key
        # bit for bit: the same stored doubles, no fill, no arithmetic
        pd.testing.assert_frame_equal(got, expected.reindex(index=got.index, columns=got.columns), check_names=False)
        assert np.array_equal(got.to_numpy(), expected.loc[got.index, got.columns].to_numpy()), key


def test_frames_cover_every_element(res_0_24):
    frames = res_0_24.get_dispatch_frames()
    expected = _expected_columns(res_0_24.grid)
    # two_area has one of each kind, so none of these is empty
    for key, cols in expected.items():
        assert list(frames[key].columns) == cols, key
        assert len(cols) > 0, key
        assert frames[key].dtypes.eq(np.float64).all(), key


def _copy_db(res, path):
    with sqlite3.connect(res.db.filename) as src, sqlite3.connect(path) as dst:
        src.backup(dst)
    return path


def test_missing_values_stay_nan(res_0_24, tmp_path):
    """A value that is not stored is NaN, not zero, and its element keeps its column."""
    dbfile = _copy_db(res_0_24, tmp_path / "gap.sqlite")
    with sqlite3.connect(dbfile) as con:
        con.execute("DELETE FROM Res_Generators WHERE indx=1 AND timestep=5")
        con.execute("DELETE FROM Res_DcBranches WHERE indx=0")
    res = powergama.Results.from_existing(dbfile, timedelta=1.0)
    frames = res.get_dispatch_frames()
    gen = frames["generator_output"]
    assert np.isnan(gen.loc[5, 1])
    assert gen.drop(index=5).notna().all().all()
    assert list(gen.columns) == list(res.grid.generator.index)
    dc = frames["dcbranch_flow"]
    assert list(dc.columns) == [0]
    assert list(dc.index) == list(range(24))
    assert dc[0].isna().all()


def test_timemaxmin_window(res_0_24):
    frames = res_0_24.get_dispatch_frames(timeMaxMin=[6, 10])
    for frame in frames.values():
        assert list(frame.index) == [6, 7, 8, 9]


def test_offset_timerange(res_24_48):
    """A run on rows 24..47 of the profile file stores timesteps 24..47."""
    frames = res_24_48.get_dispatch_frames()
    for key, (table, column) in FRAME_SOURCES.items():
        assert list(frames[key].index) == list(range(24, 48)), key
        expected = _sql_pivot(res_24_48, table, column)
        assert np.array_equal(frames[key].to_numpy(), expected.loc[frames[key].index, frames[key].columns].to_numpy())
    assert res_24_48.is_complete()
    assert res_24_48.is_complete(range(24, 48))
    assert not res_24_48.is_complete(range(0, 24))


def test_is_complete_on_truncated_database(res_0_24, tmp_path):
    assert res_0_24.is_complete()
    assert res_0_24.is_complete(range(0, 24))
    assert not res_0_24.is_complete(range(0, 25))
    # a copy of the database that lost its last timesteps, as an interrupted run leaves it
    truncated = _copy_db(res_0_24, tmp_path / "truncated.sqlite")
    with sqlite3.connect(truncated) as con:
        con.execute("DELETE FROM Res_ObjFunc WHERE timestep >= 20")
    res = powergama.Results.from_existing(truncated, timedelta=1.0)
    assert not res.is_complete()
    assert res.is_complete(range(0, 20))


def test_from_existing(res_0_24):
    reopened = powergama.Results.from_existing(res_0_24.db.filename, timedelta=1.0)
    assert reopened.is_complete()
    frames = reopened.get_dispatch_frames()
    original = res_0_24.get_dispatch_frames()
    for key in FRAME_SOURCES:
        pd.testing.assert_frame_equal(frames[key], original[key], check_names=False)


def test_from_existing_offset_timerange(res_24_48):
    """The database stores the cut profiles, so the reopened run needs its time range."""
    reopened = powergama.Results.from_existing(res_24_48.db.filename, timedelta=1.0, timerange=range(24, 48))
    assert reopened.timerange == list(range(24, 48))
    assert reopened.is_complete()
    frames = reopened.get_dispatch_frames()
    assert list(frames["generator_output"].index) == list(range(24, 48))
    # without it, the reopened run assumes rows 0..23 and is not complete
    assert not powergama.Results.from_existing(res_24_48.db.filename, timedelta=1.0).is_complete()


def test_result_frame_whitelist(res_0_24):
    frame = res_0_24.db.getResultFrame("Res_Nodes", "nodalprice", [0, 24])
    assert frame.shape == (24, len(res_0_24.grid.node))
    with pytest.raises(ValueError, match="not a result"):
        res_0_24.db.getResultFrame("Res_Generators", "output; DROP TABLE Res_ObjFunc", [0, 24])
    with pytest.raises(ValueError, match="not a result"):
        res_0_24.db.getResultFrame("data_generator", "pmax", [0, 24])
