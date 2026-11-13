"""Grid data through the results database and back.

``Results`` writes the grid tables into the database and ``Results.from_existing`` (through
``Database.get_grid_data`` and ``GridData.from_dict``) rebuilds a ``GridData`` from them.
That grid is what a re-run from a database starts from, so it must be the grid that was
written: the inter-area rules included, a table with a column named ``index`` intact, and no
columns that the run itself added.
"""

import sqlite3

import numpy as np
import pandas as pd
import pytest
from conftest import _load_9bus
from rt_case import datapath_two_area, two_area_data

import powergama


def _reopen(data, dbfile):
    powergama.Results(data, dbfile, replace=True)
    return powergama.Results.from_existing(dbfile, timedelta=data.timeDelta)


def _solve(data, dbfile):
    lp = powergama.LpProblem(data)
    res = powergama.Results(data, dbfile, replace=True)
    lp.solve(res, solver="appsi_highs", solve_args={})
    return res


def _objective(res):
    with sqlite3.connect(res.db.filename) as con:
        return pd.read_sql("SELECT * FROM Res_ObjFunc ORDER BY timestep", con)["value"].to_numpy()


# --- 1. inter_area_rules survives from_dict / from_existing -----------------------------


def test_from_existing_keeps_inter_area_rules(tmp_path):
    data = two_area_data()
    grid = _reopen(data, tmp_path / "two_area.sqlite").grid
    assert grid.inter_area_rules is not None
    pd.testing.assert_frame_equal(grid.inter_area_rules, data.inter_area_rules)


def test_from_dict_without_inter_area_rules_key():
    data = two_area_data()
    grid = powergama.GridData()
    grid.from_dict(
        {
            "node": data.node,
            "branch": data.branch,
            "dcbranch": data.dcbranch,
            "generator": data.generator,
            "consumer": data.consumer,
            "profiles": data.profiles,
            "storval_time": None,
            "storval_filling": None,
        },
        timedelta=1.0,
    )
    assert grid.inter_area_rules is None


def test_rerun_from_database_has_the_inter_area_constraints(tmp_path):
    """A run on the grid rebuilt from the database is the original run, NTC included.

    Before the fix the rebuilt grid had no inter-area rules, so the re-run left the border
    unconstrained and was cheaper than the run that wrote the database.
    """
    data = two_area_data()
    original = _solve(data, tmp_path / "original.sqlite")
    grid = powergama.Results.from_existing(original.db.filename, timedelta=1.0).grid
    lp = powergama.LpProblem(grid)
    assert len(lp._inter_area_rules_rows) == 1
    assert len(lp.model.cInterAreaRulesForward) > 0
    rerun = powergama.Results(grid, tmp_path / "rerun.sqlite", replace=True)
    lp.solve(rerun, solver="appsi_highs", solve_args={})
    np.testing.assert_allclose(_objective(rerun), _objective(original), rtol=1e-9, atol=1e-6)


# --- 2. a table with a column named "index" -------------------------------------------


def _9bus_with_index_column():
    data = _load_9bus()
    # Like some real node tables: an "index" column that is not the row index and is
    # not unique, so it cannot pass for one.
    data.node.insert(0, "index", [7, 7, 3, 3, 1, 1, 0, 0, 2][: len(data.node)])
    data.node.index = data.node.index + 100
    return data


def test_table_with_index_column_round_trips(tmp_path):
    data = _9bus_with_index_column()
    grid = _reopen(data, tmp_path / "index_col.sqlite").grid
    pd.testing.assert_frame_equal(grid.node, data.node)
    for name in ("branch", "generator", "consumer"):
        pd.testing.assert_frame_equal(getattr(grid, name), getattr(data, name))
    pd.testing.assert_frame_equal(grid.profiles, data.profiles)


def test_table_with_index_column_from_old_database(tmp_path):
    """Databases written before the explicit index label read back the same way."""
    data = _9bus_with_index_column()
    dbfile = tmp_path / "old.sqlite"
    powergama.Results(data, dbfile, replace=True)
    with sqlite3.connect(dbfile) as con:
        # what older versions wrote: pandas' own label for the row index (level_0 here)
        data.node.to_sql("data_node", con, if_exists="replace", index=True)
        assert {"index", "level_0"} <= set(pd.read_sql("SELECT * FROM data_node", con).columns)
    stored = powergama.database.Database(dbfile).get_grid_data()["node"]
    pd.testing.assert_frame_equal(stored, data.node, check_dtype=False)


def test_level_0_column_is_rejected(tmp_path):
    data = _load_9bus()
    data.node["level_0"] = 0
    with pytest.raises(ValueError, match="level_0"):
        powergama.Results(data, tmp_path / "level_0.sqlite", replace=True)


def test_9bus_round_trips_with_dtypes(tmp_path):
    """All-NaN float columns (gen_lat, ramp_up_pu, ...) come back float, not object."""
    data = _load_9bus()
    assert data.generator["gen_lat"].isna().all()
    grid = _reopen(data, tmp_path / "9bus.sqlite").grid
    for name in ("node", "branch", "generator", "consumer"):  # dcbranch is empty here
        pd.testing.assert_frame_equal(getattr(grid, name), getattr(data, name))


# --- 3. getDcBranchAreas has no side effect ---------------------------------------------


def test_get_dc_branch_areas_leaves_dcbranch_alone():
    data = two_area_data()
    before = data.dcbranch.copy()
    areas = data.getDcBranchAreas()
    pd.testing.assert_frame_equal(data.dcbranch, before)
    assert list(areas["area_from"]) == ["AREA_A"]
    assert list(areas["area_to"]) == ["AREA_B"]
    assert areas.index.equals(data.dcbranch.index)


def test_get_dc_branch_areas_on_a_non_default_index():
    data = two_area_data()
    data.dcbranch.index = [5]
    assert data.getInterAreaBranches(area_from="AREA_A", area_to="AREA_B", acdc="dc") == {
        "branches_pos": [5],
        "branches_neg": [],
    }


def test_database_dcbranch_has_no_area_columns(tmp_path):
    data = two_area_data()
    lp = powergama.LpProblem(data)  # builds the inter-area rules, which call getDcBranchAreas
    assert lp._inter_area_rules_rows[0]["dc_pos"] == [0]
    assert "area_from" not in data.dcbranch.columns
    dbfile = tmp_path / "two_area.sqlite"
    powergama.Results(data, dbfile, replace=True)
    with sqlite3.connect(dbfile) as con:
        stored = pd.read_sql("SELECT * FROM data_dcbranch", con)
    assert not {"area_from", "area_to"} & set(stored.columns)


# --- 4. block_group ---------------------------------------------------------------------


GENERATORS_CSV = datapath_two_area / "generators.csv"


def _read_two_area_grid(generators):
    data = powergama.GridData()
    data.readGridData(
        nodes=datapath_two_area / "nodes.csv",
        ac_branches=datapath_two_area / "branches.csv",
        dc_branches=datapath_two_area / "dcbranches.csv",
        generators=generators,
        consumers=datapath_two_area / "consumers.csv",
        inter_area_rules=datapath_two_area / "inter_area_rules.csv",
    )
    return data


def test_block_group_default_is_empty_string():
    data = two_area_data()
    assert "block_group" not in pd.read_csv(GENERATORS_CSV).columns
    assert list(data.generator["block_group"]) == [""] * len(data.generator)


def _two_area_with_blocks(tmp_path):
    """two_area with gas_a split into two price blocks of one group (rows 0 and 4), from CSV."""
    table = pd.read_csv(GENERATORS_CSV)
    table["block_group"] = ["gas_a", "", "", "7"]
    block = table.iloc[[0]].assign(desc="gas_a_b2", pmax=100, fuelcost=25)
    table = pd.concat([table, block], ignore_index=True)
    path = tmp_path / "generators.csv"
    table.to_csv(path, index=False)
    data = _read_two_area_grid(path)
    data.readProfileData(
        filename=datapath_two_area / "profiles.csv",
        storagevalue_filling=datapath_two_area / "profiles_storval_filling.csv",
        storagevalue_time=datapath_two_area / "profiles_storval_time.csv",
        timerange=range(0, 24),
        timedelta=1.0,
    )
    return data


def test_block_group_read_as_text(tmp_path):
    data = _two_area_with_blocks(tmp_path)
    # "7" stays the text "7", an empty cell is ""
    assert list(data.generator["block_group"]) == ["gas_a", "", "", "7", "gas_a"]
    assert data.generator["block_group"].map(type).eq(str).all()


def test_block_group_survives_database(tmp_path):
    data = _two_area_with_blocks(tmp_path)
    reopened = _reopen(data, tmp_path / "blocks.sqlite")
    pd.testing.assert_series_equal(reopened.grid.generator["block_group"], data.generator["block_group"])


def test_database_without_block_group_reads_default(tmp_path):
    """A database written before block_group existed reads back with the default ""."""
    data = two_area_data()
    dbfile = tmp_path / "old.sqlite"
    powergama.Results(data, dbfile, replace=True)
    with sqlite3.connect(dbfile) as con:
        data.generator.drop(columns="block_group").to_sql("data_generator", con, if_exists="replace", index=True)
    grid = powergama.Results.from_existing(dbfile, timedelta=1.0).grid
    assert list(grid.generator["block_group"]) == [""] * len(data.generator)


def test_block_group_from_parquet_keeps_missing_as_default(tmp_path):
    table = pd.read_csv(GENERATORS_CSV)
    table["block_group"] = pd.Series(["g", None, None, "g"], dtype=object)
    path = tmp_path / "generators.parquet"
    table.to_parquet(path)
    data = _read_two_area_grid(path)
    assert list(data.generator["block_group"]) == ["g", "", "", "g"]
