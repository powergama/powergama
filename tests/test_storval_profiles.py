"""Storage-value time curves as ordinary profile columns.

Without ``storagevalue_time``, ``storval_time_ref`` (generators with storage) and
``flex_storval_time`` (consumers with flexible load) name profile columns; ``""``,
``"const"`` and NaN are a factor of 1. The deprecated separate table is still read when
given, and then every time reference is a column of it. The filling curves stay a
separate table of rows 0..100. Everything is checked once, when ``LpProblem`` is built,
and only for the rows the LP values.
"""

import sqlite3

import numpy as np
import pandas as pd
import pytest
from rt_case import N_STEPS, datapath_two_area

import powergama

GEN_STORAGE = 2  # hydro_pump
GEN_GAS = 0
CONS_FIXED = 0
CONS_FLEX = 1
RESULT_TABLES = [
    "Res_ObjFunc",
    "Res_Generators",
    "Res_Pumping",
    "Res_Storage",
    "Res_FlexibleLoad",
    "Res_Branches",
    "Res_DcBranches",
    "Res_Nodes",
]


def _time_curves(path):
    """Time curves that vary (the two_area ones are flat), as a separate table."""
    t = np.arange(48)
    curves = pd.DataFrame(
        {
            "const": 1,
            "hydro": 1.0 + 0.2 * np.sin(2 * np.pi * t / 24),
            "flex": 1.0 + 0.1 * np.cos(2 * np.pi * t / 12),
        }
    )
    curves.to_csv(path, index=False)
    return path


def _grid(data):
    data.readGridData(
        nodes=datapath_two_area / "nodes.csv",
        ac_branches=datapath_two_area / "branches.csv",
        dc_branches=datapath_two_area / "dcbranches.csv",
        generators=datapath_two_area / "generators.csv",
        consumers=datapath_two_area / "consumers.csv",
        inter_area_rules=datapath_two_area / "inter_area_rules.csv",
    )
    return data


def _legacy(tmp_path, gen_ref="hydro", flex_ref="flex"):
    data = _grid(powergama.GridData())
    with pytest.warns(DeprecationWarning, match="storagevalue_time"):
        data.readProfileData(
            filename=datapath_two_area / "profiles.csv",
            storagevalue_filling=datapath_two_area / "profiles_storval_filling.csv",
            storagevalue_time=_time_curves(tmp_path / "storval_time.csv"),
            timerange=range(0, N_STEPS),
            timedelta=1.0,
        )
    data.generator.loc[GEN_STORAGE, "storval_time_ref"] = gen_ref
    data.consumer.loc[CONS_FLEX, "flex_storval_time"] = flex_ref
    return data


def _profiles(tmp_path, gen_ref="svt_hydro", flex_ref="svt_flex"):
    """The same curves as profile columns, renamed (as unique profile column names need)."""
    curves = pd.read_csv(_time_curves(tmp_path / "storval_time.csv")).drop(columns="const")
    curves = curves.rename(columns=lambda c: f"svt_{c}")
    path = tmp_path / "profiles_storval_time.csv"
    curves.to_csv(path, index=False)
    data = _grid(powergama.GridData())
    data.readProfileData(
        filename=[datapath_two_area / "profiles.csv", path],
        storagevalue_filling=datapath_two_area / "profiles_storval_filling.csv",
        timerange=range(0, N_STEPS),
        timedelta=1.0,
    )
    data.generator.loc[GEN_STORAGE, "storval_time_ref"] = gen_ref
    data.consumer.loc[CONS_FLEX, "flex_storval_time"] = flex_ref
    return data


def _solve(data, dbfile, **kwargs):
    lp = powergama.LpProblem(data, **kwargs)
    res = powergama.Results(data, dbfile, replace=True)
    lp.solve(res, solver="appsi_highs", solve_args={})
    return res


def _read_all(res, table):
    order = "timestep" if table == "Res_ObjFunc" else "timestep, indx"
    with sqlite3.connect(res.db.filename) as con:
        return pd.read_sql_query(f"SELECT * FROM {table} ORDER BY {order}", con)  # nosec B608


def _assert_same_results(a, b):
    for table in RESULT_TABLES:
        pd.testing.assert_frame_equal(_read_all(a, table), _read_all(b, table), check_exact=True, obj=table)


# --- the profile path gives the legacy results bit for bit --------------------------------


@pytest.mark.parametrize(
    "mode",
    [{}, {"objective_mode": "multi_period", "horizon_hours": 24, "commit_hours": 24}],
    ids=["single_period", "multi_period"],
)
def test_profile_columns_equal_legacy_table(tmp_path, mode):
    legacy = _solve(_legacy(tmp_path), tmp_path / "legacy.sqlite", **mode)
    profiles = _solve(_profiles(tmp_path), tmp_path / "profiles.sqlite", **mode)
    _assert_same_results(legacy, profiles)
    # the curves matter: flat curves give another result
    flat = _solve(_legacy(tmp_path, "const", "const"), tmp_path / "flat.sqlite", **mode)
    assert not _read_all(flat, "Res_ObjFunc").equals(_read_all(legacy, "Res_ObjFunc"))


@pytest.mark.parametrize("ref", ["const", "", np.nan])
def test_no_profile_is_a_factor_of_one(tmp_path, ref):
    """``"const"``, ``""`` and NaN: as the legacy table's all-ones ``const`` column."""
    legacy = _solve(_legacy(tmp_path, "const", "const"), tmp_path / "legacy.sqlite")
    data = _profiles(tmp_path, ref, ref)
    lp = powergama.LpProblem(data)
    assert lp._ctx.storval_time_pos[GEN_STORAGE] == -1
    assert lp._ctx.storval_flex_time_pos[CONS_FLEX] == -1
    res = powergama.Results(data, tmp_path / "profiles.sqlite", replace=True)
    lp.solve(res, solver="appsi_highs", solve_args={})
    _assert_same_results(legacy, res)


def test_legacy_mode_is_not_mixed(tmp_path):
    """With the separate table, a time reference is a column of it, not a profile column."""
    data = _legacy(tmp_path, gen_ref="wind")  # a profile column, not a time-table column
    with pytest.raises(ValueError, match="storval_time_ref 'wind' is not a column of storagevalue_time"):
        powergama.LpProblem(data)


# --- what raises, and what does not ------------------------------------------------------


def test_unknown_time_ref_on_storage_row_raises(tmp_path):
    data = _profiles(tmp_path, gen_ref="no_such_curve")
    with pytest.raises(ValueError, match="generator storval_time_ref 'no_such_curve' is not a column of the profiles"):
        powergama.LpProblem(data)
    data = _profiles(tmp_path, flex_ref="no_such_flex_curve")
    with pytest.raises(ValueError, match="consumer flex_storval_time 'no_such_flex_curve'"):
        powergama.LpProblem(data)


@pytest.mark.parametrize("ref", ["no_such_filling", "", np.nan])
def test_missing_filling_ref_raises(tmp_path, ref):
    data = _profiles(tmp_path)
    data.generator.loc[GEN_STORAGE, "storval_filling_ref"] = ref
    with pytest.raises(ValueError, match="storval_filling_ref '.*' is not a column of storagevalue_filling"):
        powergama.LpProblem(data)
    data = _profiles(tmp_path)
    data.consumer.loc[CONS_FLEX, "flex_storval_filling"] = ref
    with pytest.raises(ValueError, match="flex_storval_filling '.*' is not a column of storagevalue_filling"):
        powergama.LpProblem(data)


@pytest.mark.parametrize("rows", [100, 102])
def test_filling_table_not_101_rows_raises(tmp_path, rows):
    data = _profiles(tmp_path)
    filling = data.storagevalue_filling
    data.storagevalue_filling = filling.reindex(range(rows)).ffill()
    with pytest.raises(ValueError, match="rows 0..100"):
        powergama.LpProblem(data)


def test_filling_table_with_other_index_raises(tmp_path):
    data = _profiles(tmp_path)
    data.storagevalue_filling.index = data.storagevalue_filling.index + 1
    with pytest.raises(ValueError, match="rows 0..100"):
        powergama.LpProblem(data)


def test_no_filling_table_raises(tmp_path):
    data = _profiles(tmp_path)
    data.storagevalue_filling = None
    with pytest.raises(ValueError, match="no storage-value filling curves"):
        powergama.LpProblem(data)


def test_rows_the_lp_does_not_value_are_not_checked(tmp_path):
    data = _profiles(tmp_path)
    data.generator.loc[GEN_GAS, ["storval_time_ref", "storval_filling_ref"]] = "nonsense"
    data.consumer.loc[CONS_FIXED, ["flex_storval_time", "flex_storval_filling"]] = "nonsense"
    reference = _solve(_profiles(tmp_path), tmp_path / "reference.sqlite")
    res = _solve(data, tmp_path / "nonsense.sqlite")
    _assert_same_results(reference, res)


# --- the database --------------------------------------------------------------------


def test_database_round_trip_without_time_table(tmp_path):
    data = _profiles(tmp_path)
    original = _solve(data, tmp_path / "original.sqlite")
    with sqlite3.connect(original.db.filename) as con:
        tables = set(pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", con)["name"])
    assert "data_storval_filling" in tables
    assert "data_storval_time" not in tables
    grid = powergama.Results.from_existing(original.db.filename, timedelta=1.0, timerange=data.timerange).grid
    assert grid.storagevalue_time is None
    pd.testing.assert_frame_equal(
        grid.storagevalue_filling, data.storagevalue_filling, check_dtype=False, check_names=False
    )
    assert {"svt_hydro", "svt_flex"} <= set(grid.profiles.columns)
    rerun = _solve(grid, tmp_path / "rerun.sqlite")
    _assert_same_results(original, rerun)


def test_database_round_trip_legacy_keeps_time_table(tmp_path):
    data = _legacy(tmp_path)
    res = powergama.Results(data, tmp_path / "legacy.sqlite", replace=True)
    grid = powergama.Results.from_existing(res.db.filename, timedelta=1.0, timerange=data.timerange).grid
    pd.testing.assert_frame_equal(grid.storagevalue_time, data.storagevalue_time, check_dtype=False, check_names=False)


def test_wide_profiles_keep_the_time_curves(tmp_path):
    """Above SQLite's column limit only the referenced profile columns are written: the time
    curves of the storage rows are among them, the unreferenced ones are not."""
    data = _profiles(tmp_path)
    wide = pd.DataFrame(0.0, index=data.profiles.index, columns=[f"x{i}" for i in range(1801)])
    data.profiles = pd.concat([data.profiles, wide], axis=1)
    data.generator.loc[GEN_GAS, "storval_time_ref"] = "x7"  # no storage: not kept
    res = powergama.Results(data, tmp_path / "wide.sqlite", replace=True)
    kept = powergama.Results.from_existing(res.db.filename, timedelta=1.0, timerange=data.timerange).grid.profiles
    assert {"svt_hydro", "svt_flex"} <= set(kept.columns)
    assert "x7" not in kept.columns
    assert len(kept.columns) < 20


def test_wide_profiles_unknown_time_ref_raises_on_write(tmp_path):
    data = _profiles(tmp_path, gen_ref="no_such_curve")
    with pytest.raises(RuntimeError, match="no_such_curve"):
        powergama.Results(data, tmp_path / "bad.sqlite", replace=True)
