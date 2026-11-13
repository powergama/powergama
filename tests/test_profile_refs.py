"""
Profile files given as a list, and the generator profile references that override a
static value: pmin_ref, pmax_ref and storage_ini_ref ("const" = no profile).
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import powergama

TWO_AREA = Path(__file__).parent / "test_data/two_area"
TOL = 1e-6
RES_TABLES = (
    "Res_Branches",
    "Res_DcBranches",
    "Res_Generators",
    "Res_Storage",
    "Res_Pumping",
    "Res_FlexibleLoad",
    "Res_Nodes",
    "Res_ObjFunc",
)


def _read_two_area(profiles, timerange=range(0, 48)):
    data = powergama.GridData()
    data.readGridData(
        nodes=TWO_AREA / "nodes.csv",
        ac_branches=TWO_AREA / "branches.csv",
        dc_branches=TWO_AREA / "dcbranches.csv",
        generators=TWO_AREA / "generators.csv",
        consumers=TWO_AREA / "consumers.csv",
        inter_area_rules=TWO_AREA / "inter_area_rules.csv",
    )
    data.readProfileData(
        filename=profiles,
        storagevalue_filling=TWO_AREA / "profiles_storval_filling.csv",
        storagevalue_time=TWO_AREA / "profiles_storval_time.csv",
        timerange=timerange,
        timedelta=1.0,
    )
    return data


def _solve(data, dbfile):
    lp = powergama.LpProblem(data)
    res = powergama.Results(data, dbfile, replace=True)
    lp.solve(res, solver="appsi_highs", solve_args={})
    with sqlite3.connect(res.db.filename) as con:
        tables = {}
        for name in RES_TABLES:
            order = "timestep" if name == "Res_ObjFunc" else "timestep, indx"
            tables[name] = pd.read_sql_query(f"SELECT * FROM {name} ORDER BY {order}", con)  # nosec B608
    return lp, tables


def _gen(data, desc):
    return int(data.generator.index[data.generator["desc"] == desc][0])


# --- profile file lists -----------------------------------------------------


def _split_profiles(tmp_path):
    """two_area's profiles as one CSV and one parquet file, split by column."""
    full = pd.read_csv(TWO_AREA / "profiles.csv")
    csv_file = tmp_path / "profiles_a.csv"
    pq_file = tmp_path / "profiles_b.parquet"
    full[["const", "wind"]].to_csv(csv_file, index=False)
    full[["hydro", "demand"]].to_parquet(pq_file)
    return csv_file, pq_file


@pytest.mark.parametrize("timerange", [range(0, 48), range(10, 30)])
def test_profile_file_list_equals_single_file(tmp_path, timerange):
    csv_file, pq_file = _split_profiles(tmp_path)
    single = _read_two_area(TWO_AREA / "profiles.csv", timerange)
    joined = _read_two_area([csv_file, pq_file], timerange)
    pdt.assert_frame_equal(joined.profiles, single.profiles)
    assert list(joined.profiles.index) == list(range(len(timerange)))


def test_profile_file_list_gives_same_results(tmp_path):
    csv_file, pq_file = _split_profiles(tmp_path)
    _, single = _solve(_read_two_area(TWO_AREA / "profiles.csv"), tmp_path / "single.sqlite")
    _, joined = _solve(_read_two_area([csv_file, pq_file]), tmp_path / "joined.sqlite")
    for name in RES_TABLES:
        pdt.assert_frame_equal(joined[name], single[name], check_exact=True, obj=name)


def test_profile_file_list_duplicate_column_raises(tmp_path):
    full = pd.read_csv(TWO_AREA / "profiles.csv")
    a = tmp_path / "a.csv"
    b = tmp_path / "b.parquet"
    full[["const", "wind", "hydro"]].to_csv(a, index=False)
    full[["hydro", "demand"]].to_parquet(b)
    with pytest.raises(ValueError, match=r"Duplicate profile column.*'hydro'"):
        _read_two_area([a, b])


def test_profile_file_list_row_mismatch_raises(tmp_path):
    full = pd.read_csv(TWO_AREA / "profiles.csv")
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    full[["const", "wind"]].to_csv(a, index=False)
    full[["hydro", "demand"]].iloc[:20].to_csv(b, index=False)
    with pytest.raises(ValueError, match=r"b\.csv' has no rows for 28 timestep"):
        _read_two_area([a, b])


# --- "const" and empty references --------------------------------------------


def test_override_refs_default_to_const():
    data = _read_two_area(TWO_AREA / "profiles.csv")
    for col in ("pmin_ref", "pmax_ref", "storage_ini_ref"):
        assert (data.generator[col] == "const").all(), col
    assert "const" in data.profiles.columns  # not needed, but present in two_area


def test_const_and_empty_refs_leave_results_unchanged(tmp_path):
    base = _read_two_area(TWO_AREA / "profiles.csv")
    _, ref = _solve(base, tmp_path / "base.sqlite")

    # Explicit const / empty string / NaN, and a grid with no "const" profile column.
    mixed = _read_two_area(TWO_AREA / "profiles.csv")
    gens = mixed.generator
    gens["pmax_ref"] = ["const", "", np.nan, " const "]
    gens["pmin_ref"] = ["", np.nan, "const", ""]
    gens["storage_ini_ref"] = [np.nan, "const", "", "const"]
    lp, tables = _solve(mixed, tmp_path / "mixed.sqlite")
    assert lp._pmax_ref is None and lp._pmin_ref is None and lp._storage_ini_ref is None
    for name in RES_TABLES:
        pdt.assert_frame_equal(tables[name], ref[name], check_exact=True, obj=name)

    # Grid data without the columns at all (e.g. read from an older results database).
    absent = _read_two_area(TWO_AREA / "profiles.csv")
    absent.generator = absent.generator.drop(columns=["pmin_ref", "pmax_ref", "storage_ini_ref"])
    _, tables = _solve(absent, tmp_path / "absent.sqlite")
    for name in RES_TABLES:
        pdt.assert_frame_equal(tables[name], ref[name], check_exact=True, obj=name)


def test_const_ref_needs_no_const_column(tmp_path):
    data = _read_two_area(TWO_AREA / "profiles.csv")
    # gas_a and coal_b use "const" as inflow_ref; give them a column under another name.
    data.profiles = data.profiles.rename(columns={"const": "one"})
    data.generator["inflow_ref"] = data.generator["inflow_ref"].replace("const", "one")
    data.consumer["demand_ref"] = data.consumer["demand_ref"].replace("const", "one")
    data.generator.loc[_gen(data, "gas_a"), "pmax_ref"] = "const"
    lp = powergama.LpProblem(data)
    assert lp._pmax_ref is None


@pytest.mark.parametrize("col", ["pmax_ref", "pmin_ref", "storage_ini_ref"])
def test_unknown_ref_raises(col):
    data = _read_two_area(TWO_AREA / "profiles.csv")
    data.generator.loc[_gen(data, "coal_b"), col] = "no_such_profile"
    with pytest.raises(ValueError, match=rf"{col}: no profile column\(s\) \['no_such_profile'\].*coal_b"):
        powergama.LpProblem(data)


# --- pmin_ref / pmax_ref overrides --------------------------------------------


def test_pmax_and_pmin_ref_override(tmp_path):
    data = _read_two_area(TWO_AREA / "profiles.csv")
    n = len(data.profiles)
    t = np.arange(n)
    # coal_b (pmax 300, the expensive unit) pinned to a slow wave; gas_a (pmax 400, cheap)
    # capped below what it would otherwise produce.
    data.profiles["coal_pin"] = 0.5 + 0.05 * np.sin(2 * np.pi * t / 24)
    data.profiles["gas_cap"] = 0.3
    coal = _gen(data, "coal_b")
    gas = _gen(data, "gas_a")
    data.generator.loc[coal, ["pmin_ref", "pmax_ref"]] = "coal_pin"
    data.generator.loc[gas, "pmax_ref"] = "gas_cap"

    _, base = _solve(_read_two_area(TWO_AREA / "profiles.csv"), tmp_path / "base.sqlite")
    _, tables = _solve(data, tmp_path / "override.sqlite")
    out = tables["Res_Generators"].pivot(index="timestep", columns="indx", values="output")
    base_out = base["Res_Generators"].pivot(index="timestep", columns="indx", values="output")

    np.testing.assert_allclose(out[coal].to_numpy(), 300 * data.profiles["coal_pin"].to_numpy(), atol=TOL)
    assert (out[gas] <= 400 * 0.3 + TOL).all()
    assert (base_out[gas] > 400 * 0.3 + 1).any(), "the gas cap never binds; the test would prove nothing"


def test_pmin_ref_alone_is_a_floor(tmp_path):
    data = _read_two_area(TWO_AREA / "profiles.csv")
    data.profiles["coal_floor"] = 0.4
    coal = _gen(data, "coal_b")
    data.generator.loc[coal, "pmin_ref"] = "coal_floor"
    _, tables = _solve(data, tmp_path / "floor.sqlite")
    out = tables["Res_Generators"].pivot(index="timestep", columns="indx", values="output")
    assert (out[coal] >= 300 * 0.4 - TOL).all()


# --- storage_ini_ref ------------------------------------------------------------


def test_storage_ini_ref_sets_initial_fill():
    data = _read_two_area(TWO_AREA / "profiles.csv")
    hydro = _gen(data, "hydro_pump")
    cap = float(data.generator.loc[hydro, "storage_cap"])
    data.profiles["hydro_fill"] = np.nan  # only the first timestep is read
    data.profiles.loc[0, "hydro_fill"] = 0.8
    data.generator.loc[hydro, "storage_ini_ref"] = "hydro_fill"

    lp = powergama.LpProblem(data)
    assert lp.storage[hydro] == pytest.approx(0.8 * cap)
    assert lp.storage[hydro] != pytest.approx(float(data.generator.loc[hydro, "storage_ini"]) * cap)
    # Generators without a reference keep storage_ini.
    base = powergama.LpProblem(_read_two_area(TWO_AREA / "profiles.csv"))
    others = [g for g in data.generator.index if g != hydro]
    pdt.assert_series_equal(lp.storage[others], base.storage[others])


def test_storage_ini_ref_first_timestep_of_timerange():
    """The run's first timestep is the first row of the timerange, not of the file."""
    full = pd.read_csv(TWO_AREA / "profiles.csv")
    data = _read_two_area(TWO_AREA / "profiles.csv", timerange=range(10, 30))
    hydro = _gen(data, "hydro_pump")
    data.profiles["hydro_fill"] = full["hydro"].iloc[10:30].to_numpy()
    data.generator.loc[hydro, "storage_ini_ref"] = "hydro_fill"
    lp = powergama.LpProblem(data)
    assert lp.storage[hydro] == pytest.approx(full["hydro"].iloc[10] * data.generator.loc[hydro, "storage_cap"])


def test_storage_ini_ref_nan_raises():
    data = _read_two_area(TWO_AREA / "profiles.csv")
    hydro = _gen(data, "hydro_pump")
    data.profiles["hydro_fill"] = 0.8
    data.profiles.loc[0, "hydro_fill"] = np.nan
    data.generator.loc[hydro, "storage_ini_ref"] = "hydro_fill"
    with pytest.raises(ValueError, match=r"storage_ini_ref.*NaN.*hydro_fill"):
        powergama.LpProblem(data)


def test_from_existing_reads_database_without_the_new_ref_columns(tmp_path):
    """A results database written before pmin_ref/pmax_ref/storage_ini_ref existed still loads."""
    dbfile = tmp_path / "old.sqlite"
    _solve(_read_two_area(TWO_AREA / "profiles.csv", range(0, 4)), dbfile)
    with sqlite3.connect(dbfile) as con:
        gen = pd.read_sql_query("SELECT * FROM data_generator", con, index_col="index")
        gen = gen.drop(columns=["pmin_ref", "pmax_ref", "storage_ini_ref"])
        gen.to_sql("data_generator", con, if_exists="replace", index=True)
    res = powergama.Results.from_existing(dbfile, timedelta=1.0)
    for col in ("pmin_ref", "pmax_ref", "storage_ini_ref"):
        assert (res.grid.generator[col] == "const").all()
