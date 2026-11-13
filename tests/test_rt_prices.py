"""RT deviation prices as unit data.

The prices are ``rt_price_up``/``rt_price_down`` (generators and consumers) and
``rt_pump_price_up``/``rt_pump_price_down`` (pumping), each times an optional profile
``*_ref``. ``price_columns`` writes them from c0, a renewables mask and its curtailment
factor, and the storage values at the DA filling level (``da_storage_values``).

The reference ``test_data/two_area/rt_type_prices_no_storage`` holds the result tables of the
RT run below made with the old ``RtExtension(balancing_fee_eur_per_mwh=1,
res_curtailment_factor=2, xborder_flow_penalty_eur_per_mwh=5)`` (025625f), which priced
units by type and marginal cost inside the extension.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rt_case import (
    CONS_FLEX,
    GEN_STORAGE,
    GEN_TARGET,
    N_STEPS,
    build_rt_case,
    read_table,
    series,
    solve_rt_case,
    two_area_data,
)

import powergama
from powergama.lp.rt import RtExtension, da_storage_values, price_columns

REFERENCE = Path(__file__).parent / "test_data/two_area/rt_type_prices_no_storage"
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
GEN_WIND = 1


def _read(res, table):
    order = "timestep" if table == "Res_ObjFunc" else "timestep, indx"
    with sqlite3.connect(res.db.filename) as con:
        return pd.read_sql_query(f"SELECT * FROM {table} ORDER BY {order}", con)  # nosec B608


def _case(tmp_path, *, storage_tracked):
    """rt_case at c0 = 1, with wind tracked too (a renewable), optionally without the storage targets."""
    data, _ = build_rt_case(tmp_path, fee=1.0)
    with sqlite3.connect(tmp_path / "da.sqlite") as con:
        gen = pd.read_sql_query("SELECT * FROM Res_Generators ORDER BY timestep, indx", con)
    steps = list(data.timerange)
    pmax = data.generator.loc[GEN_WIND, "pmax"]
    data.profiles["rt_target_wind"] = series(gen, GEN_WIND, "output").reindex(steps).values / pmax
    data.generator.loc[GEN_WIND, "rt_target_ref"] = "rt_target_wind"
    if not storage_tracked:
        data.generator.loc[GEN_STORAGE, ["rt_target_ref", "rt_pump_target_ref"]] = ""
        data.consumer.loc[CONS_FLEX, "rt_target_ref"] = ""
    return data


def test_reproduces_type_prices_without_storage(tmp_path):
    """Without a tracked storage unit, the RT run is the one with the old type-based pricing, bit for bit."""
    data = _case(tmp_path, storage_tracked=False)
    _, res = solve_rt_case(data, tmp_path / "rt.sqlite")
    for table in RESULT_TABLES:
        expected = pd.read_parquet(REFERENCE / f"{table}.parquet")
        pd.testing.assert_frame_equal(_read(res, table), expected, check_exact=True, obj=table)


def _filling_factor(grid, column, level, cap):
    rows = np.rint(np.clip(level / cap, 0.0, 1.0) * 100).astype(int)
    return grid.storagevalue_filling[column].to_numpy(float)[rows]


def test_storage_prices_at_the_da_filling(tmp_path):
    """Storage units are priced at the DA filling: storage_price x F(DA filling) x T(t).

    The other units keep the old type-based prices, c0 +/- fuelcost and -c0 / factor x c0 for wind.
    """
    data = _case(tmp_path, storage_tracked=True)
    da = powergama.Results(two_area_data(), tmp_path / "da.sqlite", replace=False)
    gen = data.generator
    fee, steps = 1.0, list(range(N_STEPS))

    # the DA filling at the start of each timestep, and the curves, read independently
    stored = series(read_table(da, "Res_Storage"), GEN_STORAGE, "storage").reindex(steps).to_numpy()
    cap = gen.loc[GEN_STORAGE, "storage_cap"]
    level = np.concatenate([[gen.loc[GEN_STORAGE, "storage_ini"] * cap], stored[:-1]])
    time = data.storagevalue_time["const"].to_numpy(float)
    value = gen.loc[GEN_STORAGE, "storage_price"] * _filling_factor(data, "hydro", level, cap) * time
    cost = gen.loc[GEN_STORAGE, "fuelcost"] + value
    assert np.ptp(value) > 0  # the filling moves, and with it the price
    np.testing.assert_array_equal(data.profiles[f"da_price_up_{GEN_STORAGE}"], fee + cost)
    np.testing.assert_array_equal(data.profiles[f"da_price_down_{GEN_STORAGE}"], fee - cost)
    credit = value * gen.loc[GEN_STORAGE, "pump_efficiency"] - gen.loc[GEN_STORAGE, "pump_deadband"]
    np.testing.assert_array_equal(data.profiles[f"da_pump_price_up_{GEN_STORAGE}"], fee - credit)
    np.testing.assert_array_equal(data.profiles[f"da_pump_price_down_{GEN_STORAGE}"], fee + credit)
    assert gen.loc[GEN_STORAGE, "rt_price_up"] == gen.loc[GEN_STORAGE, "rt_pump_price_up"] == 1.0
    assert gen.loc[GEN_STORAGE, "rt_price_up_ref"] == f"da_price_up_{GEN_STORAGE}"

    # the flexible load, at its DA reservoir level
    cons = data.consumer.loc[CONS_FLEX]
    flex_cap = cons["flex_storage"] * cons["flex_fraction"] * cons["demand_avg"]
    flex_stored = series(read_table(da, "Res_FlexibleLoad"), CONS_FLEX, "storage").reindex(steps).to_numpy()
    flex_level = np.concatenate([[cons["flex_storagelevel_init"] * flex_cap], flex_stored[:-1]])
    flex_value = cons["flex_basevalue"] * _filling_factor(data, "flexdemand", flex_level, flex_cap)
    np.testing.assert_array_equal(data.profiles[f"da_cons_price_up_{CONS_FLEX}"], fee + flex_value)
    np.testing.assert_array_equal(data.profiles[f"da_cons_price_down_{CONS_FLEX}"], fee - flex_value)

    # the model's prices, every timestep
    lp, _ = solve_rt_case(data, tmp_path / "rt.sqlite")
    m = lp.model
    for t in steps:
        lp._updateLpProblem([t])
        assert m.p_rt_deviation_price_gen_pos[GEN_TARGET].value == fee + gen.loc[GEN_TARGET, "fuelcost"]
        assert m.p_rt_deviation_price_gen_neg[GEN_TARGET].value == fee - gen.loc[GEN_TARGET, "fuelcost"]
        assert m.p_rt_deviation_price_gen_pos[GEN_WIND].value == -fee
        assert m.p_rt_deviation_price_gen_neg[GEN_WIND].value == 2.0 * fee
        assert m.p_rt_deviation_price_gen_pos[GEN_STORAGE].value == fee + cost[t]
        assert m.p_rt_deviation_price_gen_neg[GEN_STORAGE].value == fee - cost[t]
        assert m.p_rt_deviation_price_pump_pos[GEN_STORAGE].value == fee - credit[t]
        assert m.p_rt_deviation_price_pump_neg[GEN_STORAGE].value == fee + credit[t]
        assert m.p_rt_deviation_price_flex_pos[CONS_FLEX].value == fee + flex_value[t]
        assert m.p_rt_deviation_price_flex_neg[CONS_FLEX].value == fee - flex_value[t]


def test_da_storage_values_are_the_da_run_values(tmp_path):
    """At the DA filling, the value is the one the (single-period) DA run used and stored."""
    data, _ = build_rt_case(tmp_path)
    da = powergama.Results(two_area_data(), tmp_path / "da.sqlite", replace=False)
    values = da_storage_values(da)
    stored = series(read_table(da, "Res_Storage"), GEN_STORAGE, "marginalprice")
    np.testing.assert_array_equal(values["generator"][GEN_STORAGE].to_numpy(), stored.to_numpy())
    flex = series(read_table(da, "Res_FlexibleLoad"), CONS_FLEX, "value")
    np.testing.assert_array_equal(values["consumer"][CONS_FLEX].to_numpy(), flex.to_numpy())
    # a window of the DA run: the same values
    cut = da_storage_values(da, timeMaxMin=[5, 9])
    pd.testing.assert_frame_equal(cut["generator"], values["generator"].loc[5:8])


# --- the extension reads the prices -----------------------------------------------------


def _priced(prices, **profiles):
    """two_area with coal tracked and the given price columns on it."""
    data = two_area_data()
    data.profiles["rt_coal"] = 0.5
    for name, values in profiles.items():
        data.profiles[name] = values
    data.generator["rt_target_ref"] = ""
    data.generator.loc[GEN_TARGET, "rt_target_ref"] = "rt_coal"
    for col, value in prices.items():
        data.generator[col] = np.nan if not isinstance(value, str) else "const"
        data.generator.loc[GEN_TARGET, col] = value
    return data


def test_price_is_constant_times_profile():
    profile = np.linspace(0.5, 2.0, N_STEPS)
    data = _priced(
        {"rt_price_up": 30.0, "rt_price_down": -4.0, "rt_price_up_ref": "mult", "rt_price_down_ref": "const"},
        mult=profile,
    )
    lp = powergama.LpProblem(data, extensions=[RtExtension()])
    for t in (0, 7, N_STEPS - 1):
        lp._updateLpProblem([t])
        assert lp.model.p_rt_deviation_price_gen_pos[GEN_TARGET].value == 30.0 * profile[t]
        assert lp.model.p_rt_deviation_price_gen_neg[GEN_TARGET].value == -4.0
        # an untracked row has no price
        assert lp.model.p_rt_deviation_price_gen_pos[0].value == 0.0


def test_price_without_ref_column_is_the_constant():
    data = _priced({"rt_price_up": 3.0, "rt_price_down": 5.0})
    lp = powergama.LpProblem(data, extensions=[RtExtension()])
    lp._updateLpProblem([0])
    assert lp.model.p_rt_deviation_price_gen_pos[GEN_TARGET].value == 3.0
    assert lp.model.p_rt_deviation_price_gen_neg[GEN_TARGET].value == 5.0


def test_split_sum_guard_still_applies():
    """A pair summing to less than 1e-6 is lifted to it, on the smaller price."""
    data = _priced({"rt_price_up": -2.0, "rt_price_down": 1.0})
    lp = powergama.LpProblem(data, extensions=[RtExtension()])
    lp._updateLpProblem([0])
    # -2 + 1 = -1 < 1e-6: the up price (the smaller) is raised by 1 + 1e-6
    assert lp.model.p_rt_deviation_price_gen_pos[GEN_TARGET].value == pytest.approx(-1.0 + 1e-6)
    assert lp.model.p_rt_deviation_price_gen_neg[GEN_TARGET].value == 1.0


@pytest.mark.parametrize("prices", [{}, {"rt_price_up": 3.0}, {"rt_price_up": 3.0, "rt_price_down": np.nan}])
def test_tracked_row_without_price_raises(prices):
    data = _priced(prices)
    with pytest.raises(ValueError, match=r"rt_price_(up|down) is missing for row\(s\) 3 \(coal_b\)"):
        powergama.LpProblem(data, extensions=[RtExtension()])


def test_tracked_pump_and_flex_rows_without_price_raise():
    data = two_area_data()
    data.profiles["zero"] = 0.0
    data.generator["rt_pump_target_ref"] = ""
    data.generator.loc[GEN_STORAGE, "rt_pump_target_ref"] = "zero"
    data.consumer["rt_target_ref"] = ""
    data.consumer.loc[CONS_FLEX, "rt_target_ref"] = "zero"
    with pytest.raises(ValueError) as err:
        powergama.LpProblem(data, extensions=[RtExtension()])
    assert "generator rt_pump_price_up is missing for row(s) 2 (hydro_pump)" in str(err.value)
    assert "consumer rt_price_up is missing for row(s) 1" in str(err.value)


def test_unknown_price_ref_raises():
    data = _priced({"rt_price_up": 3.0, "rt_price_down": 5.0, "rt_price_up_ref": "no_such_column"})
    with pytest.raises(ValueError, match=r"rt_price_up_ref: no profile column\(s\) \['no_such_column'\]"):
        powergama.LpProblem(data, extensions=[RtExtension()])


def test_unknown_price_ref_on_untracked_row_raises():
    """Refs are checked on every row, like pmax_ref."""
    data = _priced({"rt_price_up": 3.0, "rt_price_down": 5.0})
    data.generator["rt_price_down_ref"] = "const"
    data.generator.loc[0, "rt_price_down_ref"] = "typo"
    with pytest.raises(ValueError, match=r"rt_price_down_ref: no profile column\(s\) \['typo'\] for row\(s\) 0"):
        powergama.LpProblem(data, extensions=[RtExtension()])


def test_nan_in_price_profile_raises():
    profile = np.ones(N_STEPS)
    profile[3] = np.nan
    data = _priced({"rt_price_up": 3.0, "rt_price_down": 5.0, "rt_price_up_ref": "mult"}, mult=profile)
    with pytest.raises(ValueError, match="rt_price_up_ref profile 'mult' has NaN"):
        powergama.LpProblem(data, extensions=[RtExtension()])


def test_positional_argument_raises():
    """The first positional argument used to be the balancing fee."""
    with pytest.raises(TypeError):
        RtExtension(10.0)


# --- price_columns ----------------------------------------------------------------------


def _tables():
    data = two_area_data()
    for name in ("generator", "consumer"):
        table = getattr(data, name)
        table["da_index"] = pd.array(table.index, dtype="Int64")
    return data


def test_price_columns_rules():
    data = _tables()
    steps = pd.Index(range(3))
    storage = pd.DataFrame({GEN_STORAGE: [10.0, 20.0, 30.0]}, index=steps)
    flex = pd.DataFrame({CONS_FLEX: [5.0, 6.0, 7.0]}, index=steps)
    wind = data.generator["type"].eq("wind")
    refs = price_columns(
        data.generator,
        data.consumer,
        fee=4.0,
        renewable=wind,
        curtailment_factor=3.0,
        storage_value=storage,
        flex_value=flex,
    )
    g = refs.generator
    assert list(g.loc[[0, 3], "rt_price_up"]) == [4.0 + 15.0, 4.0 + 40.0]
    assert list(g.loc[[0, 3], "rt_price_down"]) == [4.0 - 15.0, 4.0 - 40.0]
    assert (g.loc[GEN_WIND, "rt_price_up"], g.loc[GEN_WIND, "rt_price_down"]) == (-4.0, 12.0)
    assert list(g.loc[[0, 1, 3], "rt_price_up_ref"]) == ["const"] * 3
    assert g.loc[GEN_STORAGE, "rt_price_up"] == 1.0
    assert g.loc[GEN_STORAGE, "rt_pump_price_down_ref"] == f"da_pump_price_down_{GEN_STORAGE}"
    assert g["rt_pump_price_up"].notna().tolist() == [False, False, True, False]
    eff, deadband = 0.8, 1.0
    np.testing.assert_array_equal(refs.profiles[f"da_price_up_{GEN_STORAGE}"], 4.0 + (0.0 + storage[GEN_STORAGE]))
    np.testing.assert_array_equal(
        refs.profiles[f"da_pump_price_up_{GEN_STORAGE}"], 4.0 - (storage[GEN_STORAGE] * eff - deadband)
    )
    np.testing.assert_array_equal(refs.profiles[f"da_cons_price_down_{CONS_FLEX}"], 4.0 - flex[CONS_FLEX])
    # only the flexible consumer is priced
    assert list(refs.consumer.index) == [CONS_FLEX]
    assert refs.consumer.loc[CONS_FLEX, "rt_price_up_ref"] == f"da_cons_price_up_{CONS_FLEX}"
    assert list(refs.profiles.columns) == [
        f"da_price_up_{GEN_STORAGE}",
        f"da_price_down_{GEN_STORAGE}",
        f"da_pump_price_up_{GEN_STORAGE}",
        f"da_pump_price_down_{GEN_STORAGE}",
        f"da_cons_price_up_{CONS_FLEX}",
        f"da_cons_price_down_{CONS_FLEX}",
    ]


def test_price_columns_raises():
    data = _tables()
    steps = pd.Index(range(3))
    flex = pd.DataFrame({CONS_FLEX: [5.0, 6.0, 7.0]}, index=steps)
    with pytest.raises(ValueError, match="DA row 2 have storage: 'storage_value' has no column"):
        price_columns(data.generator, data.consumer, fee=1.0, flex_value=flex)
    storage = pd.DataFrame({GEN_STORAGE: [1.0, np.nan, 3.0]}, index=steps)
    with pytest.raises(ValueError, match="'storage_value' has NaN for DA row 2"):
        price_columns(data.generator, data.consumer, fee=1.0, storage_value=storage, flex_value=flex)
    gen = data.generator.copy()
    gen.loc[GEN_STORAGE, "da_index"] = pd.NA
    with pytest.raises(ValueError, match="row 2 has storage but no DA row"):
        price_columns(gen, data.consumer, fee=1.0, storage_value=storage.fillna(0.0), flex_value=flex)
    with pytest.raises(ValueError, match="'renewable' must be a boolean pandas Series"):
        price_columns(data.generator, data.consumer, fee=1.0, renewable=[True])


# --- the results database keeps the price profiles ----------------------------------------

PRICE_REF_COLUMNS = {
    "generator": ["rt_price_up_ref", "rt_price_down_ref", "rt_pump_price_up_ref", "rt_pump_price_down_ref"],
    "consumer": ["rt_price_up_ref", "rt_price_down_ref"],
}


def _price_profile_refs(grid):
    refs = set()
    for table_name, columns in PRICE_REF_COLUMNS.items():
        table = getattr(grid, table_name)
        for col in columns:
            for ref in table[col].dropna().astype(str).str.strip():
                if ref not in ("", "const", "nan"):
                    refs.add(ref)
    return refs


def test_wide_profiles_keep_price_columns_and_reopen(tmp_path):
    """Above SQLITE_MAX_COLUMNS_SOFT profile columns the database stores only referenced
    columns: the price profiles are among them, and the reopened grid resolves its price refs."""
    data, _ = build_rt_case(tmp_path)
    refs = _price_profile_refs(data)
    assert refs, "the case must have price profile columns"
    assert {r for r in refs if r.startswith("da_price_")}
    filler = pd.DataFrame(
        np.zeros((len(data.profiles), 1900)),
        index=data.profiles.index,
        columns=[f"unused_{i}" for i in range(1900)],
    )
    data.profiles = pd.concat([data.profiles, filler], axis=1)
    assert len(data.profiles.columns) > powergama.database.Database.SQLITE_MAX_COLUMNS_SOFT
    dbfile = tmp_path / "rt_wide.sqlite"
    _, res = solve_rt_case(data, dbfile)
    reopened = powergama.Results.from_existing(dbfile, timedelta=1.0)
    stored = reopened.grid.profiles
    assert "unused_0" not in stored.columns
    assert _price_profile_refs(reopened.grid) == refs
    for ref in sorted(refs):
        np.testing.assert_array_equal(stored[ref].to_numpy(), data.profiles[ref].to_numpy())
    # The reopened grid's price refs resolve as the extension resolves them.
    from powergama.lp.rt.context import PRICE_COLUMNS, _price_refs

    problems = []
    for table_name, up, down, _target in PRICE_COLUMNS.values():
        table = getattr(reopened.grid, table_name)
        for column in (up, down):
            _price_refs(table, table_name, column, [], _target, stored, problems)
    assert problems == []


@pytest.mark.parametrize(
    "table_name, column",
    [
        ("generator", "rt_price_up_ref"),
        ("generator", "rt_price_down_ref"),
        ("generator", "rt_pump_price_up_ref"),
        ("generator", "rt_pump_price_down_ref"),
        ("consumer", "rt_price_up_ref"),
        ("consumer", "rt_price_down_ref"),
    ],
)
def test_input_tables_without_solve_raise_on_missing_price_profile(tmp_path, table_name, column):
    """Writing the input tables alone (no LP, no solve) checks the price refs like the LP does."""
    data = two_area_data()
    table = getattr(data, table_name)
    table[column] = "const"
    table.loc[table.index[0], column] = "no_such_price"
    with pytest.raises(RuntimeError, match="no_such_price"):
        powergama.Results(data, tmp_path / "inputs.sqlite", replace=True)


def test_input_tables_without_solve_accept_const_and_empty_price_refs(tmp_path):
    data = two_area_data()
    data.generator["rt_price_up_ref"] = "const"
    data.generator["rt_price_down_ref"] = ""
    data.generator["rt_pump_price_up_ref"] = np.nan
    data.consumer["rt_price_up_ref"] = "const"
    powergama.Results(data, tmp_path / "inputs.sqlite", replace=True)
