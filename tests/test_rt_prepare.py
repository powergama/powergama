"""prepare_rt: from a day-ahead run to the inputs of the balancing run, through files.

The hand-built RT case (tests/rt_case.py) derives its soft targets from the baseline run by
hand; prepare_rt must give the same values, write them to files that read back into the
same RT run, and line the DA timesteps up with the rows of the profile files.
"""

import sqlite3

import numpy as np
import pandas as pd
import pytest
from rt_case import (
    CONS_FIXED,
    CONS_FLEX,
    GEN_PINNED,
    GEN_STORAGE,
    GEN_TARGET,
    N_STEPS,
    datapath_two_area,
    read_table,
    series,
    two_area_data,
)

import powergama
from powergama.lp.rt import (
    ZERO_TARGET,
    CarryCurtailment,
    DeviationPrices,
    PreparedRt,
    RtExtension,
    carry_curtailment_columns,
    clear_refs,
    combine_columns,
    initial_storage_columns,
    pin_columns,
    prepare_rt,
    summarize_channels,
    target_columns,
)

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


def _data(timerange, profiles=None):
    data = powergama.GridData()
    data.readGridData(
        nodes=datapath_two_area / "nodes.csv",
        ac_branches=datapath_two_area / "branches.csv",
        dc_branches=datapath_two_area / "dcbranches.csv",
        generators=datapath_two_area / "generators.csv",
        consumers=datapath_two_area / "consumers.csv",
        inter_area_rules=datapath_two_area / "inter_area_rules.csv",
    )
    data.readProfileData(
        filename=profiles if profiles is not None else datapath_two_area / "profiles.csv",
        storagevalue_filling=datapath_two_area / "profiles_storval_filling.csv",
        storagevalue_time=datapath_two_area / "profiles_storval_time.csv",
        timerange=timerange,
        timedelta=1.0,
    )
    return data


def _solve(data, dbfile, extensions=()):
    lp = powergama.LpProblem(data, extensions=list(extensions))
    res = powergama.Results(data, dbfile, replace=True)
    lp.solve(res, solver="appsi_highs", solve_args={})
    return res


def _read_all(res, table):
    order = "timestep" if table == "Res_ObjFunc" else "timestep, indx"
    with sqlite3.connect(res.db.filename) as con:
        return pd.read_sql_query(f"SELECT * FROM {table} ORDER BY {order}", con)  # nosec B608


def _rt():
    return RtExtension()


def _prices(res_da):
    """c0 = 10, wind priced as a renewable, storage at the DA filling level."""
    return DeviationPrices(fee=10.0, renewable=res_da.grid.generator["type"].eq("wind"))


@pytest.fixture(scope="module")
def da(tmp_path_factory):
    return _solve(two_area_data(), tmp_path_factory.mktemp("da") / "da.sqlite")


@pytest.fixture(scope="module")
def prepared(da, tmp_path_factory):
    return prepare_rt(da, tmp_path_factory.mktemp("rt_inputs"), text_output=True, prices=_prices(da))


@pytest.fixture(scope="module")
def rt_profiles(tmp_path_factory):
    """The RT profile file: less wind than the DA run planned for, as in rt_case."""
    profiles = pd.read_csv(datapath_two_area / "profiles.csv")
    profiles["wind"] = 0.7 * profiles["wind"]
    path = tmp_path_factory.mktemp("rt_profiles") / "profiles_rt.csv"
    profiles.to_csv(path, index=False)
    return path


# --- (a) the values are the hand-built ones ------------------------------------------


def test_columns_equal_hand_built(da, prepared):
    """The same values rt_case.build_rt_case computes by hand from the same DA run."""
    gen = read_table(da, "Res_Generators")
    pump = read_table(da, "Res_Pumping")
    flex = read_table(da, "Res_FlexibleLoad")
    grid = da.grid
    steps = list(range(N_STEPS))
    sol = prepared.da_solution
    assert list(sol.index) == steps
    assert sol.dtypes.eq(np.float64).all()

    hand = {
        f"da_gen_{GEN_TARGET}": series(gen, GEN_TARGET, "output").reindex(steps).values
        / grid.generator.loc[GEN_TARGET, "pmax"],
        f"da_gen_{GEN_STORAGE}": series(gen, GEN_STORAGE, "output").reindex(steps).values
        / grid.generator.loc[GEN_STORAGE, "pmax"],
        f"da_pump_{GEN_STORAGE}": series(pump, GEN_STORAGE, "output").reindex(steps).fillna(0.0).values
        / grid.generator.loc[GEN_STORAGE, "pump_cap"],
        f"da_cons_{CONS_FLEX}": series(flex, CONS_FLEX, "demand").reindex(steps).values
        / grid.consumer.loc[CONS_FLEX, "demand_avg"],
    }
    for col, values in hand.items():
        assert np.array_equal(sol[col].to_numpy(), np.clip(values, 0.0, 1.0)), col
        # the case is chosen so that nothing is clipped and the comparison is meaningful
        assert np.array_equal(values, np.clip(values, 0.0, 1.0)), col
        assert sol[col].gt(0).any(), col

    gens = prepared.tables["generator"]
    assert gens.loc[GEN_TARGET, "rt_target_ref"] == f"da_gen_{GEN_TARGET}"
    assert gens.loc[GEN_STORAGE, "rt_target_ref"] == f"da_gen_{GEN_STORAGE}"
    assert gens.loc[GEN_STORAGE, "rt_pump_target_ref"] == f"da_pump_{GEN_STORAGE}"
    assert prepared.tables["consumer"].loc[CONS_FLEX, "rt_target_ref"] == f"da_cons_{CONS_FLEX}"


def test_written_files(prepared):
    assert isinstance(prepared, PreparedRt)
    names = {"node", "branch", "dcbranch", "generator", "consumer", "inter_area_rules", "da_solution"}
    names |= {"summary", "channels"}
    assert set(prepared.files) == names
    for path in prepared.files.values():
        assert path.is_file()
    assert (prepared.files["da_solution"].parent / "da_solution.csv").is_file()
    back = pd.read_parquet(prepared.files["da_solution"])
    pd.testing.assert_frame_equal(back, prepared.da_solution)
    # only referenced columns, and no zero column when nothing needs it
    refs = set(prepared.tables["generator"]["rt_target_ref"]) | set(prepared.tables["generator"]["rt_pump_target_ref"])
    refs |= set(prepared.tables["consumer"]["rt_target_ref"])
    # and the deviation prices of the storage units (the others are constants)
    for table, cols in (
        ("generator", ["rt_price_up_ref", "rt_price_down_ref", "rt_pump_price_up_ref", "rt_pump_price_down_ref"]),
        ("consumer", ["rt_price_up_ref", "rt_price_down_ref"]),
    ):
        for col in cols:
            refs |= set(prepared.tables[table][col])
    assert set(prepared.da_solution.columns) == refs - {"", "const"}
    assert ZERO_TARGET not in prepared.da_solution.columns


# --- (b) who gets a target ---------------------------------------------------------


def test_every_unit_with_capacity_has_a_target(da, prepared):
    grid = da.grid
    gens = prepared.tables["generator"]
    assert list(gens["da_index"]) == list(grid.generator.index)
    for i in grid.generator.index:
        assert (gens.loc[i, "rt_target_ref"] == f"da_gen_{i}") == (grid.generator.loc[i, "pmax"] > 0)
        has_pump = grid.generator.loc[i, "pump_cap"] > 0
        assert (gens.loc[i, "rt_pump_target_ref"] == f"da_pump_{i}") == has_pump
        if not has_pump:
            assert gens.loc[i, "rt_pump_target_ref"] == ""
    cons = prepared.tables["consumer"]
    assert cons.loc[CONS_FLEX, "rt_target_ref"] == f"da_cons_{CONS_FLEX}"
    # a consumer without flexible load has no target
    assert grid.consumer.loc[CONS_FIXED, "flex_fraction"] == 0
    assert cons.loc[CONS_FIXED, "rt_target_ref"] == ""


def test_added_rows_and_zero_capacity(da):
    frames = da.get_dispatch_frames()
    gen = da.grid.generator.copy()
    gen["da_index"] = pd.array(gen.index, dtype="Int64")
    gen.loc[0, "pmax"] = 0.0  # no capacity: no target
    added = gen.loc[[GEN_TARGET, GEN_STORAGE]].copy()  # rows for RT only, e.g. peak tranches
    added["da_index"] = pd.NA
    two_from_one = gen.loc[[GEN_TARGET]].copy()  # a second RT row from the same DA row
    rt_gen = pd.concat([gen, added, two_from_one], ignore_index=True)
    n = len(gen)
    cons = da.grid.consumer.copy()
    cons["da_index"] = pd.array(cons.index, dtype="Int64")
    cons_added = cons.loc[[CONS_FLEX, CONS_FIXED]].copy()
    cons_added["da_index"] = pd.NA
    rt_cons = pd.concat([cons, cons_added], ignore_index=True)

    tc = target_columns(frames, rt_gen, rt_cons, da_generator=da.grid.generator, da_consumer=da.grid.consumer)
    refs = tc.generator
    assert refs.loc[0, "rt_target_ref"] == ""
    assert refs.loc[n, "rt_target_ref"] == ZERO_TARGET
    assert refs.loc[n + 1, "rt_target_ref"] == ZERO_TARGET
    assert refs.loc[n + 1, "rt_pump_target_ref"] == ZERO_TARGET
    assert refs.loc[n, "rt_pump_target_ref"] == ""
    assert refs.loc[n + 2, "rt_target_ref"] == refs.loc[GEN_TARGET, "rt_target_ref"] == f"da_gen_{GEN_TARGET}"
    assert tc.consumer.loc[len(cons)] == ZERO_TARGET  # added flexible consumer
    assert tc.consumer.loc[len(cons) + 1] == ""  # added consumer without flexible load
    assert (tc.profiles[ZERO_TARGET] == 0.0).all()
    assert list(tc.profiles.columns).count(f"da_gen_{GEN_TARGET}") == 1
    assert "da_gen_0" not in tc.profiles.columns

    # without the DA tables, the capacity comes from the RT rows of each DA row
    tc2 = target_columns(frames, rt_gen, rt_cons)
    pd.testing.assert_frame_equal(tc2.profiles, tc.profiles)


# --- (c) the file round trip is the in-memory RT run, bit for bit --------------------


def test_file_round_trip(da, prepared, rt_profiles, tmp_path):
    timerange = range(0, N_STEPS)

    from_files = powergama.GridData()
    from_files.readGridData(**prepared.grid_files())
    from_files.readProfileData(
        filename=[rt_profiles, prepared.files["da_solution"]],
        storagevalue_filling=datapath_two_area / "profiles_storval_filling.csv",
        storagevalue_time=datapath_two_area / "profiles_storval_time.csv",
        timerange=timerange,
        timedelta=1.0,
    )

    in_memory = _data(timerange, profiles=rt_profiles)
    for name, table in prepared.tables.items():
        setattr(in_memory, name, table.copy())
    sol = prepared.da_solution.loc[list(timerange)]
    in_memory.profiles = pd.concat([in_memory.profiles, sol.set_axis(in_memory.profiles.index)], axis=1)

    # the references survive the CSV round trip ("" becomes an empty cell, which is no target)
    for table, col in [
        ("generator", "rt_target_ref"),
        ("generator", "rt_pump_target_ref"),
        ("consumer", "rt_target_ref"),
    ]:
        written = prepared.tables[table][col]
        read = getattr(from_files, table)[col].fillna("")
        assert list(read) == list(written), (table, col)
    assert list(from_files.generator["da_index"]) == list(prepared.tables["generator"]["da_index"])
    pd.testing.assert_frame_equal(
        from_files.profiles[prepared.da_solution.columns],
        prepared.da_solution.reset_index(drop=True),
        check_exact=True,
    )

    res_files = _solve(from_files, tmp_path / "rt_files.sqlite", [_rt()])
    res_mem = _solve(in_memory, tmp_path / "rt_mem.sqlite", [_rt()])
    assert res_files.is_complete() and res_mem.is_complete()
    for table in RESULT_TABLES:
        a, b = _read_all(res_files, table), _read_all(res_mem, table)
        assert len(a) > 0, table
        pd.testing.assert_frame_equal(a, b, check_exact=True, obj=table)
    # and the RT run is not the DA run: less wind moves the dispatch
    assert not _read_all(res_files, "Res_Generators")["output"].equals(_read_all(da, "Res_Generators")["output"])


# --- (d) DA and RT on rows 24..47 of the profile files line up -----------------------


def test_offset_timerange(tmp_path):
    timerange = range(24, 48)
    da = _solve(_data(timerange), tmp_path / "da_offset.sqlite")
    prepared = prepare_rt(da, tmp_path / "rt_inputs", prices=_prices(da))
    assert list(prepared.da_solution.index) == list(timerange)

    # the same from the DA database alone
    reopened = powergama.Results.from_existing(da.db.filename, timedelta=1.0, timerange=timerange)
    again = prepare_rt(reopened, tmp_path / "rt_inputs_again", prices=_prices(reopened))
    pd.testing.assert_frame_equal(again.da_solution, prepared.da_solution)

    data = powergama.GridData()
    data.readGridData(**prepared.grid_files())
    data.readProfileData(
        filename=[datapath_two_area / "profiles.csv", prepared.files["da_solution"]],
        storagevalue_filling=datapath_two_area / "profiles_storval_filling.csv",
        storagevalue_time=datapath_two_area / "profiles_storval_time.csv",
        timerange=timerange,
        timedelta=1.0,
    )
    # RT timestep h reads profile row 24 + h, which is DA timestep 24 + h
    da_out = da.get_dispatch_frames()["generator_output"][GEN_TARGET].to_numpy()
    pmax = float(da.grid.generator.loc[GEN_TARGET, "pmax"])
    assert np.array_equal(data.profiles[f"da_gen_{GEN_TARGET}"].to_numpy(), (da_out / pmax).clip(0.0, 1.0))
    res = _solve(data, tmp_path / "rt_offset.sqlite", [_rt()])
    assert res.is_complete()
    assert res.db.getTimerange() == list(timerange)

    # an RT run on other rows than DA wrote fails loudly instead of misaligning
    with pytest.raises(ValueError, match="no rows"):
        data.readProfileData(
            filename=[datapath_two_area / "profiles.csv", prepared.files["da_solution"]],
            timerange=range(0, 24),
        )


# --- (e) errors -------------------------------------------------------------------


def test_nan_in_da_raises(da):
    frames = da.get_dispatch_frames()
    frames["generator_output"].loc[3, GEN_TARGET] = np.nan
    gen = da.grid.generator.copy()
    gen["da_index"] = pd.array(gen.index, dtype="Int64")
    cons = da.grid.consumer.copy()
    cons["da_index"] = pd.array(cons.index, dtype="Int64")
    with pytest.raises(ValueError, match="NaN"):
        target_columns(frames, gen, cons)
    # NaN where nothing needs it is fine: that generator has no capacity, so no target
    gen.loc[GEN_TARGET, "pmax"] = 0.0
    tc = target_columns(frames, gen, cons)
    assert tc.generator.loc[GEN_TARGET, "rt_target_ref"] == ""


def test_frames_on_other_timesteps_raise(da):
    frames = da.get_dispatch_frames()
    gen = da.grid.generator.copy()
    gen["da_index"] = pd.array(gen.index, dtype="Int64")
    cons = da.grid.consumer.copy()
    cons["da_index"] = pd.array(cons.index, dtype="Int64")
    with pytest.raises(ValueError, match="expected timesteps"):
        target_columns(frames, gen, cons, timesteps=range(1, N_STEPS + 1))
    frames["pumping"] = frames["pumping"].iloc[:-1]
    with pytest.raises(ValueError, match="'pumping' is not on the expected timesteps"):
        target_columns(frames, gen, cons)
    with pytest.raises(ValueError, match="da_index"):
        target_columns(da.get_dispatch_frames(), da.grid.generator, cons)


def test_incomplete_da_raises(da, tmp_path):
    truncated = tmp_path / "truncated.sqlite"
    with sqlite3.connect(da.db.filename) as src, sqlite3.connect(truncated) as dst:
        src.backup(dst)
    with sqlite3.connect(truncated) as con:
        con.execute("DELETE FROM Res_ObjFunc WHERE timestep >= 20")
    res = powergama.Results.from_existing(truncated, timedelta=1.0)
    with pytest.raises(ValueError, match="not complete"):
        prepare_rt(res, tmp_path / "rt_inputs")


def test_misspelled_reference_raises(prepared, tmp_path):
    data = powergama.GridData()
    data.readGridData(**prepared.grid_files())
    data.readProfileData(
        filename=[datapath_two_area / "profiles.csv", prepared.files["da_solution"]],
        storagevalue_filling=datapath_two_area / "profiles_storval_filling.csv",
        storagevalue_time=datapath_two_area / "profiles_storval_time.csv",
        timerange=range(0, N_STEPS),
        timedelta=1.0,
    )
    data.generator.loc[GEN_TARGET, "rt_target_ref"] = f"da_gen_{GEN_TARGET}x"
    with pytest.raises(RuntimeError, match=f"da_gen_{GEN_TARGET}x"):
        _solve(data, tmp_path / "rt_typo.sqlite", [_rt()])


# --- (f) pins and carried curtailment ------------------------------------------

GEN_WIND = 1  # wind_a


def _rt_tables(da):
    gen = da.grid.generator.copy()
    gen["da_index"] = pd.array(gen.index, dtype="Int64")
    cons = da.grid.consumer.copy()
    cons["da_index"] = pd.array(cons.index, dtype="Int64")
    return gen, cons


def _rows(table, *rows):
    return pd.Series(table.index.isin(rows), index=table.index)


def test_pins_share_the_target_columns(da):
    frames = da.get_dispatch_frames()
    gen, cons = _rt_tables(da)
    targets = target_columns(frames, gen, cons)
    pins = pin_columns(
        frames,
        gen,
        fix=_rows(gen, GEN_PINNED),
        cap=_rows(gen, GEN_TARGET, GEN_STORAGE),
        zero_pmin=_rows(gen, GEN_TARGET),
    )
    for d in (GEN_PINNED, GEN_TARGET, GEN_STORAGE):
        name = f"da_gen_{d}"
        assert np.array_equal(pins.profiles[name].to_numpy(), targets.profiles[name].to_numpy())
        assert targets.generator.loc[d, "rt_target_ref"] == name
    assert (pins.profiles[ZERO_TARGET] == 0.0).all()
    assert pins.profiles.dtypes.eq(np.float64).all()

    refs = pins.generator
    assert list(refs.index) == [GEN_PINNED, GEN_STORAGE, GEN_TARGET]
    assert refs.loc[GEN_PINNED, "pmin_ref"] == refs.loc[GEN_PINNED, "pmax_ref"] == f"da_gen_{GEN_PINNED}"
    assert refs.loc[GEN_TARGET, "pmax_ref"] == f"da_gen_{GEN_TARGET}"
    assert refs.loc[GEN_TARGET, "pmin_ref"] == ZERO_TARGET
    assert pd.isna(refs.loc[GEN_STORAGE, "pmin_ref"])  # cap: pmin_ref is left as it is
    assert (refs["inflow_ref"] == "const").all()
    assert (refs["inflow_fac"] == 1.0).all()

    both = combine_columns(targets, pins)
    assert list(both.profiles.columns).count(f"da_gen_{GEN_TARGET}") == 1
    assert both.generator.loc[GEN_TARGET, "rt_target_ref"] == f"da_gen_{GEN_TARGET}"
    assert both.generator.loc[GEN_TARGET, "pmax_ref"] == f"da_gen_{GEN_TARGET}"
    assert pd.isna(both.generator.loc[GEN_WIND, "pmax_ref"])  # not pinned: left as it is
    assert both.consumer.loc[CONS_FLEX, "rt_target_ref"] == f"da_cons_{CONS_FLEX}"


def test_pin_masks_raise(da):
    frames = da.get_dispatch_frames()
    gen, _ = _rt_tables(da)
    with pytest.raises(ValueError, match="both 'fix' and 'cap'"):
        pin_columns(frames, gen, fix=_rows(gen, GEN_TARGET), cap=_rows(gen, GEN_TARGET))
    with pytest.raises(ValueError, match="not in 'cap'"):
        pin_columns(frames, gen, fix=_rows(gen, GEN_TARGET), zero_pmin=_rows(gen, GEN_TARGET))
    with pytest.raises(ValueError, match="RT table's index"):
        pin_columns(frames, gen, fix=_rows(gen, GEN_TARGET).iloc[:-1])
    with pytest.raises(ValueError, match="boolean"):
        pin_columns(frames, gen, fix=_rows(gen, GEN_TARGET).astype(int))

    no_capacity = gen.copy()
    no_capacity.loc[GEN_TARGET, "pmax"] = 0.0
    with pytest.raises(ValueError, match="pmax"):
        pin_columns(frames, no_capacity, cap=_rows(gen, GEN_TARGET))

    added = pd.concat([gen, gen.loc[[GEN_TARGET]].assign(da_index=pd.NA)], ignore_index=True)
    with pytest.raises(ValueError, match="no DA row"):
        pin_columns(frames, added, fix=_rows(added, len(gen)))

    nan = da.get_dispatch_frames()
    nan["generator_output"].loc[5, GEN_TARGET] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        pin_columns(nan, gen, fix=_rows(gen, GEN_TARGET))
    pin_columns(nan, gen, fix=_rows(gen, GEN_PINNED))  # NaN where no pin needs it is fine


def test_combine_columns_raises(da):
    frames = da.get_dispatch_frames()
    gen, cons = _rt_tables(da)
    targets = target_columns(frames, gen, cons)
    pins = pin_columns(frames, gen, fix=_rows(gen, GEN_TARGET))
    other = pins.profiles.copy()
    other[f"da_gen_{GEN_TARGET}"] *= 1.0 + 1e-15
    with pytest.raises(ValueError, match="different values"):
        combine_columns(targets, type(pins)(profiles=other, generator=pins.generator, consumer=pins.consumer))
    with pytest.raises(ValueError, match="same timesteps"):
        combine_columns(targets, type(pins)(profiles=pins.profiles.iloc[1:], generator=pins.generator, consumer=None))
    # a unit both pinned and capped by the carried curtailment: two pmax_ref for one row
    avail = pd.DataFrame({GEN_TARGET: 300.0}, index=frames["generator_output"].index)
    carry = carry_curtailment_columns(frames, gen, _rows(gen, GEN_TARGET), da_available=avail, rt_available=avail)
    with pytest.raises(ValueError, match="'pmax_ref' is set to different values"):
        combine_columns(targets, pins, carry)


def test_carry_curtailment_formula():
    index = pd.Index([10, 11, 12, 13, 14], name="timestep")
    pmax = 100.0
    da_out = pd.Series([50.0, 60.0, 100.0, 0.0, 20.0], index=index)
    da_avail = pd.Series([50.0, 80.0, 100.0, 30.0, 20.0], index=index)
    rt_avail = pd.Series([40.0, 90.0, 120.0, 10.0, 30.1], index=index)
    # by hand: rt + DA output - DA availability, within [0, pmax], over pmax
    expected = np.array([0.4, 0.7, 1.0, 0.0, 30.1 / 100.0])

    gen = pd.DataFrame({"pmax": [pmax, pmax], "da_index": pd.array([7, 7], dtype="Int64")})
    frames = {"generator_output": pd.DataFrame({7: da_out})}
    mask = pd.Series([True, False])
    refs = carry_curtailment_columns(
        frames, gen, mask, da_available=pd.DataFrame({7: da_avail}), rt_available=pd.DataFrame({7: rt_avail})
    )
    col = refs.profiles["da_carry_7"]
    assert np.allclose(col.to_numpy(), expected, rtol=0, atol=1e-15)
    # bit for bit the reference arithmetic: the carry limit, then / pmax, clip
    reference = ((rt_avail + da_out - da_avail).clip(lower=0.0, upper=float(pmax)) / pmax).clip(0.0, 1.0)
    assert np.array_equal(col.to_numpy(), reference.to_numpy())
    assert list(refs.generator.index) == [0]
    assert refs.generator.loc[0, "pmax_ref"] == "da_carry_7"
    assert pd.isna(refs.generator.loc[0, "pmin_ref"])
    assert refs.generator.loc[0, "inflow_ref"] == "const" and refs.generator.loc[0, "inflow_fac"] == 1.0

    fixed = carry_curtailment_columns(
        frames,
        gen,
        pd.Series([True, True]),
        da_available=pd.DataFrame({7: da_avail}),
        rt_available=pd.DataFrame({7: rt_avail}),
        pin="fix",
    )
    assert list(fixed.profiles.columns) == ["da_carry_7"]  # both rows share it
    assert (fixed.generator["pmin_ref"] == "da_carry_7").all()

    kwargs = {"da_available": pd.DataFrame({7: da_avail}), "rt_available": pd.DataFrame({7: rt_avail})}
    with pytest.raises(ValueError, match="pin must be"):
        carry_curtailment_columns(frames, gen, mask, pin="lock", **kwargs)
    with pytest.raises(ValueError, match="NaN"):
        carry_curtailment_columns(
            frames,
            gen,
            mask,
            da_available=pd.DataFrame({7: da_avail.where(index != 12)}),
            rt_available=kwargs["rt_available"],
        )
    with pytest.raises(ValueError, match="no column"):
        carry_curtailment_columns(
            frames, gen, mask, da_available=pd.DataFrame({8: da_avail}), rt_available=kwargs["rt_available"]
        )
    with pytest.raises(ValueError, match="expected timesteps"):
        carry_curtailment_columns(
            frames, gen, mask, da_available=pd.DataFrame({7: da_avail}).iloc[1:], rt_available=kwargs["rt_available"]
        )


def test_rt_run_with_pins(da, rt_profiles, tmp_path):
    """Fixed units produce their DA output, capped ones at most that, the curtailed one at most its limit."""
    grid = da.grid
    gen = grid.generator
    steps = list(range(N_STEPS))
    wind_pmax = float(gen.loc[GEN_WIND, "pmax"])
    wind_avail = wind_pmax * float(gen.loc[GEN_WIND, "inflow_fac"]) * grid.profiles["wind"].to_numpy()
    rt_wind = pd.read_csv(rt_profiles)["wind"].to_numpy()[steps]
    da_available = pd.DataFrame({GEN_WIND: wind_avail}, index=steps)
    rt_available = pd.DataFrame({GEN_WIND: wind_pmax * float(gen.loc[GEN_WIND, "inflow_fac"]) * rt_wind}, index=steps)
    prepared = prepare_rt(
        da,
        tmp_path / "rt_inputs",
        fix=_rows(gen, GEN_PINNED),
        cap=_rows(gen, GEN_TARGET),
        zero_pmin=_rows(gen, GEN_TARGET),
        carry_curtailment=CarryCurtailment(_rows(gen, GEN_WIND), da_available, rt_available),
        prices=_prices(da),
    )
    tables = prepared.tables["generator"]
    assert tables.loc[GEN_PINNED, "pmin_ref"] == tables.loc[GEN_PINNED, "pmax_ref"] == f"da_gen_{GEN_PINNED}"
    assert tables.loc[GEN_TARGET, "pmin_ref"] == ZERO_TARGET
    assert tables.loc[GEN_WIND, "pmax_ref"] == f"da_carry_{GEN_WIND}"
    assert tables.loc[GEN_WIND, "pmin_ref"] == "const"
    assert tables.loc[GEN_STORAGE, "pmax_ref"] == "const"  # not pinned
    assert list(tables.loc[[GEN_PINNED, GEN_TARGET, GEN_WIND], "inflow_ref"]) == ["const"] * 3
    assert tables.loc[GEN_STORAGE, "inflow_ref"] == "hydro"
    assert {ZERO_TARGET, f"da_carry_{GEN_WIND}"} <= set(prepared.da_solution.columns)

    data = powergama.GridData()
    data.readGridData(**prepared.grid_files())
    data.readProfileData(
        filename=[rt_profiles, prepared.files["da_solution"]],
        storagevalue_filling=datapath_two_area / "profiles_storval_filling.csv",
        storagevalue_time=datapath_two_area / "profiles_storval_time.csv",
        timerange=range(0, N_STEPS),
        timedelta=1.0,
    )
    res = _solve(data, tmp_path / "rt_pins.sqlite", [_rt()])
    assert res.is_complete()
    rt_out = res.get_dispatch_frames()["generator_output"]
    da_out = da.get_dispatch_frames()["generator_output"]
    np.testing.assert_allclose(rt_out[GEN_PINNED], da_out[GEN_PINNED], rtol=1e-9, atol=1e-6)
    assert (rt_out[GEN_TARGET] <= da_out[GEN_TARGET] + 1e-6).all()
    limit = (rt_available[GEN_WIND] + da_out[GEN_WIND] - da_available[GEN_WIND]).clip(0.0, wind_pmax)
    assert (rt_out[GEN_WIND] <= limit + 1e-6).all()
    # the less windy RT run has less wind than DA where DA did not curtail
    assert (rt_out[GEN_WIND] < da_out[GEN_WIND] - 1e-6).any()


# --- (g) initial storage from DA ------------------------------------------------

MODES = {
    "single_period": {},
    "multi_period": {"objective_mode": "multi_period", "horizon_hours": 24, "commit_hours": 24},
}


def _storage_rows(table, *rows):
    rows = rows or (GEN_STORAGE,)
    return pd.Series(table.index.isin(rows), index=table.index)


def _rt_data(prepared, timerange):
    data = powergama.GridData()
    data.readGridData(**prepared.grid_files())
    data.readProfileData(
        filename=[datapath_two_area / "profiles.csv", prepared.files["da_solution"]],
        storagevalue_filling=datapath_two_area / "profiles_storval_filling.csv",
        storagevalue_time=datapath_two_area / "profiles_storval_time.csv",
        timerange=timerange,
        timedelta=1.0,
    )
    return data


@pytest.mark.parametrize("mode", list(MODES))
def test_initial_storage_from_da(mode, tmp_path):
    """RT starts from the filling DA had at the RT run's first timestep."""
    da_data = _data(range(0, 2 * N_STEPS))
    da_lp = powergama.LpProblem(da_data, **MODES[mode])
    da_initial = float(da_lp._storage[GEN_STORAGE])
    da = powergama.Results(da_data, tmp_path / "da.sqlite", replace=True)
    da_lp.solve(da, solver="appsi_highs", solve_args={})
    grid = da.grid
    cap = float(grid.generator.loc[GEN_STORAGE, "storage_cap"])
    assert da_initial == grid.generator.loc[GEN_STORAGE, "storage_ini"] * cap

    prepared = prepare_rt(da, tmp_path / "rt_inputs", initial_storage=_storage_rows(grid.generator), prices=_prices(da))
    gens = prepared.tables["generator"]
    assert gens.loc[GEN_STORAGE, "storage_ini_ref"] == f"da_storage_{GEN_STORAGE}"
    assert (gens.drop(GEN_STORAGE)["storage_ini_ref"] == "const").all()
    column = prepared.da_solution[f"da_storage_{GEN_STORAGE}"]
    level = da.get_dispatch_frames()["storage"][GEN_STORAGE]
    # the start of each timestep: storage_ini at the first, the end of the timestep before after it
    assert column.iloc[0] == grid.generator.loc[GEN_STORAGE, "storage_ini"]
    assert np.array_equal(column.iloc[1:].to_numpy(), (level.iloc[:-1].to_numpy() / cap).clip(0.0, 1.0))

    # an RT run from the DA start starts from the DA initial filling, bit for bit
    rt_lp = powergama.LpProblem(_rt_data(prepared, range(0, N_STEPS)), extensions=[_rt()])
    assert rt_lp._storage[GEN_STORAGE] == da_initial

    # an RT run from timestep 24 starts from the DA filling at the end of timestep 23
    offset = _rt_data(prepared, range(N_STEPS, 2 * N_STEPS))
    rt_lp = powergama.LpProblem(offset, extensions=[_rt()])
    assert rt_lp._storage[GEN_STORAGE] == pytest.approx(level.loc[N_STEPS - 1], rel=1e-15)
    assert level.loc[N_STEPS - 1] != da_initial  # the case is chosen so that this means something
    res = powergama.Results(offset, tmp_path / "rt_offset.sqlite", replace=True)
    rt_lp.solve(res, solver="appsi_highs", solve_args={})
    assert res.is_complete()

    # the same column when prepare_rt writes the second day only
    cut = prepare_rt(
        da, tmp_path / "rt_inputs_cut", timeMaxMin=[N_STEPS, 2 * N_STEPS], initial_storage=_storage_rows(grid.generator)
    )
    pd.testing.assert_series_equal(
        cut.da_solution[column.name], column.loc[N_STEPS:], check_exact=True, check_index_type=False
    )


def test_initial_storage_columns(da):
    frames = da.get_dispatch_frames()
    gen = da.grid.generator.copy()
    gen["da_index"] = pd.array(gen.index, dtype="Int64")
    cap = float(gen.loc[GEN_STORAGE, "storage_cap"])

    # two RT rows from the same DA row share the column
    added = pd.concat([gen, gen.loc[[GEN_STORAGE]]], ignore_index=True)
    refs = initial_storage_columns(
        frames, added, _storage_rows(added, GEN_STORAGE, len(gen)), da_generator=da.grid.generator
    )
    assert list(refs.profiles.columns) == [f"da_storage_{GEN_STORAGE}"]
    assert list(refs.generator["storage_ini_ref"]) == [f"da_storage_{GEN_STORAGE}"] * 2
    assert list(refs.generator.index) == [GEN_STORAGE, len(gen)]

    # DA started from a storage_ini_ref profile: its first row, as is
    da_gen = da.grid.generator.copy()
    da_gen.loc[GEN_STORAGE, "storage_ini_ref"] = "fill"
    da_profiles = da.grid.profiles.assign(fill=np.linspace(0.3, 0.9, len(da.grid.profiles)))
    refs = initial_storage_columns(frames, gen, _storage_rows(gen), da_generator=da_gen, da_profiles=da_profiles)
    column = refs.profiles[f"da_storage_{GEN_STORAGE}"]
    assert column.iloc[0] == 0.3
    level = frames["storage"][GEN_STORAGE]
    assert column.iloc[1] == (level.iloc[0] / cap)
    with pytest.raises(ValueError, match="da_profiles"):
        initial_storage_columns(frames, gen, _storage_rows(gen), da_generator=da_gen)
    da_profiles.loc[da_profiles.index[0], "fill"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        initial_storage_columns(frames, gen, _storage_rows(gen), da_generator=da_gen, da_profiles=da_profiles)

    # storage_ini is clipped to [0, 1]
    da_gen = da.grid.generator.copy()
    da_gen.loc[GEN_STORAGE, "storage_ini"] = 1.5
    refs = initial_storage_columns(frames, gen, _storage_rows(gen), da_generator=da_gen)
    assert refs.profiles[f"da_storage_{GEN_STORAGE}"].iloc[0] == 1.0

    # later timesteps only
    refs = initial_storage_columns(
        frames, gen, _storage_rows(gen), da_generator=da.grid.generator, timesteps=range(5, N_STEPS)
    )
    assert refs.profiles.iloc[0, 0] == level.loc[4] / cap


def test_initial_storage_raises(da):
    frames = da.get_dispatch_frames()
    gen = da.grid.generator.copy()
    gen["da_index"] = pd.array(gen.index, dtype="Int64")
    da_gen = da.grid.generator
    with pytest.raises(ValueError, match="no storage"):
        initial_storage_columns(frames, gen, _storage_rows(gen, GEN_TARGET), da_generator=da_gen)
    added = pd.concat([gen, gen.loc[[GEN_STORAGE]].assign(da_index=pd.NA)], ignore_index=True)
    with pytest.raises(ValueError, match="no DA row"):
        initial_storage_columns(frames, added, _storage_rows(added, len(gen)), da_generator=da_gen)
    with pytest.raises(ValueError, match="boolean"):
        initial_storage_columns(frames, gen, _storage_rows(gen).astype(int), da_generator=da_gen)
    with pytest.raises(ValueError, match="no timestep"):
        initial_storage_columns(frames, gen, _storage_rows(gen), da_generator=da_gen, timesteps=range(1, N_STEPS + 1))
    nan = da.get_dispatch_frames()
    nan["storage"].loc[7, GEN_STORAGE] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        initial_storage_columns(nan, gen, _storage_rows(gen), da_generator=da_gen)
    # a DA row without storage, for an RT row with storage
    no_storage = da_gen.copy()
    no_storage.loc[GEN_STORAGE, "storage_cap"] = 0.0
    with pytest.raises(ValueError, match="no storage"):
        initial_storage_columns(frames, gen, _storage_rows(gen), da_generator=no_storage)


# --- (h) the channel summary ------------------------------------------------------------


@pytest.fixture(scope="module")
def summarized(da, tmp_path_factory):
    """prepare_rt with targets, fix, cap (+ zero pmin), carried curtailment and initial storage."""
    gen = da.grid.generator
    steps = list(range(N_STEPS))
    avail = pd.DataFrame({GEN_WIND: 200.0}, index=steps)
    return prepare_rt(
        da,
        tmp_path_factory.mktemp("rt_summary"),
        fix=_rows(gen, GEN_PINNED),
        cap=_rows(gen, GEN_TARGET),
        zero_pmin=_rows(gen, GEN_TARGET),
        carry_curtailment=CarryCurtailment(
            _rows(gen, GEN_WIND), avail, 0.9 * avail, pin="fix", source="RT wind from profiles_rt.csv"
        ),
        initial_storage=_rows(gen, GEN_STORAGE),
        prices=_prices(da),
    )


def test_summary_files_written(summarized):
    assert summarized.files["summary"].name == "prepare_rt_summary.md"
    assert summarized.files["channels"].name == "channels.csv"
    assert summarized.files["summary"].parent == summarized.files["da_solution"].parent
    assert summarized.files["summary"].is_file() and summarized.files["channels"].is_file()


def test_summary_off(da, tmp_path):
    prepared = prepare_rt(da, tmp_path, summary=False)
    assert "summary" not in prepared.files
    assert not (tmp_path / "prepare_rt_summary.md").exists()
    assert not (tmp_path / "channels.csv").exists()


def test_channels_csv(summarized):
    channels = pd.read_csv(summarized.files["channels"], keep_default_na=False).set_index("name")
    # one row per DA-derived column, nothing else
    assert set(channels.index) == set(summarized.da_solution.columns)
    assert channels["family"].to_dict() == {
        f"da_gen_{GEN_PINNED}": "da_gen",
        f"da_gen_{GEN_WIND}": "da_gen",
        f"da_gen_{GEN_STORAGE}": "da_gen",
        f"da_gen_{GEN_TARGET}": "da_gen",
        f"da_pump_{GEN_STORAGE}": "da_pump",
        f"da_cons_{CONS_FLEX}": "da_cons",
        ZERO_TARGET: "zero",
        f"da_carry_{GEN_WIND}": "da_carry",
        f"da_storage_{GEN_STORAGE}": "da_storage",
        f"da_price_up_{GEN_STORAGE}": "da_price",
        f"da_price_down_{GEN_STORAGE}": "da_price",
        f"da_pump_price_up_{GEN_STORAGE}": "da_price",
        f"da_pump_price_down_{GEN_STORAGE}": "da_price",
        f"da_cons_price_up_{CONS_FLEX}": "da_price",
        f"da_cons_price_down_{CONS_FLEX}": "da_price",
    }
    usage = channels["usage"].to_dict()
    rows = channels["rt_rows"].to_dict()
    assert rows[f"da_gen_{GEN_PINNED}"] == f"generator:{GEN_PINNED}"
    assert usage[f"da_gen_{GEN_PINNED}"] == "target+fix"
    assert usage[f"da_gen_{GEN_TARGET}"] == "target+cap"
    assert usage[f"da_gen_{GEN_WIND}"] == "target" and usage[f"da_carry_{GEN_WIND}"] == "fix"
    assert usage[f"da_gen_{GEN_STORAGE}"] == "target"
    assert usage[f"da_pump_{GEN_STORAGE}"] == "pump_target"
    assert rows[f"da_cons_{CONS_FLEX}"] == f"consumer:{CONS_FLEX}"
    assert usage[f"da_cons_{CONS_FLEX}"] == "cons_target"
    assert rows[ZERO_TARGET] == f"generator:{GEN_TARGET}" and usage[ZERO_TARGET] == "pmin"
    assert usage[f"da_storage_{GEN_STORAGE}"] == "storage_ini"
    assert usage[f"da_price_up_{GEN_STORAGE}"] == "price_up"
    assert usage[f"da_pump_price_down_{GEN_STORAGE}"] == "pump_price_down"
    assert rows[f"da_cons_price_up_{CONS_FLEX}"] == f"consumer:{CONS_FLEX}"
    assert usage[f"da_cons_price_up_{CONS_FLEX}"] == "cons_price_up"
    assert channels.loc[f"da_pump_price_up_{GEN_STORAGE}", "da_row"] == str(GEN_STORAGE)
    assert channels.loc[f"da_gen_{GEN_PINNED}", "usages"] == "target;fix"
    assert channels.loc[ZERO_TARGET, "da_row"] == ""
    assert channels.loc[f"da_carry_{GEN_WIND}", "da_row"] == str(GEN_WIND)
    assert (channels["n_rows"] == 1).all()

    # desc, node, area (through the node table) and type of the referencing rows
    grid_gen = summarized.tables["generator"]
    areas = summarized.tables["node"].set_index("id")["area"]
    for d in (GEN_PINNED, GEN_WIND, GEN_STORAGE, GEN_TARGET):
        row = channels.loc[f"da_gen_{d}"]
        assert row["desc"] == grid_gen.loc[d, "desc"] and row["type"] == grid_gen.loc[d, "type"]
        assert row["node"] == grid_gen.loc[d, "node"] and row["area"] == areas[grid_gen.loc[d, "node"]]
    cons_node = summarized.tables["consumer"].loc[CONS_FLEX, "node"]
    assert channels.loc[f"da_cons_{CONS_FLEX}", "area"] == areas[cons_node]

    # and the summary agrees with the tables written
    tables = {"generator": summarized.tables["generator"], "consumer": summarized.tables["consumer"]}
    for name, row in channels.iterrows():
        for entry in row["rt_rows"].split(";"):
            what, r = entry.split(":")
            assert name in set(tables[what].loc[int(r)].astype(str)), (name, entry)


def test_summary_markdown(summarized):
    text = summarized.files["summary"].read_text(encoding="utf-8")
    assert "**DA timesteps**: 0..23 (24)" in text
    assert "fix 1, cap 1, zero_pmin 1, carry_curtailment (pin=fix) 1, initial_storage 1" in text
    assert "`da_solution.parquet`" in text
    assert "## `da_gen_<d>`: DA generator output (4 channels)" in text
    assert (
        "Used as tracking target (`rt_target_ref`) by 4 generators: "
        "area AREA_A 2 (gas 1, wind 1); area AREA_B 2 (coal 1, hydro 1)"
    ) in text
    assert "Used as fixed (`pmin_ref` = `pmax_ref`) by 1 generator: area AREA_A 1 (gas 1)" in text
    assert "Used as upper bound (`pmax_ref` only) by 1 generator: area AREA_B 1 (coal 1)" in text
    assert "availabilities: RT wind from profiles_rt.csv" in text
    assert "Used as lower bound (`pmin_ref` only) by 1 generator: area AREA_B 1 (coal 1)" in text
    assert "Used as initial filling (`storage_ini_ref`) by 1 generator: area AREA_B 1 (hydro 1)" in text
    assert "prices (c0 10 EUR/MWh, 1 renewable at 2 x c0 down)" in text
    assert (
        "## `da_[pump_|cons_]price_<up|down>_<d>`: deviation price of a storage unit (EUR/MWh) (6 channels)"
    ) in text
    assert "storage values at the DA filling level from DA run" in text
    assert "Used as pumping deviation price up (`rt_pump_price_up_ref`) by 1 generator: area AREA_B 1 (hydro 1)" in text
    assert "Used as flexible-demand deviation price down (consumer `rt_price_down_ref`) by 1 consumer" in text
    # the non-flexible consumer is not covered
    assert "- 1 consumer without any channel:\n  - no flexible load (1): area AREA_A 1" in text
    # counts only, no listing of the rows
    assert "gas_a" not in text and "coal_b" not in text


def test_summarize_channels_direct(da):
    """On RT tables the caller builds: an unreferenced channel, a zero-capacity row, a header, sources."""
    frames = da.get_dispatch_frames()
    gen, cons = _rt_tables(da)
    gen.loc[GEN_WIND, "pmax"] = 0.0
    refs = combine_columns(target_columns(frames, gen, cons), pin_columns(frames, gen, fix=_rows(gen, GEN_PINNED)))
    refs = type(refs)(profiles=refs.profiles.assign(my_column=0.5), generator=refs.generator, consumer=refs.consumer)
    text, channels = summarize_channels(refs, gen, cons, sources={"da_gen": "DA database x"}, header={"Run": "test"})
    assert "- **Run**: test" in text
    assert "- Source: DA database x." in text
    assert "no capacity (`pmax <= 0`) (1)" in text
    other = channels.set_index("name").loc["my_column"]
    assert other["family"] == "other" and other["n_rows"] == 0 and pd.isna(other["da_row"])
    # no node table and no area column: grouped by type only
    assert "Used as fixed (`pmin_ref` = `pmax_ref`) by 1 generator: gas 1" in text

    # a channel keyed on DA row d referenced by a row of another DA row raises
    wrong = gen.copy()
    wrong.loc[GEN_PINNED, "da_index"] = GEN_TARGET
    with pytest.raises(ValueError, match="da_index"):
        summarize_channels(refs, wrong, cons)


def test_clear_refs(da):
    frames = da.get_dispatch_frames()
    gen, cons = _rt_tables(da)
    steps = range(N_STEPS)
    cleared = clear_refs(gen, _rows(gen, GEN_WIND, GEN_STORAGE), ["pmin_ref", "pmax_ref"], timesteps=steps)
    assert list(cleared.profiles.index) == list(steps) and cleared.profiles.columns.empty
    assert list(cleared.generator.index) == [GEN_WIND, GEN_STORAGE]
    assert (cleared.generator == "").all().all()
    assert cleared.consumer.empty
    # it joins the other parts; clearing a cell another part sets raises
    targets = target_columns(frames, gen, cons)
    both = combine_columns(targets, pin_columns(frames, gen, fix=_rows(gen, GEN_PINNED)), cleared)
    assert both.generator.loc[GEN_WIND, "pmax_ref"] == ""
    assert both.generator.loc[GEN_PINNED, "pmax_ref"] == f"da_gen_{GEN_PINNED}"
    with pytest.raises(ValueError, match="different values"):
        combine_columns(pin_columns(frames, gen, fix=_rows(gen, GEN_WIND)), cleared)
    cons_cleared = clear_refs(cons, _rows(cons, CONS_FLEX), "rt_target_ref", timesteps=steps, which="consumer")
    assert cons_cleared.consumer.loc[CONS_FLEX, "rt_target_ref"] == ""
    with pytest.raises(ValueError, match="which"):
        clear_refs(gen, _rows(gen, GEN_WIND), "pmin_ref", timesteps=steps, which="branch")
