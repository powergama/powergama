"""RT peak tranches: ``peak_tranche_columns`` and ``prepare_rt(peak_tranches=...)``.

On ``two_area`` with gas_a (400 MW) split into two price blocks of one ``block_group``
(rows 0, 250 MW, and 4, 150 MW) and coal_b (row 3) as a unit of one: each unit gets one extra row at the peak price,
whose ``pmax_ref`` is the unit's DA headroom.
"""

import numpy as np
import pandas as pd
import pytest
from rt_case import N_STEPS, datapath_two_area

import powergama
from powergama.lp.rt import (
    ZERO_TARGET,
    DeviationPrices,
    PeakTranches,
    RtExtension,
    combine_columns,
    peak_tranche_columns,
    prepare_rt,
    rt_tables_from_da,
    target_columns,
)

GAS, WIND, HYDRO, COAL, GAS_B2 = 0, 1, 2, 3, 4
PRICE = 300.0
STEPS = list(range(N_STEPS))


def _blocks_data(tmp_path, profiles=None):
    table = pd.read_csv(datapath_two_area / "generators.csv")
    table["block_group"] = ["gas_a", "", "", ""]
    block = table.iloc[[GAS]].assign(desc="gas_a_b2", pmax=150, fuelcost=20)
    table.loc[GAS, "pmax"] = 250
    table = pd.concat([table, block], ignore_index=True)
    path = tmp_path / "generators.csv"
    table.to_csv(path, index=False)
    data = powergama.GridData()
    data.readGridData(
        nodes=datapath_two_area / "nodes.csv",
        ac_branches=datapath_two_area / "branches.csv",
        dc_branches=datapath_two_area / "dcbranches.csv",
        generators=path,
        consumers=datapath_two_area / "consumers.csv",
        inter_area_rules=datapath_two_area / "inter_area_rules.csv",
    )
    data.readProfileData(
        filename=profiles if profiles is not None else datapath_two_area / "profiles.csv",
        storagevalue_filling=datapath_two_area / "profiles_storval_filling.csv",
        storagevalue_time=datapath_two_area / "profiles_storval_time.csv",
        timerange=range(0, N_STEPS),
        timedelta=1.0,
    )
    return data


def _solve(data, dbfile, extensions=()):
    lp = powergama.LpProblem(data, extensions=list(extensions))
    res = powergama.Results(data, dbfile, replace=True)
    lp.solve(res, solver="appsi_highs", solve_args={})
    return res


@pytest.fixture(scope="module")
def da(tmp_path_factory):
    path = tmp_path_factory.mktemp("da_blocks")
    return _solve(_blocks_data(path), path / "da.sqlite")


def _rows(table, *rows):
    return pd.Series(table.index.isin(rows), index=table.index)


def _tranches(da, **kwargs):
    tables = rt_tables_from_da(da)
    gen = tables["generator"]
    out, refs = peak_tranche_columns(
        da.get_dispatch_frames(), gen, _rows(gen, GAS, GAS_B2, COAL), price=PRICE, **kwargs
    )
    return gen, out, refs


def test_rows_order_and_pmax(da):
    gen, out, refs = _tranches(da)
    assert list(out.index) == [*gen.index, 5, 6]
    # the RT rows are unchanged (fuelcost may become float)
    pd.testing.assert_frame_equal(out.loc[gen.index], gen, check_dtype=False)
    assert out["da_index"].dtype == "Int64"
    assert out.loc[5:6, "da_index"].isna().all()
    # groups by their first row: gas_a (rows 0 and 4) first, then coal_b
    assert list(out.loc[[5, 6], "desc"]) == ["gas_a", "coal_b"]
    assert list(out.loc[[5, 6], "pmax"]) == [400.0, 300.0]
    assert list(out.loc[[5, 6], "fuelcost"]) == [PRICE, PRICE]
    assert list(out.loc[[5, 6], "pmax_ref"]) == [f"da_headroom_{GAS}", f"da_headroom_{COAL}"]
    assert list(out.loc[[5, 6], "pmin_ref"]) == ["", ""]
    assert list(out.loc[[5, 6], "inflow_ref"]) == ["const", "const"]
    assert list(out.loc[[5, 6], "inflow_fac"]) == [1.0, 1.0]
    assert list(out.loc[[5, 6], "block_group"]) == ["gas_a", ""]
    # no ramp limit by default (the unit's 0.25 is not inherited)
    assert out.loc[[5, 6], ["ramp_up_pu", "ramp_down_pu"]].isna().all().all()
    assert out.dtypes.drop(["da_index", "fuelcost"]).equals(gen.dtypes.drop(["da_index", "fuelcost"]))
    assert list(refs.profiles.columns) == [f"da_headroom_{GAS}", f"da_headroom_{COAL}"]
    assert list(refs.generator.index) == [5, 6]
    assert refs.consumer.empty


def test_headroom_formula(da):
    _, _, refs = _tranches(da)
    out = da.get_dispatch_frames()["generator_output"]
    assert out[GAS].gt(0).any() and out[GAS_B2].gt(0).any(), "both blocks run in DA somewhere"
    gas = ((400.0 - (0.0 + out[GAS] + out[GAS_B2])) / 400.0).clip(0.0, 1.0)
    coal = ((300.0 - (0.0 + out[COAL])) / 300.0).clip(0.0, 1.0)
    assert np.array_equal(refs.profiles[f"da_headroom_{GAS}"].to_numpy(), gas.to_numpy())
    assert np.array_equal(refs.profiles[f"da_headroom_{COAL}"].to_numpy(), coal.to_numpy())
    hand = 1.0 - (out[GAS].to_numpy() + out[GAS_B2].to_numpy()) / 400.0
    np.testing.assert_allclose(refs.profiles[f"da_headroom_{GAS}"].to_numpy(), hand, rtol=0, atol=1e-12)
    assert refs.profiles.dtypes.eq(np.float64).all()
    assert list(refs.profiles.index) == STEPS


def test_tranches_track_zero(da):
    _, out, refs = _tranches(da)
    targets = target_columns(da.get_dispatch_frames(), out, rt_tables_from_da(da)["consumer"])
    assert list(targets.generator.loc[[5, 6], "rt_target_ref"]) == [ZERO_TARGET, ZERO_TARGET]
    assert targets.generator.loc[GAS_B2, "rt_target_ref"] == f"da_gen_{GAS_B2}"
    combined = combine_columns(targets, refs)
    assert {ZERO_TARGET, f"da_headroom_{GAS}"} <= set(combined.profiles.columns)


@pytest.mark.parametrize("ramp_pu", [None, 0.0, 0.6])
def test_ramp_pu(da, ramp_pu):
    _, out, _ = _tranches(da, ramp_pu=ramp_pu)
    ramps = out.loc[[5, 6], ["ramp_up_pu", "ramp_down_pu"]]
    if ramp_pu is None:
        assert ramps.isna().all().all()
    else:
        assert (ramps == ramp_pu).all().all()
    # the unit's own rows keep their ramp limits
    assert list(out.loc[[GAS, COAL], "ramp_up_pu"]) == [0.25, 0.25]


@pytest.mark.parametrize("ramp_pu", [-0.1, float("nan"), float("inf")])
def test_ramp_pu_invalid(da, ramp_pu):
    with pytest.raises(ValueError, match="ramp_pu must be"):
        _tranches(da, ramp_pu=ramp_pu)


def test_without_block_group_every_row_is_a_unit(da):
    gen, out, _ = _tranches(da, block_group="no_such_column")
    assert list(out.loc[len(gen) :, "pmax"]) == [250.0, 300.0, 150.0]
    assert list(out.loc[len(gen) :, "pmax_ref"]) == [f"da_headroom_{d}" for d in (GAS, COAL, GAS_B2)]


def test_zero_capacity_group_is_skipped(da):
    tables = rt_tables_from_da(da)
    gen = tables["generator"].copy()
    gen.loc[COAL, "pmax"] = 0.0
    out, refs = peak_tranche_columns(da.get_dispatch_frames(), gen, _rows(gen, COAL, GAS, GAS_B2), price=PRICE)
    assert len(out) == len(gen) + 1
    assert list(refs.profiles.columns) == [f"da_headroom_{GAS}"]


def test_raises(da):
    frames = da.get_dispatch_frames()
    gen = rt_tables_from_da(da)["generator"]
    with pytest.raises(ValueError, match="only part of block_group 'gas_a'"):
        peak_tranche_columns(frames, gen, _rows(gen, GAS), price=PRICE)
    with pytest.raises(ValueError, match="pump_cap > 0|storage_cap > 0"):
        peak_tranche_columns(frames, gen, _rows(gen, HYDRO), price=PRICE)
    no_da = gen.copy()
    no_da.loc[COAL, "da_index"] = pd.NA
    with pytest.raises(ValueError, match="no DA row"):
        peak_tranche_columns(frames, no_da, _rows(gen, COAL), price=PRICE)
    shifted = gen.copy()
    shifted.index = shifted.index + 1
    with pytest.raises(ValueError, match="already in use"):
        peak_tranche_columns(frames, shifted, _rows(shifted, COAL + 1), price=PRICE)


# --- prepare_rt(peak_tranches=...) ------------------------------------------------------


@pytest.fixture(scope="module")
def prepared(da, tmp_path_factory):
    """The DA units are fixed to their DA output (and the storage unit to its generation)."""
    gen = da.grid.generator
    return prepare_rt(
        da,
        tmp_path_factory.mktemp("rt_inputs"),
        fix=_rows(gen, GAS, GAS_B2, COAL, HYDRO),
        peak_tranches=PeakTranches(_rows(gen, GAS, GAS_B2, COAL), PRICE, source="test price"),
        prices=DeviationPrices(fee=10.0, renewable=_rows(gen, WIND)),
    )


def test_prepare_rt_tables(da, prepared):
    gen = prepared.tables["generator"]
    assert list(gen.index) == [*da.grid.generator.index, 5, 6]
    assert list(gen.loc[[5, 6], "rt_target_ref"]) == [ZERO_TARGET, ZERO_TARGET]
    assert list(gen.loc[[5, 6], "pmax_ref"]) == [f"da_headroom_{GAS}", f"da_headroom_{COAL}"]
    # the other options' masks are on the DA rows and leave the tranches alone
    assert gen.loc[GAS_B2, "pmin_ref"] == gen.loc[GAS_B2, "pmax_ref"] == f"da_gen_{GAS_B2}"
    _, _, refs = _tranches(da)
    for col in refs.profiles.columns:
        assert np.array_equal(prepared.da_solution[col].to_numpy(), refs.profiles[col].to_numpy())
    # the tranches are priced around their own price, as any unit around its fuelcost
    assert list(gen.loc[[5, 6], "rt_price_up"]) == [10.0 + PRICE] * 2
    assert list(gen.loc[[5, 6], "rt_price_down"]) == [10.0 - PRICE] * 2
    assert list(gen.loc[[5, 6], "rt_price_up_ref"]) == ["const"] * 2


def test_prepare_rt_summary_family(prepared):
    channels = pd.read_csv(prepared.files["channels"])
    head = channels[channels["family"] == "da_headroom"].set_index("name")
    assert list(head.index) == [f"da_headroom_{GAS}", f"da_headroom_{COAL}"]
    assert list(head["rt_rows"]) == ["generator:5", "generator:6"]
    assert list(head["usage"]) == ["cap", "cap"]
    assert list(head["da_row"]) == [GAS, COAL]
    text = prepared.files["summary"].read_text(encoding="utf-8")
    assert "## `da_headroom_<d>`: peak-tranche headroom (2 channels)" in text
    assert "test price" in text
    assert "peak_tranches 3 (+2 tranche rows at 300 EUR/MWh, free ramp)" in text


def test_rt_run_tranche_within_headroom(da, prepared, tmp_path):
    """RT has three times the DA demand in area A: the tranches cover it, within the headroom."""
    data = powergama.GridData()
    data.readGridData(**prepared.grid_files())
    data.readProfileData(
        filename=[datapath_two_area / "profiles.csv", prepared.files["da_solution"]],
        storagevalue_filling=datapath_two_area / "profiles_storval_filling.csv",
        storagevalue_time=datapath_two_area / "profiles_storval_time.csv",
        timerange=range(0, N_STEPS),
        timedelta=1.0,
    )
    assert np.isnan(data.generator.loc[5, "ramp_up_pu"])  # free ramp (the default)
    data.consumer.loc[0, "demand_avg"] *= 3.0
    res = _solve(data, tmp_path / "rt.sqlite", [RtExtension()])
    assert res.is_complete()
    rt_out = res.get_dispatch_frames()["generator_output"]
    da_out = da.get_dispatch_frames()["generator_output"]
    for tranche, rows, cap in ((5, [GAS, GAS_B2], 400.0), (6, [COAL], 300.0)):
        headroom = cap - sum((da_out[r] for r in rows), pd.Series(0.0, index=da_out.index))
        assert (rt_out[tranche] <= headroom.clip(lower=0.0) + 1e-6).all()
        assert rt_out[tranche].gt(1.0).any(), tranche
        # the blocks stay at their DA output
        for r in rows:
            np.testing.assert_allclose(rt_out[r], da_out[r], rtol=1e-9, atol=1e-6)
