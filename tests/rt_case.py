"""The two-area real-time test case, shared by the RT test modules.

A two-step case: a baseline (day-ahead) run, then RT inputs derived from it --
soft targets on a generator, the storage unit's generation and pumping and the
flexible load, a generator pinned to its baseline output and a consumer with
fixed demand (both as profile columns, ``pmin_ref = pmax_ref`` and ``demand_ref``),
and the inter-area rule held to its DA summed flow (``da_flow_ref``) at a balancing penalty and
within a lock window. The deviation prices are unit data written by ``price_columns``,
with the storage values at the DA filling level.
"""

import sqlite3
from pathlib import Path

import pandas as pd

import powergama
from powergama.lp.rt import RtExtension, apply_refs, da_storage_values, price_columns

datapath_two_area = Path(__file__).parent / "test_data/two_area"

N_STEPS = 24
GEN_PINNED = 0  # gas_a: pinned to its baseline output (pmin_ref = pmax_ref)
GEN_TARGET = 3  # coal_b: soft dispatch target
GEN_STORAGE = 2  # hydro_pump: soft generation and pumping targets
CONS_FIXED = 0  # consumer at a2: demand fixed through demand_ref
CONS_FLEX = 1  # consumer at b2: flexible load with a soft target
BRANCH_BORDER = 1  # a2-b1, the AC branch between the two areas
DCBRANCH_BORDER = 0  # a1-b2 HVDC link
XBORDER_PENALTY = 5.0  # EUR/MWh of deviation of the summed inter-area flow from DA
LOCK_SHARE = 0.6  # lambda: the summed flow within DA - 0.4 x 150 MW and DA + 0.4 x 250 MW
LOCK_WIDTH_FORWARD = 100.0  # (1 - LOCK_SHARE) x ntc_forward
LOCK_WIDTH_BACKWARD = 60.0  # (1 - LOCK_SHARE) x ntc_backward
FIXED_DEMAND = 140.0  # MW, against a profile-driven demand_avg of 150
FEE = 10.0  # c0, the balancing fee of the case (EUR/MWh)
CURTAILMENT_FACTOR = 2.0


def two_area_data():
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
        filename=datapath_two_area / "profiles.csv",
        storagevalue_filling=datapath_two_area / "profiles_storval_filling.csv",
        storagevalue_time=datapath_two_area / "profiles_storval_time.csv",
        timerange=range(0, N_STEPS),
        timedelta=1.0,
    )
    return data


def read_table(res, table):
    with sqlite3.connect(res.db.filename) as con:
        return pd.read_sql_query(f"SELECT * FROM {table} ORDER BY timestep, indx", con)  # nosec B608


def series(df, indx, col):
    return df[df["indx"] == indx].set_index("timestep")[col]


def add_prices(data, res_da, *, fee=FEE, curtailment_factor=CURTAILMENT_FACTOR):
    """Write the deviation prices of every row into *data*, from c0 and the DA storage values.

    Wind is priced as a renewable; the storage unit and the flexible load at their DA
    storage value. Every RT row is its DA row (``da_index``). Returns the price columns.
    """
    for name in ("generator", "consumer"):
        table = getattr(data, name)
        table["da_index"] = pd.array(table.index, dtype="Int64")
    values = da_storage_values(res_da)
    refs = price_columns(
        data.generator,
        data.consumer,
        fee=fee,
        renewable=data.generator["type"].eq("wind"),
        curtailment_factor=curtailment_factor,
        storage_value=values["generator"],
        flex_value=values["consumer"],
    )
    data.generator = apply_refs(data.generator, refs.generator)
    data.consumer = apply_refs(data.consumer, refs.consumer)
    prices = refs.profiles.loc[list(data.timerange)].set_axis(data.profiles.index)
    data.profiles = pd.concat([data.profiles, prices], axis=1)
    return refs


def build_rt_case(tmp_path, fee=FEE):
    """A two-step case: a baseline (day-ahead) run, then RT inputs derived from it.

    Returns the RT GridData, with soft targets, the generator pin and the fixed demand as
    profile columns, the deviation prices at c0 = *fee* (:func:`add_prices`), and the
    inter-area rule's DA summed flow (``da_flow_0``) with a balancing penalty of
    :data:`XBORDER_PENALTY` and a lock share of :data:`LOCK_SHARE`.
    """
    base = two_area_data()
    lp = powergama.LpProblem(base)
    res = powergama.Results(base, tmp_path / "da.sqlite", replace=True)
    lp.solve(res, solver="appsi_highs", solve_args={})
    gen = read_table(res, "Res_Generators")
    pump = read_table(res, "Res_Pumping")
    flex = read_table(res, "Res_FlexibleLoad")
    ac = read_table(res, "Res_Branches")
    dc = read_table(res, "Res_DcBranches")

    data = two_area_data()
    steps = list(data.timerange)

    # Soft targets, as fractions the way the profile columns are interpreted.
    data.profiles["rt_target_coal"] = (
        series(gen, GEN_TARGET, "output").reindex(steps).values / data.generator.loc[GEN_TARGET, "pmax"]
    )
    data.profiles["rt_gen_hydro"] = (
        series(gen, GEN_STORAGE, "output").reindex(steps).values / data.generator.loc[GEN_STORAGE, "pmax"]
    )
    data.profiles["rt_pump_hydro"] = (
        series(pump, GEN_STORAGE, "output").reindex(steps).fillna(0.0).values
        / data.generator.loc[GEN_STORAGE, "pump_cap"]
    )
    data.profiles["rt_target_flex"] = (
        series(flex, CONS_FLEX, "demand").reindex(steps).values / data.consumer.loc[CONS_FLEX, "demand_avg"]
    )
    data.generator["rt_target_ref"] = ""
    data.generator.loc[GEN_TARGET, "rt_target_ref"] = "rt_target_coal"
    data.generator.loc[GEN_STORAGE, "rt_target_ref"] = "rt_gen_hydro"
    data.generator["rt_pump_target_ref"] = ""
    data.generator.loc[GEN_STORAGE, "rt_pump_target_ref"] = "rt_pump_hydro"
    data.consumer["rt_target_ref"] = ""
    data.consumer.loc[CONS_FLEX, "rt_target_ref"] = "rt_target_flex"

    # A generator pinned to its baseline output, and a fixed demand: profile columns,
    # as fractions of pmax and demand_avg. The pin is the whole availability, so the
    # inflow must not scale it again.
    gen_pin = series(gen, GEN_PINNED, "output").reindex(steps).values
    data.profiles["rt_pin_gas"] = gen_pin / data.generator.loc[GEN_PINNED, "pmax"]
    data.generator.loc[GEN_PINNED, ["pmin_ref", "pmax_ref", "inflow_ref"]] = ["rt_pin_gas", "rt_pin_gas", "const"]
    data.generator.loc[GEN_PINNED, "inflow_fac"] = 1.0
    data.profiles["rt_fixed_demand"] = FIXED_DEMAND / data.consumer.loc[CONS_FIXED, "demand_avg"]
    data.consumer.loc[CONS_FIXED, "demand_ref"] = "rt_fixed_demand"

    # The inter-area rule held to its DA summed flow (both branches go from A to B).
    ac_flow = series(ac, BRANCH_BORDER, "flow").reindex(steps).values
    dc_flow = series(dc, DCBRANCH_BORDER, "flow").reindex(steps).values
    data.profiles["da_flow_0"] = ac_flow + dc_flow
    data.inter_area_rules.loc[0, ["da_flow_ref", "balancing_penalty", "lock_share"]] = ["da_flow_0", XBORDER_PENALTY, LOCK_SHARE]

    add_prices(data, res, fee=fee)

    # The real-time deviation: less wind than the day-ahead run planned for.
    data.profiles["wind"] = 0.7 * data.profiles["wind"]
    return data, {"ac_flow": ac_flow, "dc_flow": dc_flow, "gen_pin": gen_pin}


def solve_rt_case(data, dbfile):
    rt = RtExtension()
    lp = powergama.LpProblem(data, extensions=[rt])
    res = powergama.Results(data, dbfile, replace=True)
    lp.solve(res, solver="appsi_highs", solve_args={})
    return lp, res
