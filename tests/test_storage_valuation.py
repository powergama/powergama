"""
How stored energy is valued, in both formulations (issue #40).

Storage has no cost per MWh generated. The energy left in storage at the end of
the LP is valued at the storage value, so drawing 1 MWh costs exactly its storage
value -- once. Every objective term is a cost rate in EUR/h, so prices are in
EUR/MWh whatever the timestep length.
"""

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

import powergama

datapath = Path(__file__).parent / "test_data/data_9bus"
TOL = 1e-6
MODES = {
    "single_period": {"objective_mode": "single_period"},
    "multi_period_h1": {"objective_mode": "multi_period", "horizon_hours": 1, "commit_hours": 1},
}


def _9bus(timerange=range(48), timedelta=1.0):
    data = powergama.GridData()
    data.readGridData(
        nodes=datapath / "9busmod_nodes.csv",
        ac_branches=datapath / "9busmod_branches.csv",
        dc_branches=None,
        generators=datapath / "9busmod_generators.csv",
        consumers=datapath / "9busmod_consumers.csv",
    )
    data.readProfileData(
        filename=datapath / "9busmod_profiles.csv",
        storagevalue_filling=datapath / "9busmod_profiles_storval_filling.csv",
        storagevalue_time=datapath / "9busmod_profiles_storval_time.csv",
        timerange=timerange,
        timedelta=timedelta,
    )
    return data


def _solve(data, dbfile, **kwargs):
    lp = powergama.LpProblem(data, **kwargs)
    res = powergama.Results(data, dbfile, replace=True)
    lp.solve(res, solver="appsi_highs", solve_args={})
    return res


def _read(res, query):
    with sqlite3.connect(res.db.filename) as con:
        return pd.read_sql_query(query, con)


@pytest.mark.parametrize("mode", MODES)
def test_storage_draw_costs_its_storage_value_once(tmp_path, mode):
    """Where the storage plant is the marginal generator, the price is its storage value.

    The CSP plant has no fuel cost, so when it is dispatched strictly between zero and
    its availability, the price at its node is its marginal cost: the storage value.
    The double-counting formulation gave twice that.
    """
    data = _9bus()
    res = _solve(data, tmp_path / f"{mode}.sqlite", **MODES[mode])
    (csp,) = data.getIdxGeneratorsWithStorage()
    gen = data.generator.loc[csp]
    node = data.node.index[data.node["id"] == gen["node"]][0]

    output = _read(res, f"SELECT timestep, output FROM Res_Generators WHERE indx={csp}").set_index("timestep")["output"]
    storage = _read(res, "SELECT timestep, storage, marginalprice FROM Res_Storage").set_index("timestep")
    price = _read(res, f"SELECT timestep, nodalprice FROM Res_Nodes WHERE indx={node}").set_index("timestep")

    marginal = 0
    for t in list(res.timerange)[1:]:
        inflow = gen["pmax"] * gen["inflow_fac"] * data.profiles.loc[t, gen["inflow_ref"]]
        available = min(gen["pmax"], inflow + storage.loc[t - 1, "storage"] / data.timeDelta)
        if 1e-3 < output[t] < available - 1e-3:
            marginal += 1
            assert price.loc[t, "nodalprice"] == pytest.approx(storage.loc[t, "marginalprice"], abs=1e-6), (
                f"t={t}: price at the storage plant's node is not its storage value"
            )
    assert marginal > 0, "fixture never makes the storage plant marginal"


@pytest.mark.parametrize("mode", MODES)
def test_prices_and_objective_do_not_depend_on_timestep_length(tmp_path, mode):
    """The objective is a cost rate, so one timestep gives the same answer at 1 h and 15 min.

    Only the first timestep is compared: from the second on, the storage has moved by
    a quarter as much and the runs legitimately diverge.
    """
    hour = _solve(_9bus(range(1), 1.0), tmp_path / "hour.sqlite", **MODES[mode])
    quarter = _solve(_9bus(range(1), 0.25), tmp_path / "quarter.sqlite", **MODES[mode])

    assert quarter.db.getResultGeneratorPowerAll(0) == pytest.approx(hour.db.getResultGeneratorPowerAll(0))
    for n in range(len(hour.grid.node)):
        assert quarter.getNodalPrices(n) == pytest.approx(hour.getNodalPrices(n))
    obj = "SELECT value FROM Res_ObjFunc"
    assert _read(quarter, obj)["value"].tolist() == pytest.approx(_read(hour, obj)["value"].tolist())


@pytest.mark.parametrize("mode", ["single_period", "multi_period_h1", "multi_period_h24"])
def test_storage_bounded_and_spills_only_when_full(tmp_path, mode):
    """0 <= level <= capacity, and inflow is spilled only from a full reservoir.

    Run with a small reservoir and high inflow so that it does fill and spill.

    The second holds for one-timestep solves only. Over a longer window the LP knows
    the reservoir will be full before the window ends, so spilling early or late
    leaves the same level at the end: both are optimal, and which one the solver
    returns is arbitrary.
    """
    data = _9bus()
    (csp,) = data.getIdxGeneratorsWithStorage()
    data.generator.loc[csp, "storage_cap"] = 400
    data.generator.loc[csp, "inflow_fac"] = 1.5
    kwargs = dict(MODES.get(mode, {"objective_mode": "multi_period", "horizon_hours": 24, "commit_hours": 24}))
    res = _solve(data, tmp_path / f"{mode}.sqlite", **kwargs)

    storage = _read(res, "SELECT timestep, storage FROM Res_Storage").set_index("timestep")["storage"]
    spilled = _read(res, f"SELECT timestep, inflow_spilled FROM Res_Generators WHERE indx={csp}").set_index("timestep")[
        "inflow_spilled"
    ]
    capacity = data.generator.loc[csp, "storage_cap"]

    assert storage.min() >= -TOL
    assert storage.max() <= capacity + TOL
    assert (spilled > TOL).any(), "fixture never spills"
    if mode != "multi_period_h24":
        assert (storage[spilled > TOL] >= capacity - 1e-4).all(), "spilled while the reservoir had room"


def test_unknown_storage_value_curve_raises(tmp_path):
    """A storage value curve name that is not in the table is an error, not another curve."""
    data = _9bus(range(2))
    (csp,) = data.getIdxGeneratorsWithStorage()
    data.generator.loc[csp, "storval_filling_ref"] = "no_such_curve"
    lp = powergama.LpProblem(data)
    res = powergama.Results(data, tmp_path / "bad_curve.sqlite", replace=True)
    with pytest.raises(KeyError, match="no_such_curve"):
        lp.solve(res, solver="appsi_highs", solve_args={})
