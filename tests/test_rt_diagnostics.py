"""The optional RT debug stream (issue #46).

``POWERGAMA_RT_SOLVER_DEBUG_JSONL`` makes RtExtension write one JSON record per
event: a header, one per solve() call, and one per solved timestep. It is
write-only instrumentation, so nothing else in the suite would notice it break.
Runs on the two-area RT case, which exercises every kind of target and the inter-area rule.
"""

import json
import sqlite3

import pandas as pd
import pytest
from rt_case import (
    BRANCH_BORDER,
    CONS_FLEX,
    DCBRANCH_BORDER,
    GEN_STORAGE,
    GEN_TARGET,
    LOCK_SHARE,
    LOCK_WIDTH_BACKWARD,
    LOCK_WIDTH_FORWARD,
    N_STEPS,
    build_rt_case,
    read_table,
    series,
    solve_rt_case,
)

ENV = "POWERGAMA_RT_SOLVER_DEBUG_JSONL"
TOL = 1e-6


def _run(tmp_path, monkeypatch, name="rt"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    data, _ = build_rt_case(tmp_path)
    path = tmp_path / f"{name}.jsonl"
    monkeypatch.setenv(ENV, str(path))
    lp, res = solve_rt_case(data, tmp_path / f"{name}.sqlite")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return lp, res, records


def _objective(res):
    with sqlite3.connect(res.db.filename) as con:
        return pd.read_sql_query("SELECT timestep, value FROM Res_ObjFunc ORDER BY timestep", con)


def _stable(record):
    """A record without the per-session id and wall-clock timestamps."""
    if isinstance(record, dict):
        return {k: _stable(v) for k, v in record.items() if k != "debug_session_id" and not k.endswith("_utc")}
    if isinstance(record, list):
        return [_stable(v) for v in record]
    return record


@pytest.fixture
def debug_run(tmp_path, monkeypatch):
    return _run(tmp_path, monkeypatch)


def test_stream_has_header_solve_start_and_one_record_per_timestep(debug_run):
    lp, _, records = debug_run
    rt = lp._extensions[0]

    events = [r["event"] for r in records]
    assert events == ["rt_debug_header", "rt_debug_solve_start"] + ["rt_timestep_debug"] * N_STEPS
    assert [r["timestep"] for r in records[2:]] == list(range(N_STEPS))

    header, start = records[0], records[1]
    assert header["debug_session_id"] == start["debug_session_id"] == rt.debug_session_id != ""
    assert "rt_balancing_fee_eur_per_mwh" not in header  # the prices are unit data
    assert "rt_xborder_flow_penalty_eur_per_mwh" not in header
    assert header["inter_area_rules_balancing_penalised"] == [0]
    assert header["inter_area_rules_locked"] == [0]
    assert start["n_timesteps"] == N_STEPS
    assert (start["first_timestep"], start["last_timestep"]) == (0, N_STEPS - 1)


def test_timestep_objective_matches_stored_result(debug_run):
    _, res, records = debug_run
    stored = _objective(res).set_index("timestep")["value"]
    logged = pd.Series({r["timestep"]: r["objective"] for r in records[2:]})
    pd.testing.assert_series_equal(logged.sort_index(), stored.sort_index(), check_names=False, atol=TOL)


def test_active_channels_match_the_case(debug_run):
    """Generator targets (one plain, one storage), a pump target, one penalised and locked inter-area rule."""
    _, _, records = debug_run
    for r in records[2:]:
        assert r["active_channels"] == {
            "rt_pump_target_count": 1,
            "rt_target_gen_count": 2,
            "rt_rule_balancing_penalty_count": 1,
            "rt_rule_lock_count": 1,
        }, f"timestep {r['timestep']}"
        rows = r["rt_pump_rows"]
        assert [row["indx"] for row in rows] == [GEN_STORAGE]
        assert 0.0 <= rows[0]["soc_rt"] <= 1.0


def test_stream_is_deterministic(tmp_path, monkeypatch):
    """Apart from the session id and timestamps, two runs write the same stream."""
    _, _, first = _run(tmp_path / "a", monkeypatch)
    _, _, second = _run(tmp_path / "b", monkeypatch)
    assert first[0]["debug_session_id"] != second[0]["debug_session_id"]
    assert [_stable(r) for r in first] == [_stable(r) for r in second]


def test_append_debug_event_adds_a_record(debug_run):
    lp, _, records = debug_run
    rt = lp._extensions[0]
    rt.append_debug_event({"event": "caller_marker", "value": 1})
    lines = rt.debug_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(records) + 1
    assert json.loads(lines[-1]) == {"event": "caller_marker", "value": 1}


def test_no_stream_without_the_variable(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    data, _ = build_rt_case(tmp_path)
    lp, res = solve_rt_case(data, tmp_path / "rt.sqlite")
    rt = lp._extensions[0]

    assert rt.debug_path is None
    assert not list(tmp_path.rglob("*.jsonl"))
    rt.append_debug_event({"event": "ignored"})  # a no-op, not an error
    assert len(_objective(res)) == N_STEPS


def test_rule_rows(debug_run):
    """One row per held inter-area rule: the RT and DA summed flows, the deviation and its cost."""
    lp, res, records = debug_run
    ac = read_table(res, "Res_Branches")
    dc = read_table(res, "Res_DcBranches")
    da_flow = lp._grid.profiles["da_flow_0"]
    for r in records[2:]:
        t = r["timestep"]
        (row,) = r["rt_rule_rows"]
        assert (row["rule"], row["area_from"], row["area_to"], row["da_flow_ref"]) == (
            0,
            "AREA_A",
            "AREA_B",
            "da_flow_0",
        )
        transfer = series(ac, BRANCH_BORDER, "flow")[t] + series(dc, DCBRANCH_BORDER, "flow")[t]
        assert row["transfer_mw"] == pytest.approx(transfer, abs=TOL)
        assert row["da_flow_mw"] == da_flow.loc[t]
        assert row["dev_pos_mw"] - row["dev_neg_mw"] == pytest.approx(transfer - da_flow.loc[t], abs=TOL)
        assert r["term_breakdown"]["xborder_dev"] == pytest.approx(row["dev_cost"], abs=TOL)
        assert row["dev_cost"] == pytest.approx(row["balancing_penalty"] * (row["dev_pos_mw"] + row["dev_neg_mw"]))
        assert row["lock_share"] == LOCK_SHARE
        assert row["lock_lb_mw"] == pytest.approx(da_flow.loc[t] - LOCK_WIDTH_BACKWARD)
        assert row["lock_ub_mw"] == pytest.approx(da_flow.loc[t] + LOCK_WIDTH_FORWARD)
        assert row["lock_lb_mw"] - TOL <= row["transfer_mw"] <= row["lock_ub_mw"] + TOL


def test_rows_are_per_unit(debug_run):
    """One row per tracked unit, with its prices; the cost terms are their sums. No type buckets."""
    lp, _, records = debug_run
    m = lp.model
    last = records[-1]
    assert "redispatch_cost_attribution_eur" not in last
    gen_rows = last["rt_gen_rows"]
    assert [row["indx"] for row in gen_rows] == sorted([GEN_TARGET, GEN_STORAGE])
    assert [row["indx"] for row in last["rt_pump_rows"]] == [GEN_STORAGE]
    assert [row["indx"] for row in last["rt_flex_rows"]] == [CONS_FLEX]
    # the last record is the model's last timestep
    for row in gen_rows:
        i = row["indx"]
        assert row["price_up"] == m.p_rt_deviation_price_gen_pos[i].value
        assert row["price_down"] == m.p_rt_deviation_price_gen_neg[i].value
    for r in records[2:]:
        terms = r["term_breakdown"]
        for key, rows in (("gen_dev", "rt_gen_rows"), ("pump_dev", "rt_pump_rows"), ("flex_dev", "rt_flex_rows")):
            assert terms[key] == pytest.approx(sum(row["dev_cost"] for row in r[rows]), abs=TOL)
        check = r["da_replay_check"]
        assert check["reachable_count"] + check["infeasible_count"] == len(r["rt_gen_rows"])
