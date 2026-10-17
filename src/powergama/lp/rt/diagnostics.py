"""
Real-time solver diagnostics and objective tracing for :class:`~powergama.LpProblem`.

This is write-only instrumentation: every entry point returns early unless the problem
is in real-time mode and a debug path has been configured, and nothing here feeds back
into the optimisation. It is gated on these environment variables:

``POWERGAMA_RT_SOLVER_DEBUG_JSONL``
    Path to a JSONL file receiving one record per solved timestep.
``POWERGAMA_OBJECTIVE_TRACE_CSV``
    Path to a CSV file receiving the per-day/per-hour objective trace.

The JSONL path may also be set per-problem by assigning a ``rt_solver_debug_jsonl``
attribute on the :class:`~powergama.GridData` object, which takes precedence over the
environment variable. That attribute is not part of the documented ``GridData`` schema
(it is absent from ``GridData.keys_powergama``) and is injected by downstream callers;
it is read via ``getattr`` with a default so ordinary grid data is unaffected.

Each function takes the :class:`LpProblem` as its first argument rather than being a
method, to keep the (large) set of model attributes it reads explicit. It is a reader of
most of the model: Pyomo components (``varGeneration``, the ``p_rt_*`` parameter family,
the index sets, ``OBJ``), simulation state (``_storage``, ``_gen_prev``), the RT index
sets and ``_grid``. That read surface is why this is instrumentation living alongside the
model rather than an independent module.
"""

import json
import os
import uuid
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from ... import constants as const


def write_objective_trace(lp):
    """Write optional per-hour objective trace CSV for mode parity diagnostics."""
    trace_path = str(os.environ.get("POWERGAMA_OBJECTIVE_TRACE_CSV", "")).strip()
    if not trace_path or not lp._hourly_objective_trace:
        return
    try:
        out_path = Path(trace_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        trace_df = pd.DataFrame(
            lp._hourly_objective_trace,
            columns=["timestep", "day_index", "hour_in_day", "objective_value"],
        )
        trace_df["objective_mode"] = str(lp._objective_mode)
        trace_df["horizon_hours"] = int(lp._horizon_hours)
        trace_df.to_csv(out_path, index=False)
        print(f"Wrote objective trace CSV: {out_path}")
    except Exception as ex:
        warnings.warn(f"Failed to write POWERGAMA_OBJECTIVE_TRACE_CSV='{trace_path}': {ex}", UserWarning)


def append_payload(lp, payload, *, warning_context=""):
    """Best-effort JSONL append for RT solver diagnostics."""
    if not lp._is_rt or lp._rt_solver_debug_path is None:
        return False
    try:
        with lp._rt_solver_debug_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
        return True
    except Exception as ex:
        if warning_context:
            warnings.warn(f"{warning_context}: {ex}", UserWarning)
        return False


def initialize_debug_stream(lp, grid):
    """Initialize optional RT solver debug stream and write a session header."""
    _rt_debug_raw = str(getattr(grid, "rt_solver_debug_jsonl", "")).strip()
    if not _rt_debug_raw:
        _rt_debug_raw = str(os.environ.get("POWERGAMA_RT_SOLVER_DEBUG_JSONL", "")).strip()
    if not (lp._is_rt and _rt_debug_raw):
        return
    try:
        _dbg_path = Path(_rt_debug_raw)
        _dbg_path.parent.mkdir(parents=True, exist_ok=True)
        lp._rt_solver_debug_path = _dbg_path
        lp._rt_debug_session_id = str(uuid.uuid4())
        _meta = {
            "event": "rt_debug_header",
            "debug_session_id": lp._rt_debug_session_id,
            "debug_header_written_at_utc": str(pd.Timestamp.utcnow().isoformat()),
            "dispatch_objective_mode": lp._rt_dispatch_objective_mode,
            "rt_target_tracking_active": bool(lp._rt_target_tracking_active),
            "rt_balancing_fee_eur_per_mwh": float(lp._rt_balancing_fee_eur_per_mwh),
            "rt_xborder_flow_penalty_eur_per_mwh": float(lp._rt_xborder_flow_penalty_eur_per_mwh),
        }
        append_payload(
            lp,
            _meta,
            warning_context=f"Failed to write RT solver debug header to '{_rt_debug_raw}'",
        )
    except Exception as ex:
        warnings.warn(f"Failed to initialize RT solver debug stream '{_rt_debug_raw}': {ex}", UserWarning)
        lp._rt_solver_debug_path = None


def append_event(lp, payload):
    """Public wrapper for optional debug append used by workflow helpers."""
    append_payload(lp, payload)


def append_hourly_debug(lp, timestep):
    """Append per-timestep RT solver internals for redispatch diagnostics."""
    if not lp._is_rt or lp._rt_solver_debug_path is None:
        return

    def _val(x):
        try:
            v = pyo.value(x, exception=False)
        except Exception:
            v = x
        if v is None:
            return None
        try:
            fv = float(v)
            if not np.isfinite(fv):
                return None
            return fv
        except Exception:
            return None

    def _sum_expr(items):
        s = 0.0
        for term in items:
            tv = _val(term)
            if tv is not None:
                s += tv
        return float(s)

    def _prof(ts, ref_name):
        if not isinstance(ref_name, str):
            return None
        ref = str(ref_name).strip()
        if not ref:
            return None
        try:
            if ref in lp._grid.profiles.columns:
                return float(lp._grid.profiles.loc[int(ts), ref])
        except Exception:
            return None
        return None

    storage_rows = []
    da_reference_storage_mismatch_cost = 0.0
    for i in sorted(int(ii) for ii in lp._rt_storage_target_indices):
        storage_cap = _val(lp._grid.generator.loc[i, "storage_cap"]) or 0.0
        pre_storage = _val(lp._storage[i]) or 0.0
        soc_rt = (pre_storage / storage_cap) if storage_cap > 0.0 else 0.0
        ref_target = lp._rt_storage_target_ref.iloc[int(i)] if lp._rt_storage_target_ref is not None else ""
        soc_da_target = _prof(timestep, ref_target)
        dt_h = float(lp.timeDelta) if np.isfinite(lp.timeDelta) and lp.timeDelta > 0.0 else 1.0
        da_gen_target_mw = _val(lp.model.p_rt_target[i]) or 0.0
        da_pump_target_mw = 0.0
        eff = (
            float(pd.to_numeric(lp._grid.generator.loc[i, "pump_efficiency"], errors="coerce") or 0.0)
            if int(i) in lp.model.s_gen_pump
            else 0.0
        )
        lhs_da_mwh = (_val(lp.model.p_rt_storage_balance_rhs[i]) or 0.0) - dt_h * da_gen_target_mw
        if int(i) in lp.model.s_gen_pump:
            lhs_da_mwh += dt_h * eff * da_pump_target_mw
        target_mwh = _val(lp.model.p_rt_storage_target[i]) or 0.0
        mismatch_da_mwh = lhs_da_mwh - target_mwh
        c_pos = _val(lp.model.p_rt_deviation_price_storage_pos[i]) or 0.0
        c_neg = _val(lp.model.p_rt_deviation_price_storage_neg[i]) or 0.0
        mismatch_da_cost = c_pos * max(0.0, mismatch_da_mwh) + c_neg * max(0.0, -mismatch_da_mwh)
        da_reference_storage_mismatch_cost += float(mismatch_da_cost)
        row = {
            "indx": int(i),
            "node": str(lp._grid.generator.loc[i, "node"]),
            "desc": str(lp._grid.generator.loc[i, "desc"]),
            "type": str(lp._grid.generator.loc[i, "type"]),
            "storage_cap_mwh": float(storage_cap),
            "pre_storage_mwh": float(pre_storage),
            "soc_rt": float(max(0.0, min(1.0, soc_rt))),
            "target_storage_mwh": _val(lp.model.p_rt_storage_target[i]) or 0.0,
            "storage_dev_pos_mwh": _val(lp.model.varRtStorageTargetDevPos[i]) or 0.0,
            "storage_dev_neg_mwh": _val(lp.model.varRtStorageTargetDevNeg[i]) or 0.0,
            "coef_storage_pos": _val(lp.model.p_rt_deviation_price_storage_pos[i]) or 0.0,
            "coef_storage_neg": _val(lp.model.p_rt_deviation_price_storage_neg[i]) or 0.0,
            "gen_mw": _val(lp.model.varGeneration[i]) or 0.0,
            "pump_mw": (_val(lp.model.varPump[i]) or 0.0) if int(i) in lp.model.s_gen_pump else 0.0,
            "gen_target_mw": _val(lp.model.p_rt_target[i]) or 0.0,
            "pump_target_mw": 0.0,
            "gen_cost_mwh": _val(lp.model.p_gen_cost[i]) or 0.0,
            "pump_cost_mwh": (_val(lp.model.p_genpump_cost[i]) or 0.0) if int(i) in lp.model.s_gen_pump else 0.0,
            "rt_storage_target_ref": str(ref_target) if isinstance(ref_target, str) else "",
            "soc_da_target_profile": float(soc_da_target) if soc_da_target is not None else None,
            "da_ref_storage_balance_lhs_mwh": float(lhs_da_mwh),
            "da_ref_storage_target_mwh": float(target_mwh),
            "da_ref_storage_balance_mismatch_mwh": float(mismatch_da_mwh),
            "da_ref_storage_mismatch_cost": float(mismatch_da_cost),
        }
        storage_rows.append(row)

    gas_rows = []
    da_replay_unavoidable_gas_dev_cost_lb = 0.0
    da_replay_infeasible_gas_count = 0
    for i in sorted(
        int(ii) for ii in lp._rt_target_gen_indices if int(ii) in (lp._idx_rt_normal_gas | lp._idx_rt_peak_gen)
    ):
        pmin_now = _val(lp.model.p_gen_pmin[i])
        pmax_now = _val(lp.model.p_gen_pmax[i])
        target_now = _val(lp.model.p_rt_target[i]) or 0.0
        target_clipped = max(
            pmin_now if pmin_now is not None else 0.0,
            min(target_now, pmax_now if pmax_now is not None else target_now),
        )
        da_replay_pos_lb = max(0.0, target_clipped - target_now)
        da_replay_neg_lb = max(0.0, target_now - target_clipped)
        coef_pos = _val(lp.model.p_rt_deviation_price_gen_pos[i]) or 0.0
        coef_neg = _val(lp.model.p_rt_deviation_price_gen_neg[i]) or 0.0
        da_replay_cost_lb = da_replay_pos_lb * coef_pos + da_replay_neg_lb * coef_neg
        ramp_prev = float(lp._gen_prev[i]) if i < len(lp._gen_prev) and np.isfinite(lp._gen_prev[i]) else None
        ramp_up = None
        if lp._ramp_up_pu is not None:
            rv = lp._ramp_up_pu[i]
            ramp_up = float(rv) if np.isfinite(rv) else None
        ramp_down = None
        if lp._ramp_down_pu is not None:
            rv = lp._ramp_down_pu[i]
            ramp_down = float(rv) if np.isfinite(rv) else None
        da_replay_reachable = (abs(da_replay_pos_lb) <= 1e-9) and (abs(da_replay_neg_lb) <= 1e-9)
        if not da_replay_reachable:
            da_replay_infeasible_gas_count += 1
        da_replay_unavoidable_gas_dev_cost_lb += float(da_replay_cost_lb)
        gas_rows.append(
            {
                "indx": int(i),
                "node": str(lp._grid.generator.loc[i, "node"]),
                "desc": str(lp._grid.generator.loc[i, "desc"]),
                "is_peak_rt_block": bool(int(i) in lp._idx_rt_peak_gen),
                "gen_mw": _val(lp.model.varGeneration[i]) or 0.0,
                "gen_target_mw": _val(lp.model.p_rt_target[i]) or 0.0,
                "gen_dev_pos_mw": _val(lp.model.varRtTargetDevPos[i]) or 0.0,
                "gen_dev_neg_mw": _val(lp.model.varRtTargetDevNeg[i]) or 0.0,
                "coef_gen_pos": coef_pos,
                "coef_gen_neg": coef_neg,
                "gen_cost_mwh": _val(lp.model.p_gen_cost[i]) or 0.0,
                "da_replay_target_reachable": bool(da_replay_reachable),
                "da_replay_prev_gen_mw": ramp_prev,
                "da_replay_ramp_up_pu": ramp_up,
                "da_replay_ramp_down_pu": ramp_down,
                "da_replay_feasible_pmin_mw": pmin_now,
                "da_replay_feasible_pmax_mw": pmax_now,
                "da_replay_target_clipped_mw": float(target_clipped),
                "da_replay_unavoidable_dev_pos_mw": float(da_replay_pos_lb),
                "da_replay_unavoidable_dev_neg_mw": float(da_replay_neg_lb),
                "da_replay_unavoidable_dev_cost_lb": float(da_replay_cost_lb),
            }
        )

    gas_actual_dev_cost = sum(
        float(row["gen_dev_pos_mw"]) * float(row["coef_gen_pos"])
        + float(row["gen_dev_neg_mw"]) * float(row["coef_gen_neg"])
        for row in gas_rows
    )
    storage_actual_dev_cost = sum(
        float(row["storage_dev_pos_mwh"]) * float(row["coef_storage_pos"])
        + float(row["storage_dev_neg_mwh"]) * float(row["coef_storage_neg"])
        for row in storage_rows
    )
    storage_da_injected_abs_balance_mismatch_mwh = sum(
        abs(float(row["da_ref_storage_balance_mismatch_mwh"])) for row in storage_rows
    )
    storage_da_injected_signed_balance_mismatch_mwh = sum(
        float(row["da_ref_storage_balance_mismatch_mwh"]) for row in storage_rows
    )

    be_balancing_dev = (
        _sum_expr(
            lp.model.varGeneration[i] - lp.model.p_rt_target[i]
            for i in lp.model.s_gen
            if int(i) in lp._rt_target_gen_indices
        )
        + _sum_expr(lp.model.varLoadShed[j] for j in lp.model.s_load)
        - _sum_expr(lp.model.varDumpLoad[j] for j in lp.model.s_load)
        - _sum_expr(
            lp.model.varFlexLoad[j] - lp.model.p_rt_flexload_target[j]
            for j in lp.model.s_load_flex
            if int(j) in lp._rt_consumer_target_indices
        )
        + _sum_expr(
            lp._border_ac_sign.get(int(b), 0.0)
            * (_val(lp.model.p_rt_io_target_active_ac[b]) or 0.0)
            * ((_val(lp.model.varAcBranchFlow[b]) or 0.0) - (_val(lp.model.p_rt_io_target_ac[b]) or 0.0))
            for b in lp.model.s_branch_ac
            if int(b) in lp._idx_border_ac
        )
        + _sum_expr(
            lp._border_dc_sign.get(int(b), 0.0)
            * (_val(lp.model.p_rt_io_target_active_dc[b]) or 0.0)
            * ((_val(lp.model.varDcBranchFlow[b]) or 0.0) - (_val(lp.model.p_rt_io_target_dc[b]) or 0.0))
            for b in lp.model.s_branch_dc
            if int(b) in lp._idx_border_dc
        )
    )

    redispatch_cost_attribution_eur: dict[str, float] = {
        "wind": 0.0,
        "solar": 0.0,
        "gas": 0.0,
        "nuclear": 0.0,
        "hydro_ror": 0.0,
        "biomass": 0.0,
        "fossil_other": 0.0,
        "storage_generation": 0.0,
        "other": 0.0,
        "storage_soc": 0.0,
        "pump_dev": 0.0,
        "flex_dev": 0.0,
    }
    redispatch_mw_attribution: dict[str, dict[str, float]] = {}

    def _bucket_for_gen(ii: int) -> str:
        if ii in lp._idx_rt_storage_gens:
            return "storage_generation"
        if ii in lp._idx_rt_wind:
            return "wind"
        if ii in lp._idx_rt_solar:
            return "solar"
        if ii in (lp._idx_rt_normal_gas | lp._idx_rt_peak_gen):
            return "gas"
        if ii in lp._idx_rt_nuclear:
            return "nuclear"
        if ii in lp._idx_rt_hydro_ror:
            return "hydro_ror"
        if ii in lp._idx_rt_biomass:
            return "biomass"
        if ii in lp._idx_rt_fossil_other:
            return "fossil_other"
        return "other"

    for i in lp.model.s_gen:
        ii = int(i)
        if ii not in lp._rt_target_gen_indices:
            continue
        dev_pos = _val(lp.model.varRtTargetDevPos[i]) or 0.0
        dev_neg = _val(lp.model.varRtTargetDevNeg[i]) or 0.0
        coef_pos = _val(lp.model.p_rt_deviation_price_gen_pos[i]) or 0.0
        coef_neg = _val(lp.model.p_rt_deviation_price_gen_neg[i]) or 0.0
        this_cost = float(dev_pos * coef_pos + dev_neg * coef_neg)
        bucket = _bucket_for_gen(ii)
        redispatch_cost_attribution_eur[bucket] = float(redispatch_cost_attribution_eur.get(bucket, 0.0) + this_cost)
        if bucket not in redispatch_mw_attribution:
            redispatch_mw_attribution[bucket] = {
                "abs_dev_mw": 0.0,
                "signed_dev_mw": 0.0,
                "up_dev_mw": 0.0,
                "down_dev_mw": 0.0,
            }
        redispatch_mw_attribution[bucket]["abs_dev_mw"] += float(dev_pos + dev_neg)
        redispatch_mw_attribution[bucket]["signed_dev_mw"] += float(dev_pos - dev_neg)
        redispatch_mw_attribution[bucket]["up_dev_mw"] += float(dev_pos)
        redispatch_mw_attribution[bucket]["down_dev_mw"] += float(dev_neg)

    redispatch_cost_attribution_eur["storage_soc"] = float(storage_actual_dev_cost)
    redispatch_cost_attribution_eur["flex_dev"] = _sum_expr(
        lp.model.p_rt_deviation_price_flex_pos[j] * lp.model.varRtFlexLoadTargetDevPos[j]
        + lp.model.p_rt_deviation_price_flex_neg[j] * lp.model.varRtFlexLoadTargetDevNeg[j]
        for j in lp.model.s_load_flex
        if int(j) in lp._rt_consumer_target_indices
    )

    gen_dev_cost_total = _sum_expr(
        lp.model.p_rt_deviation_price_gen_pos[i] * lp.model.varRtTargetDevPos[i]
        + lp.model.p_rt_deviation_price_gen_neg[i] * lp.model.varRtTargetDevNeg[i]
        for i in lp.model.s_gen
        if int(i) in lp._rt_target_gen_indices
    )
    redispatch_cost_attribution_eur["meta_be_gen_dev_cost_accounted"] = float(
        sum(
            redispatch_cost_attribution_eur.get(k, 0.0)
            for k in [
                "wind",
                "solar",
                "gas",
                "nuclear",
                "hydro_ror",
                "biomass",
                "fossil_other",
                "storage_generation",
                "other",
            ]
        )
    )
    redispatch_cost_attribution_eur["meta_model_gen_dev_cost_total"] = float(gen_dev_cost_total)

    payload = {
        "event": "rt_timestep_debug",
        "timestep": int(timestep),
        "objective": _val(lp.model.OBJ()),
        "flags": {
            "rt_deviation_objective_active": bool(lp._rt_deviation_objective_active),
        },
        "residual": {
            "rt_balancing_deviation_mw": float(be_balancing_dev),
        },
        "term_breakdown": {
            "loadshed": _sum_expr(const.loadshedcost * lp.model.varLoadShed[j] for j in lp.model.s_load),
            "dumpload": _sum_expr(const.loadshedcost * lp.model.varDumpLoad[j] for j in lp.model.s_load),
            "gen_dev": _sum_expr(
                lp.model.p_rt_deviation_price_gen_pos[i] * lp.model.varRtTargetDevPos[i]
                + lp.model.p_rt_deviation_price_gen_neg[i] * lp.model.varRtTargetDevNeg[i]
                for i in lp.model.s_gen
                if int(i) in lp._rt_target_gen_indices
            ),
            "flex_dev": _sum_expr(
                lp.model.p_rt_deviation_price_flex_pos[j] * lp.model.varRtFlexLoadTargetDevPos[j]
                + lp.model.p_rt_deviation_price_flex_neg[j] * lp.model.varRtFlexLoadTargetDevNeg[j]
                for j in lp.model.s_load_flex
                if int(j) in lp._rt_consumer_target_indices
            ),
            "storage_dev": _sum_expr(
                lp.model.p_rt_deviation_price_storage_pos[i] * lp.model.varRtStorageTargetDevPos[i]
                + lp.model.p_rt_deviation_price_storage_neg[i] * lp.model.varRtStorageTargetDevNeg[i]
                for i in lp.model.s_gen_storage
                if int(i) in lp._rt_storage_target_indices
            ),
            "da_ref_storage_mismatch_cost": float(da_reference_storage_mismatch_cost),
        },
        "active_channels": {
            "rt_storage_target_count": int(len(lp._rt_storage_target_indices)),
            "rt_target_gen_count": int(len(lp._rt_target_gen_indices)),
            "foreign_gen_lock_rows": int(
                sum(
                    1
                    for i in lp.model.s_gen
                    if lp._foreign_gen_lock is not None and (int(timestep), int(i)) in lp._foreign_gen_lock.index
                )
            ),
            "foreign_cons_lock_rows": int(
                sum(
                    1
                    for j in lp.model.s_load
                    if lp._foreign_cons_lock is not None and (int(timestep), int(j)) in lp._foreign_cons_lock.index
                )
            ),
            "rt_io_target_active_ac": int(
                sum(
                    int(_val(lp.model.p_rt_io_target_active_ac[b]) or 0)
                    for b in lp.model.s_branch_ac
                    if int(b) in lp._idx_border_ac
                )
            ),
            "rt_io_target_active_dc": int(
                sum(
                    int(_val(lp.model.p_rt_io_target_active_dc[b]) or 0)
                    for b in lp.model.s_branch_dc
                    if int(b) in lp._idx_border_dc
                )
            ),
        },
        "da_injection_check": {
            "storage_check_available": True,
            "gas_da_injected_dev_cost": 0.0,
            "storage_da_injected_dev_cost": float(da_reference_storage_mismatch_cost),
            "storage_da_injected_abs_balance_mismatch_mwh": float(storage_da_injected_abs_balance_mismatch_mwh),
            "storage_da_injected_signed_balance_mismatch_mwh": float(storage_da_injected_signed_balance_mismatch_mwh),
            "da_replay_gas_reachable_count": int(len(gas_rows) - da_replay_infeasible_gas_count),
            "da_replay_gas_infeasible_count": int(da_replay_infeasible_gas_count),
            "da_replay_unavoidable_gas_dev_cost_lb": float(da_replay_unavoidable_gas_dev_cost_lb),
            "actual_gas_rt_dev_cost": float(gas_actual_dev_cost),
            "actual_storage_rt_dev_cost": float(storage_actual_dev_cost),
            "actual_total_rt_dev_cost": float(gas_actual_dev_cost + storage_actual_dev_cost),
        },
        "redispatch_cost_attribution_eur": redispatch_cost_attribution_eur,
        "redispatch_mw_attribution": redispatch_mw_attribution,
        "rt_storage_rows": storage_rows,
        "rt_gen_rows": gas_rows,
    }

    append_payload(
        lp,
        payload,
        warning_context=f"Failed writing RT solver debug at timestep={timestep}",
    )


def append_multi_period_debug(lp, m, hi, timestep):
    """Append per-timestep RT debug rows for multi_period joint solves."""
    if not lp._is_rt or lp._rt_solver_debug_path is None:
        return

    def _val(x):
        try:
            v = pyo.value(x, exception=False)
            if v is None:
                return 0.0
            fv = float(v)
            if not np.isfinite(fv):
                return 0.0
            return fv
        except Exception:
            return 0.0

    def _prof(ts, ref_name):
        if not isinstance(ref_name, str):
            return None
        ref = str(ref_name).strip()
        if not ref:
            return None
        try:
            if ref in lp._grid.profiles.columns:
                return float(lp._grid.profiles.loc[int(ts), ref])
        except Exception:
            return None
        return None

    storage_rows = []
    for i in sorted(int(ii) for ii in lp._rt_storage_target_indices):
        storage_cap = float(pd.to_numeric(lp._grid.generator.loc[i, "storage_cap"], errors="coerce") or 0.0)
        rt_storage_mwh = _val(m.varStorage[i, hi]) if i in lp.model.s_gen_storage else 0.0
        soc_rt = (rt_storage_mwh / storage_cap) if storage_cap > 0.0 else 0.0

        target_storage_mwh = 0.0
        ref_target = lp._rt_storage_target_ref.iloc[int(i)] if lp._rt_storage_target_ref is not None else ""
        soc_da_target = _prof(timestep, ref_target)
        if lp._rt_storage_target_ref is not None:
            ref = lp._rt_storage_target_ref.iloc[int(i)]
            rv = _prof(timestep, ref)
            if rv is not None and storage_cap > 0.0:
                target_storage_mwh = max(0.0, storage_cap * max(0.0, min(1.0, rv)))

        gen_mw = _val(m.varGeneration[i, hi])
        pump_mw = _val(m.varPump[i, hi]) if i in lp.model.s_gen_pump else 0.0

        storage_rows.append(
            {
                "indx": int(i),
                "node": str(lp._grid.generator.loc[i, "node"]),
                "desc": str(lp._grid.generator.loc[i, "desc"]),
                "rt_storage_mwh": float(rt_storage_mwh),
                "target_storage_mwh": float(target_storage_mwh),
                "storage_dev_pos_mwh": float(max(0.0, rt_storage_mwh - target_storage_mwh)),
                "storage_dev_neg_mwh": float(max(0.0, target_storage_mwh - rt_storage_mwh)),
                "soc_rt": float(max(0.0, min(1.0, soc_rt))),
                "rt_storage_target_ref": str(ref_target) if isinstance(ref_target, str) else "",
                "soc_da_target_profile": float(soc_da_target) if soc_da_target is not None else None,
                "gen_mw": float(gen_mw),
                "pump_mw": float(pump_mw),
            }
        )

    gas_rows = []
    rt_gas_idx = lp._idx_rt_normal_gas | lp._idx_rt_peak_gen
    for i in sorted(int(ii) for ii in lp._rt_target_gen_indices if int(ii) in rt_gas_idx):
        gen_mw = _val(m.varGeneration[i, hi])
        target_mw = 0.0
        if lp._rt_target_ref is not None:
            ref = lp._rt_target_ref.iloc[int(i)]
            rv = _prof(timestep, ref)
            if rv is not None:
                pmax_base = float(pd.to_numeric(lp._grid.generator.loc[i, "pmax"], errors="coerce") or 0.0)
                pmax_fac = 1.0
                if lp._pmax_ref is not None:
                    pmax_ref = lp._pmax_ref.iloc[int(i)]
                    pmax_ref_val = _prof(timestep, pmax_ref)
                    if pmax_ref_val is not None:
                        pmax_fac = float(pmax_ref_val)
                target_mw = max(0.0, pmax_base * pmax_fac * rv)

        gas_rows.append(
            {
                "indx": int(i),
                "node": str(lp._grid.generator.loc[i, "node"]),
                "desc": str(lp._grid.generator.loc[i, "desc"]),
                "is_peak_rt_block": bool(int(i) in lp._idx_rt_peak_gen),
                "gen_mw": float(gen_mw),
                "gen_target_mw": float(target_mw),
                "gen_dev_pos_mw": float(max(0.0, gen_mw - target_mw)),
                "gen_dev_neg_mw": float(max(0.0, target_mw - gen_mw)),
            }
        )

    payload = {
        "event": "rt_timestep_debug",
        "debug_path_mode": "multi_period",
        "timestep": int(timestep),
        "hour_in_window": int(hi),
        "da_injection_check": {
            "storage_check_available": False,
            "gas_da_injected_dev_cost": 0.0,
            "storage_da_injected_dev_cost": None,
            "storage_da_injected_abs_balance_mismatch_mwh": None,
            "storage_da_injected_signed_balance_mismatch_mwh": None,
            "actual_gas_rt_dev_cost": None,
            "actual_storage_rt_dev_cost": None,
            "actual_total_rt_dev_cost": None,
            "note": "multi_period debug mode does not compute DA-reference storage mismatch diagnostics",
        },
        "redispatch_cost_attribution_eur": {},
        "redispatch_mw_attribution": {},
        "rt_storage_rows": storage_rows,
        "rt_gen_rows": gas_rows,
    }

    append_payload(
        lp,
        payload,
        warning_context=f"Failed writing multi_period RT solver debug at timestep={timestep}",
    )
