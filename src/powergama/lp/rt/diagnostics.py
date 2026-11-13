"""
Real-time solver diagnostics, written by :class:`~powergama.lp.rt.RtExtension`.

This is write-only instrumentation: every entry point returns early unless a debug
path has been configured, and nothing here feeds back into the optimisation. The path
comes from this environment variable:

``POWERGAMA_RT_SOLVER_DEBUG_JSONL``
    Path to a JSONL file receiving one record per solved timestep.

The JSONL path may also be set per-problem by assigning a ``rt_solver_debug_jsonl``
attribute on the :class:`~powergama.GridData` object, which takes precedence over the
environment variable. That attribute is not part of the documented ``GridData`` schema
(it is absent from ``GridData.keys_powergama``) and is injected by downstream callers;
it is read via ``getattr`` with a default so ordinary grid data is unaffected.

Each function takes the :class:`LpProblem` and the extension (``rt``) rather than being
a method, to keep the (large) set of model attributes it reads explicit. It is a reader of
most of the model: Pyomo components (``varGeneration``, the ``p_rt_*`` parameter family,
the index sets, ``OBJ``), simulation state (``_storage``), the RT index
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


def append_payload(rt, payload, *, warning_context=""):
    """Best-effort JSONL append for RT solver diagnostics."""
    if rt.debug_path is None:
        return False
    try:
        with rt.debug_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
        return True
    except Exception as ex:
        if warning_context:
            warnings.warn(f"{warning_context}: {ex}", UserWarning)
        return False


def initialize_debug_stream(rt, grid):
    """Initialize optional RT solver debug stream and write a session header."""
    _rt_debug_raw = str(getattr(grid, "rt_solver_debug_jsonl", "")).strip()
    if not _rt_debug_raw:
        _rt_debug_raw = str(os.environ.get("POWERGAMA_RT_SOLVER_DEBUG_JSONL", "")).strip()
    if not _rt_debug_raw:
        return
    try:
        _dbg_path = Path(_rt_debug_raw)
        _dbg_path.parent.mkdir(parents=True, exist_ok=True)
        rt.debug_path = _dbg_path
        rt.debug_session_id = str(uuid.uuid4())
        _meta = {
            "event": "rt_debug_header",
            "debug_session_id": rt.debug_session_id,
            "debug_header_written_at_utc": str(pd.Timestamp.utcnow().isoformat()),
            # RT always minimises deviation; kept for readers of the header format.
            "dispatch_objective_mode": "deviation",
            "rt_target_tracking_active": True,
            "rt_xborder_flow_penalty_eur_per_mwh": float(rt.xborder_flow_penalty_eur_per_mwh),
        }
        append_payload(
            rt,
            _meta,
            warning_context=f"Failed to write RT solver debug header to '{_rt_debug_raw}'",
        )
    except Exception as ex:
        warnings.warn(f"Failed to initialize RT solver debug stream '{_rt_debug_raw}': {ex}", UserWarning)
        rt.debug_path = None


def append_hourly_debug(lp, rt, timestep):
    """Append per-timestep RT solver internals for redispatch diagnostics."""
    if rt.debug_path is None:
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

    # One row per tracked unit, with its prices and deviations: nothing is grouped by type.
    gen_rows = []
    for i in sorted(int(ii) for ii in rt.ctx.target_gen_indices):
        pmin_now = _val(lp.model.p_gen_pmin[i, 0])
        pmax_now = _val(lp.model.p_gen_pmax[i, 0])
        target_now = _val(lp.model.p_rt_target[i]) or 0.0
        # What replaying the DA target costs at least: the part of it outside this
        # timestep's limits (caps and ramps) must deviate.
        target_clipped = max(
            pmin_now if pmin_now is not None else 0.0,
            min(target_now, pmax_now if pmax_now is not None else target_now),
        )
        da_replay_pos_lb = max(0.0, target_clipped - target_now)
        da_replay_neg_lb = max(0.0, target_now - target_clipped)
        price_up = _val(lp.model.p_rt_deviation_price_gen_pos[i]) or 0.0
        price_down = _val(lp.model.p_rt_deviation_price_gen_neg[i]) or 0.0
        dev_pos = _val(lp.model.varRtTargetDevPos[i]) or 0.0
        dev_neg = _val(lp.model.varRtTargetDevNeg[i]) or 0.0
        gen_rows.append(
            {
                "indx": int(i),
                "node": str(lp._grid.generator.loc[i, "node"]),
                "desc": str(lp._grid.generator.loc[i, "desc"]),
                "type": str(lp._grid.generator.loc[i, "type"]),
                "gen_mw": _val(lp.model.varGeneration[i, 0]) or 0.0,
                "gen_target_mw": target_now,
                "gen_dev_pos_mw": dev_pos,
                "gen_dev_neg_mw": dev_neg,
                "price_up": price_up,
                "price_down": price_down,
                "dev_cost": float(dev_pos * price_up + dev_neg * price_down),
                "da_replay_target_reachable": bool(da_replay_pos_lb <= 1e-9 and da_replay_neg_lb <= 1e-9),
                "da_replay_feasible_pmin_mw": pmin_now,
                "da_replay_feasible_pmax_mw": pmax_now,
                "da_replay_target_clipped_mw": float(target_clipped),
                "da_replay_unavoidable_dev_cost_lb": float(da_replay_pos_lb * price_up + da_replay_neg_lb * price_down),
            }
        )

    pump_rows = []
    for i in sorted(int(ii) for ii in rt.ctx.pump_target_indices):
        storage_cap = _val(lp._grid.generator.loc[i, "storage_cap"]) or 0.0
        pre_storage = _val(lp._storage[i]) or 0.0
        soc_rt = (pre_storage / storage_cap) if storage_cap > 0.0 else 0.0
        ref_target = rt.ctx.pump_target_ref.iloc[int(i)]
        price_up = _val(lp.model.p_rt_deviation_price_pump_pos[i]) or 0.0
        price_down = _val(lp.model.p_rt_deviation_price_pump_neg[i]) or 0.0
        dev_pos = _val(lp.model.varRtPumpTargetDevPos[i]) or 0.0
        dev_neg = _val(lp.model.varRtPumpTargetDevNeg[i]) or 0.0
        pump_rows.append(
            {
                "indx": int(i),
                "node": str(lp._grid.generator.loc[i, "node"]),
                "desc": str(lp._grid.generator.loc[i, "desc"]),
                "type": str(lp._grid.generator.loc[i, "type"]),
                "storage_cap_mwh": float(storage_cap),
                "pre_storage_mwh": float(pre_storage),
                "soc_rt": float(max(0.0, min(1.0, soc_rt))),
                "pump_mw": _val(lp.model.varPump[i, 0]) or 0.0,
                "pump_target_mw": _val(lp.model.p_rt_pump_target[i]) or 0.0,
                "pump_dev_pos_mw": dev_pos,
                "pump_dev_neg_mw": dev_neg,
                "price_up": price_up,
                "price_down": price_down,
                "dev_cost": float(dev_pos * price_up + dev_neg * price_down),
                "rt_pump_target_ref": str(ref_target) if isinstance(ref_target, str) else "",
                "pump_target_profile": _prof(timestep, ref_target),
            }
        )

    flex_rows = []
    for j in sorted(int(jj) for jj in rt.ctx.consumer_target_indices):
        price_up = _val(lp.model.p_rt_deviation_price_flex_pos[j]) or 0.0
        price_down = _val(lp.model.p_rt_deviation_price_flex_neg[j]) or 0.0
        dev_pos = _val(lp.model.varRtFlexLoadTargetDevPos[j]) or 0.0
        dev_neg = _val(lp.model.varRtFlexLoadTargetDevNeg[j]) or 0.0
        flex_rows.append(
            {
                "indx": int(j),
                "node": str(lp._grid.consumer.loc[j, "node"]),
                "flex_mw": _val(lp.model.varFlexLoad[j, 0]) or 0.0,
                "flex_target_mw": _val(lp.model.p_rt_flexload_target[j]) or 0.0,
                "flex_dev_pos_mw": dev_pos,
                "flex_dev_neg_mw": dev_neg,
                "price_up": price_up,
                "price_down": price_down,
                "dev_cost": float(dev_pos * price_up + dev_neg * price_down),
            }
        )

    da_replay_infeasible = sum(1 for row in gen_rows if not row["da_replay_target_reachable"])

    be_balancing_dev = (
        _sum_expr(
            lp.model.varGeneration[i, 0] - lp.model.p_rt_target[i]
            for i in lp.model.s_gen
            if int(i) in rt.ctx.target_gen_indices
        )
        - _sum_expr(
            lp.model.varPump[i, 0] - lp.model.p_rt_pump_target[i]
            for i in lp.model.s_gen_pump
            if int(i) in rt.ctx.pump_target_indices
        )
        + _sum_expr(lp.model.varLoadShed[j, 0] for j in lp.model.s_load)
        - _sum_expr(lp.model.varDumpLoad[j, 0] for j in lp.model.s_load)
        - _sum_expr(
            lp.model.varFlexLoad[j, 0] - lp.model.p_rt_flexload_target[j]
            for j in lp.model.s_load_flex
            if int(j) in rt.ctx.consumer_target_indices
        )
        + _sum_expr(
            rt.ctx.border_ac_sign.get(int(b), 0.0)
            * (_val(lp.model.p_rt_io_target_active_ac[b]) or 0.0)
            * ((_val(lp.model.varAcBranchFlow[b, 0]) or 0.0) - (_val(lp.model.p_rt_io_target_ac[b]) or 0.0))
            for b in lp.model.s_branch_ac
            if int(b) in rt.ctx.idx_border_ac
        )
        + _sum_expr(
            rt.ctx.border_dc_sign.get(int(b), 0.0)
            * (_val(lp.model.p_rt_io_target_active_dc[b]) or 0.0)
            * ((_val(lp.model.varDcBranchFlow[b, 0]) or 0.0) - (_val(lp.model.p_rt_io_target_dc[b]) or 0.0))
            for b in lp.model.s_branch_dc
            if int(b) in rt.ctx.idx_border_dc
        )
    )

    payload = {
        "event": "rt_timestep_debug",
        "timestep": int(timestep),
        "objective": _val(lp.model.OBJ()),
        "flags": {
            "rt_deviation_objective_active": True,
        },
        "residual": {
            "rt_balancing_deviation_mw": float(be_balancing_dev),
        },
        "term_breakdown": {
            "loadshed": _sum_expr(const.loadshedcost * lp.model.varLoadShed[j, 0] for j in lp.model.s_load),
            "dumpload": _sum_expr(const.loadshedcost * lp.model.varDumpLoad[j, 0] for j in lp.model.s_load),
            "gen_dev": float(sum(row["dev_cost"] for row in gen_rows)),
            "flex_dev": float(sum(row["dev_cost"] for row in flex_rows)),
            "pump_dev": float(sum(row["dev_cost"] for row in pump_rows)),
        },
        "active_channels": {
            "rt_pump_target_count": int(len(rt.ctx.pump_target_indices)),
            "rt_target_gen_count": int(len(rt.ctx.target_gen_indices)),
            "rt_io_target_active_ac": int(
                sum(
                    int(_val(lp.model.p_rt_io_target_active_ac[b]) or 0)
                    for b in lp.model.s_branch_ac
                    if int(b) in rt.ctx.idx_border_ac
                )
            ),
            "rt_io_target_active_dc": int(
                sum(
                    int(_val(lp.model.p_rt_io_target_active_dc[b]) or 0)
                    for b in lp.model.s_branch_dc
                    if int(b) in rt.ctx.idx_border_dc
                )
            ),
        },
        "da_replay_check": {
            "reachable_count": int(len(gen_rows) - da_replay_infeasible),
            "infeasible_count": int(da_replay_infeasible),
            "unavoidable_dev_cost_lb": float(sum(row["da_replay_unavoidable_dev_cost_lb"] for row in gen_rows)),
        },
        "rt_gen_rows": gen_rows,
        "rt_pump_rows": pump_rows,
        "rt_flex_rows": flex_rows,
    }

    append_payload(
        rt,
        payload,
        warning_context=f"Failed writing RT solver debug at timestep={timestep}",
    )


def append_solve_start(lp, rt, timesteps, continue_from_last, results):
    """Append the record that opens one solve() call."""
    if rt.debug_path is None:
        return
    last_saved = None
    if continue_from_last and (results is not None):
        try:
            last_saved = results.get_last_timestep_in_results()
        except Exception:
            last_saved = None
    payload = {
        "event": "rt_debug_solve_start",
        "debug_session_id": rt.debug_session_id,
        "written_at_utc": str(pd.Timestamp.utcnow().isoformat()),
        "continue_from_last": bool(continue_from_last),
        "last_saved_timestep": (int(last_saved) if last_saved is not None else None),
        "first_timestep": (int(timesteps[0]) if timesteps else None),
        "last_timestep": (int(timesteps[-1]) if timesteps else None),
        "n_timesteps": int(len(timesteps)),
    }
    append_payload(rt, payload)
