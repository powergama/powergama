"""Optional per-timestep objective trace, for comparing the simulation modes.

Write-only instrumentation, gated on an environment variable:

``POWERGAMA_OBJECTIVE_TRACE_CSV``
    Path to a CSV file receiving the per-day/per-hour objective trace.
"""

import os
import warnings
from pathlib import Path

import pandas as pd


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
