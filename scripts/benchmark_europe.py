"""Time a simulation of the Europe 2014 test case, split by phase.

Used as the speed gate for changes to the LP formulation: run it on the base
branch and on the change, same machine, same arguments, and compare.

    poetry run python scripts/benchmark_europe.py --steps 168
    poetry run python scripts/benchmark_europe.py --steps 8760 --json full_year.json
    poetry run python scripts/benchmark_europe.py --mode multi_period --steps 168

The phases are measured by wrapping the LpProblem methods the solve loop calls,
so "solve+other" is everything in the loop that is neither the per-window
update nor the result extraction: mainly the solver call, plus APPSI's update of
the solver's copy of the model.
"""

import argparse
import json
import sqlite3
import tempfile
import time
from pathlib import Path

import powergama

DATAPATH = Path(__file__).resolve().parent.parent / "tests" / "test_data" / "data_europe2014"


def load_grid(steps):
    data = powergama.GridData()
    data.readGridData(
        nodes=DATAPATH / "2014_nodes.csv",
        ac_branches=DATAPATH / "2014_branches.csv",
        dc_branches=DATAPATH / "2014_hvdc.csv",
        generators=DATAPATH / "2014_generators.csv",
        consumers=DATAPATH / "2014_consumers.csv",
    )
    data.readProfileData(
        filename=DATAPATH / "profiles.csv",
        storagevalue_filling=DATAPATH / "profiles_storval_filling.csv",
        storagevalue_time=DATAPATH / "profiles_storval_time.csv",
        timerange=range(0, steps),
        timedelta=1.0,
    )
    return data


def timed(lp, name, totals):
    """Replace lp.<name> with a wrapper that accumulates its wall time.

    Methods the LpProblem does not have are skipped, so the same script times
    older versions too.
    """
    inner = getattr(lp, name, None)
    if inner is None:
        return

    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return inner(*args, **kwargs)
        finally:
            totals[name] = totals.get(name, 0.0) + time.perf_counter() - t0

    setattr(lp, name, wrapper)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=168)
    parser.add_argument("--mode", default="single_period", choices=["single_period", "multi_period"])
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--solver", default="appsi_highs")
    parser.add_argument("--json", default=None, help="write the timings to this file")
    args = parser.parse_args()

    t0 = time.perf_counter()
    data = load_grid(args.steps)
    t_read = time.perf_counter() - t0

    t0 = time.perf_counter()
    lp = powergama.LpProblem(
        data,
        objective_mode=args.mode,
        horizon_hours=args.horizon,
        commit_hours=args.horizon,
    )
    t_build = time.perf_counter() - t0

    totals = {}
    timed(lp, "_updateLpProblem", totals)
    timed(lp, "_storeResultsAndUpdateStorage", totals)
    timed(lp, "_solve_and_store_day_joint", totals)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        res = powergama.Results(data, Path(tmp) / "bench.sqlite3", replace=True)
        t0 = time.perf_counter()
        lp.solve(res, solver=args.solver, solve_args={})
        t_solve = time.perf_counter() - t0
        price = sum(res.getAreaPricesAverage().values())
        con = sqlite3.connect(res.db.filename)
        objective = con.execute("SELECT SUM(value) FROM Res_ObjFunc").fetchone()[0]
        con.close()
        del res

    update = totals.get("_updateLpProblem", 0.0)
    store = totals.get("_storeResultsAndUpdateStorage", 0.0)
    report = {
        "mode": args.mode,
        "steps": args.steps,
        "solver": args.solver,
        "read_s": round(t_read, 2),
        "build_s": round(t_build, 2),
        "solve_loop_s": round(t_solve, 2),
        "update_s": round(update, 2),
        "store_s": round(store, 2),
        "solve_plus_other_s": round(t_solve - update - store, 2),
        "total_s": round(t_build + t_solve, 2),
        "per_step_ms": round(1000 * t_solve / args.steps, 1),
        "objective_sum": objective,
        "area_price_sum": price,
    }
    print(json.dumps(report, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
