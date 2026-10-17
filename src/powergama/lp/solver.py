"""Solver construction for the two solve paths.

Nothing here knows anything about the power system model beyond needing a
``ConcreteModel`` to hand to a persistent solver, so it is the one piece of
``LpProblem`` that genuinely stands alone.

Note the duplication that is preserved rather than fixed. Both paths read the same
``POWERGAMA_HIGHS_*`` environment variables, and they do it differently:

======================================  ====================  ===================
                                        single-period          multi-period
======================================  ====================  ===================
``POWERGAMA_HIGHS_RANDOM_SEED``         warns on bad input     ignored on bad input
``POWERGAMA_HIGHS_THREADS``             warns on bad input     ignored on bad input
``POWERGAMA_HIGHS_PARALLEL``            honoured               **not read at all**
======================================  ====================  ===================

Unifying them changes behaviour for anyone who sets those variables, so it is a
change rather than cleanup and belongs with solver work. The two parsers are kept
as separate functions, side by side, so that the divergence is at least visible to
whoever fixes it.
"""

import os
import warnings

import pyomo.environ as pyo
from pyomo.contrib import appsi


def _highs_env_options_single_period(opt):
    """Apply POWERGAMA_HIGHS_* to a HiGHS solver on the single-period path.

    Warns on malformed input and honours PARALLEL. See the module docstring.
    """
    seed_raw = str(os.environ.get("POWERGAMA_HIGHS_RANDOM_SEED", "")).strip()
    if seed_raw:
        try:
            opt.highs_options["random_seed"] = int(seed_raw)
        except Exception:
            warnings.warn(
                f"Ignoring invalid POWERGAMA_HIGHS_RANDOM_SEED='{seed_raw}' (expected integer).",
                UserWarning,
            )

    threads_raw = str(os.environ.get("POWERGAMA_HIGHS_THREADS", "")).strip()
    if threads_raw:
        try:
            threads = int(threads_raw)
            if threads >= 1:
                opt.highs_options["threads"] = threads
        except Exception:
            warnings.warn(
                f"Ignoring invalid POWERGAMA_HIGHS_THREADS='{threads_raw}' (expected integer >= 1).",
                UserWarning,
            )

    parallel_raw = str(os.environ.get("POWERGAMA_HIGHS_PARALLEL", "")).strip().lower()
    if parallel_raw in {"off", "on", "choose"}:
        opt.highs_options["parallel"] = parallel_raw


def _highs_env_options_window(opt):
    """Apply POWERGAMA_HIGHS_* to a HiGHS solver on the multi-period path.

    Silent on malformed input and does not read PARALLEL. See the module docstring.
    """

    def getenv_int(name: str) -> int | None:
        """Helper to get integer environment value, or None if it does not exist"""
        value = os.environ.get(name)
        if value is None:
            return None
        try:
            return int(value.strip())
        except ValueError:
            return None

    if (seed := getenv_int("POWERGAMA_HIGHS_RANDOM_SEED")) is not None:
        opt.highs_options["random_seed"] = seed

    if (threads := getenv_int("POWERGAMA_HIGHS_THREADS")) is not None and threads >= 1:
        opt.highs_options["threads"] = threads


def create_solver(solver, model, solver_path=None, solve_args=None):
    """Build the solver for the single-period timestep loop.

    `solve_args` is mutated for gurobi_persistent, which consumes
    symbolic_solver_labels once at set_instance time rather than per solve.
    """
    if solver == "gurobi":
        opt = pyo.SolverFactory("gurobi", solver_io="python")
        print(":) Using direct python interface to solver")
    elif solver == "gurobi_direct":  # think this is the same as gurobi above
        opt = pyo.SolverFactory(solver)
        print(":) Using gurobi_direct")
    elif solver == "gurobi_persistent":
        opt = pyo.SolverFactory("gurobi_persistent")
        print(":) Using persistent (in-memory) python interface to solver")
        print("-- Experimental --")
        symbolic_solver_labels = True
        if "symbolic_solver_labels" in solve_args:
            symbolic_solver_labels = solve_args["symbolic_solver_labels"]
            solve_args.pop("symbolic_solver_labels")
        opt.set_instance(model, symbolic_solver_labels=symbolic_solver_labels)
    elif solver == "appsi_highs":
        # opt = pyo.SolverFactory(solver)
        opt = appsi.solvers.Highs()
        # Keep solve() from raising on infeasible/unbounded runs; load values explicitly on optimal.
        opt.config.load_solution = False

        # Optional reproducibility controls for APPsi/HiGHS.
        # These keep workflow APIs unchanged while enabling seed/thread control per run.
        _highs_env_options_single_period(opt)

        if opt.available():
            print(":) Found solver")
        else:
            print(":( Could not find solver {}. Returning.".format(solver))
            raise Exception("Could not find LP solver {}".format(solver))
    else:
        solver_io = None
        # Some solver plugins (e.g. pyomo.contrib.highs) do not accept
        # the executable kwarg in their ConfigDict.
        if solver_path:
            opt = pyo.SolverFactory(solver, executable=solver_path, solver_io=solver_io)
        else:
            opt = pyo.SolverFactory(solver, solver_io=solver_io)
        if opt.available():
            opt_exec = None
            if hasattr(opt, "executable"):
                try:
                    opt_exec = opt.executable()
                except Exception:
                    opt_exec = None
            if opt_exec:
                print(":) Found solver here: {}".format(opt_exec))
            else:
                print(":) Found solver")
        else:
            print(":( Could not find solver {}. Returning.".format(solver))
            raise Exception("Could not find LP solver {}".format(solver))
    return opt


def create_dual_handle(opt, model):
    """Return the mapping that result extraction reads duals from.

    Either way the caller ends up with a mapping from constraint data to float,
    which is all result extraction needs -- but the two paths produce it very
    differently:

    - APPSI returns a plain dict from get_duals(), replaced after every solve.
    - Every other solver imports duals through a pyo.Suffix, which has to be a
      component of the model the solver is handed. Attaching it to the LpProblem
      instead would not raise: the solver would simply never see it, and every
      nodal price would come back zero.
    """
    if isinstance(opt, appsi.solvers.highs.Highs):
        return {}
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    return model.dual


def create_window_solver(solver_name):
    """Build the APPSI HiGHS solver for one multi-period window.

    A fresh instance per window: the model is rebuilt each time, so there is
    nothing for a persistent solver to hold on to.
    """
    # Only APPSI HiGHS supports dual extraction here.
    if solver_name != "appsi_highs":
        warnings.warn(
            f"multi_period mode with solver='{solver_name}': only 'appsi_highs' supports "
            "dual extraction for nodal prices. Proceeding but prices may be zero.",
            UserWarning,
        )

    day_opt = appsi.solvers.highs.Highs()
    _highs_env_options_window(day_opt)
    return day_opt
