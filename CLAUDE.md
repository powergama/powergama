# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Dependency management is Poetry; the package lives in `src/powergama` (src-layout).

```bash
poetry install                       # install deps + package in editable mode
poetry run pytest                    # full test suite (CI runs this bare, from repo root)
poetry run pytest tests/test_sim_dayhead.py::test_multi_period_regression  # single test
poetry run pytest -k ipynb --nbmake  # execute the example notebooks as tests
poetry run coverage run -m pytest && poetry run coverage xml
poetry run ruff check --fix && poetry run ruff format    # line-length 120, double quotes
poetry run pre-commit run --all-files                    # ruff + bandit
```

An LP solver must be on PATH. Tests use **GLPK** (`choco install glpk` on Windows); HiGHS is
available in-process via the `highspy` dependency (`solver="appsi_highs"`), which is the fastest
option and what most tests and the user guide use. CBC and Gurobi are also supported.

CI (`.github/workflows/build.yml`) runs the matrix {ubuntu, macos, windows} × {py3.12, py3.14}.

## Architecture

A simulation is a pipeline of four objects:

`GridData` → `LpProblem` → `Results` → `Database` (sqlite3)

- **[GridData.py](src/powergama/GridData.py)** holds the static grid model as pandas DataFrames
  (`node`, `branch`, `dcbranch`, `generator`, `consumer`, `inter_area_ntc`) plus time series in
  `profiles`, `storagevalue_filling`, `storagevalue_time`. The dict `GridData.keys_powergama` is the
  **single source of truth for the input schema**: every column name, which are required
  (`default=None`) and the default for optional ones. Adding an input field means adding it there.
  `read_csv_files` accepts a filename, a *list* of filenames (concatenated), or CSV/Parquet.
  Generators/consumers reference profile *columns by name* (e.g. `inflow_ref`, `demand_ref`,
  `storval_filling_ref`, and the optional `pmax_ref`/`pmin_ref`/`rt_target_ref`), so a string in a
  grid-data column must match a column in `data.profiles`.

- **[lp/](src/powergama/lp/)** is the optimisation package (`powergama.LpProblem`). Both
  simulation modes solve **one formulation**, an LP over a window of `H` timesteps
  (`lp/formulation/`): single-period is `H=1`, multi-period is `H=horizon_hours`. The old
  `LpProblemPyomo` module is gone; import `LpProblem` from `powergama` or `powergama.lp`.

  | module | holds |
  |---|---|
  | [lp/problem.py](src/powergama/lp/problem.py) | the `LpProblem` class: construction order, `solve()`, mutable state, the subclass seams |
  | [lp/config.py](src/powergama/lp/config.py) | `LpConfig` — validated constructor arguments, the window (`window_hours`, `window_commit`) and the guards |
  | [lp/context.py](src/powergama/lp/context.py) | `GridContext` — everything derived once from `GridData` |
  | [lp/solver.py](src/powergama/lp/solver.py) | solver construction, HiGHS env vars, the dual handle |
  | [lp/storage_value.py](src/powergama/lp/storage_value.py) | storage and flexible-load valuation |
  | [lp/extension.py](src/powergama/lp/extension.py) | `ModelExtension` — the hooks an optional addition to the model plugs into |
  | [lp/objective_trace.py](src/powergama/lp/objective_trace.py) | optional per-timestep objective CSV (`POWERGAMA_OBJECTIVE_TRACE_CSV`) |
  | `lp/formulation/{build,data,extract}.py` | the LP over a window: the model, the data written into it for each window, and the results read out |
  | `lp/rt/{extension,context,build,data,locks,diagnostics}.py` | `RtExtension`, the balancing extension, and everything only it uses |

  **`cfg` and `ctx` are what make the builders plain functions.** Construction code reads nothing
  mutable, so it takes `(model, grid, ctx, cfg)` explicitly. The per-timestep updaters and the
  extractors take the `LpProblem` instead, because they read and write simulation state and an
  eight-argument signature there would be a longer way of spelling `self`. That asymmetry is
  deliberate.

  **The model is built once in `__init__` and then mutated in place** by
  `_updateLpProblem(timesteps)` for every window — parameters for inflow, demand, storage values
  and bounds are rewritten rather than the model rebuilt. Only a multi-period window cut short at
  the end of the run gets a second model, for its length (`_use_window_model`). Every component
  is indexed by hour in the window, `[·, h]`; single-period is `h = 0`. The window's first hour
  takes the simulation state through Params; later hours are coupled to the hour before through
  constraints that only exist for `h ≥ 1`, so `H=1` is exactly the old single-period LP. Storage
  filling level lives in `self._storage` / `self._storage_flexload` (seed them with the public
  `set_storage_state()`) and is what makes the problem sequential. `self._gen_prev` carries
  previous-timestep output for ramp constraints. `formulation.data.update_window` is vectorised
  and writes Params with `store_values(check=False)`: per-element pandas lookups and Param writes
  used to cost more than the solve on the Europe case. Keep it column-wise.

  Four things to know:
  - `__getattr__` resolves names against `_cfg`, then `_ctx`, then the model — so `lp._lossmethod`,
    `lp._idx_generatorsWithStorage` and `lp.varGeneration` all still work. **Reads only.**
    `self.cFoo = pyo.Constraint(...)` sets a plain attribute and the component is never added;
    construction must always say `model.cFoo`. The redirected names are listed explicitly in
    `CONFIG_ATTRS` / `CONTEXT_ATTRS` so a new field cannot silently shadow an attribute.
  - Inside a constraint rule, Pyomo components come off the rule's `model` argument and everything
    else off `ctx`/`cfg`/`grid`. `tests/test_model_invariants.py` enforces both directions across
    the whole package.
  - Between multi-period windows the HiGHS solver's state is cleared
    (`solver.discard_warm_start`): warm-starting a 24-hour window from the previous day's basis
    skips presolve and was about 5× slower than a fresh solve on the Europe case. Single-period
    keeps the warm start, which pays off between consecutive timesteps.
  - A solver must only ever be handed the model it was created for: APPSI's incremental update
    and `gurobi_persistent.set_instance` key off the object. Each window length keeps its own
    model and solver in `self._window_models`; `self.model` is the active one.
  - **Declaration order is load-bearing.** Pyomo writes LP columns in the order components are
    added, so `formulation.build.build_model()`'s sequence is not arbitrary: it is what keeps
    single-period results bit-identical to before the merge. Components only windows of `H > 1`
    have are declared after the per-hour ones, and extensions declare theirs after the core's, in
    each phase.

- **[timestep_result.py](src/powergama/timestep_result.py)** holds `TimestepResult`, the dataclass
  that names the 21-field LP→database contract. `formulation.extract.store_window` builds one per
  committed timestep and calls `.store(res)`.

- **[Results.py](src/powergama/Results.py)** is a thin analysis/plotting layer over the database.
  `ResultsBaseClass` is the write path (no plotting, importable without heavy plot deps); `Results`
  adds the `get*`/`plot*` methods. `Results.from_existing(dbfile, timedelta)` reconstructs a full
  `GridData` from a results database — grid data is persisted alongside results.

- **[database.py](src/powergama/database.py)** writes `Data_*` tables (grid + profiles, chunked into
  `data_profiles`, `data_profiles_aux_N` to stay under sqlite's column limit) and `Res_*` tables
  (`Res_Nodes`, `Res_Branches`, `Res_BranchesSens`, `Res_Generators`, `Res_Storage`, `Res_Pumping`,
  `Res_FlexibleLoad`, `Res_ObjFunc`). All result queries are raw SQL here; `Results` never touches
  sqlite directly.

### Three simulation modes

Selected via `LpProblem` constructor arguments (see [usage.md](userguide/usage.md)):

1. **Single-period** (`objective_mode="single_period"`, default) — one LP per timestep, solved
   sequentially, storage state passed forward: windows of one timestep, committing one.
2. **Multi-period** (`objective_mode="multi_period"`) — an LP over `horizon_hours`, of which
   only the first `commit_hours` are committed (rolling horizon). Losses and extensions need
   `horizon_hours=1`.
3. **Balancing / real-time** (`extensions=[RtExtension(...)]`, from `powergama.lp.rt`) — a
   single-period solve with penalties on deviation from a day-ahead schedule supplied through
   `rt_target_ref` / `rt_pump_target_ref` profile columns and the DA lock parquets
   (`lp/rt/locks.py`). It is the second step after a day-ahead run, and not a third formulation:
   `RtExtension` plugs into the model through the `ModelExtension` hooks
   (`lp/extension.py`), including editing the objective's named terms of each hour. Windows of
   one timestep only (`supports_multi_period=False`). **The core never names RT** — RT settings, state, locks and
   components all live on the extension, and `tests/test_rt_isolation.py` fails on any RT
   reference in `lp/` outside `lp/rt/`. A new RT feature goes behind a hook; if no hook fits, add
   a general one to `ModelExtension` rather than an RT special case in the core.

The mode was called `"hourly"` until the rename; a timestep need not be an hour. The old spelling
raises, with no alias.

**There is one formulation** (#42), described component by component in
[model_formulations.md](userguide/model_formulations.md). In short:

- **Storage is valued only at the end of the window** (issue #40): no cost per MWh drawn, and
  `−v·S_end/dt` in the objective, `v` from `storage_value.generator_storage_values` at the
  filling level the window starts from. Single-period re-prices every timestep; `multi_period`
  freezes `v` for the window, so it does not respond to the level within a window.
- **Flexible load is a bounded reservoir**, `varFlexStorage` in `[0, capacity]` (no floor where
  `flex_on_off > 1`), valued per MWh consumed by `lp/storage_value.py`.
- **Ramp limits for energy-limited generators.** The first hour of a window applies
  `pmin_now = min(pmin_now, pmax_now)`, dropping the ramp-down floor when the availability bound
  is tighter; later hours have a ramp-down slack priced at `loadshedcost` (`varRampSlack`). Scope
  ramp tests to energy-unlimited generators. `ramp_daily_reset` releases at absolute midnight in
  both modes, whatever `commit_hours` is.
- **Multi-period results are only unique up to ties.** A window LP often has several equally
  cheap dispatches, and adding or reordering columns changes which one the solver returns. Pin
  window objectives and prices in multi-period tests, not flows.

### Conventions

- The **external interface is in MW and EUR/MWh**; the DC power flow equations work in per-unit with
  `constants.baseS` = 100 MVA. Conversions via `const.baseMVA` are applied at the constraint level
  (`formulation.build.create_constraint_powerflow_equation` and `…_powerbalance`).
- Marginal cost for generators with storage comes from the storage-value curves
  (filling-level × time) — this is the core modelling idea, described in
  [model_description.md](userguide/model_description.md). It enters as the value of the energy
  left in storage, and `fuelcost` applies on top as a level-independent cost. Pumping is worth
  `v·pump_efficiency`, less `pump_deadband`.
- **Every objective term is a cost rate in EUR/h** (MW × EUR/MWh), whatever `timeDelta` is, so
  duals are prices in EUR/MWh with no `dt` correction. Amounts in EUR (the terminal storage
  value) are divided by `dt`. Keep it that way: the user wants results readable the same at
  1 h and 15 min.
- Load shedding and dump load are priced at `constants.loadshedcost` (1000/MWh), so they act as
  penalty slack and keep every timestep feasible.
- `lossmethod` on `LpProblem`: `0` = lossless (default), `1` = piecewise-linearised losses (needs a
  pre-solve of the first timestep to initialise flows), `2` = losses added as load.
- `solve(..., continue_from_last=True)` resumes from the last timestep in an existing database,
  reloading storage, previous generation and, with losses, the branch flows the loss linearisation
  is taken around. The resume is exact: anything a timestep carries forward must be restorable from
  the database, which is why `lossmethod=1` linearises around net flows (the stored quantity)
  rather than the directional `flow12`/`flow21`.

### Other modules

`scenarios.py` — read/write scenario CSV files that rescale a base `GridData` per area and
technology. `analysis.py`, `costbenefit.py`, `filters.py` — post-processing helpers.
`plots.py` (folium map via `plotMap`) and `plots2.py` (geopandas/matplotlib). `fault_scenarios/` is a
subpackage for adequacy studies: it generates fault situations, re-simulates **only** the faulted
timesteps starting from a baseline results database, via `LpFaultProblem` — see
[fault_scenario_analysis.md](userguide/fault_scenario_analysis.md). It subclasses `LpProblem` and
overrides six methods, so `_updateLpProblem`, `_get_fault_start`, `_update_progress`,
`_relax_and_retry` and `_get_timesteps_to_solve` are **extension seams**, not merely long methods:
they stay on the class and delegate to `lp/formulation/`. The seams take a window:
`_updateLpProblem(timesteps)`, `_storeResultsAndUpdateStorage(timesteps, commit, objective_values,
results)`; fault simulation runs in windows of one timestep.

## Tests

`tests/conftest.py` provides the `testcase_9bus_data` / `testcase_9bus_res` fixtures over
`tests/test_data/data_9bus` (48 timesteps). The CSV parse and the baseline solve are both cached at
session scope; `testcase_9bus_data` hands out a `deepcopy` because several tests mutate it, and
`testcase_9bus_res` rebinds the cached database so `res.grid is testcase_9bus_data` still holds.

Test data sets:

- `data_9bus` — the main case, but note what it *lacks*: no pumping, no DC branch, no flexible load,
  a single area, one storage generator. Its `ntc_interarea.csv` references an area with no nodes, so
  every NTC constraint is skipped.
- `two_area` — covers exactly those gaps (two areas with a binding inter-area NTC, an HVDC link,
  a pumped-storage generator, a flexible load, ramp limits). Use this when touching NTC, pumping,
  flexible load or multi-period window logic. `tests/rt_case.py` builds the RT case on it: a
  baseline run, then every kind of RT target and DA lock derived from it. Use that for RT work;
  `test_rt_diagnostics.py` covers the JSONL debug stream with it.
- `mount_synth` — small synthetic case where storage is needed to avoid load shedding.
- `data_europe2014` — 1539 nodes; marked `slow`.

Three kinds of test, and the distinction matters when changing the model:

- **Property tests** (`test_properties.py`) assert invariants, not values — storage continuity,
  capacity limits, table shapes, run-to-run determinism. These survive changes to storage valuation
  and are the primary regression net. `test_model_invariants.py` is the source-level counterpart:
  an AST check that constraint rules read Pyomo components off their `model` argument and
  everything else off `self`. Run it after touching any rule; it takes no measurable time.
- **Numeric regression tests** with hard-coded flows/prices. If a modelling change moves these, the
  change needs justifying, not the expectations silently updating.
- **Strict xfails** (`xfail(strict=True)`) assert behaviour we *want* for known defects, so a fix
  flips them green and an accidental fix fails loudly as XPASS. Never pin a golden number over a
  known defect.

Write outputs to pytest's `tmp_path`, not the repo root. Skip the heavy cases with
`poetry run pytest -m "not slow"`.

### Known defects pinned by tests

None at present. Pin the next one as a strict xfail.

**Don't repeat a constraint row as a variable bound.** If a limit is both a row whose dual is read
and an identical `setlb`/`setub` on the variable, the solver prices it on the bound (reduced cost)
and the row's dual is zero. That is what made single-period branch sensitivities identically zero
(issue #39); `test_properties.py` guards it. DA-locked border windows are still set as variable
bounds, so a branch held by a tighter lock reports zero sensitivity for that timestep.

## Docs

`userguide/*.md` is the maintained documentation and worth updating alongside behaviour changes.
[model_formulations.md](userguide/model_formulations.md) maps the single-period and multi-period
formulations onto each other component by component — update it when either model changes.
`doc/` is a legacy Sphinx setup (still references the removed `powergim` module). `CHANGES.md` is
updated per release together with the version in `pyproject.toml`.

## Notes

- `main` is the default branch; feature work happens on branches with PRs into `main`.
- The README ends with a workspace-specific note pointing at WILLOW T45 runbooks in a sibling
  `../willow` directory — that is not part of this package.
- `lp/context.py` reads one project-specific environment variable,
  `T45_NUCLEAR_OPERATIONAL_LIMITS`, which relaxes nuclear ramp/daily-reset constraints when
  `pmax_ref`/`pmin_ref` profile bounds would otherwise make the day-joint model infeasible.
