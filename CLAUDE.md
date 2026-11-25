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

No external LP solver is needed. HiGHS runs in-process via the `highspy` dependency
(`solver="appsi_highs"`, the default of `LpProblem.solve`) and is the only solver the tests, CI and
the user guide use. Gurobi (`gurobi`, `gurobi_direct`, `gurobi_persistent`, licence required) and
other Pyomo solver names are still accepted by `lp/solver.py`, but are not tested.

CI (`.github/workflows/build.yml`) runs the matrix {ubuntu, macos, windows} × {py3.12, py3.14}.

## Architecture

A simulation is a pipeline of four objects:

`GridData` → `LpProblem` → `Results` → `Database` (sqlite3)

- **[GridData.py](src/powergama/GridData.py)** holds the static grid model as pandas DataFrames
  (`node`, `branch`, `dcbranch`, `generator`, `consumer`, `inter_area_rules`) plus time series in
  `profiles` and `storagevalue_filling` (101 rows, one per percent of filling). The storage-value
  *time* curves are ordinary profile columns (`storval_time_ref`/`flex_storval_time`; `"const"`,
  `""` or NaN is a factor of 1); `storagevalue_time` is the deprecated separate table, used for every
  row when given. `GridContext` resolves and checks both once, for the rows the LP values only
  (generators with storage, consumers with flexible load). The dict `GridData.keys_powergama` is the
  **single source of truth for the input schema**: every column name, which are required
  (`default=None`) and the default for optional ones. Adding an input field means adding it there.
  `read_csv_files` accepts a filename, a *list* of filenames (concatenated), or CSV/Parquet;
  `readProfileData(filename=...)` likewise, but each file is cut to `timerange` and the files are
  joined *by column* (duplicate column names, or a file lacking rows of `timerange`, raise).
  Generators/consumers reference profile *columns by name* (e.g. `inflow_ref`, `demand_ref`,
  `storval_filling_ref`, `rt_target_ref`), so a string in a grid-data column must match a column
  in `data.profiles`. The override refs `pmax_ref`, `pmin_ref`, `storage_ini_ref` default to
  `"const"`, which there means *no profile* (no `const` column needed; empty/NaN read the same).
  Any other value must be a profile column: `GridContext._build_profile_refs` raises at
  `LpProblem` construction otherwise. When set, `pmax_eff(t) = pmax·profile(t)`,
  `pmin_eff(t) = pmax·profile(t)` (both fractions of *installed* `pmax`; `pmax_ref` also scales
  inflow), and the initial filling is `storage_cap·profile` at the first timestep (fresh runs only).

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
  | `lp/rt/{extension,context,build,data,diagnostics}.py` | `RtExtension`, the balancing extension, and everything only it uses |
  | [lp/rt/prepare.py](src/powergama/lp/rt/prepare.py) | `prepare_rt` / `target_columns` / `price_columns`: a DA run to the RT inputs (tables with `da_index` from `rt_tables_from_da`, `da_solution.parquet`, deviation prices) |
  | [lp/rt/summary.py](src/powergama/lp/rt/summary.py) | `summarize_channels`: `channels.csv` and `prepare_rt_summary.md`, which RT rows use which DA-derived column how |

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
  `GridData` from a results database — grid data is persisted alongside results, and the
  round trip (`Database.get_grid_data` → `GridData.from_dict`) must give back the grid that
  was written, `inter_area_rules` included (`tests/test_grid_roundtrip.py`).

- **[database.py](src/powergama/database.py)** writes `Data_*` tables (grid + profiles, chunked into
  `data_profiles`, `data_profiles_aux_N` to stay under sqlite's column limit; through
  `_write_data_table`/`_read_data_table`, which keep the row index in `index`, or `level_0` if
  the table has an `index` column) and `Res_*` tables
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
   `rt_target_ref` / `rt_pump_target_ref` profile columns and the inter-area rules' DA summed
   flows (`inter_area_rules.da_flow_ref`). The deviation prices are unit data, `rt_price_up`/`rt_price_down`
   (`rt_pump_price_*` for pumping; consumers too) times an optional `*_ref` profile, resolved
   in `RtContext._build_prices` (a tracked row without a price raises); the extension applies
   no pricing rule but the split-sum guard. `prepare.price_columns` / `prepare_rt(prices=...)`
   write them (c0 ± marginal cost, renewables −c0 / factor·c0, storage value at the DA
   filling from `da_storage_values`), so `RtExtension()` takes no argument. The cross-border
   penalty and lock are per inter-area rule, on its summed flow (the NTC constraint's branches
   and signs): `inter_area_rules.balancing_penalty` (the cross-border balancing penalty) prices
   the deviation from the DA summed flow in
   `da_flow_ref`, and `lock_share` λ keeps it within DA − (1 − λ)·`ntc_backward` and
   DA + (1 − λ)·`ntc_forward` (`cRtRuleLockForward`/`Backward`, rows with mutable bounds);
   `prepare_rt(inter_area=...)` writes them. Pinned generators and fixed demand are not locks:
   they are `pmin_ref = pmax_ref` and `demand_ref` profile columns. It is the second step after a day-ahead run, and not a third formulation:
   `RtExtension` plugs into the model through the `ModelExtension` hooks
   (`lp/extension.py`), including editing the objective's named terms of each hour. Windows of
   one timestep only (`supports_multi_period=False`). **The core never names RT** — RT settings, state and
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
- **Caps always win over ramps.** The first hour of a window clips the ramp window into the cap
  interval (`pmax' = clip(prev + up, pmin, pmax)`, `pmin' = clip(prev − down, pmin, pmax)`, the caps
  from `pmin`/`pmax`, the `*_ref` profiles, inflow and storage), so a unit goes to a rising floor or
  a falling ceiling and a pinned unit follows its pin. Later hours lift the ramp limit into an hour
  to the cap's step where the cap moves faster (`max(up, pmin[h] − pmin[h−1])`,
  `max(down, pmax[h−1] − pmax[h])`, in `data._update_ramp_limits`), so ramps alone can't make a
  window infeasible; a ramp-down slack priced at `loadshedcost` (`varRampSlack`) remains. Every
  place a cap wins is a conflict in the input data: recorded in `lp.cap_ramp_conflicts`
  `(timestep, generator, kind, MW)` and warned once per window (`data._report_cap_ramp_conflicts`).
  Scope ramp tests to energy-unlimited generators. There is no daily reset: a unit that may change its
  level quickly needs ramp rates that allow it.
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
`plots.py` (folium map via `plotMap`) and `plots2.py` (geopandas/matplotlib).
`_updateLpProblem`, `_relax_and_retry` and `_get_timesteps_to_solve` are kept as methods on
`LpProblem` so that a subclass can override them; they delegate to `lp/formulation/`. The seams take
a window: `_updateLpProblem(timesteps)`, `_storeResultsAndUpdateStorage(timesteps, commit,
objective_values, results)`. Fault and contingency analysis is not part of PowerGAMA: the former
`fault_scenarios` subpackage was removed in favour of external reliability tools (e.g. modrel).
They use the public route instead: `solve(results, timesteps=..., state=SimulationState(...))`
re-solves a few timesteps from a given state ([lp/state.py](src/powergama/lp/state.py)), and
`MemoryResults` ([results_memory.py](src/powergama/results_memory.py)) keeps the results in memory.

## Tests

`tests/conftest.py` provides the `testcase_9bus_data` / `testcase_9bus_res` fixtures over
`tests/test_data/data_9bus` (48 timesteps). The CSV parse and the baseline solve are both cached at
session scope; `testcase_9bus_data` hands out a `deepcopy` because several tests mutate it, and
`testcase_9bus_res` rebinds the cached database so `res.grid is testcase_9bus_data` still holds.

Test data sets:

- `data_9bus` — the main case, but note what it *lacks*: no pumping, no DC branch, no flexible load,
  a single area, one storage generator. Its `inter_area_rules.csv` references an area with no nodes, so
  every NTC constraint is skipped.
- `two_area` — covers exactly those gaps (two areas with a binding inter-area NTC, an HVDC link,
  a pumped-storage generator, a flexible load, ramp limits). Use this when touching NTC, pumping,
  flexible load or multi-period window logic. `tests/rt_case.py` builds the RT case on it: a
  baseline run, then every kind of RT target, a generator pin and a fixed demand (as
  `pmin_ref`/`pmax_ref` and `demand_ref` profiles), the deviation prices (`add_prices`) and the inter-area
  rule's DA summed flow with a balancing penalty and a lock share. Use that for RT work; `test_rt_diagnostics.py` covers the JSONL debug stream with it,
  and `test_rt_prices.py` the prices (`test_data/two_area/rt_type_prices_no_storage` is the old type-based run it must
  reproduce bit for bit).
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
(issue #39); `test_properties.py` guards it. The RT lock on an inter-area rule's summed flow
is a row (`cRtRuleLockForward`/`Backward`), not a bound, so branch sensitivities are unaffected.

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
