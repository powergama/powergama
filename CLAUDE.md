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

- **[LpProblemPyomo.py](src/powergama/LpProblemPyomo.py)** (`powergama.LpProblem`) **holds** a
  `pyo.ConcreteModel` as `self.model`. The crucial design point: **the model is built once in
  `__init__` and then mutated in place** by `_updateLpProblem(timestep)` on every iteration —
  parameters for inflow, demand, storage values and bounds are rewritten rather than the model
  rebuilt. Results extraction and the storage-state carry-over happen in
  `_storeResultsAndUpdateStorage`. Storage filling level lives in `self._storage` /
  `self._storage_flexload` (seed them with the public `set_storage_state()`) and is what makes
  the problem sequential. `self._gen_prev` carries previous-timestep output for ramp constraints.

  Three things to know about the composition:
  - `__getattr__` falls through to the model, so `lp.varGeneration` still resolves. **Reads
    only.** `self.cFoo = pyo.Constraint(...)` sets a plain attribute and the component is never
    added — construction must always say `self.model.cFoo`.
  - Inside a constraint rule, Pyomo components come off the rule's `model` argument and
    everything else off `self`. `tests/test_model_invariants.py` enforces both directions.
  - `self.model`'s object identity must survive the whole of `solve()`: APPSI's incremental
    update and `gurobi_persistent.set_instance` key off it.

- **[timestep_result.py](src/powergama/timestep_result.py)** holds `TimestepResult`, the dataclass
  that names the 21-field LP→database contract. Both solve paths build one and call `.store(res)`,
  so `Results.addResultsFromTimestep` has a single caller-side definition. The extraction *bodies*
  stay per-mode — they read from different objects — so this shares the contract, not the code.

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

1. **Hourly** (`objective_mode="hourly"`, default) — one LP per timestep, solved sequentially,
   storage state passed forward.
2. **Day-ahead / multi-period** (`objective_mode="multi_period"`) — a joint LP over
   `horizon_hours` built by `_build_day_joint_model`, of which only the first
   `commit_hours` are committed (rolling horizon). This is a **separate model-building
   and result-extraction code path** from the hourly one; changes to constraints usually need to be
   made in both.
3. **Balancing / real-time** (`is_rt=True`) — hourly solve with penalties on deviation from a
   day-ahead schedule supplied through `rt_target_ref` / `rt_storage_target_ref` profile columns.

Where the two model-building paths deliberately differ:

- **Storage values are frozen per window in `multi_period`.** Both generator storage and flexible
  load are priced once per window from the filling level it starts at, because the whole window is
  one LP and the value has to be a constant coefficient. Hourly re-prices from the current level
  before every timestep.
- **Flexible load is a bounded variable in `multi_period`, a clipped state in hourly.**
  `varFlexStorage` carries the reservoir with a continuity equality and hard `[0, capacity]`
  bounds, so it cannot leave range. Hourly has no bounds at all: it lets the level overshoot and
  prices that at `constants.flexload_outside_cost` on the *next* timestep, which does mean it can
  report a filling level above capacity. The multi-period floor is skipped where `flex_on_off > 1`,
  since the per-hour charge limit is then below the drain and no schedule could hold the level
  non-negative. One valuation function, `LpProblem._flexload_storage_value`, serves both.
- **Ramp limits bind differently for energy-limited generators.** Hourly applies
  `pmin_now = min(pmin_now, pmax_now)`, which drops the ramp-down floor when the availability bound
  is tighter — a CSP plant at sunset ramps down past its limit. `multi_period` has no such escape
  hatch. Scope hourly ramp tests to energy-unlimited generators.

### Conventions

- The **external interface is in MW and EUR/MWh**; the DC power flow equations work in per-unit with
  `constants.baseS` = 100 MVA. Conversions via `const.baseMVA` are applied at the constraint level
  (`_create_constraint_powerflow_equation`, `_create_constraint_powerbalance`).
- Marginal cost for generators with storage comes from the storage-value curves
  (filling-level × time), not `fuelcost` — this is the core modelling idea, described in
  [model_description.md](userguide/model_description.md).
- Load shedding and dump load are priced at `constants.loadshedcost` (1000/MWh), so they act as
  penalty slack and keep every timestep feasible.
- `lossmethod` on `LpProblem`: `0` = lossless (default), `1` = piecewise-linearised losses (needs a
  pre-solve of the first timestep to initialise flows), `2` = losses added as load.
- `solve(..., continue_from_last=True)` resumes from the last timestep in an existing database,
  reloading storage and previous generation from it.

### Other modules

`scenarios.py` — read/write scenario CSV files that rescale a base `GridData` per area and
technology. `analysis.py`, `costbenefit.py`, `filters.py` — post-processing helpers.
`plots.py` (folium map via `plotMap`) and `plots2.py` (geopandas/matplotlib). `fault_scenarios/` is a
subpackage for adequacy studies: it generates fault situations, re-simulates **only** the faulted
timesteps starting from a baseline results database, via `LpFaultProblem` — see
[fault_scenario_analysis.md](userguide/fault_scenario_analysis.md).

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
  flexible load or multi-period window logic.
- `mount_synth` — small synthetic case where storage is needed to avoid load shedding.
- `data_europe2014` — 1539 nodes; marked `slow`.

Three kinds of test, and the distinction matters when changing the model:

- **Property tests** (`test_properties.py`) assert invariants, not values — storage continuity,
  capacity limits, table shapes, run-to-run determinism. These survive changes to storage valuation
  and are the primary regression net. `test_model_invariants.py` is the source-level counterpart:
  an AST check that constraint rules read Pyomo components off their `model` argument and
  everything else off `self`. Run it after touching any rule; it takes no measurable time.
- **Numeric regression tests** with hard-coded flows/prices. If a modelling change moves these, the
  change needs justifying, not the expectations silently updating. For `multi_period` these record
  *current* behaviour, which is not the same as correct — see the storage double-count below.
- **Strict xfails** (`xfail(strict=True)`) assert behaviour we *want* for known defects, so a fix
  flips them green and an accidental fix fails loudly as XPASS. Never pin a golden number over a
  known defect.

Write outputs to pytest's `tmp_path`, not the repo root. Skip the heavy cases with
`poetry run pytest -m "not slow"`.

### Known defects pinned by tests

- **Storage draw is priced twice in `multi_period` mode** — the objective charges the storage value
  per MWh generated *and* subtracts a terminal storage value, which the continuity equality links,
  giving an effective cost of `storval[h] + storval[last]*dt`. Affects every horizon, not just
  `horizon_hours=1`. Pinned by `test_sim_hourly_rolling_equiv.py`.
- **`continue_from_last` is not exactly resumable with `lossmethod` 1 or 2** — the loss linearisation
  operating point lives in `Var.value` rather than being persisted, so a resumed run reseeds it from
  a pre-solve instead of the real preceding solution.
- **Branch capacity sensitivities are identically zero in hourly mode** — `Res_BranchesSens` holds
  nothing but zeros for every hourly run measured (9-bus and two-area, glpk and `appsi_highs`, all
  three loss methods), while the same case in `multi_period` produces non-zero values. Branches sit
  at their rating with nodal price differences of up to 990 EUR/MWh across them, so the limit is
  genuinely binding. Not the constraint form: splitting the two-sided `cMaxFlowAc` into two
  one-sided rows does not recover the dual. Pinned by `test_properties.py`, with the working
  multi-period path as the reference case next to it.

## Docs

`userguide/*.md` is the maintained documentation and worth updating alongside behaviour changes.
`doc/` is a legacy Sphinx setup (still references the removed `powergim` module). `CHANGES.md` is
updated per release together with the version in `pyproject.toml`.

## Notes

- `main` is the default branch; feature work happens on branches with PRs into `main`.
- The README ends with a workspace-specific note pointing at WILLOW T45 runbooks in a sibling
  `../willow` directory — that is not part of this package.
- `LpProblemPyomo.py` reads one project-specific environment variable,
  `T45_NUCLEAR_OPERATIONAL_LIMITS`, which relaxes nuclear ramp/daily-reset constraints when
  `pmax_ref`/`pmin_ref` profile bounds would otherwise make the day-joint model infeasible.
