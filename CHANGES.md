# Changes

## unpublished

### Inter-area rules in RT
2026-11-13

Added
- `inter_area_rules` has three optional columns, read by `RtExtension` only (a DA run
  ignores them): `balancing_penalty` (EUR/MWh, ≥ 0, default 0), `lock_share` (λ in [0, 1],
  default NaN: no lock) and `da_flow_ref` (profile column with the DA summed flow of the
  rule, MW, `area_from` → `area_to`; default `""`). The results database keeps the
  `da_flow_ref` columns like the other profile refs
- cross-border balancing penalty on the **summed** flow: a rule with a `da_flow_ref`
  and `balancing_penalty` p > 0 costs p per MWh of |T − DA|, T the summed flow of the
  rule's NTC constraint (`varRtRuleDevPos`/`Neg`, `cRtRuleFlowTracking`)
- lock on the **summed** flow: a rule with a `da_flow_ref` and a `lock_share` λ holds
  T within DA − (1 − λ)·`ntc_backward` and DA + (1 − λ)·`ntc_forward`
  (`cRtRuleLockForward`/`cRtRuleLockBackward`, rows with mutable bounds
  `p_rt_rule_lock_ub`/`_lb` written each timestep; λ = 1 pins T to DA, an infinite NTC
  leaves that side open), together with its penalty if it has one
- `prepare_rt(..., inter_area=InterAreaFlows(mask, balancing_penalty, lock_share, source))`
  writes `da_flow_<k>` (the stored DA AC and DC branch flows across the boundary of rule k,
  summed with the signs of the NTC constraint) and fills the three columns in;
  `da_flow_columns(...)` is the rule on its own. `RefColumns` gains an `inter_area_rules`
  part, and the channel summary the `da_flow` family (usage `flow_ref`, rows
  `inter_area_rules:<k>`, `desc` `balancing_penalty <p>, lock_share <λ>`)
- the JSONL debug stream reports `rt_rule_rows` (with `balancing_penalty` per entry),
  `term_breakdown.xborder_dev`, `rt_rule_balancing_penalty_count`, and in the header
  `inter_area_rules_balancing_penalised` / `inter_area_rules_locked`

Removed and renamed
- **breaking (RT):** `RtExtension` no longer takes `xborder_flow_penalty_eur_per_mwh` (use
  `inter_area_rules.balancing_penalty`); `RtExtension()` takes no argument
- **breaking (RT):** the per-branch border-flow targets and windows are removed:
  `border_ac_flow_lock_parquet`, `border_dc_flow_lock_parquet`,
  `border_ac_flow_bounds_parquet` and `border_dc_flow_bounds_parquet` (use
  `balancing_penalty` / `lock_share`). Gone with them: `powergama.lp.rt.locks` (`DaLocks`,
  `load_da_locks`, `LOCK_ATTRS`, `REMOVED_LOCK_ATTRS`), `rt.data.apply_da_locks`,
  `RtExtension.locks`, `p_rt_io_*`, `varRtIo*`, `cRtIoTargetTracking*`,
  `add_border_flow_target_tracking`, `update_border_flow_targets`,
  `RtContext.idx_border_*` / `border_*_sign` and `_be_oriented_sign`. The debug stream's
  `rt_balancing_deviation_mw` no longer includes a border term (it was oriented on a "BE"
  node prefix), and the header loses `rt_xborder_flow_penalty_eur_per_mwh`
- the profile columns `be_da_net_export_mw` and `be_da_lock_share` (WILLOW T45, Belgium)
  are no longer kept in a results database above SQLite's column limit when nothing
  references them

Changes results? Day-ahead runs: no (they ignore the new columns). RT runs: yes, for users
of the per-branch border-flow targets or windows:
- the penalty and the lock act on the summed flow of a boundary, so flow may shift between
  the parallel branches of a boundary at no cost (on `two_area` with penalty 5, 30 MW moves
  from the HVDC link to the AC branch in hours 9-19; dispatch, objective and prices are
  unchanged)
- the lock's width comes from the NTC and `lock_share`, not from per-branch bounds; on
  `two_area` λ = 0.6 (DA − 60, DA + 100 MW) in place of ±30 MW per branch gives the same
  dispatch, objective and prices. Being a row rather than a variable bound, the lock no
  longer zeroes branch sensitivities
- RT runs without border-flow targets or windows are unchanged

### DA to RT preparation
2026-11-13

Added
- `powergama.lp.rt.prepare_rt(res_da, outdir)` turns a complete day-ahead run into the
  inputs of the balancing run: the RT grid tables as CSV, each generator and consumer row
  with `da_index` (its DA row), and `da_solution.parquet`, a profile file indexed by the
  stored DA timestep with the target columns the tables reference: `da_gen_<d>` (DA output /
  `pmax`), `da_pump_<d>` (DA pumping / `pump_cap`), `da_cons_<d>` (DA flexible demand /
  `demand_avg`), clipped to [0, 1], and `rt_target_zero` for rows with no DA row. Every
  generator with capacity gets a target; consumers without flexible load get none. It reads
  only `res_da.grid` and `res_da.get_dispatch_frames()`, and raises unless
  `res_da.is_complete()`. `target_columns(...)` is the same rule for RT tables built by the
  caller, and `rt_tables_from_da(res_da)` the copy of the DA grid tables it starts from
  (node, branch, dcbranch, generator, consumer and `inter_area_rules` if any, with
  `da_index`; it drops `area_from`/`area_to` from `dcbranch`, which DA databases written
  before carry), the same for the DA run's `Results` and one read back with
  `Results.from_existing`
- pins to DA: keyword-only `fix=`, `cap=`, `zero_pmin=` (boolean masks on the generator
  rows) set `pmin_ref = pmax_ref = da_gen_<d>` (fix), `pmax_ref = da_gen_<d>` (cap) and
  `pmin_ref = rt_target_zero` (cap with `zero_pmin`), sharing the target column;
  `carry_curtailment=CarryCurtailment(mask, da_available, rt_available, pin, source=None)`
  sets `pmax_ref` (and with `pin="fix"` also `pmin_ref`) to `da_carry_<d>` = (RT
  availability + DA output − DA availability, within [0, `pmax`]) / `pmax`. Pinned rows get
  `inflow_ref = "const"`, `inflow_fac = 1`. The rules on their own: `pin_columns`,
  `carry_curtailment_columns`, joined with `combine_columns` and set with `apply_refs`;
  `RefColumns` holds profile columns and references (`TargetColumns` is an alias);
  `clear_refs(table, mask, columns, timesteps=...)` builds one that clears references (`""`)
- RT start storage: keyword-only `initial_storage=` (a boolean mask on the generator
  rows) sets `storage_ini_ref = da_storage_<d>`, the DA filling at the start of each DA
  timestep / `storage_cap` (the DA initial filling at the DA run's first timestep,
  `Res_Storage[t − 1] / storage_cap` after it). The rule on its own:
  `initial_storage_columns`
- peak tranches: `peak_tranches=PeakTranches(mask, price, ramp_pu=None,
  source=None)` adds one RT row per unit (`block_group`, or the row alone) of the masked
  generators: a copy of the unit's first row with `pmax` = the unit's summed `pmax`,
  `fuelcost = price`, `da_index` NA (target `rt_target_zero`), `pmin_ref = ""`,
  `inflow_ref = "const"`, `inflow_fac = 1` and `pmax_ref = da_headroom_<d>` =
  `(cap − DA output of the unit) / cap`, clipped to [0, 1]; `ramp_up_pu` =
  `ramp_down_pu` = `ramp_pu` (>= 0), or NaN, no ramp limit, when it is None (the default;
  the unit's ramp limits are not inherited). The rule on its own: `peak_tranche_columns(frames, generator, mask, *, price, ...)`
- channel summary: `prepare_rt` writes `channels.csv` (one row per DA-derived column:
  family, formula, DA row, and the RT rows using it and how — `target`, `pump_target`,
  `cons_target`, `fix`, `cap`, `pmin`, `storage_ini` — with their `desc`, `node`, `area`,
  `type`) and `prepare_rt_summary.md` (per family: formula, source, counts by usage, area
  and type; the rows without any channel). Keyword-only `summary=True`, `sources=`,
  `summary_header=`. For RT tables the caller builds: `summarize_channels(...)` and
  `write_channel_summary(...)` (`powergama.lp.rt.summary`)
- RT deviation prices as unit data: every tracked row has `rt_price_up`/`rt_price_down`
  (generators and consumers) or `rt_pump_price_up`/`rt_pump_price_down` (pumping), EUR/MWh,
  each times an optional profile `<column>_ref` (default `"const"`).
  `powergama.lp.rt.price_columns` writes them from c0, a renewables mask and its curtailment
  factor, and a storage value per storage unit; `prepare_rt(prices=DeviationPrices(...))`
  does it with the storage value at the DA filling level (`da_storage_values(res_da)`).
  Time-varying prices are profile columns of the `da_price` family (`da_price_up_<d>`,
  `da_pump_price_down_<d>`, `da_cons_price_up_<d>`, ...), listed in the channel summary
- the results database treats the price refs (`rt_price_up_ref`, `rt_price_down_ref`,
  `rt_pump_price_up_ref`, `rt_pump_price_down_ref`; consumer `rt_price_up_ref`,
  `rt_price_down_ref`) like the other profile refs: writing the input tables raises on a ref
  to a missing profile column, and with more than 1,800 profile columns the referenced
  price columns are among those stored

Removed and renamed
- **breaking (RT):** `RtExtension` no longer takes `balancing_fee_eur_per_mwh` or
  `res_curtailment_factor` (`xborder_flow_penalty_eur_per_mwh` is now keyword-only) and no
  longer classifies
  generators by type (`RtContext.idx_*` and `_build_classification` are gone). A tracked
  row without a price, an unknown `*_ref` or a NaN in a referenced profile raises when the
  `LpProblem` is built. Only the split-sum guard stays in the extension
- **breaking (RT):** the lock files `foreign_gen_lock_parquet` and
  `foreign_consumer_lock_parquet` are removed.
  Pin a generator with `pmin_ref = pmax_ref` pointing at a profile column of its DA output
  (fraction of `pmax`, with `inflow_ref = "const"`, `inflow_fac = 1`; caps win over ramps,
  so it follows the pin), and fix a demand with `demand_ref` pointing at one of the DA
  demand (fraction of `demand_avg`). `DaLocks` loses `foreign_gen_lock` /
  `foreign_cons_lock`, `LOCK_ATTRS` the two attributes. The border-flow locks are unchanged
- **breaking (RT):** setting `generator.rt_storage_target_ref` or the
  `da_storage_marginalprice_parquet` attribute no longer raises an error; both have been
  without effect since 2026-10-21
- **breaking (RT diagnostics):** the JSONL debug stream reports per unit: `rt_gen_rows`
  (every tracked generator, with `price_up`, `price_down`, `dev_cost` and the DA-replay
  bounds), `rt_pump_rows`, `rt_flex_rows`, and `da_replay_check`. The type buckets
  (`redispatch_cost_attribution_eur`, `redispatch_mw_attribution`), the gas-only
  `da_injection_check`, the pump rows' `gen_cost_mwh`/`pump_credit_mwh`, the header's
  `rt_balancing_fee_eur_per_mwh` and the `active_channels` counts `foreign_gen_lock_rows` /
  `foreign_cons_lock_rows` are gone. `RtContext.idx_peak_gen` (`is_peak_rt_block`) takes the
  `fossil_gas` rows without a DA row when the generator table has `da_index`, instead of
  those with `[RT]` in `desc`

Changes results? Day-ahead runs: no. RT runs:
- storage units (generation, pumping, flexible load) are priced at the storage value of the
  DA filling level instead of the RT filling level, so RT results change where a storage
  unit is tracked. Other units keep their prices exactly (c0 ± fuelcost; −c0 / factor × c0
  for renewables): an RT run without a tracked storage unit is bit-identical to before
  (reference results in `tests/test_data/two_area/rt_type_prices_no_storage`)
- the prices must now be given as unit data: an RT setup that passed
  `balancing_fee_eur_per_mwh` writes the price columns (`price_columns`) instead
- the foreign locks expressed as pins and fixed demand give the same RT results on the
  two-area case
- `prepare_rt` and the channel summary only write input files

### Results API
2026-11-13

Added
- `Results.get_dispatch_frames(timeMaxMin)` returns the dispatch as wide frames (index =
  stored timestep, columns = element index): `generator_output`, `pumping`,
  `flexload_demand`, `storage`, `branch_flow` and `dcbranch_flow`, with a column for every
  element of each kind and NaN where nothing is stored (never filled)
- `Results.is_complete(timerange)` tells whether the stored timesteps are exactly the run's
  time range (or the one given), e.g. to check that a run finished before reading it
- `Database.getResultFrame(table, column, timeMaxMin)` reads one result variable as such a
  frame; only the result tables with one value per element and timestep are accepted
  (`Database.RESULT_FRAME_COLUMNS`)
- `Results.from_existing(databasefile, timedelta, *, timerange=None)` takes the profile rows
  the run used, for runs that did not start at row 0 (the database stores the profiles
  already cut to the run's time range). A `timerange` whose length differs from the stored
  profiles raises `ValueError`

Changes results? No. These functions only read a results database; nothing written by a run
changes. Without `timerange`, `from_existing` behaves as before.

### Ramps and caps
2026-11-13

Added
- caps always win over ramps. In the first hour of a window the ramp window is clipped into
  the cap interval: `pmax' = clip(prev + ramp_up·pmax, pmin, pmax)` and
  `pmin' = clip(prev − ramp_down·pmax, pmin, pmax)`, the caps being the bounds from
  `pmin`/`pmax`, the `pmin_ref`/`pmax_ref` profiles, inflow and storage. It replaces the
  one-sided guard `pmin' = min(pmin', pmax')`, under which a falling ceiling won over the
  ramp but the ramp won over a rising floor
- in multi-period windows, caps win over ramps in the later hours too: the ramp limits into
  hour `h` are lifted where a cap moves faster than the ramp,
  `max(ramp_up·pmax, pmin[h] − pmin[h−1])` and `max(ramp_down·pmax, pmax[h−1] − pmax[h])`,
  with the bounds of each hour as written into `p_gen_pmin`/`p_gen_pmax`. No new variable
- cap/ramp conflict report: wherever a cap wins over a ramp (the first-hour clip moves a
  bound, or a later hour's limit is lifted), which consistent input data never need,
  `LpProblem` records it in `cap_ramp_conflicts` as `(timestep, generator, kind, MW)` (kind
  `"floor_up"` or `"ceiling_down"`) and warns once per window (`UserWarning`, up to ten
  conflicts listed). A storage unit running out of energy is not counted. A pinned unit
  with ramp rates conflicts at every jump of its pin; give it NaN ramps

Removed
- **breaking:** `ramp_daily_reset` is removed. It released a generator's ramp limit at every
  midnight, so a unit with zero ramp rates picked a level at midnight and held it all day.
  Give the unit its real ramp rates in `ramp_up_pu`/`ramp_down_pu` instead (NaN for no limit)
- **breaking:** the project-specific environment variable `T45_NUCLEAR_OPERATIONAL_LIMITS` is
  no longer read. It exempted nuclear generators with matching `pmax_ref`/`pmin_ref`
  profiles from their ramp limits (and their daily reset); the ramp limits now apply to
  every generator with ramp rates. A unit fully pinned by its profiles should have NaN ramp
  rates
- **breaking:** `ModelExtension.exempt_pinned_from_ramp` (and `RtExtension`'s setting of it)
  is removed. A pinned generator (`pmin_ref = pmax_ref`) follows its pin because caps win
  over ramps, in RT and in a plain run alike

Changes results? Yes, where the data have caps that move faster than the ramps:
- a unit now goes to a floor that rises faster than its ramp-up allows (it used to stay
  below it); a multi-period window with such a floor used to be infeasible and now solves
- a ceiling falling faster than the ramp-down is followed at no cost instead of through the
  priced `varRampSlack`
- a pinned generator with ramp rates follows its pin in a plain run too (it used to be held
  back by the ramp; in RT it was already exempt)
- units with `ramp_daily_reset` or covered by `T45_NUCLEAR_OPERATIONAL_LIMITS` are held to
  their ramp rates all day; data that relied on either must be changed
- units whose caps do not conflict with their ramps are unchanged

### Input data and grid
2026-11-13

Added
- `readProfileData(filename=...)` accepts a list of CSV or Parquet files, each cut to
  `timerange` and joined by column; a column name repeated across files, or a file without
  the rows of `timerange`, raises
- the generator columns `pmax_ref`, `pmin_ref` and the new `storage_ini_ref` are in
  `keys_powergama` with default `"const"`, meaning no profile. `storage_ini_ref` sets the
  initial filling to `storage_cap` times the profile at the first timestep, in place of
  `storage_ini`
- new optional generator column `block_group` (text, default `""`): the rows of one physical
  unit (e.g. price blocks) share a value, `""` is a group of one. The model does not read it.
  It survives the results database; databases written before read back with `""`.
  `GridData.from_dict` converts only the typed columns a table has, and a text column read
  from Parquet keeps a missing value missing (it became the text `"None"`)
- the storage-value time curves are ordinary profile columns: with no `storagevalue_time`,
  `storval_time_ref` (generators with storage) and `flex_storval_time` (consumers with
  flexible load) name a column of `grid.profiles`, read at the same row as every other profile
  reference; `"const"`, `""` and NaN are a factor of 1. Put the curve file in
  `readProfileData(filename=[...])`. The curves of every valued row are resolved and checked
  once when `LpProblem` is built (`GridContext.storval_*`), only for the rows the LP values
- grid data round trip through the results database: `GridData.from_dict`, and so
  `Results.from_existing`, keeps the inter-area table (`inter_area_rules`) that
  `Database.get_grid_data` reads (it was silently dropped); a grid table with a column named
  `index` survives (the row index is written under `index`, or `level_0` when the table has
  an `index` column, and read back from `level_0` when a table has both; databases written
  before read back the same way; a column named `level_0` raises); float columns missing
  throughout (e.g. `gen_lat`, `ramp_up_pu`, stored as NULL) read back as float, not `object`
  (`test_io::test_via_database` passes again); `Results.from_existing` adds missing optional
  columns with their defaults, so databases written before `pmin_ref`, `pmax_ref`,
  `storage_ini_ref` and `block_group` existed are read
- the results database writes `data_storval_time` only when there is a separate time table,
  and `Database.get_grid_data` / `GridData.from_dict` read a filling table without one. When
  a wide profile table (more than 1800 columns) is reduced for SQLite, the time curves of the
  storage rows and flexible consumers are kept with the other references, and an unknown one
  raises when the database is created

Removed and renamed
- **breaking:** the inter-area table `inter_area_ntc` is renamed `inter_area_rules`
  everywhere: the `readGridData(inter_area_rules=...)` keyword (no alias), the
  `GridData.inter_area_rules` attribute and `keys_powergama` entry, and the LP names
  `s_inter_area_rules` and `cInterAreaRulesForward`/`cInterAreaRulesBackward`. Its columns,
  and the NTC limits on the summed flow between two areas, are unchanged. Results databases
  store it as `data_inter_area_rules`
- **breaking:** the `LpProblem` argument `storage_initial_fill_scale` is removed. It scaled
  the initial filling level of generators by `type` (pumped hydro and batteries only); set
  `storage_ini` (or `storage_ini_ref`) in the generator data instead
- **breaking:** a `pmax_ref`, `pmin_ref` or `storage_ini_ref` that is neither `"const"`,
  empty, nor a profile column raises when the `LpProblem` is constructed (it used to mean no
  profile, silently)
- **deprecation:** `readProfileData(storagevalue_time=...)` warns (`DeprecationWarning`). It
  still works as before: every time reference is then a column of that table, for every row
  (no mixing). `FullProfiles` (fault scenarios) reads it only when given
- **breaking:** the filling curves (`storagevalue_filling`) must have the rows 0..100 and a
  column for every `storval_filling_ref` of a generator with storage and every
  `flex_storval_filling` of a consumer with flexible load; a grid with either and no filling
  table raises too. An unknown time curve name raises `ValueError` when `LpProblem` is built
  (it used to raise a `KeyError` at the first solve). No fallback
- `GridData.getDcBranchAreas` returns a copy of `dcbranch` with `area_from`/`area_to` and no
  longer adds them to `self.dcbranch`, so new results databases no longer store the two
  columns in `data_dcbranch`. It maps areas by node id instead of aligning on row position,
  which only differs for a DC branch table whose index is not `0..n-1`

Changes results? Only in these cases; existing valid data give the same results:
- a grid rebuilt from a results database (a re-run on `Results.from_existing(...).grid`) now
  has the inter-area NTC constraints the original run had: on the two-area case a re-run
  from the database reproduces the original objective, where it was lower without the NTC
- a run that used `storage_initial_fill_scale` starts from `storage_ini` unless the data are
  changed accordingly
- the storage-value time curves as profile columns give bit-identical results to the
  separate table on the two-area case; the strict references and checks only turn silent
  mistakes into errors

### Earlier
2026-11-09
- **breaking:** the inter-area table `inter_area_ntc` is renamed `inter_area_rules` everywhere:
  the `readGridData(inter_area_rules=...)` keyword (no alias), the `GridData.inter_area_rules`
  attribute and `keys_powergama` entry, and the LP names `s_inter_area_rules` and
  `cInterAreaRulesForward`/`cInterAreaRulesBackward`. Its columns are unchanged. Results
  databases store it as `data_inter_area_rules`
- **breaking:** the `LpProblem` argument `storage_initial_fill_scale` is removed. It scaled the
  initial filling level of generators by `type` (pumped hydro and batteries only); set
  `storage_ini` in the generator data instead

2026-10-21
- `readProfileData(filename=...)` accepts a list of CSV or Parquet files, each cut to
  `timerange` and joined by column; a column name repeated across files, or a file without
  the rows of `timerange`, raises. The generator columns `pmax_ref`, `pmin_ref` and the new
  `storage_ini_ref` are in `keys_powergama` with default `"const"`, meaning no profile.
  **behaviour change:** any other value that is not a profile column now raises when the
  `LpProblem` is constructed (it used to mean no profile, silently); empty cells still mean
  no profile. `storage_ini_ref` sets the initial filling to `storage_cap` times the profile
  at the first timestep, in place of `storage_ini`. Results are unchanged for existing data
- **breaking:** RT storage is settled on its power, not its filling level. Pumping is tracked
  against the day-ahead schedule through a new `generator.rt_pump_target_ref` profile column
  (a fraction of `pump_cap`), priced at c0 ± (deadband − storage value × pump_efficiency),
  and storage generation through `rt_target_ref` like any other generator (ported from
  761d88f by Valentin Chabaud). `generator.rt_storage_target_ref` and the
  `da_storage_marginalprice_parquet` lock file are removed and raise an error if set. RT
  used to drop the pumping cost with the operating cost, so pumping was free and a storage
  unit could pump and generate at once; now generating η MWh and pumping 1 MWh more costs
  c0(1 + η) + deadband. The RT debug stream reports `rt_pump_rows` and
  `rt_pump_target_count` in place of `rt_storage_rows` and `rt_storage_target_count`
- the storage value plot draws the pump threshold at storage value × pump_efficiency −
  deadband, the price below which pumping pays
- **behaviour change:** spilled power is limited to the inflow, so only inflow that cannot be
  stored is spilled, never stored or pumped energy (#33). Storage without inflow, such as
  batteries and pure pumped storage, can no longer spill, and in a surplus can no longer pump
  in power and spill it as a free sink in place of dump load. This replaces the
  `spill_cap_frac` generator column, which is now ignored with a warning, and removes the
  storage overflow slack (`varStorageOverflow`), which spill bounded by inflow makes
  unnecessary. The initial storage level is clipped into `[0, storage_cap]`, as
  `set_storage_state` already did
- the single-period and multi-period formulations are merged into one, `powergama.lp.formulation`:
  an LP over a window of timesteps, built once per window length and rewritten for every window
  (#42). Single-period is a window of one timestep; its results are unchanged
- **breaking:** model components are indexed by hour in the window, `[·, h]`; in single-period
  mode `h` is 0. `lp.model.varGeneration[i]` becomes `lp.model.varGeneration[i, 0]`, likewise
  `p_gen_pmin`, `p_demand`, `varAcBranchFlow`, the loss variables and the rest. Parameters that
  do not vary by hour (`p_gen_cost`, `p_genpump_cost`, `p_storage_prev`, `p_storage_value`,
  `p_flex_storage_prev`) keep their index
- **breaking:** for extensions, the hook is `edit_objective_terms(model, terms, h)`, called once
  per hour of the window; `LpProblem._updateLpProblem` takes the window's timesteps, and
  `_storeResultsAndUpdateStorage` takes the window, the number of committed timesteps and their
  objective values. `powergama.lp.single_period` and `powergama.lp.multi_period` are gone
- multi-period mode is no longer rebuilt for every window, which with HiGHS's state cleared
  between windows makes it 50-60% faster on the Europe 2014 case, and gains what the
  single-period solve loop had: any solver (it used to build its own HiGHS solver, and gave zero prices with any
  other), `warmstart`, `savefiles`, `_relax_and_retry`, progress reporting, and losses and
  extensions with `horizon_hours=1`
- `ramp_daily_reset` releases the ramp limit at midnight in both modes, whatever `commit_hours`
  is; multi-period mode no longer requires `commit_hours=24` with it
- multi-period results can differ from before between equally cheap dispatches: the window
  objectives, prices and storage values are the same, the tie between them is broken differently
- `solve(solver="appsi_highs")` without `solve_args` works: the default arguments are for the
  SolverFactory solvers, which the APPSI interface does not accept
- **behaviour change:** flexible load is a bounded reservoir in single-period mode too, as in
  multi-period (#42). Its level is an LP variable kept within `[0, capacity]` (no floor where
  `flex_on_off > 1`), instead of a state that could overshoot and was pushed back by
  `flexload_outside_cost` on the next timestep. Results change where the level used to leave
  that range
- **behaviour change:** in multi-period mode, a ramp-down limit that cannot be kept after a
  window's first hour (an energy-limited generator running out) no longer makes the window
  infeasible: the shortfall is penalised like load shedding. Where the limit can be kept,
  results differ at most between equally cheap schedules
- a multi-period window of one timestep now gives the single-period results in every respect,
  including flexible load
2026-10-19
- **behaviour change:** storage is modelled the same way in both modes, and multi-period mode no
  longer prices storage draw twice (#40). The objective no longer charges the storage value per
  MWh generated; it values the energy left in storage at the end of the LP (the timestep, or the
  window) at the storage value, with the storage level an LP variable in both modes. A
  multi-period window of one timestep now gives the single-period results, except with
  flexible load. Effects on results:
  - multi-period: storage was charged its storage value per MWh generated *and* through the
    terminal value, so it was held back and prices were set at about twice the storage value.
    Expect storage to be used more and prices to change
  - both modes: pumping is worth the storage value times `pump_efficiency`, less
    `pump_deadband`, instead of the storage value less `pump_deadband`. Pumping losses were
    ignored before. If `pump_deadband` was set to stand in for them, they now count twice
  - both modes: `fuelcost` applies to generators with storage too, as a cost per MWh on top of
    the storage value. It used to be ignored for them
  - single-period: spilled inflow loses its storage value, and spilled power is limited to
    `spill_cap_frac` × `pmax` (default 1), as it already was in multi-period. Energy that fits
    neither in the storage nor through the generator nor in the spill limit is penalised like
    load shedding, which can set a price of ±1000 at the generator's node. Multi-period runs that
    were infeasible in that situation now solve
  - multi-period: the first hour of each window applies the single-period ramp guard, so a
    generator that runs out of energy breaks its ramp-down limit instead of making the window
    infeasible
  - multi-period: inflow not used by generators without storage (e.g. curtailed wind) is now
    reported as spilled, as in single-period
- the objective is a cost rate in EUR/h in both modes, whatever `timeDelta` is, so nodal prices
  are in EUR/MWh at any timestep length. `Res_ObjFunc` stays a cost rate, with the value of the
  energy left in storage replaced by the value of the energy taken from it
- single-period runs are faster: the per-timestep update is vectorised and writes the model
  parameters in bulk, and the objective is evaluated once per timestep instead of twice. On the
  Europe 2014 case this more than offsets the larger LP
- pytest now treats every `xfail` as strict
2026-10-18
- fix `continue_from_last` with `lossmethod` 1 or 2 not reproducing an uninterrupted run (#41).
  Losses are linearised around the preceding timestep's flows; a resumed run pre-solved its first
  timestep instead. It now seeds them from the stored flows
- `lossmethod=1` now linearises DC branch losses around the net flow, as for AC branches since
  #29, instead of the directional flows. Results change only after a timestep with simultaneous
  flow in both directions on a DC branch
- **breaking:** real-time (balancing) simulation is now an extension, `RtExtension`, passed to
  the problem as `extensions=[...]` (#44). `LpProblem` no longer takes `is_rt`,
  `rt_balancing_fee_eur_per_mwh`, `rt_p_res_curtailment_factor`,
  `rt_xborder_flow_penalty_eur_per_mwh` or `rt_dispatch_objective_mode`, and the core model code
  no longer refers to RT at all. Extensions plug into the single-period model through the hooks
  of `powergama.lp.ModelExtension`, and the objective is built from named terms an extension can
  edit. To migrate:
  - `LpProblem(data, is_rt=True, rt_balancing_fee_eur_per_mwh=a, rt_p_res_curtailment_factor=b,
    rt_xborder_flow_penalty_eur_per_mwh=c)` becomes
    `LpProblem(data, extensions=[RtExtension(balancing_fee_eur_per_mwh=a, res_curtailment_factor=b,
    xborder_flow_penalty_eur_per_mwh=c)])`, with `from powergama.lp.rt import RtExtension`.
    Drop `rt_dispatch_objective_mode`; it never affected the model
  - RT state is read from the extension object, not the problem: `lp._idx_rt_wind` becomes
    `rt.ctx.idx_wind`, `lp._rt_target_gen_indices` becomes `rt.ctx.target_gen_indices`, the lock
    tables (`lp._foreign_gen_lock`, ...) become `rt.locks.foreign_gen_lock`, ...
  - `lp.append_rt_solver_debug_event(payload)` becomes `rt.append_debug_event(payload)`
  - `powergama.lp.project_locks` moved to `powergama.lp.rt.locks`
  - the input data is unchanged: `rt_target_ref` / `rt_storage_target_ref` profile columns and the
    lock-file attributes on `GridData` work as before
- RT inputs and model parts exist only with `RtExtension`. A normal run no longer declares the
  RT parameters and deviation variables (with `appsi_highs` these were unused solver columns in
  every run), derives RT context or reads the lock files
- **behaviour change:** the day-ahead lock parquets (`foreign_gen_lock_parquet` and friends) are
  RT inputs and are read and applied only by `RtExtension`. They used to apply in every mode,
  including multi-period, where they are now ignored
- RT results can differ from before in timesteps with several equally cheap solutions, because
  the RT variables and constraints are now declared after the core ones. Objective values are
  unchanged
- removed RT parameters that were written but never read, and the multi-period RT debug output,
  which could never be reached
- multi-period mode now applies the `POWERGAMA_HIGHS_*` environment variables the same way as
  single-period (#45): it honours `POWERGAMA_HIGHS_PARALLEL`, which it used to ignore, and
  warns on a malformed `POWERGAMA_HIGHS_RANDOM_SEED` or `POWERGAMA_HIGHS_THREADS` instead of
  silently dropping it
- fix branch capacity sensitivities being identically zero in single-period mode (#39). The
  branch rating was set both as a constraint and as a variable bound, and the solver attributed
  the binding limit to the bound, leaving the constraint dual that is reported at zero. With
  `lossmethod=1` the same applied to the directional flows, whose bounds are now 1% above
  capacity; this lets the unphysical two-way flow of #29 grow by up to 1% of capacity

2026-10-17
- rename simulation mode: `objective_mode="hourly"` is now `"single_period"`, since a timestep
  need not be an hour. The old spelling raises; there is no alias (this name was never released)
- reorganise the optimisation code into the `powergama.lp` package, split by formulation:
  `lp/single_period/`, `lp/multi_period/` and `lp/rt/` for the balancing side-path, each with
  its own model building, input data and result extraction. `powergama.LpProblem` is unchanged,
  so code using the documented import needs no edit
- **removed** the `powergama.LpProblemPyomo` module. It was only ever a path to `LpProblem`,
  which has always been exported as `powergama.LpProblem`; import that, or `powergama.lp`.
  Unlike the renames above this name did ship, so `from powergama.LpProblemPyomo import ...`
  will now raise `ModuleNotFoundError`
- the two formulations no longer share any model state, so either can be built without the other
- add [model_formulations.md](userguide/model_formulations.md), mapping the single-period and
  multi-period models onto each other and recording where they disagree

2026-10-16
- rename multi-period API: `objective_mode="daily_24h"` is now `"multi_period"`, and
  `objective_day_horizon_hours`/`objective_day_commit_hours` are now
  `horizon_hours`/`commit_hours` (these names were never released)
- fix wrong branch capacity sensitivities in multi-period mode (transposed constraint index)
- unsupported mode combinations now raise instead of silently computing the wrong thing:
  `is_rt` with `multi_period`, `lossmethod != 0` with `multi_period`, `ramp_daily_reset`
  with `commit_hours != 24`, and fault simulation outside `single_period` mode
- `continue_from_last=True` on an empty results database now raises a clear error

2026-10-01
- add multi-period and balancing simulation modes

## v1.5.1
2026-01-14
- improve simulation with losses (lossmethod=1)

## v1.5.0
2025-10-17
- add option to read model, including grid data, from sqlite file
- improved map plot
- add possibility to read grid data from multiple files (v1.4.2)
- add possibility to extend previous simulation (v1.4.1)


## v1.4.0
2025-01-29
- added support for Highs solver

## v1.3.0
2024-12-02
- added feature for fault scenario analysis

## v1.2.0
2024-06-12
- update user documentation
- speedup of model creation
- add option to use persistent solver
- change from pyomo AbstractModel to ConcreteModel formulation
- introduce more tests

## v1.1.4
2023-03-06
- moved repository to github
- moved powergim into separate repository (powergama/powergim)
- added pytest tests

## v1.1.3
2020-09-11
- powergim update (v0.1)

## v1.1.2 
2019-07-19
- map plotting using leaflet

## v1.1.1
2017-07-10

## v1.1.0
2017-01-12
- added Power Grid Investment Module (PowerGIM)
- small changes in input data format

## v1.0.4
- using pandas to read/write to file, and storing grid model data
- minor changes in input data format (csv file column headers)
- misc minor improvements

## v1.0.3 
2016-05-31 -- Version 1.
- added possibility for time dependence on flexible demand strategy (new column in input file)

## v1.0.2 
2015-03-27 -- Version 1.03-27
- improvements to result plotting
- python 3 compatible
- misc bug fixes

## v0.8.0
2014-12-04 -- Test version
- added pumped storage
- added flexible demand

## v0.7.0
2014-08-05 -- Test version
 
