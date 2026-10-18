# Changes

## unpublished
2026-10-18
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
 
