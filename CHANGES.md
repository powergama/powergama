# Changes

## unpublished
2026-10-20
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
 
