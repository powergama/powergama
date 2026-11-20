# Changes

## v2.0.0 (planned)
2026-xx-xx
- add multi-period simulation mode (rolling horizon, `objective_mode="multi_period"`)
- add real-time / balancing simulation as a second step after day-ahead
  (`RtExtension`), with `prepare_rt` to build its inputs from a day-ahead run
- add `ModelExtension` hooks for optional additions to the model
- single-period and multi-period share one LP formulation over a window of timesteps
- **breaking:** storage is valued at the end of the window instead of per MWh drawn;
  pumping now accounts for `pump_efficiency`, and `fuelcost` also applies to storage units.
  Results change
- **breaking:** spilling is limited to inflow; `spill_cap_frac` is ignored
- flexible load is modelled as a bounded reservoir
- caps (`pmin`/`pmax`, profiles, storage) always win over ramp limits; conflicts are reported
- objective is a cost rate in EUR/h, so prices are in EUR/MWh at any timestep length
- new optional inputs: `pmax_ref`, `pmin_ref`, `storage_ini_ref` profiles, `block_group`,
  inter-area NTC rules (`inter_area_rules`); storage-value time curves as profile columns
- `readProfileData` accepts a list of CSV/Parquet files
- invalid profile references now raise instead of being silently ignored
- Results API: `get_dispatch_frames`, `is_complete`, `from_existing(timerange=...)`;
  grid data round-trips through the results database
- fix branch capacity sensitivities being zero in single-period mode
- fix `continue_from_last` with losses not reproducing an uninterrupted run
- **breaking:** `powergama.LpProblemPyomo` removed; import `LpProblem` from `powergama`

## v1.5.2
2026-09-29
- support Python 3.14

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
 
