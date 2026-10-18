# Changes

## unpublished
2026-10-18
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
 
