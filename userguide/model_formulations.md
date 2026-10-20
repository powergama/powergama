# One formulation, over a window of timesteps

PowerGAMA solves one LP, over a **window** of `H` consecutive timesteps. The two
simulation modes differ only in the window:

| | single-period | multi-period |
|---|---|---|
| selected by | `objective_mode="single_period"` (default) | `objective_mode="multi_period"` |
| window length `H` | 1 | `horizon_hours` |
| timesteps stored from each window | 1 | `commit_hours` (at most `horizon_hours`) |
| losses (`lossmethod` 1, 2) | yes | only with `horizon_hours=1` |
| extensions (e.g. `RtExtension`) | yes | only with `horizon_hours=1` |

A multi-period run with `horizon_hours=1` *is* a single-period run, and
`tests/test_sim_single_period_rolling_equiv.py` checks it gives the same results.

| | module |
|---|---|
| the model | [`lp/formulation/build.py`](../src/powergama/lp/formulation/build.py) |
| the data for each window | [`lp/formulation/data.py`](../src/powergama/lp/formulation/data.py) |
| the results of each window | [`lp/formulation/extract.py`](../src/powergama/lp/formulation/extract.py) |
| the solve loop | [`lp/problem.py`](../src/powergama/lp/problem.py) |

The model is **built once per window length**, and its mutable `Param`s are
rewritten for every window rather than the model rebuilt. Only a multi-period run
whose last window is cut short builds a second model, for that length.

## Index sets

Every variable, and every constraint that applies hour by hour, is indexed by the
entity and the hour `h` in the window: `varGeneration[i, h]`, `cPowerbalance[n, h]`.
In single-period mode `h` is always 0.

| set | |
|---|---|
| nodes, generators, loads, AC/DC branches | `s_node`, `s_gen`, `s_load`, `s_branch_ac`, `s_branch_dc` |
| pumping, storage, flexible load | `s_gen_pump`, `s_gen_storage`, `s_load_flex` |
| hours in the window | `s_h` |
| (generator, hour) pairs with a ramp limit to the hour before | `s_ramp_up`, `s_ramp_dn` — hours ≥ 1 only |
| NTC rows | `s_inter_area_ntc` |

They are initialised from the lists [`GridContext`](../src/powergama/lp/context.py)
holds.

## Variables

| quantity | |
|---|---|
| generation, pumping, flexible load | `varGeneration`, `varPump`, `varFlexLoad` |
| load shed, dump load | `varLoadShed`, `varDumpLoad` |
| AC/DC flow, voltage angle | `varAcBranchFlow`, `varDcBranchFlow`, `varVoltageAngle` |
| generator storage level, at the end of each hour | `varStorage` |
| inflow spilled | `varSpill` |
| flexible-load reservoir level | `varFlexStorage` |
| ramp-down shortfall, hours ≥ 1 | `varRampSlack` |
| directional flow and losses (`lossmethod=1`) | `varAcBranchFlow12/21`, `varLossAc12/21`, and the DC pair |
| RT deviations (`RtExtension`) | the `varRt*` family |

## Constraints

| physics | |
|---|---|
| nodal power balance | `cPowerbalance` |
| flow vs. angle difference, reference node | `cFlowAngle`, `cReferenceNode` |
| branch capacity | `cMaxFlowAc`, `cMaxFlowDc` — two-sided rows, whose duals are the capacity sensitivities |
| inter-area NTC | `cInterAreaNtcForward`/`Backward` |
| generator output limits | `cGenMaxLimit`/`cGenMinLimit`, against `p_gen_pmax`/`p_gen_pmin` |
| pumping | `cPump`, against `p_pump_pmax` |
| flexible load | `cFlexload`; reservoir `cFlexStorCont`, with capacity and floor as variable bounds |
| storage continuity | `cStorCont`, with capacity as a variable bound; spill limited to inflow by `cSpillInflow` |
| losses (`lossmethod=1`) | `cFlowAc`/`cFlowDc`, `cLossAc12/21`, `cLossDc12/21` |
| between consecutive hours (hours ≥ 1) | `cGenUbStorAvail`, `cPumpRoom`, `cRampUp`, `cRampDn` |
| RT target tracking (`RtExtension`) | `cRtTargetTracking` and four siblings |

### The first hour, and the hours after it

The window's first hour depends on the simulation state, which `data.py` computes
before the solve and writes into `Param`s:

- `p_storage_prev` and `p_flex_storage_prev`, the levels the window starts from;
- the generator bounds `p_gen_pmin[i, 0]`/`p_gen_pmax[i, 0]`, which include the
  availability from storage and the ramp limits around the previous dispatch. Where
  a generator cannot keep to its ramp-down limit — a CSP plant that runs empty — the
  floor is dropped (`pmin = min(pmin, pmax)`) rather than making the LP infeasible;
- `p_pump_pmax[i, 0]`, limited by the room left in the storage.

Later hours depend on the hour before, which is solved together with them, so the
same limits are constraints on its variables: `cStorCont` and `cFlexStorCont` chain
the levels, `cGenUbStorAvail` limits generation from storage to what the hour
before left, `cPumpRoom` limits pumping to the room it left, and `cRampUp`/`cRampDn`
limit the change in output. A ramp-down limit that cannot be kept has a slack,
`varRampSlack`, priced like load shedding, so a window is never infeasible. None of
these exist in a window of one timestep.

A generator with `ramp_daily_reset` is free to change its output at midnight
(`timestep % 24 == 0`): in the first hour its previous dispatch is forgotten, and
in later hours its ramp limit into midnight is lifted.

## Objective

`OBJ` minimises cost. `build.objective_terms` gives the terms of each hour by name
— operating cost, load shedding and other penalty slacks, and in the last hour the
value of the energy left in storage — and an extension may edit them
(`ModelExtension.edit_objective_terms`).

Storage is valued only at the end of the window: nothing per MWh generated from
storage, and the energy left in it valued at the storage value. That value is
fixed for the window from the filling level at its start, so in a multi-period
window it does not respond to the level within the window.

Every term is a cost rate in EUR/h; the terminal storage value is an amount in EUR
and is divided by `dt`. The objective stored for each timestep is that hour's
terms, with the value of the energy left in storage replaced by the value of the
energy taken from it during the hour.

## History

The two modes used to be separate formulations, `lp/single_period/` and
`lp/multi_period/`, which disagreed: multi-period priced storage draw twice (#40),
bounded flexible load where single-period did not, applied ramp limits without
single-period's guard, and was rebuilt for every window without the single-period
solve loop's features (`solver` choice and duals, `warmstart`, `savefiles`,
`_relax_and_retry`, progress reporting). They were made to agree and then merged
(#42).
