# The two formulations, side by side

PowerGAMA poses the dispatch problem in two ways. They model the same power
system and should agree, but they do not — and the differences are not all
deliberate.

This document maps one onto the other, component by component, so that the
differences are visible rather than discovered. It is the reference for merging
them, which is the intended direction of travel.

| | single-period | multi-period |
|---|---|---|
| selected by | `objective_mode="single_period"` (default) | `objective_mode="multi_period"` |
| one LP covers | one timestep | `horizon_hours` timesteps |
| built by | [`lp/single_period/build.py`](../src/powergama/lp/single_period/build.py) | [`lp/multi_period/build.py`](../src/powergama/lp/multi_period/build.py) |
| data from | [`single_period/data.py`](../src/powergama/lp/single_period/data.py) | [`multi_period/data.py`](../src/powergama/lp/multi_period/data.py) |
| built | once, then mutated in place each timestep | rebuilt for every window |
| time-varying data enters as | mutable `Param`s rewritten per timestep | constants baked in at build time |
| storage state | an LP variable, `varStorage`, carried on the `LpProblem` between solves | an LP variable, `varStorage`, carried between windows |
| supports losses | yes (`lossmethod` 0/1/2) | no — rejected at construction |
| supports real-time | yes (`RtExtension`) | no — rejected at construction |

A window of length 1 is the single-period problem, except for flexible load.
See **Where they disagree**.

## Index sets

Every set is the same content in both, except that the multi-period model adds
an hour dimension. Each variable and constraint indexed `[x]` in one is indexed
`[x, h]` in the other.

| set | single-period | multi-period |
|---|---|---|
| nodes, generators, loads, AC/DC branches | `s_node`, `s_gen`, `s_load`, `s_branch_ac`, `s_branch_dc` | same |
| pumping, storage, flexible load | `s_gen_pump`, `s_gen_storage`, `s_load_flex` | same |
| hours in the window | — | `s_h` |
| NTC rows | `s_inter_area_ntc` | `s_ntc` |
| areas | `s_area` (declared, unused) | — |

Both are initialised from the same lists, which
[`GridContext`](../src/powergama/lp/context.py) now holds, so neither formulation
depends on the other having been built.

## Variables

| quantity | single-period | multi-period |
|---|---|---|
| generation, pumping, flexible load | `varGeneration`, `varPump`, `varFlexLoad` | same, `[·, h]` |
| load shed, dump load | `varLoadShed`, `varDumpLoad` | same, `[·, h]` |
| AC/DC flow, voltage angle | `varAcBranchFlow`, `varDcBranchFlow`, `varVoltageAngle` | same, `[·, h]` |
| generator storage level | `varStorage` | `varStorage` |
| inflow spilled | `varSpill` | `varSpill` |
| storage overflow (penalised slack) | `varStorageOverflow` | `varStorageOverflow` |
| flexible-load reservoir | *not a variable* — state on the `LpProblem` | `varFlexStorage` |
| directional flow and losses | `varAcBranchFlow12/21`, `varLossAc12/21`, and the DC pair | — (no losses) |
| RT deviations | the `varRt*` family | — (no RT) |

## Constraints

Only `cPowerbalance` and `cFlowAngle` keep their names across both. The rest are
the same physics under different names, or genuinely only exist on one side.

| physics | single-period | multi-period |
|---|---|---|
| nodal power balance | `cPowerbalance[n]` | `cPowerbalance[n, h]` |
| flow vs. angle difference | `cFlowAngle[b]` | `cFlowAngle[b, h]` |
| reference node | `cReferenceNode` constraint | angle variables *fixed* in a loop, no component |
| branch capacity | `cMaxFlowAc`, `cMaxFlowDc` — one two-sided `pyo.inequality` | `cAcFlowLb`/`cAcFlowUb`, `cDcFlowLb`/`cDcFlowUb` — four one-sided rows |
| inter-area NTC | `cInterAreaNtcForward`/`Backward` | `cNtcFwd`/`cNtcBwd` |
| generator output limits | `cGenMaxLimit`/`cGenMinLimit`, against mutable `p_gen_pmax`/`p_gen_pmin` | `cGenUbNs`, `cGenLbNs`, plus `cGenUbStorAvail`, `cGenUbStorCap`, `cGenLbStor` for storage |
| pumping | `cPump`, with `p_pump_pmax` recomputed per timestep | `cPumpCap` + `cPumpRoom` |
| flexible load | `cFlexload` | `cFlexUb` |
| ramp limits | **no constraint** — variable bounds tightened each timestep | `cRampUp`, `cRampDn` |
| storage continuity | `cStorCont`; capacity and spill limits as variable bounds | the same |
| flexible-load reservoir | **no constraint** — level tracked outside the LP | `cFlexStorCont`, `cFlexStorCap`, `cFlexStorFloor` |
| losses | `cFlowAc`/`cFlowDc`, `cLossAc12/21`, `cLossDc12/21` | — |
| RT target tracking | `cRtTargetTracking` and three siblings | — |

The pattern: wherever the single-period path keeps state *outside* the LP and
enforces it between solves, the multi-period path has to make it a variable with
a continuity constraint, because the whole window is solved at once. Generator
storage is the exception: both make its level a variable, and the single-period
path carries it from one solve to the next.

## Objective

Both are named `OBJ` and minimise cost, and both value storage the same way:
nothing per MWh generated from storage, and the energy left in storage at the end
of the LP valued at the storage value, as a **terminal storage value** term. In
single-period the end is the end of the timestep; in multi-period, of the window.
The storage value is fixed for the LP from the filling level at its start, which
in multi-period means it does not respond to the level within a window.

Every per-step term is a cost rate in EUR/h; the terminal term is an amount in
EUR and is divided by `dt`. The multi-period objective sums the per-step terms
over `h`.

## Where they disagree

A window of 1 is the single-period problem for everything but flexible load:
`tests/test_sim_single_period_rolling_equiv.py` checks that dispatch, storage,
flows and prices agree on `data_9bus` and on `two_area` without its flexible load.
Storage used to be the main exception — multi-period priced it twice, per MWh
generated and through the terminal value (issue #40).

**1. Ramp limits bind differently for energy-limited generators, after the first
hour of a window.** The single-period path applies `pmin_now = min(pmin_now,
pmax_now)`, which drops the ramp-down floor whenever the availability bound is
tighter — a CSP plant that empties ramps down past its limit rather than making
the timestep infeasible. Multi-period applies the same guard to the first hour of
each window, whose availability is known before the solve, but not to later
hours, where it depends on the solve. A window can therefore still be infeasible
where single-period steps would not be.

**2. Flexible load is bounded in multi-period, unbounded in single-period.**
`varFlexStorage` carries the reservoir with a continuity equality and hard
`[0, capacity]` bounds, so it cannot leave range. The single-period path has no
bounds at all: it lets the level overshoot and prices that at
`constants.flexload_outside_cost` on the *next* timestep, which means it can
report a filling level above capacity. The multi-period floor is skipped where
`flex_on_off > 1`, since the per-hour charge limit is then below the drain and no
schedule could hold the level non-negative. Pinned as a strict xfail in
`tests/test_sim_single_period_rolling_equiv.py`.

## Features the multi-period path silently ignores

Supported by the single-period solve loop and not implemented for windows:
`savefiles`, `warmstart`, `solver_path`, `aclossmultiplier`/`dclossmultiplier`,
`_relax_and_retry`, `_update_progress`, the shared `dual` handle, and the
`lossmethod` pre-solve. Deciding which of these a merged formulation gains and
which get deleted is part of the merge, not a precondition for it.
