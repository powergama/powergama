# Real-time (RT) run: model description

This page describes the model behind the real-time (RT) extension: what it represents, how
deviations from the day-ahead schedule are priced, and its formulation. How to set up an
RT run is in [Usage_RT.md](Usage_RT.md). The approach was developed in the WILLOW project
(Horizon Europe) to predict wind curtailment; WILLOW deliverable D4.5 and its modelling
annex give the full justification and a calibrated case study.

## Two stages: day-ahead, then real time

Electricity markets do not clear in one step. The day-ahead (DA) market clears the
dispatch of the next day, typically around noon the day before, on **forecasts** of wind,
solar and demand. When the delivery hour comes, the actual values differ from the
forecasts, and the difference has to be balanced:

$$\Delta P_\text{net}(t) = P_\text{actual}(t) - P_\text{forecast}(t).$$

PowerGAMA represents this with two runs:

1. a **day-ahead run** on the forecast profiles, usually in multi-period mode with a
   rolling horizon (see [model_description.md](model_description.md)), whose dispatch is
   the day-ahead schedule;
2. an **RT run** on the actual profiles, time step by time step, which starts from that
   schedule and pays for every deviation from it.

In real systems, the corrections are made in the intraday market (continuous trading in
15-minute products up to shortly before delivery) and by the transmission system operator
activating reserves in the balancing market (on a minute scale). The RT run models
neither of these mechanisms. It keeps the resolution of the day-ahead run, typically one
hour, and represents the **need** for corrections and the order in which the system makes
them. "Real time" means "this delivery hour", and "balancing" means resolving the mismatch
between the schedule and what happens.

## Why deviations from the schedule are priced

Simply re-running the dispatch on the actual profiles would let the optimisation
re-shuffle the whole system at no cost, as if the day-ahead schedule, the commitments of
power plants and the cross-border exchanges it settled could be undone freely. That
overestimates the flexibility of the system, and underestimates for example how much wind
has to be curtailed.

The RT run instead minimises the cost of **deviating** from the schedule. For each
generator, pump and flexible load with a day-ahead target, the deviation is split into an
upward part $d^+$ (more than scheduled) and a downward part $d^-$ (less than scheduled),
each with its own price:

$$\min \sum_i \left( c^+_i\, d^+_i(t) + c^-_i\, d^-_i(t) \right) + \text{border deviations} + c_\text{shed} \sum_n \left( P^\text{shed}_n(t) + P^\text{dump}_n(t) \right).$$

Load shedding and dump load keep their cost, so they remain the last resort. The prices
are input data, one pair per unit, so a study decides how each unit balances.

## Deviation prices and the merit order

The prices set the **merit order** in which the system balances a forecast error. A usual
choice, which `prepare_rt` writes (`DeviationPrices`), builds them around a **balancing
fee** $c_0$ and the unit's marginal cost $c_i$:

$$c^+_i = c_0 + c_i, \qquad c^-_i = c_0 - c_i.$$

Producing more than scheduled costs the fuel plus the fee; producing less saves the fuel
but still costs the fee. $c_i$ is the opportunity cost of the unit: its fuel cost, and
for a unit with storage its storage value $v$ (for pumping, the value of the pumped energy
$v\,\eta - $ deadband). The storage value is taken at the filling level the unit had in
the day-ahead run at that hour.

The fee has two roles. It makes every deviation cost something, so the RT run does not
move units away from their schedule for nothing. And it protects the day-ahead schedule
from the RT run's short view: the day-ahead run saw a whole window ahead, while the RT run
sees one time step, and without a fee it would, for example, empty a storage unit early
because it cannot see the peak that the storage was kept for. The fee should stay
moderate, so as not to hide the cost differences between the flexibility options.

Renewable units are priced differently:

$$c^+_\text{RES} = -c_0, \qquad c^-_\text{RES} = f\, c_0, \quad f > 1.$$

When there is more wind or sun than forecast, producing it is profitable ($-c_0$), so the
RT run uses it rather than keeping to the lower schedule. Reducing a renewable unit below
its schedule costs more than any other downward option ($f\,c_0 > c_0 - c_i$), so
curtailment comes last. During a surplus, the resulting order is, for example: reduce the
flexible thermal units within their ramp limits, charge storage and use flexible demand,
export within what the borders allow (see below), and only then curtail renewables. Dump
load, at the cost of load shedding, should never be needed.

## Units that cannot balance

Some units cannot change their schedule at short notice, and some deviations are decided
elsewhere. These are inputs to the RT run, not prices:

- **Pinned units** produce exactly their day-ahead output (minimum and maximum output both
  set to it), or at most it (they can go down but not up). Typical examples are inflexible
  baseload plants committed in the day-ahead market.
- **Carried curtailment.** A unit can be held to its actual availability less the
  curtailment the day-ahead run already decided, so that curtailment is decided once. This
  suits units whose curtailment is managed outside the transmission grid, for example
  solar connected to distribution networks.
- **Peak tranches.** Units that may only go down from their schedule can be given an extra,
  more expensive RT-only unit for upward balancing, with the capacity they did not sell in
  the day-ahead market: installed capacity minus the day-ahead output of the whole unit.
  It represents fast peaking plants (open-cycle gas turbines, say) that are not committed
  in the day-ahead market.
- **Storage** starts the RT run from its day-ahead filling, and is settled on its power
  (generation and pumping), not on its filling level. A study can give the RT run its own
  storage value curves, for example flat ones that let short-term storage use its whole
  capacity for balancing.

## Cross-border flows

Cross-border capacity cannot be freely re-allocated at short notice: the day-ahead market
settles the commercial exchanges between areas up to their net transfer capacities (NTC),
and in real time only a residual margin is available for balancing across borders. An RT
run can represent this per inter-area rule, on the **net transfer** $T$ between two areas,
the same quantity the NTC limits:

- a **lock** keeps $T$ within a corridor around its day-ahead value,
  $\text{DA}(t) - (1-\lambda)\,\text{NTC}_{B\to A} \le T \le \text{DA}(t) + (1-\lambda)\,\text{NTC}_{A\to B}$,
  where the lock share $\lambda$ is the part of the NTC held to the day-ahead exchange
  ($\lambda = 1$: no change; $\lambda = 0$: the full NTC is available);
- a **penalty** $p$ per MWh of $|T - \text{DA}(t)|$, which makes the RT run balance within
  an area before it uses its borders.

Both act on the net transfer, not on each branch: flow may move between the parallel
branches of a border (an AC line and an HVDC link, say) as long as the net transfer keeps
to the corridor. The NTC itself still holds.

## Formulation

The extension drops the operating cost and the storage
value and adds the priced deviations from the DA schedule (load shedding and the other
slacks stay):

    Σ_i (c⁺_i(t) · d⁺_i + c⁻_i(t) · d⁻_i)  +  Σ_r p_r · (d⁺_r + d⁻_r)

over the tracked generators (output), pumps (pumping) and flexible consumers (flexible
demand), with `x − target = d⁺ − d⁻`, `d⁺, d⁻ ≥ 0`, and the inter-area rules r with a
`da_flow_ref` and `balancing_penalty` p_r > 0 (`varRtRuleDevPos`/`Neg`, `cRtRuleFlowTracking`):

    T_r − DA_r(t) = d⁺_r − d⁻_r,
    T_r = Σ_{ac_pos} f_b − Σ_{ac_neg} f_b + Σ_{dc_pos} f_b − Σ_{dc_neg} f_b

T_r is the summed flow of the NTC rows (`cInterAreaRulesForward`/`Backward`, the same
branches and signs) and DA_r(t) the rule's `da_flow_ref` profile, MW. A rule with a
`da_flow_ref` and a `lock_share` λ_r is also held to a window around DA
(`cRtRuleLockForward`/`Backward`, constraints with mutable right-hand sides written each
timestep):

    DA_r(t) − (1 − λ_r) · NTC⁻_r  ≤  T_r  ≤  DA_r(t) + (1 − λ_r) · NTC⁺_r

with NTC⁺/NTC⁻ the rule's `ntc_forward`/`ntc_backward`; λ_r = 1 is T_r = DA_r(t), and an
infinite NTC drops that side (for λ_r < 1). The penalty and the lock are on the sum: flow
moving between parallel branches of one boundary costs nothing and leaves the
window as it is. The prices
are unit data: c⁺_i(t) = `rt_price_up` × profile(t) of `rt_price_up_ref` (1 for
`"const"`), c⁻_i(t) likewise from `rt_price_down` (`rt_pump_price_*` for pumping), with
the one guard c⁺ + c⁻ ≥ 1e-6 (the smaller is raised). Nothing in the extension depends on
the unit's type, cost or filling level; `price_columns` writes c0 ± marginal cost, with the
storage value at the DA filling level (see
[Usage_RT.md](Usage_RT.md), section 4, "Deviation prices").
