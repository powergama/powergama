# Real-time (RT) runs

An RT run reproduces, in a simplified way, what happens to a day-ahead (DA) schedule when
the delivery hour comes and wind, sun and demand turn out different from their forecasts.
It starts from the schedule a day-ahead run produced and finds the cheapest way to balance
the forecast errors: which units deviate from their schedule, in which order, and what
cannot be absorbed (for example wind that has to be curtailed).

**What it is not.** In European power markets, the intraday market trades in 15-minute
products up to shortly before delivery, and the balancing market activates reserves on a
minute scale to keep the frequency. An RT run models neither: it stays at the resolution
of the day-ahead run, typically one hour, and represents the need for corrections to the
day-ahead schedule and the merit order in which they are made, not the internal mechanisms
of these markets. "Real-time" here means "this delivery hour", and "balancing" means
resolving the mismatch between the schedule and what actually happens.

**Why not simply re-run with the actual weather?** A second dispatch on the actual
profiles would re-shuffle the whole system for free, as if the day-ahead schedule, the
commitments it implies and the cross-border exchanges it settled could be undone at no
cost. An RT run instead keeps every unit on its day-ahead schedule unless deviating is
worth its price. Deviation prices set the merit order: a unit that may deviate cheaply is
used first, renewable curtailment can be made the last resort, and units that cannot
change their schedule at short notice are pinned to it.

This approach was developed for the WILLOW project (Horizon Europe), to predict wind
curtailment in Belgium; the modelling choices and their justification are in WILLOW
deliverable D4.5 and its modelling annex. The formulation is in
[model_description_RT.md](model_description_RT.md).

Technically, the RT run is an extension of the single-period mode: `RtExtension` attached
to an `LpProblem` with windows of one timestep. Everything it needs to know about the
day-ahead schedule is ordinary input data, so an RT run is set up with the same `GridData`
as any run, plus the columns of section 2. Without `RtExtension` these columns are ignored.

A complete example, on a European grid model from the WILLOW project, is the notebook
[example_notebook_willow_da_rt.ipynb](../tests/example_notebook_willow_da_rt.ipynb).

## 1. From a day-ahead run to an RT run

A day-ahead run and the RT run after it are **separate runs**, connected only by
files. `powergama.lp.rt.prepare_rt` reads a finished day-ahead run and writes the inputs of
section 2 for the RT run: a copy of the day-ahead grid tables with the RT
columns filled in, and one profile file with the day-ahead schedule.

```python
import powergama
from powergama.lp.rt import DeviationPrices, RtExtension, prepare_rt

# 1. DA run
data_da = powergama.GridData()
data_da.readGridData(nodes=..., ac_branches=..., dc_branches=..., generators=..., consumers=...)
data_da.readProfileData(filename=["profiles_da.csv"], timerange=range(24, 48), ...)
res_da = powergama.Results(data_da, "da.sqlite3")
powergama.LpProblem(data_da).solve(res_da, solver="appsi_highs")

# 2. prepare RT (also works on Results.from_existing("da.sqlite3", 1.0, timerange=range(24, 48)))
prices = DeviationPrices(fee=10, renewable=res_da.grid.generator["type"].isin(["wind", "solar"]))
prepared = prepare_rt(res_da, "rt_inputs", prices=prices)

# 3. RT run: the RT grid tables, and the DA-solution file next to the RT profiles
data_rt = powergama.GridData()
data_rt.readGridData(**prepared.grid_files())
data_rt.readProfileData(
    filename=["profiles_rt.csv", prepared.files["da_solution"]], timerange=range(24, 48), ...
)
res_rt = powergama.Results(data_rt, "rt.sqlite3")
powergama.LpProblem(data_rt, extensions=[RtExtension()]).solve(res_rt, ...)
```

## 2. The inputs of an RT run

In the example below, `rt_lock_profile` is the day-ahead schedule for the given
generator, as a fraction of its `pmax`, and its deviations cost 100 + `fuelcost` per MWh
above it and 100 − `fuelcost` per MWh below it.

```python
from powergama.lp.rt import RtExtension

data.profiles["rt_lock_profile"] = ...
data.generator.loc[index_gen, "rt_target_ref"] = "rt_lock_profile"
fuel = data.generator.loc[index_gen, "fuelcost"]
data.generator.loc[index_gen, ["rt_price_up", "rt_price_down"]] = [100 + fuel, 100 - fuel]

lp = powergama.LpProblem(data, extensions=[RtExtension()])
```

**Targets.** The day-ahead schedule of a unit is its target, a profile column named in
the unit's row:

| table | column | target of | as a fraction of |
|---|---|---|---|
| generator | `rt_target_ref` | its output | `pmax` |
| generator | `rt_pump_target_ref` | its pumping | `pump_cap` |
| consumer | `rt_target_ref` | its flexible demand | `demand_avg` |

Storage is settled on its power, not its filling level: the generation and the pumping of
a storage unit are tracked, and the storage value enters through the deviation prices.

**Deviation prices** are unit data, in EUR/MWh, like `pmax` and `pmax_ref`. "Up" is more
than the day-ahead schedule, "down" less:

| table | columns | prices the deviation of |
|---|---|---|
| generator | `rt_price_up`, `rt_price_down` | its output from `rt_target_ref` |
| generator | `rt_pump_price_up`, `rt_pump_price_down` | its pumping from `rt_pump_target_ref` |
| consumer | `rt_price_up`, `rt_price_down` | its flexible demand from `rt_target_ref` |

Each has an optional profile multiplier, the same name with `_ref` (`rt_price_up_ref`,
...); `"const"` (the default), an empty cell or NaN is a factor of 1. The price at
timestep t is `constant × profile(t)`. A row with a target needs both prices: a missing or
NaN price raises when the `LpProblem` is built, as does a `*_ref` that is not a profile
column (checked on every row) or a referenced profile with NaN. The one rule `RtExtension`
applies itself is the split-sum guard: a pair with `up + down < 1e-6` has the smaller one
raised so that they sum to 1e-6, so that both split deviation variables can't grow
together. The usual price rule (the balancing fee plus or minus the marginal cost, with
storage priced at its storage value) is written by `prepare_rt` (section 3).

**Units that follow the day-ahead schedule exactly.** A generator pinned to its day-ahead
output, or a consumer whose demand is fixed to the day-ahead value, needs no RT
input: it is ordinary grid data. Pin a generator by pointing both `pmin_ref` and
`pmax_ref` at a profile column holding its DA output as a fraction of `pmax`, with
`inflow_ref = "const"` and `inflow_fac = 1` so that the availability is not applied twice
(see [input_data.md](input_data.md), "Profile references that override a value"). Caps
always win over ramps, so such a generator stays on its output whatever its ramp rates.
Fix a demand by pointing `demand_ref` at a profile column holding the DA demand as a
fraction of `demand_avg`.

**Cross-border flows.** The inter-area rules table, which limits the summed flow between
two areas by its NTC (see [input_data.md](input_data.md), "Inter-area capacities"), can
also hold that summed flow to its day-ahead value in an RT run. Three optional
columns do this:

| column | meaning |
|---|---|
| `da_flow_ref` | profile column with the DA summed flow DA(t) of the rule, MW, `area_from` → `area_to` |
| `balancing_penalty` | p, EUR/MWh: each MWh of \|T − DA(t)\| costs p (default 0) |
| `lock_share` | λ in [0, 1]: T stays within DA(t) − (1 − λ)·`ntc_backward` and DA(t) + (1 − λ)·`ntc_forward` (default NaN: no lock) |

T is the summed flow the NTC constraint limits: AC and DC branches from an `area_from`
node to an `area_to` node positive, the other way negative. λ = 1 pins T to DA(t), λ = 0
leaves the full NTC either side of it, and an infinite NTC leaves that side open (unless
λ = 1). Penalty and lock act on the sum, not per branch: flow moving between parallel
branches of one boundary (an AC line and an HVDC link, say) costs nothing, as long as the
boundary's total keeps to the day-ahead value. A penalty and a lock on one rule work
together: deviations within the window cost p. A rule without `da_flow_ref` gets neither,
and a day-ahead run ignores all three columns. Building the `LpProblem` raises on a
`da_flow_ref` that is not a profile column or has NaN, a negative penalty, a `lock_share`
outside [0, 1], or a `da_flow_ref` on a rule with no branch across its boundary. The
formulation is in [model_formulations.md](model_formulations.md), "Objective".

## 3. What `prepare_rt` writes


`prepare_rt` reads the day-ahead run only through `res_da.grid` and
`res_da.get_dispatch_frames()` (see [output_data_and_analysis.md](output_data_and_analysis.md)),
and raises unless the run is complete. It always sets a target for every unit from its
day-ahead dispatch. The keyword options add the rest; each sets inputs that section 2
describes, or that any run reads:

| option | what it does | inputs it sets |
|---|---|---|
| (always) | the day-ahead dispatch as the target of every unit | `rt_target_ref`, `rt_pump_target_ref` |
| `fix`, `cap`, `zero_pmin` | pin units to their day-ahead output: exactly (`fix`) or at most (`cap`, optionally down to zero) | `pmin_ref`, `pmax_ref` |
| `carry_curtailment=CarryCurtailment(...)` | a unit keeps the curtailment it had in day-ahead, on its RT availability | `pmax_ref` (and `pmin_ref`) |
| `initial_storage` | storage units start from their day-ahead filling | `storage_ini_ref` |
| `peak_tranches=PeakTranches(...)` | adds, per unit, an RT-only row that offers what the unit did not sell in day-ahead, at a given price | new generator rows |
| `prices=DeviationPrices(...)` | deviation prices: balancing fee ± marginal cost, storage at its storage value at the day-ahead filling, renewables by their own rule | `rt_price_*`, `rt_pump_price_*` |
| `inter_area=InterAreaFlows(...)` | the day-ahead summed flow of each inter-area rule, with its penalty and lock | `da_flow_ref`, `balancing_penalty`, `lock_share` |

PowerGAMA knows no energy types here: which units are pinned, which are priced as
renewables and so on is decided by the caller, as boolean masks on the generator rows,
for example from `generator.type`:

```python
gen = res_da.grid.generator
prepared = prepare_rt(
    res_da,
    "rt_inputs",
    fix=gen["type"].eq("coal"),          # produces exactly its DA output
    cap=gen["type"].eq("gas"),           # produces at most its DA output
    zero_pmin=gen["type"].eq("gas"),     # ... and can go down to zero
    initial_storage=gen["storage_cap"].gt(0),
)
```

It writes into the output folder the RT grid tables (`prepared.grid_files()`
gives them as keyword arguments of `readGridData`), `da_solution.parquet` with every
day-ahead-derived profile column, and a summary of those columns and of the rows that use
them (`channels.csv`, `prepare_rt_summary.md`). Every option is also a function on its own,
for callers that build the RT tables themselves. The formulas, the files, the
checks and these functions are in
section 4.

## 4. Reference: `prepare_rt` and the functions behind it

The formula of every column `prepare_rt` derives from the day-ahead run, the files it
writes, the checks it makes, and each option as a function on its own, for callers that
build the RT tables themselves.

### Files and row mapping

`prepare_rt(res_da, outdir, *, timeMaxMin=None, text_output=False, ...)` reads only the DA grid
(`res_da.grid`) and the DA dispatch (`res_da.get_dispatch_frames()`, see
[output_data_and_analysis.md](output_data_and_analysis.md)), and raises unless
`res_da.is_complete()`. It writes into `outdir`:

- the RT grid tables, one CSV each: `node.csv`, `branch.csv`, `dcbranch.csv`,
  `generator.csv`, `consumer.csv` and the inter-area table if the DA grid has one.
  `prepared.grid_files()` gives them as keyword arguments of `readGridData`;
- `da_solution.parquet`, one wide profile file with the DA-derived columns, and only those
  the RT tables reference. Its index is the stored DA timestep, which is the row of the
  profile files the DA timestep was read from. `readProfileData` cuts it at the rows of
  `timerange` like the other profile files, so an RT run on the same rows lines up with
  the DA run, and an RT run on rows the DA run did not cover raises;
- with `text_output=True`, also `da_solution.csv`, for reading only;
- `channels.csv` and `prepare_rt_summary.md`, what the DA-derived columns are and which
  RT rows use them how (see "Channel summary").

Each RT generator and consumer row has a `da_index` column: the DA row it comes from, or
empty for a row added for RT only. The DA-derived columns are named after the DA row, so
RT rows from the same DA row share a column. The targets are fractions, clipped to
`[0, 1]`, in the units of the column they are read against:

| column | value | referenced by |
|---|---|---|
| `da_gen_<d>` | DA output of d / `pmax` of d | `generator.rt_target_ref`, every generator with `pmax > 0` |
| `da_pump_<d>` | DA pumping of d / `pump_cap` of d | `generator.rt_pump_target_ref`, every generator with `pump_cap > 0` |
| `da_cons_<d>` | DA flexible demand of d / `demand_avg` of d | `consumer.rt_target_ref`, every consumer with flexible load |
| `rt_target_zero` | 0 | the rows above that have no DA row |

Generators with `pmax <= 0` and consumers without flexible load get no target. A unit is
excluded from balancing by pinning it (`pmin_ref = pmax_ref`), not by dropping its
target. The references are strict: a name that is not a profile column raises when the
results database is created, so a misspelled name is never read as zero. A NaN in a DA
value that a target needs raises in `prepare_rt`.

`powergama.lp.rt.target_columns(frames, generator, consumer, ...)` is the rule on its own:
given the DA frames and RT tables with `da_index`, it returns the target columns and the
references, for callers that build the RT tables themselves. Such a caller starts from
`powergama.lp.rt.rt_tables_from_da(res_da)`, the table copy `prepare_rt` starts from: a
dict of copies of the DA node, branch, dcbranch, generator, consumer and (if any)
`inter_area_rules` tables, keyed like the `GridData` attributes, with `da_index` on every
generator and consumer row. Rows added for RT only get `da_index` NA. It gives the same
tables for the DA run's own `Results` and for one read back with `Results.from_existing`.

### Scaling

RT reads a target as a fraction of the RT row's own capacity: the generation
target is `pmax × profile`, the pumping target `pump_cap × profile`, the flexible-load
target `demand_avg × profile`. When an RT row keeps the capacity of its DA row, as in the
tables `prepare_rt` writes, that is the DA value. The same holds for `pmax_ref` and
`pmin_ref`, which override the bounds with `profile × pmax`. The inflow, on the other hand,
stays `pmax × inflow_fac × inflow_ref`: if `pmax_ref` points to a DA-derived column (a unit
pinned to its DA output), set `inflow_ref = "const"` for that unit, otherwise the
availability is applied twice. PowerGAMA does not warn about this.

### Pins

Which units can balance is decided by the caller, as boolean masks on the
generator rows (`pandas.Series` on the RT generator index; in `prepare_rt`, which keeps the
DA rows, that is the DA generator index). PowerGAMA knows no energy types here: a policy
such as "gas follows DA" is a mask the caller builds, e.g. from `generator.type`.

```python
gen = res_da.grid.generator
prepared = prepare_rt(
    res_da,
    "rt_inputs",
    fix=gen["type"].eq("coal"),          # produces exactly its DA output
    cap=gen["type"].eq("gas"),           # produces at most its DA output
    zero_pmin=gen["type"].eq("gas"),     # ... and can go down to zero
    carry_curtailment=CarryCurtailment(  # its RT availability less the DA curtailment
        gen["type"].eq("solar"), da_available=da_avail_mw, rt_available=rt_avail_mw, pin="fix"
    ),
)
```

| option | `pmin_ref` | `pmax_ref` |
|---|---|---|
| `fix` | `da_gen_<d>` | `da_gen_<d>` |
| `cap` | unchanged | `da_gen_<d>` |
| `cap` and `zero_pmin` | `rt_target_zero` | `da_gen_<d>` |
| `carry_curtailment`, `pin="cap"` (default) | unchanged | `da_carry_<d>` |
| `carry_curtailment`, `pin="fix"` | `da_carry_<d>` | `da_carry_<d>` |

`da_gen_<d>` is the unit's target column: the pin and the target read the same values.
Every pinned row also gets `inflow_ref = "const"` and `inflow_fac = 1`, so the
availability is not applied twice (see "Scaling"). `da_carry_<d>` carries the DA
curtailment `da_available − DA output` into RT:

    da_carry_<d> = ((rt_available + DA output − da_available).clip(0, pmax) / pmax).clip(0, 1)

`da_available` and `rt_available` are the unit's availability in MW (for example
`pmax × inflow_fac × inflow profile` of the DA and the RT profiles), computed by the
caller: frames with index = the DA timesteps written and one column per **DA row**. A row
in both `fix` and `cap`, `zero_pmin` outside `cap`, a pinned row with `pmax <= 0` or
without a DA row, a unit both pinned and in `carry_curtailment`, and NaN in a value that
is needed raise `ValueError`.

The same rules are available on their own for RT tables the caller builds:
`pin_columns(frames, generator, fix=..., cap=..., zero_pmin=...)` and
`carry_curtailment_columns(frames, generator, mask, da_available=..., rt_available=...,
pin=...)` return a `RefColumns` (profile columns and references, like `target_columns`);
`combine_columns(*parts)` joins them, raising if a column name has different values in
two parts or a table cell is set to two values, and `apply_refs(table, refs)` sets the
references on a table.

### Initial storage

With `initial_storage=mask` (a boolean `pandas.Series` on the
generator rows; in `prepare_rt`, which keeps the DA rows, that is the DA generator
index), the masked storage units start the RT run from the filling DA had at the RT
run's first timestep. Their `storage_ini_ref` points to `da_storage_<d>`, the DA filling
of d at the **start** of each DA timestep as a fraction of `storage_cap`:

| DA timestep | `da_storage_<d>` |
|---|---|
| the DA run's first | the filling DA started from: `storage_ini` clipped to [0, 1], or the DA `storage_ini_ref` profile at its first row |
| t after it | `Res_Storage` of d at t − 1 (the filling at the end of t − 1) / `storage_cap`, clipped to [0, 1] |

`LpProblem` reads `storage_ini_ref` at the run's first timestep, so an RT run on the DA
rows starts from the DA initial filling exactly, and one that starts at a later row t
from the DA filling at the end of t − 1 (up to the rounding of `/ storage_cap ×
storage_cap`). A masked row without storage, without a DA row, or NaN in a value that is
needed raise `ValueError`.

`initial_storage_columns(frames, generator, mask, da_generator=..., da_profiles=...)` is
the rule on its own, returning a `RefColumns` like `target_columns`; `frames["storage"]`
must start at the DA run's first timestep; it joins the other parts with
`combine_columns`.

### Peak tranches

With `peak_tranches=PeakTranches(mask, price, ramp_pu=None,
source=None)` (the mask on the DA generator rows), `prepare_rt` adds one RT row per unit
of the masked generators: a reserve the unit did not sell in DA, offered at `price`. A
unit is a `block_group` (the rows of one physical unit, e.g. its price blocks); a row with
an empty `block_group` is a unit of its own, and a `block_group` must be masked whole.
For each unit, in the order of its first row, the tranche is a copy of that first row with

| column | value |
|---|---|
| `pmax` | cap = the sum of the unit's `pmax` |
| `fuelcost` | `price` |
| `da_index` | NA (no DA row of its own), so its target is `rt_target_zero` |
| `pmax_ref` | `da_headroom_<d>`, d = the DA row of the unit's first row |
| `pmin_ref`, `inflow_ref`, `inflow_fac` | `""`, `"const"`, 1 |
| `ramp_up_pu`, `ramp_down_pu` | `ramp_pu` (>= 0); NaN, no ramp limit, with `ramp_pu=None` (the default) |

    da_headroom_<d> = ((cap − DA output of the unit's rows) / cap).clip(0, 1)

The tranche can produce what the unit had left over in DA. Everything else (`desc`,
`node`, `type`, `pmin`, ...) is the first row's. A unit whose cap is not positive gets no
tranche; a masked row without a DA row or with storage or pumping raises. The tranches
are appended at the row labels `len(generator)`, `len(generator) + 1`, ..., before the
targets and the other options are computed, whose masks stay on the DA rows. With
`prices`, a tranche is priced like any unit, around its own `fuelcost` (`price`).
`peak_tranche_columns(frames, generator, mask, *, price, ...)` is the rule on its own: it
returns the table with the tranche rows and their `RefColumns`, to compute the targets on
and join with `combine_columns`.

### Deviation prices

With `prices=DeviationPrices(fee, renewable=None,
curtailment_factor=2.0, storage_value=None, source=None)`, `prepare_rt` writes the
deviation prices of every RT row (section 2), the tranches included. `fee` is the
balancing fee c0, paid on every MWh of deviation; `renewable` is a boolean mask on the DA
generator rows (PowerGAMA knows no energy types: the caller decides which units are
priced as renewables); c is the unit's marginal cost, its `fuelcost` plus, for a unit with
storage, the storage value v at the **DA** filling level:

| rows | up | down |
|---|---|---|
| generators in `renewable` | −c0 | `curtailment_factor` × c0 |
| other generators | c0 + c | c0 − c |
| generators with pumping (pumping) | c0 − (v × `pump_efficiency` − `pump_deadband`) | c0 + (v × `pump_efficiency` − `pump_deadband`) |
| consumers with flexible load | c0 + v | c0 − v |

A price that does not vary in time is written as the constant (`*_ref = "const"`). Where it
varies, for storage units, the constant is 1 and the `*_ref` names a profile column in
`da_solution.parquet` holding the price in EUR/MWh: `da_price_up_<d>`,
`da_price_down_<d>`, `da_pump_price_up_<d>`, `da_pump_price_down_<d>`,
`da_cons_price_up_<d>`, `da_cons_price_down_<d>` (the `da_price` family, d the DA row).

v is `da_storage_values(res_da)`, unless `storage_value` gives other values
(`{"generator": frame, "consumer": frame}`, index = DA timesteps, columns = DA rows): the
value of stored energy at the filling the unit had in DA at the **start** of each
timestep, evaluated with PowerGAMA's own storage-value functions on the DA grid,
`storage_price × F(filling) × T(t)` for a generator and `flex_basevalue × F(filling) × T(t)`
for a flexible load. That is the value a single-period DA run used at that timestep
(`Res_Storage.marginalprice`).

`price_columns(generator, consumer, *, fee, renewable=None, curtailment_factor=2.0,
storage_value=None, flex_value=None, timesteps=None)` is the rule on its own, returning a
`RefColumns` like `target_columns`. Every generator row, every row with pumping and every
flexible consumer gets a price; rows without a target ignore theirs. A storage row
without a DA row or without a value, and NaN, raise `ValueError`.

### Inter-area rules

With `inter_area=InterAreaFlows(mask=None, balancing_penalty=None,
lock_share=None, source=None)`, `prepare_rt` writes the DA summed flow of each inter-area
rule, `da_flow_<k>` (k the rule's row; the `da_flow` family), and sets the rule's
`da_flow_ref` to it. The flow is the sum of the stored DA flows of the AC and DC branches
across the boundary, signed like the NTC constraint (from an `area_from` node to an
`area_to` node positive), in MW and not clipped. `mask` (boolean, on the rule rows)
selects the rules; by default every rule with a branch across its boundary, a rule
without one being skipped (one a mask selects raises). `balancing_penalty` (the
cross-border balancing penalty, EUR/MWh, ≥ 0) and
`lock_share` (λ in [0, 1]) are a scalar for every selected rule or a Series/dict by rule
row; `None` keeps the table's own value (by default `balancing_penalty` 0, `lock_share` NaN: no
lock). How an RT run uses them is in "Cross-border flows" in section 2.
`da_flow_columns(frames, rules, *, grid, mask=None, balancing_penalty=None, lock_share=None,
timesteps=None)` is the rule on its own, returning a `RefColumns` whose `inter_area_rules`
part holds the rule columns (`apply_refs` sets them on the table).


### Clearing references

`clear_refs(table, mask, columns, timesteps=..., which="generator")`
returns a `RefColumns` that sets `columns` to `""` (no profile) on the masked rows, with an
empty profile frame on `timesteps`, to join with the other parts in `combine_columns`, for
example to drop a unit's DA `pmin_ref`/`pmax_ref` in RT. A cell that another part sets to
a profile raises there.

### Channel summary

The DA-derived columns are the *channels* of the RT run. Next to
`da_solution.parquet`, `prepare_rt` writes (unless `summary=False`):

- `channels.csv`: one row per channel: `name`, `family` (`da_gen`, `da_pump`, `da_cons`,
  `da_carry`, `da_storage`, `da_headroom`, `da_price`, `da_flow`, `zero`, or `other` for a
  column the caller named), `formula`,
  `da_row` (from the name; for `da_flow_<k>` the rule row), `n_rows`, `usages` (the set,
  `;`-joined), and per referencing RT row, `;`-joined in the same order: `rt_rows`
  (`generator:<row>`, `consumer:<row>` or `inter_area_rules:<row>`), `usage`, `desc`,
  `node`, `area` (through the node table) and `type`. An inter-area rule has `area`
  `<area_from>-><area_to>` and `desc` `balancing_penalty <p>, lock_share <λ>`;
- `prepare_rt_summary.md`: a short overview: the DA run, its timesteps, the option counts
  and the files written; then per family the formula, the source, and how many rows use it
  how, grouped by area and then type (counts only); then the rows without any channel
  (generators with `pmax <= 0`, consumers without flexible load), counted the same way.

| usage | reference |
|---|---|
| `target` | generator `rt_target_ref` |
| `pump_target` | generator `rt_pump_target_ref` |
| `cons_target` | consumer `rt_target_ref` |
| `fix` | generator `pmin_ref` and `pmax_ref`, both to the channel |
| `cap` | generator `pmax_ref` only |
| `pmin` | generator `pmin_ref` only |
| `storage_ini` | generator `storage_ini_ref` |
| `price_up`, `price_down` | generator `rt_price_up_ref`, `rt_price_down_ref` |
| `pump_price_up`, `pump_price_down` | generator `rt_pump_price_up_ref`, `rt_pump_price_down_ref` |
| `cons_price_up`, `cons_price_down` | consumer `rt_price_up_ref`, `rt_price_down_ref` |
| `flow_ref` | inter-area rule `da_flow_ref` |

A row using one channel in two ways has both, joined by `+` (e.g. `target+fix`). The
sources default to the DA database for the DA families; `CarryCurtailment(...,
source="RT solar availability from profiles_rt.csv")` describes where the availabilities
come from, and `sources={"da_gen": ...}` and `summary_header={label: text}` override or add
lines. `summarize_channels(refs, generator, consumer, *, node=None, inter_area_rules=None,
sources=None, header=None)` returns the same as `(markdown, channels)` for RT tables the caller builds
(a cell that `refs` leaves unset is read from the table), and `write_channel_summary(refs,
generator, consumer, outdir, ...)` writes both files. Nothing in it knows energy types or
areas: it groups by the `area` and `type` columns as they are.

### Debug stream

With the JSONL debug stream on, an RT run reports one `rt_rule_rows` entry per held
inter-area rule (summed RT and DA flows, deviations, `balancing_penalty`, cost, the lock
window `lock_lb_mw`/`lock_ub_mw`) and their total as `term_breakdown.xborder_dev`; its
header lists `inter_area_rules_balancing_penalised` and `inter_area_rules_locked`.
