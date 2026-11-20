# Real-time (RT) runs

An RT run takes the schedule produced by a day-ahead (DA) run and replays the same period
with the **actual** wind, solar and demand instead of their forecasts. Every unit starts
from its day-ahead schedule, and the RT run finds the cheapest way to cover the
differences: which units produce more or less than scheduled, in which order, and what
cannot be absorbed (for example wind that has to be curtailed). PowerGAMA does this hour
by hour, as a simplified representation of how forecast errors are balanced when the
delivery hour comes. It does not model the intraday or balancing markets themselves.

[model_description_RT.md](model_description_RT.md) explains the ideas: why deviations are
priced, how the prices set the order in which units are used, and the formulation. This
page explains how to set up and run it. A complete example, on a European grid model from
the WILLOW project, is the notebook
[example_notebook_willow_da_rt.ipynb](../tests/example_notebook_willow_da_rt.ipynb).

## 1. The workflow in three steps

1. **Day-ahead run.** An ordinary PowerGAMA run on the forecast profiles, usually in
   multi-period mode with a rolling horizon (see [usage.md](usage.md)). Its dispatch is the
   day-ahead schedule.
2. **Preparing the RT run.** The function `prepare_rt` reads the results of the day-ahead
   run and writes the input files of the RT run into a folder.
3. **RT run.** An ordinary PowerGAMA run on those files and on the actual profiles, with
   `RtExtension` added to the problem. `RtExtension` replaces the usual cost of generation
   by the cost of deviating from the day-ahead schedule, and solves one timestep at a time.

```python
import powergama
from powergama.lp.rt import DeviationPrices, RtExtension, prepare_rt

# 1. Day-ahead run, on the forecasts
data_da = powergama.GridData()
data_da.readGridData(nodes=..., ac_branches=..., dc_branches=..., generators=..., consumers=...)
data_da.readProfileData(filename=["profiles_da.csv"], timerange=range(24, 48), ...)
res_da = powergama.Results(data_da, "da.sqlite3")
lp_da = powergama.LpProblem(data_da, objective_mode="multi_period", horizon_hours=24, commit_hours=12)
lp_da.solve(res_da, solver="appsi_highs")

# 2. Write the inputs of the RT run into the folder "rt_inputs"
renewable = res_da.grid.generator["type"].isin(["wind", "solar"])
prepared = prepare_rt(res_da, "rt_inputs", prices=DeviationPrices(fee=10, renewable=renewable))

# 3. RT run, on the actual profiles plus the file of day-ahead values
data_rt = powergama.GridData()
data_rt.readGridData(**prepared.grid_files())
data_rt.readProfileData(
    filename=["profiles_rt.csv", prepared.files["da_solution"]], timerange=range(24, 48), ...
)
res_rt = powergama.Results(data_rt, "rt.sqlite3")
powergama.LpProblem(data_rt, extensions=[RtExtension()]).solve(res_rt, solver="appsi_highs")
```

`prepare_rt` writes three kinds of file into the folder:

- **The grid tables of the RT run** (`node.csv`, `generator.csv`, ...). They are copies of
  the day-ahead grid tables, so the RT run has the same grid and the same units. What
  differs are the columns that tell the RT run about the day-ahead schedule: for each
  generator and consumer, which profile column holds its day-ahead schedule (its
  *target*), what it costs to deviate from it (its *deviation prices*), and, for units that
  must follow the schedule, a fixed output. `prepared.grid_files()` gives the files as
  keyword arguments of `readGridData`.
- **`da_solution.parquet`**, a profile file with the day-ahead values these columns refer
  to, for example the day-ahead output of each generator in each hour, as a fraction of its
  installed capacity. It is read together with the actual profiles, as in step 3. Its rows
  are numbered like the profile files of the day-ahead run, so the RT run must use the same
  `timerange` (or part of it).
- **A summary**, `prepare_rt_summary.md` and `channels.csv`, listing every day-ahead value
  written and which units use it, and how. Reading it is the quickest way to check an RT
  setup.

`prepare_rt` reads only the grid and the dispatch of the day-ahead run, through `Results`.
It can therefore also start from a run saved earlier:
`res_da = powergama.Results.from_existing("da.sqlite3", 1.0, timerange=range(24, 48))`.
It raises an error if the day-ahead run did not complete every timestep.

## 2. Deciding how each unit behaves in the RT run

By default, `prepare_rt` gives every unit its day-ahead dispatch as target, and lets it
deviate at the price it is given. Real units are not all that flexible at short notice,
and some decisions were already taken at the day-ahead stage. The keyword options of
`prepare_rt` describe this, one aspect each:

| option | what it represents |
|---|---|
| `prices=DeviationPrices(...)` | the cost of deviating from the schedule, which sets the order in which units are used |
| `fix=...` | units that produce exactly their day-ahead output, for example inflexible baseload plants |
| `cap=...`, `zero_pmin=...` | units that may reduce their output but not exceed their day-ahead output, optionally down to zero |
| `carry_curtailment=CarryCurtailment(...)` | units that keep the curtailment decided at the day-ahead stage, on their actual availability |
| `peak_tranches=PeakTranches(...)` | an extra, more expensive unit added to the RT run, offering the capacity a unit did not sell in day-ahead |
| `initial_storage=...` | storage units start the RT run from their day-ahead filling |
| `inter_area=InterAreaFlows(...)` | the exchange between areas is held close to its day-ahead value |

Which units an option applies to is given as a **mask**: a True/False value for each row of
the generator table, as a `pandas.Series` on its index. PowerGAMA does not know energy
types, so the study decides, typically from the column `type`:

```python
gen = res_da.grid.generator
gas = gen["type"].eq("gas") & gen["pmax"].gt(0)
prepared = prepare_rt(
    res_da,
    "rt_inputs",
    prices=DeviationPrices(fee=50, renewable=gen["type"].isin(["wind", "solar"]), curtailment_factor=1.1),
    fix=gen["type"].isin(["coal", "biomass"]) & gen["pmax"].gt(0),
    cap=gas,
    zero_pmin=gas,
    peak_tranches=PeakTranches(gas, price=100.0),
    initial_storage=gen["storage_cap"].gt(0),
)
```

**Deviation prices.** `DeviationPrices(fee, renewable=None, curtailment_factor=2.0)` writes
two prices per unit, in EUR/MWh: one for producing more than scheduled ("up"), one for
producing less ("down"). They are built around a balancing fee $c_0$ (`fee`) and the
unit's marginal cost $c$:

| units | up | down |
|---|---|---|
| renewable units (the `renewable` mask) | $-c_0$ | `curtailment_factor` × $c_0$ |
| other generators | $c_0 + c$ | $c_0 - c$ |
| pumping of storage units | $c_0 - (v \cdot$ `pump_efficiency` $-$ `pump_deadband`$)$ | $c_0 + (v \cdot$ `pump_efficiency` $-$ `pump_deadband`$)$ |
| consumers with flexible load | $c_0 + v$ | $c_0 - v$ |

$c$ is the unit's `fuelcost`, plus, for a unit with storage, its storage value $v$. The
storage value is taken at the filling level the unit had in the day-ahead run at that
hour, so it varies in time. Why the prices have this shape is explained in
[model_description_RT.md](model_description_RT.md), "Deviation prices and the merit
order".

**Fixed and capped units.** `fix` holds a unit at exactly its day-ahead output. `cap`
allows it to produce less, but not more; with `zero_pmin` as well, it may go down to zero
whatever its `pmin`. Technically, both set `pmin_ref` and `pmax_ref` (see
[input_data.md](input_data.md)) to the unit's day-ahead output.

**Carried curtailment.** Some units, typically solar connected to distribution networks,
should not be curtailed by the RT run, but keep the curtailment the day-ahead run decided.
`CarryCurtailment(mask, da_available, rt_available, pin="cap")` sets their output to the
actual availability less the day-ahead curtailment. It needs the availability of each unit
in the day-ahead and in the actual profiles, in MW, as two tables with one row per
day-ahead timestep and one column per generator row: for example installed capacity ×
`inflow_fac` × the inflow profile. The notebook shows how to compute them. `pin="fix"`
holds the unit at that output; `pin="cap"` (the default) only caps it there.

**Peak tranches.** A unit that may only go down in the RT run (with `cap`) cannot help when
more power is needed. `PeakTranches(mask, price, ramp_pu=None)` adds, for each masked unit,
one extra generator row to the RT run, offering the capacity the unit did not sell in
day-ahead (installed capacity minus day-ahead output) at the given `price`. It stands for
fast peaking plants that were not scheduled day-ahead. A unit split into several rows
(price blocks, marked by `block_group`, see [input_data.md](input_data.md)) gets one
tranche for the whole unit. By default the tranche has no ramp limit; `ramp_pu` sets one.

**Start filling of storage.** With `initial_storage`, the masked storage units start the RT
run with the filling they had in the day-ahead run at the first hour of the RT run.

**Exchanges between areas.** With `inter_area=InterAreaFlows()`, `prepare_rt` writes the
day-ahead net transfer of each inter-area rule (the summed flow over the border, as the NTC
limits it), so that the RT run can hold the exchange close to it. How close is set per
rule by two columns of the inter-area table: `balancing_penalty`, a cost per MWh of
deviation, and `lock_share`, the share of the NTC that stays fixed at its day-ahead value
(see "Exchanges between areas" in section 3). `InterAreaFlows(balancing_penalty=...,
lock_share=...)` can set them for all rules at once instead.

## 3. What the RT run reads

`prepare_rt` writes everything below. A study that builds the inputs of its RT run in
another way needs to provide the same columns.

**Targets.** The day-ahead schedule of each unit is a profile column, named in the unit's
row:

| table | column | the profile holds | as a fraction of |
|---|---|---|---|
| generator | `rt_target_ref` | the day-ahead output | the installed capacity `pmax` |
| generator | `rt_pump_target_ref` | the day-ahead pumping | `pump_cap` |
| consumer | `rt_target_ref` | the day-ahead flexible demand | `demand_avg` |

Storage units are settled on their output and pumping, like any unit; their filling level
has no target. A unit that should not deviate at all is pinned (see below), not left
without a target.

**Deviation prices.** Each unit with a target has two prices, in EUR/MWh: "up" for more
than its target, "down" for less.

| table | columns | price the deviation of |
|---|---|---|
| generator | `rt_price_up`, `rt_price_down` | the output from `rt_target_ref` |
| generator | `rt_pump_price_up`, `rt_pump_price_down` | the pumping from `rt_pump_target_ref` |
| consumer | `rt_price_up`, `rt_price_down` | the flexible demand from `rt_target_ref` |

A price can vary in time: each column has an optional companion with `_ref` added
(`rt_price_up_ref`, ...) that names a profile column, and the price at timestep $t$ is the
column value times that profile. `"const"`, an empty cell or NaN means a factor of 1.
Building the `LpProblem` raises an error if a unit with a target lacks one of its two
prices, if a `_ref` names a column that does not exist, or if a referenced profile has a
NaN. `RtExtension` applies no pricing rule of its own, apart from one guard: if the up and
down prices of a unit sum to less than 1e-6, the smaller is raised so that they do (the two
deviations of a unit could otherwise grow together at no cost).

**Units that follow their schedule exactly** need no RT column. A generator is pinned by
pointing both `pmin_ref` and `pmax_ref` to a profile column with its day-ahead output, as a
fraction of its installed capacity; set `inflow_ref = "const"` and `inflow_fac = 1` as
well, otherwise its availability is counted twice (see [input_data.md](input_data.md),
"Limits that vary in time"). A consumer's demand is fixed by pointing `demand_ref` to a
profile with its day-ahead demand, as a fraction of `demand_avg`.

**Exchanges between areas.** The inter-area table, which limits the net transfer between
two areas by its NTC (see [input_data.md](input_data.md), "Inter-area capacities"), has
three optional columns that only an RT run reads:

| column | meaning |
|---|---|
| `da_flow_ref` | profile column with the day-ahead net transfer DA(t), in MW, from `area_from` to `area_to` |
| `balancing_penalty` | cost $p$, in EUR/MWh, of each MWh of difference between the net transfer $T$ and DA(t) (default 0) |
| `lock_share` | $\lambda$ between 0 and 1: $T$ stays between DA(t) − (1 − $\lambda$) × `ntc_backward` and DA(t) + (1 − $\lambda$) × `ntc_forward` (default NaN: no lock) |

$\lambda = 1$ holds the exchange at its day-ahead value; $\lambda = 0$ leaves the full NTC
available either way. A rule without `da_flow_ref` gets neither penalty nor lock. Both act
on the net transfer, not on each line: flow may shift between the lines of a border as long
as the net transfer stays in its corridor. Building the `LpProblem` raises an error if
`da_flow_ref` names a missing column or one with NaN, if `balancing_penalty` is negative,
if `lock_share` is outside [0, 1], or if a rule with `da_flow_ref` has no line across its
border.

A small RT setup by hand, for one generator whose day-ahead schedule is the profile
`schedule_g1` (fraction of `pmax`) and whose deviations cost 100 + `fuelcost` per MWh up
and 100 − `fuelcost` down:

```python
from powergama.lp.rt import RtExtension

data.profiles["schedule_g1"] = ...
data.generator.loc[g1, "rt_target_ref"] = "schedule_g1"
fuel = data.generator.loc[g1, "fuelcost"]
data.generator.loc[g1, ["rt_price_up", "rt_price_down"]] = [100 + fuel, 100 - fuel]
lp = powergama.LpProblem(data, extensions=[RtExtension()])
```

## 4. Reference

### The columns `prepare_rt` writes

In `da_solution.parquet`, one column per unit and quantity. `<d>` is the row number of the
unit in the day-ahead generator (or consumer) table; `<k>` is the row number of an
inter-area rule. Values given as fractions are clipped to [0, 1].

| column | value | used in the RT tables by |
|---|---|---|
| `da_gen_<d>` | day-ahead output / installed capacity `pmax` | `rt_target_ref` of every generator with `pmax > 0`; `pmin_ref`/`pmax_ref` of fixed and capped units |
| `da_pump_<d>` | day-ahead pumping / `pump_cap` | `rt_pump_target_ref` of every generator with `pump_cap > 0` |
| `da_cons_<d>` | day-ahead flexible demand / `demand_avg` | `rt_target_ref` of every consumer with flexible load |
| `rt_target_zero` | 0 | the target of rows added for the RT run only (peak tranches); `pmin_ref` with `zero_pmin` |
| `da_carry_<d>` | (actual availability − (day-ahead availability − day-ahead output)) / `pmax` | `pmax_ref` (and with `pin="fix"` `pmin_ref`) of units with carried curtailment |
| `da_storage_<d>` | filling at the start of each day-ahead hour / `storage_cap` | `storage_ini_ref` of units in `initial_storage` |
| `da_headroom_<d>` | (installed capacity of the unit − its day-ahead output) / installed capacity | `pmax_ref` of the peak tranche of the unit |
| `da_price_up_<d>`, `da_price_down_<d>`, `da_pump_price_up_<d>`, `da_pump_price_down_<d>`, `da_cons_price_up_<d>`, `da_cons_price_down_<d>` | deviation price, EUR/MWh, where it varies in time (units with storage) | the `_ref` of the matching price column, whose own value is then 1 |
| `da_flow_<k>` | day-ahead net transfer of rule `<k>`, MW | `da_flow_ref` of the rule |

Some more detail on three of them:

- `da_storage_<d>` at the day-ahead run's first hour is the filling that run started from
  (`storage_ini`, or its `storage_ini_ref` profile); at each later hour it is the filling at
  the end of the hour before. The RT run reads it at its own first hour.
- `da_headroom_<d>` is computed per unit: the installed capacity of all rows of the unit
  (its `block_group`) minus their summed day-ahead output.
- `da_flow_<k>` sums the day-ahead flows of every AC and DC line across the border of rule
  `<k>`, counted positive from an `area_from` node to an `area_to` node, as the NTC
  constraint counts them.

Every generator row that `fix`, `cap` or `carry_curtailment` pins also gets
`inflow_ref = "const"` and `inflow_fac = 1`. Each RT generator and consumer row has a column
`da_index` with its row number in the day-ahead table, empty for rows added for the RT run
only. A peak tranche row is a copy of the first row of its unit, with `pmax` the installed
capacity of the whole unit, `fuelcost` the tranche price, `pmax_ref = da_headroom_<d>`,
no `pmin_ref`, and the ramp limit `ramp_pu` (none by default).

### Checks

`prepare_rt` raises an error, naming the rows concerned, if:

- the day-ahead run did not complete every timestep;
- a value it needs is NaN;
- a row is in both `fix` and `cap`, a row in `zero_pmin` is not in `cap`, or a unit is both
  pinned and in `carry_curtailment`;
- a pinned row has no installed capacity or no day-ahead row;
- a row masked for peak tranches has no day-ahead row, or has storage or pumping, or only
  part of its `block_group` is masked (a unit without installed capacity simply gets no
  tranche);
- a row in `initial_storage` has no storage;
- a storage unit has no storage value to price it with;
- a rule selected in `InterAreaFlows` has no line across its border.

### Options of `prepare_rt` not shown above

- `timeMaxMin=[start, end]`: use only these day-ahead timesteps.
- `text_output=True`: also write `da_solution.csv`, for reading.
- `summary=False`: skip the summary files; `sources=` and `summary_header=` add lines to it.
- `DeviationPrices(storage_value=...)`: other storage values than the day-ahead ones.

### The rules as separate functions

Each option is also available as a function, for studies that build the RT tables
themselves. Each returns profile columns and the references to set on the tables (a
`RefColumns`); `combine_columns` joins several, and raises an error if two of them set the
same cell to different values; `apply_refs` writes the references into a table. The
arguments are described in the docstrings.

| function | rule |
|---|---|
| `rt_tables_from_da` | copies of the day-ahead grid tables, with `da_index` |
| `target_columns` | targets |
| `price_columns`, `da_storage_values` | deviation prices; storage values at the day-ahead filling |
| `pin_columns`, `carry_curtailment_columns` | fixed and capped units; carried curtailment |
| `initial_storage_columns` | start filling of storage |
| `peak_tranche_columns` | peak tranches |
| `da_flow_columns` | day-ahead net transfer between areas |
| `clear_refs` | remove a profile reference from some rows |
| `summarize_channels`, `write_channel_summary` | the summary |

### The summary files

`channels.csv` has one row per column of `da_solution.parquet`: its name, its family
(`da_gen`, `da_storage`, ...), its formula, and the RT rows that use it, with how they use
it (as target, as upper limit, as start filling, ...) and their node, area and type.
`prepare_rt_summary.md` gives the same in short: the day-ahead run, the options used, and
per family how many rows of each area and type use it.

### Debug output

With the debug stream switched on, an RT run reports for each held inter-area rule the RT
and day-ahead net transfer, the deviation, its cost and the lock corridor
(`rt_rule_rows`), and their total cost (`term_breakdown.xborder_dev`).
