
# Input data formats

Input files are comma separated text files (CSV), with a comma as
delimiter and period as the decimal symbol, or Parquet files (`.parquet`).
The first line in a CSV file contains a header, with unique keys associated
with each column. The ordering of columns is arbitrary. Keys are case
sensitive and should be all lower case. Columns labelled OPT are optional;
an optional column that is missing gets its default value. Each grid table
may also be given as a list of files, which are concatenated row-wise.

## Grid data

There are 5 input files associated with nodes, AC branches, DC branches,
consumers and generators, and an optional sixth one with the inter-area
rules. These files contain references to additional
files which have information about normalised storage values, energy
inflow profiles and power demand profiles. The reference identifier
(integer number or string) in the generator and consumer files should
match an identifier in the relevant storage value or profile files.

### Nodes

Nodes need to have unique identifier strings. Area information is used
for scenario generation (preprocessing), and for plotting and
presentation of results. Latitude and longitude information is only used
for plotting the grid on a map.

  column key  | description                   |  type  |   units
  ------------|-------------------------------|--------|---------
  "id"       | Unique string identifier       |  string|   
  "lat"      | Latitude                       |  float |   degrees
  "lon"      | Longitude                      |  float |   degrees
  "area"     | Area/country code              |  string|   
  "zone"     | Zone (level below area) (OPT)  |  string|   

### AC Branches

Branches have from and to references that must match a node identifier
in the list of nodes. Impedance should be given as per unit system with
the base power being the global one (powergama.constants.baseS).

  column key     | description      |  type  |  units
  ---------------|------------------|--------|-------
  "node_from\"   | Node identifier  |  string  | 
  "node_to\"     | Node identifier  |  string  | 
  "capacity\"    | Capacity         |  float   | MW
  "reactance\"   | Reactance        |  float   | p.u
  "resistance\"  | Resistance (OPT) |  float   | p.u.

### DC Branches

DC branches have from and to references that must match a node
identifier in the list of nodes.

  column key    | description     | type   | units
  --------------|-----------------|--------|-------
  "node_from\"  | Node identifier |  string|   
  "node_to\"    | Node identifier |  string|   
  "capacity\"   | Capacity        |  float |   MW
  "resistance\" | Resistance (OPT)| float  | Ohm

### Inter-area capacities
Limits on power exchange between areas can be specified as net transfer capacities (NTCs).
These can be different in the two directions. "Forward" is from->to, and "backward" is to->from.

If NTCS are specified, these are added as flow constraints in addition to the capacity limits of individual lines.
They are read from the optional inter-area rules table, `readGridData(..., inter_area_rules="inter_area_rules.csv")`
(`GridData.inter_area_rules`). One row per pair of areas:

  column key      | description     | type  | units
  ----------------|-----------------|-------|-------
  "area_from\"    | Area identifier | str   |
  "area_to\"      | Area identifier | str   |
  "ntc_forward\"  | NTC value       | float | MW
  "ntc_backward\" | NTC value       | float | MW
  "da_flow_ref\"  | RT runs only: profile with the day-ahead net transfer (OPT, default "": none) | str | MW
  "balancing_penalty\" | RT runs only: cost of a deviation from it (OPT, default 0) | float | EUR/MWh
  "lock_share\"   | RT runs only: lock share λ in [0, 1] (OPT, default NaN: no lock) | float |

The NTC acts on the **net** summed flow over the boundary, not on each branch: the AC and
DC branches from an `area_from` node to an `area_to` node count positive, those the other
way negative (areas from `node.area`, case-sensitive), and the net transfer is held within
−`ntc_backward` ≤ net ≤ `ntc_forward` (constraints `cInterAreaRulesForward`/`Backward`, in
both the single-period and the multi-period formulation). An infinite or empty NTC leaves
that direction unconstrained; 0 allows no net exchange in that direction. A pair of areas
with no branch between them gets no constraint.

The NTC is the capacity the market may use, which is usually less than the physical
capacity of the lines and can differ between the two directions. For example:

```csv
area_from,area_to,ntc_forward,ntc_backward
BE,FR,650,2000
BE,NL,1400,950
```

limits the net transfer from BE to FR to 650 MW and from FR to BE to 2000 MW. Directional
NTC data for Europe are published, for example, on the ENTSO-E Transparency Platform.

The last three columns are read only in an RT run (`RtExtension`), where they hold the
net transfer close to its day-ahead value, with a penalty, a hard window, or both; any
other run ignores them. See [Usage_RT.md](Usage_RT.md), section 3, "Exchanges between areas".


### Consumers

Consumers are loads connected to nodes. There may be any number of
consumers per node, although zero or one is typical.

`demand_avg` gives the average demand, which is easily computed from the
annual demand if necessary. `demand_ref` gives the name of the demand
profile which gives the variation over time. Demand profiles should be
normalised and have an annual average of 1.

  column key  | description  |      type  |   units
  ------------|--------------|------------|--------
  "node\"                 |  Node identifier                                      | string  | 
  "demand_avg\"           |  Average demand                                       | float   | MW
  "demand_ref\"           |  Profile reference                                    | string  | 
  "flex_fraction\"        |  Fraction of demand which is flexible (OPT)           | float   | 
  "flex_on_off\"          |  Flexibility on/off ratio (OPT)                       | float   | 
  "flex_storage\"         |  Maximum flexibility (OPT)                            | float   | hours
  "flex_basevalue\"       |  Base storage value (OPT)                             | float   | €/MWh
  "flex_storval_filling\" |  Profile ref, storage value filling dependence (OPT)  | string  | 
  "flex_storval_time\"    |  Profile ref, storage value time dependence (OPT)     | string  | 
  "flex_storagelevel_init\"| Initial flexibility buffer state (OPT)               | float   | 1

### Generators

Generators are the most complex data structure and require the most
input data.

  column key              |description                                          | type   | units
  ------------------------|-----------------------------------------------------|--------|-------
  "node\"                 | Node identifier                                     |  string|   
  "desc\"                 | Description or name                                 |  string|   
  "type\"                 | Generator type                                      |  string|   
  "pmax\"                 | Installed capacity                                  |  float |   MW
  "pmin\"                 | Minimum production                                  |  float |   MW
  "fuelcost\"             | Cost of generation                                  |  float |   €/MWh
  "inflow_fac\"           | Inflow factor                                       |  float |   
  "inflow_ref\"           | Inflow profile reference                            |  string|   
  "storage_cap\"          | Storage capacity                                    |  float |   MWh
  "storage_price\"        | Base for storage value                              |  float |   €/MWh
  "storval_filling_ref\"  | Profile ref, storage value filling level dependence |  string|   
  "storval_time_ref\"     | Profile ref, storage value time dependence          |  string|   
  "storage_ini\"          | Initial storage filling level                       |  float |   1
  "pump_cap"              | Pumping capacity (OPT)                              |  float |   MW
  "pump_efficiency"       | Pumping efficiency (OPT)                            |  float |   
  "pump_deadband"         | Pumping price dead-band (OPT)                       |  float |   €/MWh
  "gen_lat"               | Latitude (OPT)                                      |  float | degrees
  "gen_lon"               | Longitude (OPT)                                     |  float | degrees
  "ramp_up_pu"            | Ramp up limit (OPT, default NaN: none)              |  float | pu/timestep
  "ramp_down_pu"          | Ramp down limit (OPT, default NaN: none)            |  float | pu/timestep
  "pmax_ref"              | Profile ref, availability (OPT, default "const")    |  string| 
  "pmin_ref"              | Profile ref, minimum output (OPT, default "const")  |  string| 
  "storage_ini_ref"       | Profile ref, initial filling (OPT, default "const") |  string| 
  "block_group"           | Rows of one physical unit (OPT, default "")         |  string| 

The columns fall into a few groups, described below: capacity, cost and inflow (every
generator); storage and pumping (hydro reservoirs, pumped hydro, batteries, solar CSP);
ramp limits; profiles that make a limit vary in time; and the grouping of rows that
belong to one physical unit.

#### Capacity, cost and inflow

`node` is the identifier of the node the generator is connected to; a node may have any
number of generators. `pmax` is the **installed capacity** of the generator, written
$P_\text{installed}$ in the formulas below. (The column keeps its historical name; the
power a generator can actually produce at a given time can be lower, see "Limits that vary
in time".) `pmin` is the minimum production, normally zero, but non-zero for units that
must run, such as some nuclear plants.

`fuelcost` is the cost of generation per MWh produced. For a generator without storage it
is the marginal cost. For a generator with storage it is an operating cost that does not
depend on the filling level, and the storage value comes on top of it; it is usually zero.

`inflow_fac` and `inflow_ref` give the power available from the primary resource (wind,
sun, water inflow). `inflow_ref` names a profile column, and the inflow at timestep $t$ is

$$
%\label{eq:inflow}
     P_\text{inflow}(t) =  P_\text{installed} \times \text{inflow factor} \times \text{profile value}(t) .
$$

If the annual inflow is known, integrating this equation gives the inflow factor:

$$
%\label{eq:inflow_annual}
     \text{inflow factor} = \frac{\text{annual inflow}}{8760~\text{h} \times P_\text{installed} \times  \text{avg}(\text{profile value})}.
$$

There are two typical ways to use the inflow factor and the inflow profile:

-   Normalised inflow profile with *maximum* value = 1: the profile gives the power output
    per installed capacity, with an average equal to the capacity factor of the
    generator. The inflow factor is then about 1, larger for good sites and smaller for bad
    ones. With an inflow factor above one, the inflow can at times exceed the installed
    capacity.

-   Normalised inflow profile with *average* value = 1: the inflow factor is the capacity
    factor, i.e. the average inflow divided by the installed capacity. Typical capacity
    factors are 0.5 for a large hydro storage system, 0.25 for wind power, and 0.22 for
    solar PV.

If the installed capacity is upgraded without the energy inflow changing (which may be
relevant if there is storage), the inflow factor must be reduced correspondingly. Many
generators may share one profile, with different inflow factors for representative
capacity factors.

#### Storage and pumping

`storage_cap` is the energy capacity of the storage (hydro reservoir, pumped hydro,
battery, solar CSP), and `storage_ini` its filling level at the start of the run, as a
fraction of `storage_cap`. `storage_price` is the base value of the stored energy: it
sets the absolute scale of the generator's storage values, and `storval_filling_ref` and
`storval_time_ref` name the curves that make it depend on the filling level and on time
(see "Storage values" below).

The three pumping columns are only needed where `pump_cap`, the pumping capacity, is
non-zero. `pump_efficiency` is the fraction of the pumped energy that reaches the
storage, and `pump_deadband` is a cost per MWh pumped: pumping pays when the price is
below the storage value times the efficiency, less the dead-band.

#### Ramp limits

Many plants cannot change their output quickly. `ramp_up_pu` and `ramp_down_pu` limit
how much the output may change from one timestep to the next, as a fraction of the
installed capacity per timestep; NaN (the default) means no limit. They apply in every
mode: in single-period runs around the output of the previous timestep, and in
multi-period runs also between the timesteps of each window.

The other limits of a generator (minimum production, available capacity, inflow,
storage) always take precedence over its ramp limits. Where one of them moves faster than
the ramp allows, the generator follows it, and PowerGAMA reports the place as an
inconsistency in the input data (see [model_description.md](model_description.md),
"Ramp limits").

#### Limits that vary in time, and the start filling

Three optional columns replace a fixed value of the generator with the value of a
profile column at each timestep. Typical uses are planned outages or the seasonal
availability of thermal plants, a must-run level, a unit held to a schedule computed
elsewhere, and reservoir levels taken from history at the start date.

| column | replaces | value at timestep $t$ |
|---|---|---|
| `pmax_ref` | the available capacity (normally the installed capacity) | $P_\text{installed} \times \text{profile}(t)$ |
| `pmin_ref` | the minimum production `pmin` | $P_\text{installed} \times \text{profile}(t)$ |
| `storage_ini_ref` | the start filling `storage_ini` | `storage_cap` × profile at the first timestep of the run |

Both `pmax_ref` and `pmin_ref` are fractions of the *installed* capacity, so pointing
both to the same column pins the output to that profile. `storage_ini_ref` reads only one
value, at the first timestep of the run (the first row of the time range); when a run is
resumed with `continue_from_last=True`, the storage levels come from the results database
instead.

The default of all three is `"const"`, which here means *no profile*: the fixed value is
used, and no column called `const` needs to exist. An empty cell or NaN means the same.
Any other value must be a column of the profile data: otherwise `LpProblem` raises an
error when it is constructed, naming the generators and the missing columns. A NaN in a
`pmax_ref` or `pmin_ref` profile at some timestep means no change at that timestep; a NaN
in the value `storage_ini_ref` reads raises an error.

**The available capacity also scales the inflow.** With a `pmax_ref` profile the inflow
equation above becomes

$$
     P_\text{inflow}(t) =  P_\text{installed} \times \text{pmax profile}(t) \times \text{inflow factor} \times \text{inflow profile}(t) .
$$

For a wind farm with an availability profile, this is what is wanted: availability and
wind multiply. But if the `pmax_ref` profile already *is* the output, for example a
dispatch taken from an earlier run, the resource would be counted twice. For such a
generator, set `inflow_ref = "const"` and `inflow_fac = 1` (with a profile column `const`
equal to 1), so that its inflow is just $P_\text{installed} \times \text{pmax
profile}(t)$. PowerGAMA does not warn about this.

#### Several rows for one unit: `block_group`

A large unit or fleet can be split into several generator rows, for example price blocks
with increasing marginal cost, so that it is dispatched progressively along the merit
order rather than all at once at a single price. Gas fleets and hydro reservoirs are
typical cases. The rows remain parts of one physical unit, and `block_group` records
which: rows with the same non-empty value belong to the same unit, and an empty value (the
default) makes a row a unit of its own. The value is text, so `7` and `07` are different
groups, and the order of the rows within a group is their order in the table.

The optimisation ignores `block_group`. It is there to keep track of the original units,
for analysis and for tools that work per unit; in PowerGAMA, the peak tranches of
`prepare_rt` use it to add one extra row per unit to an RT run (see
[Usage_RT.md](Usage_RT.md)).

## Time dependence of power consumption, power inflow and storage values

The following quantities vary with time, each through a profile column
named in the generator or consumer row:

  quantity | named by | profile value
  ---------|----------|--------------
  power inflow (wind, solar radiation, rain) | `inflow_ref` | inflow per installed capacity × `inflow_fac` (see above)
  consumer load (power demand) | `demand_ref` | demand per `demand_avg`, annual average 1
  storage value, time dependence | `storval_time_ref`, `flex_storval_time` | factor on the storage value (see "Storage values")
  available capacity (OPT) | `pmax_ref` | fraction of the installed capacity `pmax`
  minimum output (OPT) | `pmin_ref` | fraction of the installed capacity `pmax`
  initial storage filling (OPT) | `storage_ini_ref` | fraction of `storage_cap`, read at the first timestep only

For inflow, load and storage values, there are two fields in the input data, one parameter that
gives the absolute scale, and a reference to a normalised profile which
entails the time profile. Multiplied together these give the absolute
variation over time, as expressed e.g. in the inflow equation above.
The reason for this splitting between absolute
scale and normalised profile is to enable multiple references to the
same profile (e.g. normalised profile for demand may be the same for all
consumers within an area), and to simplify the task of creating
scenarios by scaling up or down the absolute scale without the need to
change the profile time series.

Power inflow is given by weather conditions. Hydro has mainly a seasonal
profile, whereas wind and solar varies from hour to hour. Solar has a
characteristic daily profile with no production in dark hours. As stated
previously, there are two alternative ways to specify inflow profile and
absolute scale (inflow factor): 1) The profile is normalised to
give power inflow per installed capacity (with average value
representing the capacity factor), and absolute scale is nominally equal
to one; 2) The profile is normalised to have an average value of unity,
and absolute scale represents the capacity factor.

  column key   | description   |  type |  units
  -------------|---------------|-------|-------
  identifier1  | values type 1 |  float|
  identifier2  | values type 2 |  float|
  ...          |               |       |    

There is one row per time step. The row labels (the first column of a CSV
file, the index of a Parquet file) are the timestep numbers that
`timerange` selects, so `timerange=range(24, 48)` reads the rows labelled
24 to 47: the second day of an hourly year.

The profiles may be split over several files, CSV or Parquet, given to
`readProfileData` as a list:

```python
data.readProfileData(
    filename=["profiles.csv", "profiles_availability.parquet", "profiles_storval_time.csv"],
    storagevalue_filling="profiles_storval_filling.csv",
    timerange=range(0, 8760),
)
```

Each file is cut to `timerange` (its row labels) on its own, and the files
are then joined side by side, column by column. Every file must have all the
rows of `timerange`, and a column name may appear in only one file;
otherwise reading the files raises an error. A single file name works as
before.

## Storage values

There are two dependencies:

-   Filling level

-   Time of year and time of day

All in all, the storage values are computed according to

$$
%\label{eq:storagevalue_calc}
     \text{storage value}(f,t) = \text{base value} \times \text{filling level profile}(f) \times \text{time profile}(t),
$$

where $f$ is the relative filling level, and $t$ is the timestep.

Time dependence of storage values reflect the time dependence of the
associated inflow, and is therefore quite different for hydro (seasonal
variation) and CSP (daily variation). The time curves are ordinary
profile columns, read with the other profiles (their file is one more
entry in the `filename` list of `readProfileData`, cut to `timerange`
like every profile). `storval_time_ref` of a generator and
`flex_storval_time` of a consumer name such a column. `"const"`, an empty
cell or NaN means a time factor of 1, and needs no column. Any other name
must be a profile column: `LpProblem` raises an error when it is built,
naming the rows and the missing columns. Only the rows the model values
are checked: generators with storage (`storage_cap > 0`) and consumers with
flexible load.

*Deprecated:* `readProfileData(..., storagevalue_time=...)` still reads the
time curves from a separate file, with a `DeprecationWarning`. Every time
reference is then a column of that file, `"const"` included; the two ways
are not mixed in one run. Results databases written that way keep their
`data_storval_time` table, and are read back the same way.

Storage value dependence on filling level is specified as follows:

  column key   | description   | type  | units
  -------------|---------------|-------|-------
  identifier1  | values type 1 |  float|   €/MWh
  identifier2  | values type 2 |  float|   €/MWh
  ...  | | |

There is one row per percentile (filling level): exactly 101 rows,
0 to 100. The filling curves are a separate input
(`readProfileData(..., storagevalue_filling=...)`), not profile columns.
`storval_filling_ref` of every generator with storage and
`flex_storval_filling` of every consumer with flexible load must name one
of its columns. Another number of rows, a missing column or a missing
table when the grid has storage or flexible load raise an error when
`LpProblem` is built. There is no fallback curve.

