
# Input data formats

Input files are comma separated text files (CSV), with a comma as
delimiter and period as the decimal symbol. The first line in the files
contains a header, with unique keys associated with each column. The
ordering of columns is arbitrary. Keys are case sensitive and should be
all lower case.

## Grid data

There are 5 input files associated with nodes, AC branches, DC branches,
consumers and generators. These files contain references to additional
files which have information about normalised storage values, energy
inflow profiles and power demand profiles. The reference identifier
(integer number or string) in the generator and consumer files should
match an identifier in the relevant storage value or profile files.

### Nodes

Nodes need to have unique identifier strings. Area information is used
for scenario generation (preprocessing), and for plotting and
presentation of results. Latitude and longitude information is only used
for plotting the grid on a map.

  column key  | description              |  type  |   units
  ------------|--------------------------|--------|---------
  "id"       | Unique string identifier |  string|   
  "lat"      | Latitude                 |  float |   degrees
  "lon"      | Longitude                |  float |   degrees
  "area"     | Area/country code        |  string|   

### AC Branches

Branches have from and to references that must match a node identifier
in the list of nodes. Impedance should be given as per unit system with
the the base power being the global one (powergama.constants.baseS)

  column key     | description      |  type  |  units
  ---------------|------------------|--------|-------
  "node_from\"   | Node identifier  |  string  | 
  "node_to\"     | Node identifier  |  string  | 
  "reactance\"   | Reactance        |  float   | p.u
  "resistance\"  | Resistance (OPT) |  float   | p.u.
  "capacity\"    | Capacity         |  float   | MW

### DC Branches

DC branches have from and to references that must match a node
identifier in the list of nodes.

  column key    | description     | type   | units
  --------------|-----------------|--------|-------
  "node_from\"  | Node identifier |  string|   
  "node_to\"    | Node identifier |  string|   
  "capacity\"   | Capacity        |  float |   MW

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
  "flex_storage\"         |  Maximum flexibility (OPT)                            | float   | MWh
  "flex_storval_filling\" |  Profile ref, storage value filling dependence (OPT)  | string  | 
  "flex_storval_time\"    |  Profile ref, storage value time dependence (OPT)     | string  | 
  "flex_basevalue\"       |  Base storage value (OPT)                             | float   | €/MWh

### Generators

Generators are the most complex data structure and require the most
input data. The three columns related to pumping only need to be filled
out if the pumping capacity is non-zero.

  column key              |description                                          | type   | units
  ------------------------|-----------------------------------------------------|--------|-------
  "node\"                 | Node identifier                                     |  string|   
  "desc\"                 | Description or name                                 |  string|   
  "type\"                 | Generator type                                      |  string|   
  "pmax\"                 | Maximum production                                  |  float |   MW
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

`node` is the string identifier of the node where the generator is
connected. There may be any number of generators per node. `pmax` is the
maximum power production, i.e. the generator capacity `pmin` is the
minimum power production. This is normally zero, but may be nonzero for
certain generator types such as nuclear power generators. `fuelcost` is
the cost of generation. For generators without storage, the marginal
cost is set equal to this value value. `storage_price` is the the base
value for storage generator's storage values. It sets the absolute scale
in the storage value calculation. `storage_capacity` is the capacity of
the storage system. This is usually relevant only for hydro power and
solar CSP. `storagevalue_ref` is the string identifier of the associated
storage value table to be used for this generator/storage system
`storage_init` is the initial relative filling level of the storage.
`inflow_fac` is the inflow factor. `inflow_ref` is the string identifier
of the associated inflow profile.

Power inflow at a given timestep $t$ is computed according to
$$ P_\text{inflow}(t) =  P_\text{max} \times \text{inflow factor} \times \text{profile value}(t) $$
In case the annual inflow is known, the inflow factor can be expressed
by integrating the above equation, giving 
$$
%\label{eq:inflow_annual}
     \text{inflow factor} = \frac{\text{annual inflow}}{8760~\text{h} \times P_\text{max} \times  \text{avg}(\text{profile value})}.
$$

There are two typical ways to use inflow factor and inflow profile:

-   Normalised inflow profile with *maximum* value = 1: profile gives
    power output per installed capacity, with average value equal to the
    capacity factor of the generator. In this case, `inflow_factor`
    should be approximately 1, larger for good sites and smaller for bad
    sites. If inflow factor is larger than one, then at times
    $P_\text{inflow}>P_\text{max}$.

-   Normalised inflow profile with *average* value = 1: `inflow_factor`
    is equal to capacity factor, i.e. average inflow divided by
    generator capacity. Typical capacity factors are 0.5 for a large
    hydro storage system, 0.25 for wind power, and 0.22 for solar PV.

It is important to keep in mind that if the generator capacity is
upgraded without the energy inflow changing (which may be relevant if
there is storage), the inflow factor must be reduced correspondingly.

If fine resolution is not needed, many generators may use the same
profile, but with different inflow factors to get representative
capacity factor.

## Time dependence of power consumption, power inflow and storage values

The following quantities vary with time:

-   Generator power inflow (wind, solar radiation, rain)

-   Consumer load (power demand)

-   Storage values

For these, there are two field in the input data, one parameter that
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
absolute scale (inflow factor) are used: 1) The profile is normalised to
give power inflow per installed capacity (with average value
representing the capacity factor), and absolute scale is nominally equal
to one; 2) The profile is normalised to have an average value of unity,
and absolute scale represents the capacity factor.

  column key   | description   |  type |  units
  -------------|---------------|-------|-------
  identifier1  | values type 1 |  float|   MW
  identifier2  | values type 2 |  float|   MW
  ...          |               |       |    

There is one row per time step.

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
variation) and CSP (daily variation). This depenency is given in the
same format as for inflow and consumption, see above.

Storage value dependence on filling level is specified as follows:

  column key   | description   | type  | units
  -------------|---------------|-------|-------
  identifier1  | values type 1 |  float|   €/MWh
  identifier2  | values type 2 |  float|   €/MWh
  ...  | | |

There is one row per percentile (filling level)

