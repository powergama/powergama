# Power Grid and Investment Module
## User guide

Contents:
* [What it does](#markdown-header-what-it-does)
* [Running an optimisation](#markdown-header-running-an-optimisation)
* [Input data](#markdown-header-input-data)
  * [Grid data](#markdown-header-grid-data)
  * [Time sample](#markdown-header-time-sample)
  * [Parameters](#markdown-header-parameters)
* [Analysis of results](#markdown-header-analysis-of-results)


## What it does
PowerGIM is a module for grid investment analyses that is included with
the PowerGAMA python package. It is a power system *expansion planning* model
that can consider both transmission and generator investments.

PowerGIM works by finding the optimal investments amongst a set of specified
candidate, such that the total system costs (capex and opex) are minimised
over the investment lifetime.

### Two-stage optimisation
PowerGIM is formulated as a two-stage optimisation problem, and there may be
a time delay between the two investment stages. First-stage variables represent
the *here-and-now* investments that are of primary interest. Second-stage
variables include operational decisions and second-stage investments.

it has the ability to account for uncertainties since it is formulated as a
two-stage stochastic program with variables related to investment decisions
in the first stage, and operational (and second phase investment) variables
in the second stage.

## Running an optimisation


## Input data

### Grid data
Grid data is imported from CSV files in almost the same format as for
PowerGAMA. There are files for *nodes*, *branches*, *generators* and
*consumers*.

#### Nodes
Node data are specified in a CSV file with one element per row, with
columns as shown below:

column | description | type | units
-------|-------------|------|------
id   | Unique string identifier | string
lat  | Latitude   | float |degrees
lon  | Longitude  | float |degrees
area | Area/country code | string
existing | Whether node already exists |boolean |0,1
offshore | Whether node is offshore | boolean | 0,1
cost_scaling | Cost scaling factor |float
type | Node (cost) type |string

#### Branches
Branch data are specified in a CSV file with one element per row, with
columns as shown below:

column | description | type | units
-------|-------------|------|------
node_from | Node identifier | string
node to   | Node identifier | string
capacity  | Existing capacity | float | MW
capacity2 | Capacity added stage 2 (OPT) | float | MW
expand    | Consider expansion in stage 1  | boolean   | 0,1
expand2    | Consider expansion in stage 2  | boolean   | 0,1
distance  | Branch length (OPT) | float | km
max_newCap    | Max new capacity (OPT) | float | km
cost_scaling  | Cost scaling factor | float
type      | Branch (cost) type | string

Branches have from and to references that must match a node identifier
in the list of nodes.
* expand/expand2 is 0 if no expansion should be considered (not part of
  optimisaion)
* distance may be left blank. Then distance is computed as the shortest
  distance between the associated nodes (based on lat/lon coordinates)
* capacity2 is already decided additional branch capacity that will be added
  at stage two (optional input).



#### Generators
Generator data are specified in a CSV file with one element per row, with
columns as shown below:

column | description | type | units
-------|-------------|------|------
node  | Node identifier |string
desc  | Description or name (OPT) |string
type  | Generator type |string
pmax  | Generator capacity |float |MW
pmax2 | Generator capacity stage 2 (OPT) |float |MW
pmin  | Minimum production |float |MW
expand  | Consider capacity expansion |boolean |0,1
expand2  | Consider capacity expansion in stage 2 |boolean |0,1
fuelcost  | Cost of generation |float |€/MWh
fuelcost_ref  | Cost profile |string
inflow_fac  | Inflow factor |float
inflow_ref  | Inflow profile reference |string
pavg  | Average power output (OPT) |float |MW
p_maxNew  | Maximum new capacity (OPT) |float |MW
cost_scaling  | Cost scaling factor (OPT) |float

* The average power constraint (pavg) is used to represent generators
  with large storage. pavg=0 means no constraint on average output is used
  (no storage constraint).
* pmax2 is already decided increase in generator capacity in stage 2


#### Consumers
Consumer (power demand) data are specified in a CSV file with one element per row, with
columns as shown below:

column | description | type | units
-------|-------------|------|------
node    | Node identifier  | string
demand_avg  | Average demand |float |MW
demand_ref  | Profile reference |string
emission_cap| Maximum CO2 emission allowed (OPT) |float |kg

* There may be any number of consumers per node, although zero or one is
  typical.
* demand_avg gives the average demand, which is easily computed from the
  annual demand if necessary.
* demand_ref gives the name of the demand profile (time sample) which gives
  the variation over time. Demand profiles should be normalised and have an annual
  average of 1.


### Time sample
A set of time series or samples are used to represent the variability in
renewable energy availability, power demand and generator fuel costs
(power prices).

These data are provided as a CSV file with one profile/sample per column, with
the column header being the profile string identifier, and one row per
timestamp.

the time samples are used together with base values to get demand, available
power and fuel costs at a given time as follows:

demand(t) = demand_avg ×  demand_ref(t)
fuelcost(t) = fuelcost ×  fuelcost_ref(t)
pmax(t) =  (pmax+pmax2) × inflow_fac ×  inflow_ref(t)


### Parameters
Investment costs and other parameters are provided in an XML file with the
following structure:
```XML
<?xml version="1.0" encoding="utf-8"?>
<powergim>
<nodetype>
  <item name="ac" L="1"	S="50e6" />
  <item name="dc" L="1"	S="1" />
</nodetype>
<branchtype>
  <item name="ac"	   B="5000e3" Bdp="1.15e3"  Bd="656e3"  CL="1562e3"  CLp="0"        CS="4813e3"  CSp="0"        maxCap="400"  lossFix="0"     lossSlope="5e-5" />
  <item name="dcmesh"   B="5000e3" Bdp="0.47e3"  Bd="680e3"  CL="0"       CLp="0"        CS="0"       CSp="0"        maxCap="2000" lossFix="0"     lossSlope="3e-5" />
  <item name="dcdirect" B="5000e3" Bdp="0.47e3"  Bd="680e3"  CL="20280e3" CLp="118.28e3" CS="129930e3" CSp="757.84e3" maxCap="2000" lossFix="0.032" lossSlope="3e-5" />
  <item name="conv"     B="0"      Bdp="0"       Bd="0"      CL="10140e3" CLp="59.14e3"  CS="64965e3" CSp="378.92e3" maxCap="2000" lossFix="0.016" lossSlope="0" />
  <item name="ac_ohl"   B="0"      Bdp="0.394e3" Bd="1187e3" CL="1562e3"  CLp="0"        CS="0"       CSp="0"        maxCap="4000" lossFix="0"     lossSlope="3e-5" />
</branchtype>
<gentype>
  <item name="alt"  CX="10" CO2="0" />
  <item name="wind" CX="0"  CO2="0" />
</gentype>

<parameters
  financeInterestrate="0.05"
  financeYears="30"
  omRate="0.02"
  curtailmentCost="0"
  CO2price="0"
  VOLL="0"
  stage2TimeDelta="1"
  stages="2"
/>
</powergim>
```

Most of the parametes in the  ```nodetype```, ```branchtype``` and ```gentype```
blocks are [cost parameters](#markdown-header-cost-model)
```branchtype``` has the following additional parameters related to
[power losses](#markdown-header-power-losses), and the maximum allowable
power rating per cable system (maxCap)

Parameters specified in the ```parameters``` block are:

* financeInterestrate = discount rate used in net present value calculation of
  generation costs and operation and maintenance costs
* financeYears = financial lifetime of investments - the period over which
  the total costs are computed (years)
* omRate = fraction specifying the annual operation and maintenance costs
  relative to the investment cost
* curtailmentCost = penalty cost for curtailment of renewable energy (EUR/MWh)
* CO2price = costs for CO2 emissions (EUR/kgCO2)
* VOLL = penalty cost for load shedding (demand not supplied) (EUR/MWh)
* stage2TimeDelta = time duration between investment stage 1 and 2 (years)
* stages = number of investment stages (2 is the only choice at the moment)

## Analysis of results


## More about the PowerGIM optimisation model

### Cost model

##### Investment cost

*Branches:*
cost_b = B + Bbd ⋅ b ⋅ d + Bd ⋅ d + Σ(Cp ⋅ p + C)

* d = branch distance (km)
* p = power rating (MW)
* B = fixed cost (EUR)
* Bdp = cost dependence on both distance and rating (EUR/km/MW)
* Bd = cost dependence on distance (EUR/km)
* C = fixed endpoint cost (CL=on land, CS=at sea) (EUR)
* Cp = endpoint cost dependence on rating (EUR/MW)

The sum is over the two branch endpoints, that may be on land or at sea.

*Nodes:*
cost_n = N

* N = fixed cost (NL=on land, NS=at sea)

*Generators:*
cost_g = CX ⋅ capacity

* CX = generator cost per power rating (EUR/MW)

##### Operational cost

cost = Pg ⋅ fuelcost

* Pg = generator output (MW)
* fuelcost = generator cost (EUR/MW)

Total costs

### Power losses

power out = power in (lossFix + lossSlope*d)

* lossFix = loss factor, fixed part
* lossSlope = loss factor dependence on branch distance (1/km)
