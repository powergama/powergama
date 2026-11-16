# Output data and results analysis

## Re-open previously simulated system
To re-open and analyse a system previously simulated and stored in a sqlite database file, you can use (where you put in the correct sqlite filename and timedelta used for the simulation):
```python
> pg_res = powergama.Results.from_existing("results.sqlite3",timedelta=1.0)
> pg_data = pg_res.grid
```
And then use the analysis

The database stores only the rows of the profiles the run used, renumbered from 0. If the
run did not start at the first row of the profile files, give its time range when you
reopen it, so that the timesteps keep their original numbers:
`Results.from_existing("results.sqlite3", timedelta=1.0, timerange=range(24, 48))`.

`pg_res.grid` is the grid that was written: the node, branch, DC branch, generator and
consumer tables, the profiles, the storage-value tables and the inter-area rules. A new
`LpProblem` on it therefore has the same constraints as the original run, the limits on
exchanges between areas included. (A grid table may have its own column named `index`:
the database then keeps the row numbers in a column `level_0` instead. A table with a
column named `level_0` cannot be stored, and writing it raises an error.)

## Reading the dispatch as frames

To work with the dispatch of a run in pandas, `Results.get_dispatch_frames(timeMaxMin=None)`
returns it as a dictionary of tables (pandas DataFrames), one per kind of element, with
one row per timestep and one column per element:

| key | values | columns |
|---|---|---|
| `generator_output` | generator output (MW) | every generator |
| `pumping` | pumping (MW) | every generator with pumping (`pump_cap > 0`) |
| `flexload_demand` | flexible demand (MW) | every consumer with flexible load |
| `storage` | storage filling (MWh) | every generator with storage (`storage_cap > 0`) |
| `branch_flow` | AC branch flow (MW) | every AC branch |
| `dcbranch_flow` | DC branch flow (MW) | every DC branch |

The columns are the row numbers of the elements in the grid tables (`grid.generator`,
`grid.consumer`, ...). The rows are the timesteps, numbered like the rows of the profile
files they were read from: a run on `timerange=range(24, 48)` gives rows 24 to 47.
`timeMaxMin = [start, end]` selects the timesteps from `start` up to, but not including,
`end`; by default, the whole run.

The values are exactly the stored ones. A value that was never stored, for instance because
the run stopped early, is NaN; it is never filled in. To check that a run is complete,
`Results.is_complete(timerange=None)` tells whether every timestep of its time range (or of
the one given) is stored:

```python
res = powergama.Results.from_existing("da.sqlite3", timedelta=1.0)
if not res.is_complete():
    raise RuntimeError("the run did not finish")
frames = res.get_dispatch_frames()
output = frames["generator_output"]  # index = timestep, columns = generator index
```

Other results can be read the same way, one variable at a time, with
`res.db.getResultFrame(table, column, timeMaxMin)`, for example the nodal prices of the
first day: `res.db.getResultFrame("Res_Nodes", "nodalprice", [0, 24])`. This works for the
result tables that hold one value per element and timestep (listed in
`Database.RESULT_FRAME_COLUMNS`).

## Output data format
The results of the simulation is saved in a sqlite3 database file

## Optimal solution

The primary result are values for the cost function (total cost of
generation) and values for all variables for each timestep in the
simulation. The variables are

-   power generation for each generator

-   voltage angles at nodes

-   power flow on AC branches (actually derived from voltage angles)

-   power flow on DC branches

-   load shedding at each node

Derived quantities include

-   storage level and marginal price for generators with storage

-   spilled power inflow (e.g. constrained/curtailed wind power)

## Sensitivities

Sensitivities are computed for the following variables:

-   AC branch capacity

-   DC branch capacity

-   Power demand at each node

These sensitivities say how much the total generation cost would
increase if branch capacity or power demand at a given branch or node
were to increase by one unit. This is useful for identifying grid
bottlenecks and nodal power prices.

## Further analysis of results

Examples of interesting analyses that can be addressed using PowerGAMA
are

-   Identification of grid bottlenecks. This is relevant for existing
    bottlenecks, but even more so with future scenarios with new
    generators installed, e.g. large amounts of renewable generation.
    Assessment of benefits by reinforcing certain connections, or
    adding more lines.

-   Identification of the potential of the grid and power system to
    absorb large amounts of renewable generation. How much new capacity
    of wind and solar power can be introduced without problems

-   Estimation of generation mix

It should be noted that PowerGAMA does not include any power market
subtleties (such as start-up costs, forecast errors, unit commitments)
and as such will tend to overestimate the ability to accommodate large
amounts of variable renewable energy. Essentially it assumes a perfect
marking based on nodal pricing without barriers between different
countries. This is naturally a gross oversimplification of the real
power system, but gives nonetheless very useful information to guide the
planning of grid developments and to assess broadly the impacts of new
generation and new interconnections.

## Included plots and other result analysis functions

See the online powergama source code documentation for a complete
overview of plotting functions and other functions to retrieve
simulation results for further analysis.
