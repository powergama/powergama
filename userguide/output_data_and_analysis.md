# Output data and results analysis

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
