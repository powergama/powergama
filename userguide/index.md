# PowerGAMA user documentation 

Specific topics: 
- [Scenario creation](scenario_generation.md)
- [Input data](input_data.md)
- [Output data and result analysis](output_data_and_analysis.md)
- [Model description](model_description.md)

# Introduction

PowerGAMA is open source software created by SINTEF Energy Research. The
expanded name is *Power Grid And Market Analysis*. This is a
Python-based lightweight simulation tool for high level analyses of
renewable energy integration in large power systems.

The simulation tool optimises the generation dispatch, i.e. the power
output from all generators in the power system, based on marginal costs
for each timestep for a given duration. It takes into account the
variable power available for solar, hydro and wind power generators. It
also takes into account the variability of demand. Moreover, it is
flow-based meaning that the power flow in the AC grid is determined by
physical power flow eqautions.

Since some generators may have an energy storage (hydro power with
reservoir and consentrated solar power with thermal storage) the optimal
solution in one timestep depends on the previous timestep, and the
problem is therefore be solved sequentially. A realistic utilisation of
energy storage is ensured through the use of storage values.

PowerGAMA does not include any power market subtleties (such as start-up
costs, limited ramp rates, forecast errors, unit commitments) and as
such will tend to overestimate the ability to accomodate large amounts
of variable renewable energy. Essentially it assumes a perfect market
based on nodal pricing without barriers between different countries.
This is naturally a gross oversimplification of the real power system,
but gives nontheless very useful information to guide the planning of
grid developments and to assess broadly the impacts of new generation
and new interconnections.


## Licence

PowerGAMA is open source software distributed under the the [MIT License](http://opensource.org/licenses/MIT),

## Dependencies

PowerGAMA is a Python package. It requires

-   Python 3
-   A solver (e.g. the free CBC solver or the GLPK solver)

