# WILLOW 2024 calibration case: 5–6 October 2024

Example data for [example_notebook_willow_da_rt.ipynb](../../example_notebook_willow_da_rt.ipynb):
a day-ahead run followed by a real-time (RT) run, on a European grid model with Belgium in
more detail.

The data come from the [WILLOW](https://willow-project.eu) project (Horizon Europe), which
studies wind curtailment in future wind-dominated grids. The grid model was built in WILLOW
Task 4.2 (deliverable D4.2); the two-stage day-ahead and real-time method and the Belgian
case study are from Task 4.5 (deliverable D4.5 and its modelling annex). This folder is an
excerpt of the 2024 calibration case of D4.5: 48 hours from 2024-10-05 00:00 UTC (row 0)
to 2024-10-06 23:00 UTC (row 47).

| file | content |
|---|---|
| `node.csv`, `branch.csv`, `dcbranch.csv`, `generator.csv`, `consumer.csv` | the grid tables: 325 nodes in 17 areas, 2,478 generators |
| `inter_area_rules.csv` | net transfer capacities between areas, with the RT penalty and lock share of each border |
| `profiles_da.parquet` | day-ahead profiles: forecasts of wind, solar and demand, and the other profiles the grid tables reference |
| `profiles_rt.parquet` | real-time profiles: the same columns, with the actual wind, solar and demand |
| `storval_filling_da.csv`, `storval_filling_rt.csv` | storage-value filling curves of the day-ahead and the real-time run |

The tables are the inputs PowerGAMA received in the WILLOW study for this window, after the
study's own processing.

## Data sources

- **Grid model:** the European grid model built in WILLOW Task 4.2 (deliverable D4.2) and
  updated in the DeCOOP project, then processed for the 2024 case of WILLOW Task 4.5
  (deliverable D4.5).
- **Generation capacities, demand, wind, solar, hydro inflow and reservoir levels, nuclear
  availability, cross-border flows:** the [ENTSO-E Transparency
  Platform](https://transparency.entsoe.eu) (for Belgium, data published by Elia through
  ENTSO-E). The day-ahead profiles use its day-ahead forecasts, the real-time profiles its
  actual values.
- **Net transfer capacities between areas:** [Ember, Europe electricity interconnection
  data tool](https://ember-energy.org/data/europe-electricity-interconnection-data-tool/).
