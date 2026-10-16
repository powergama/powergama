# Inter-Area NTC (Net Transmission Capacity) Constraints

## Overview

The inter-area NTC constraint feature allows you to enforce directional transmission capacity limits between geographic areas (countries, regions) in PowerGAMA simulations. This is distinct from individual branch capacity constraints and represents market-available cross-zonal capacity.

## Background

There are two different concepts for transmission capacity:

1. **Physical Line Capacity** (symmetric): The maximum power rating of transmission lines. Both directions can independently carry this capacity.
2. **Market NTC (Directional)**: The net transmission capacity available for market exchanges after coordinated capacity calculation and security constraints. Can be asymmetric in the two directions.

For example, a 5.5 GW physical transmission link between Belgium and France may have market NTC of only 600 MW in one direction and 2000 MW in the other, due to network congestion, flow patterns, and security margins.

## File Format

Create a CSV file named `inter_area_ntc.csv` in your dataset directory with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| area_from | str | Source area identifier (must match entries in node.csv area column) |
| area_to | str | Destination area identifier |
| ntc_forward | float | NTC limit from area_from → area_to (MW) |
| ntc_backward | float | NTC limit from area_to → area_from (MW) |

### Example

```csv
area_from,area_to,ntc_forward,ntc_backward
BE,FR,650,2000
BE,NL,1400,950
```

This constrains:
- Belgium → France: max 650 MW
- France → Belgium: max 2000 MW
- Belgium → Netherlands: max 1400 MW
- Netherlands → Belgium: max 950 MW

## How It Works

1. **Branch Identification**: For each area pair, the constraint identifies all AC and DC branches connecting them (using `GridData.getInterAreaBranches()`).

2. **Flow Aggregation**:
   - **Forward direction**: Sums all branch flows where direction matches area_from → area_to
   - **Backward direction**: Sums all branch flows in the opposite direction

3. **Constraint Application**: Pyomo constraints enforce:
   - Sum of forward flows ≤ ntc_forward
   - Sum of backward flows ≤ ntc_backward

## Usage

### 1. Create the inter_area_ntc.csv File

Place the file in your dataset directory alongside `node.csv`, `branch.csv`, etc.

### 2. Run Simulation

No code changes needed. The simulation will automatically detect and load `inter_area_ntc.csv` if present:

```python
import powergama
data = powergama.GridData()
data.readGridData(
    nodes="dataset/node.csv",
    ac_branches="dataset/branch.csv",
    dc_branches="dataset/dcbranch.csv",
    generators="dataset/generator.csv",
    consumers="dataset/consumer.csv",
    inter_area_ntc="dataset/inter_area_ntc.csv"  # Optional
)
```

Or in the Willow workflow, it's automatic if the file exists in the dataset directory.

## Design Principles

- **Optional**: If `inter_area_ntc.csv` is missing, no inter-area constraints are applied.
- **Clean Merging**: All code follows existing PowerGAMA patterns for easy upstream merging.
- **Directional**: Forward and backward capacities can be asymmetric, matching real market operations.
- **Aggregate**: Constraints sum across all branches between two areas, not per-branch.

## Notes

- Set ntc_forward or ntc_backward to a non-positive value to disable that direction.
- Area identifiers are case-sensitive and must match the `area` column in `node.csv`.
- Multiple area pairs can be specified; one pair per row.
- The constraint is implemented as a Pyomo constraint created in `LpProblemPyomo._create_constraint_inter_area_ntc()`.

## Data Sources

- **Ember**: [Europe Interconnection Data](https://data.ember.dev/datasets/) - provides REF_NTC (reference) data
- **ENTSO-E Transparency Platform**: Offers similar directional NTC data
- **National Grid / TSO Publications**: Physical asset capacities (different from market NTC)
