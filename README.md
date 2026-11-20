[![GitHub license](https://img.shields.io/github/license/powergama/powergama)](https://github.com/powergama/powergama/blob/main/LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/powergama.svg)](https://python.org)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![build](https://github.com/powergama/powergama/actions/workflows/build.yml/badge.svg)](https://github.com/powergama/powergama/actions/workflows/build.yml)
[![GitHub version](https://badge.fury.io/gh/powergama%2Fpowergama.svg)](https://badge.fury.io/gh/powergama%2Fpowergama)
[![PyPI version](https://img.shields.io/pypi/v/powergama.svg)](https://pypi.python.org/pypi/powergama)
[![codecov](https://codecov.io/gh/powergama/powergama/graph/badge.svg?token=EHHS8UO7H1)](https://codecov.io/gh/powergama/powergama)

# PowerGAMA - Power Grid And Market Analysis

## Introduction
PowerGAMA is an open-source Python package for power system grid and market analyses. It finds the
least-cost dispatch of generation, energy storage and flexible demand on a linearised power flow model
of the grid. It runs one timestep after another over a year or more, and reports power flows, nodal
prices, storage use, load shedding and other characteristics.

Energy storage and flexible demand mean that the best decision in one timestep depends on the timesteps
before it. The problem is therefore solved in sequence. Storage values give the stored energy a price
that ensures realistic use of the storage.

PowerGAMA can be run in three modes:

- **Single-period** (default): one optimisation per timestep, with the storage level carried forward.
  This is fast and suits long simulations of large systems.
- **Multi-period**: a rolling horizon. Each optimisation looks a number of hours ahead (for example
  24) and keeps only the first ones. With ramp limits, this imitates day-ahead market clearing.
- **Real-time (balancing)**: a second step after a day-ahead run. It re-dispatches the system for
  forecast errors or outages, with penalties for straying from the day-ahead schedule.

PowerGAMA is a deliberately simplified model. It assumes a perfect market with nodal pricing, where
trade between areas can optionally be limited by transfer capacities (NTC). It does not model unit
commitment, start-up costs or strategic bidding. As a result, it tends to overestimate how much
variable renewable energy the system can take. Even so, it gives very useful information for planning
grid developments and for broadly assessing the impact of new generation and new interconnections.


## Documentation

- [Online user guide](userguide/index.md)

- HG Svendsen and OC Spro, *PowerGAMA: A new simplified modelling approach for analyses of large interconnected power systems, applied to a 2030 Western Mediterranean case study*, J. Renewable and Sustainable Energy 8 (2016), [doi.org/10.1063/1.4962415](https://doi.org/10.1063/1.4962415)
## Install use-only version
Install latest PowerGAMA release from PyPi:
```
pip install powergama
```

## Install development version
Prerequisites:
- [Poetry](https://python-poetry.org/docs/#installation) (used by CI and `poetry.lock`); [uv](https://docs.astral.sh/uv/) also works
- No separate LP solver is needed: HiGHS is included in the package and used by default. Gurobi (licence required) and other Pyomo-supported solvers can also be used, but are not tested.

### Install
```
git clone git@github.com:powergama/powergama.git
cd powergama
poetry install                 # or: uv sync
poetry run pre-commit install  # or: uv run pre-commit install
poetry run pytest              # or: uv run pytest
```


### GitHub Actions Pipelines
These pipelines are defined:

1. Build: Building and testing on multiple OS and python versions. Triggered on any push to GitHub.
2. Release: Create a release. Triggered when a version tag v* is created.
3. Publish: Publish to PyPI. Triggered when a release is published

## Contribute to the code
You are welcome to contribute to the improvement of the code.

* Use Issues to describe and track needed improvements and bug fixes
* Use branches for development and pull requests to merge into main
* Use [Pre-commit hooks](https://pre-commit.com/)

### Contact

[Harald G Svendsen](https://www.sintef.no/en/all-employees/employee/?empid=3414)  
SINTEF Energy Research


