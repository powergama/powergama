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
PowerGAMA is an open-source Python package for power system grid and market analyses.

Since some generators may have an energy storage (hydro power with reservoir and concentrated solar power with thermal storage) the optimal solution in one timestep depends on the previous timestep, and the problem should therefore be solved sequentially. A realistic utilisation of energy storage is ensured through the use of storage values.

PowerGAMA does not include any power market subtleties (such as start-up costs, limited ramp rates, forecast errors, unit commitments) and as such will tend to overestimate the ability to accommodate large amounts of variable renewable energy. Essentially it assumes a perfect market based on nodal pricing without barriers between different countries. This is naturally a gross oversimplification of the real power system, but gives nonetheless very useful information to guide the planning of grid developments and to assess broadly the impacts of new generation and new interconnections.

## Documentation

- [Online user guide](userguide/index.md)

- HG Svendsen and OC Spro, *PowerGAMA: A new simplified modelling approach for analyses of large interconnected power systems, applied to a 2030 Western Mediterranean case study*, J. Renewable and Sustainable Energy 8 (2016), [doi.org/10.1063/1.4962415](https://doi.org/10.1063/1.4962415)
## Install use-only version
Install latest PowerGAMA release from PyPi:
```
pip install powergama
```

## Install development version
Prerequisite: 
- [Poetry](https://python-poetry.org/docs/#installation)
- [Pre-commit](https://pre-commit.com/)
- An LP solver, e.g. the free [CBC solver](https://projects.coin-or.org/Cbc).
Clone or download the code and install it as a python package. The tests rely on the [GLPK solver](https://www.gnu.org/software/glpk/).


### Install 
1. `git clone git@github.com:powergama/powergama.git`
2. `cd powergama`
3. `poetry install`
4. `poetry shell`
5. `poetry run pytest tests`


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

