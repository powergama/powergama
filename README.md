[![GitHub license](https://img.shields.io/github/license/powergama/powergama)](https://github.com/powergama/powergama/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10-blue.svg)](https://python.org)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
# PowerGAMA - Power Grid And Market Analysis

PowerGAMA is a Python package for hour-by-hour optimal power flow analysis of interconnected power systems with variable energy sources and storage systems.

## 



## Getting started
Install latest PowerGAMA release from PyPi:
```
pip install powergama
```

## Install development version
Prerequisite: 
- [Poetry](https://python-poetry.org/docs/#installation)
- [Pre-commit](https://pre-commit.com/)
- An LP solver, e.g. the free [CBC solver](https://projects.coin-or.org/Cbc).
Clone or download the code and install it as a python package. 


### Install 
1. `git clone git@github.com:powergama/powergama.git`
2. `cd powergama`
3. `poetry install`
4. `poetry shell`
5. `poetry run pytest tests`


### GitHub Actions Pipelines
These pipelines are defined:

1. Build: Building and testing on multiple OS and python versions. Triggered on any push to GitHub.

## Contribute to the code
You are welcome to contribute to the improvement of the code.

* Use Issues to describe and track needed improvements and bug fixes
* Use branches for development and pull requests to merge into main
* Use [Pre-commit hooks](https://pre-commit.com/)

### Contact

[Harald G Svendsen](https://www.sintef.no/en/all-employees/employee/?empid=3414)  
SINTEF Energy Research

