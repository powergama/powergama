"""LP problem formulation for PowerGAMA.

The package is organised by *formulation*, so that the two ways of posing the
dispatch problem sit side by side and can be read against each other:

``single_period``
    One LP per timestep, solved in sequence, with storage state carried forward
    between solves.
``multi_period``
    A joint LP over a window of ``horizon_hours``, of which the first
    ``commit_hours`` are committed before the window advances.

``rt`` holds the balancing ("real-time") side-path, which extends the
single-period formulation with deviation tracking against a day-ahead schedule.

A window of one period *should* be the single-period problem, and merging the two
formulations is the intended direction of travel. They do not agree today -- see
``userguide/model_formulations.md`` for the component-by-component correspondence
and the places they deliberately differ.
"""

from .problem import LpProblem

__all__ = ["LpProblem"]
