"""LP problem formulation for PowerGAMA.

Both simulation modes solve the same LP, :mod:`.formulation`, over a window of
timesteps: single-period mode is a window of one timestep, solved in sequence with
storage state carried forward between solves; multi-period mode is a window of
``horizon_hours``, of which the first ``commit_hours`` are committed before the
window advances. The model is built once per window length and its parameters are
rewritten for every window.

Extensions (:mod:`.extension`) add to the model through fixed hooks without the
core naming them. ``rt`` holds the one that ships with the package:
:class:`~powergama.lp.rt.RtExtension`, the balancing ("real-time") simulation,
which tracks a day-ahead schedule.

``userguide/model_formulations.md`` describes the formulation component by
component, and what a window longer than one timestep adds.
"""

from .extension import ModelExtension
from .problem import LpProblem

__all__ = ["LpProblem", "ModelExtension"]
