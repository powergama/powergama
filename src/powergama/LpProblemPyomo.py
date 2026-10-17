"""Backwards-compatible import location for :class:`powergama.LpProblem`.

The implementation moved to the :mod:`powergama.lp` package, which is organised by
formulation (``single_period`` / ``multi_period``, plus the ``rt`` side-path). This
module is kept because ``powergama.LpProblemPyomo`` is a plausible import path in
code outside this repository.

New code should import from :mod:`powergama` or :mod:`powergama.lp`.
"""

from .lp.problem import LpProblem

__all__ = ["LpProblem"]
