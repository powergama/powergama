"""The balancing ("real-time") side-path.

RT is not a third formulation. It extends the single-period one: the same
physics, plus deviation tracking against a day-ahead schedule, and an objective
that minimises deviation instead of cost. It is rejected outright in
multi-period mode, which has no RT layer at all.

RT is the second step of a two-step workflow. A day-ahead run produces a
schedule; an external prepare-rt step turns it into RT inputs -- hard locks
(parquet files: pinned foreign generators, fixed foreign demand, border-flow
windows) and soft targets (``rt_target_ref`` / ``rt_storage_target_ref`` profile
columns, border-flow targets, DA storage prices) whose deviations are penalised.
Everything else is optimised freely.

None of that exists in a normal run. Without ``cfg.is_rt`` the lock files are not
read (a grid carrying them gets a warning), the RT context fields stay empty, and
no RT parameter, variable or constraint is declared: each function in
:mod:`.build` returns without declaring anything.

Everything here is reached from a handful of named seams in
``lp/single_period/``. That is the weak form of making RT a plugin: the core no
longer has 450 lines of RT woven through it, but it does still name RT at those
seams.

The strong form -- RT registering itself, with no ``is_rt`` in the core at all --
needs one more thing that is deliberately not done here. In deviation mode the
RT objective *replaces* the cost expression rather than adding to it, so RT
cannot be a purely additive extension until the objective is assembled from
contributed terms. That is a change to the model, not to file organisation.
"""
