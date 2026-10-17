"""The balancing ("real-time") side-path.

RT is not a third formulation. It extends the single-period one: the same
physics, plus deviation tracking against a day-ahead schedule, and an objective
that minimises deviation instead of cost. It is rejected outright in
multi-period mode, which has no RT layer at all.

Everything here is reached from a handful of named seams in
``lp/single_period/``, each guarded by ``cfg.is_rt``. That is the weak form of
making RT a plugin: the core no longer has 450 lines of RT woven through it, but
it does still name RT at those seams.

The strong form -- RT registering itself, with no ``is_rt`` in the core at all --
needs one more thing that is deliberately not done here. In deviation mode the
RT objective *replaces* the cost expression rather than adding to it, so RT
cannot be a purely additive extension until the objective is assembled from
contributed terms. That is a change to the model, not to file organisation.
"""
