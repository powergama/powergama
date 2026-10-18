"""Optional additions to the single-period model.

An extension adds to the model without the core naming it. It is passed to
:class:`~powergama.LpProblem` as ``extensions=[...]``, and the core calls its
hooks at fixed points; each hook does nothing unless overridden.

====================================  =============================================
hook                                  called
====================================  =============================================
``prepare(lp, grid)``                 in ``LpProblem.__init__``, before the model is
                                      built; derive the extension's own data here
``add_parameters(model, grid, ctx)``  after the core parameters are declared
``add_variables(model, grid, ctx)``   after the core variables are declared
``edit_objective_terms(model, t)``    while the objective is built; *t* is the
                                      ordered dict of named core terms, to add to,
                                      drop from or replace in place
``add_constraints(model, grid, ctx)`` after the core constraints are declared
``update_timestep(lp, timestep)``     after the core has updated the model for the
                                      timestep, before it is solved
``on_solve_start(lp, timesteps,``     once, at the start of ``solve()``, with the
``continue_from_last, results)``      timesteps about to be solved
``after_timestep(lp, timestep)``      after each timestep is solved, before its
                                      results are stored
====================================  =============================================

Construction hooks follow the same rules as the core builders: inside a
constraint rule, Pyomo components come off the rule's ``model`` argument and
everything else off ``ctx``, ``grid`` or the extension itself.

Two class attributes describe the extension to the core:

``supports_multi_period``
    Extensions hook into the single-period formulation only. The multi-period
    model has no hooks, so combining an extension that sets this False with
    ``objective_mode="multi_period"`` raises rather than being silently ignored.
``exempt_pinned_from_ramp``
    Skip ramp-rate clipping for generators whose limits pin them to one output
    (pmin ≈ pmax). Clipping such a generator lowers pmax below pmin, and the
    guard that follows then lowers pmin, moving it off the pinned output.
"""


class ModelExtension:
    """Base class for an optional addition to the single-period model."""

    supports_multi_period = False
    exempt_pinned_from_ramp = False

    def prepare(self, lp, grid):
        pass

    def add_parameters(self, model, grid, ctx):
        pass

    def add_variables(self, model, grid, ctx):
        pass

    def edit_objective_terms(self, model, terms):
        pass

    def add_constraints(self, model, grid, ctx):
        pass

    def update_timestep(self, lp, timestep):
        pass

    def on_solve_start(self, lp, timesteps, continue_from_last, results):
        pass

    def after_timestep(self, lp, timestep):
        pass
