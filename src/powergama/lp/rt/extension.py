"""The real-time (balancing) extension."""

from ..extension import ModelExtension
from . import build as rt_build
from . import data as rt_data
from . import diagnostics as rt_diagnostics
from . import locks as rt_locks
from .context import RtContext


class RtExtension(ModelExtension):
    """Re-simulate a day-ahead schedule in real time, minimising deviation from it.

    The second step of a two-step workflow: a day-ahead run produces a schedule,
    and an external prepare-rt step turns it into this extension's inputs.

    - **Soft targets**, as profile columns referenced from the grid data:
      ``generator.rt_target_ref`` (dispatch, fraction of pmax),
      ``generator.rt_pump_target_ref`` (pumping, fraction of pump_cap)
      and ``consumer.rt_target_ref`` (flexible load, fraction of demand_avg).
      Deviations from them are priced, not forbidden.
    - **Border-flow locks**, as parquet files named by attributes on the GridData
      object (see :data:`~powergama.lp.rt.locks.LOCK_ATTRS`): border-flow windows
      (hard) and border-flow targets (soft).
    - **Pinned generators and fixed demand** are ordinary grid data, not locks:
      ``pmin_ref = pmax_ref`` pointing at a profile column of the DA output
      (fraction of pmax, with ``inflow_ref="const"`` and ``inflow_fac=1``), and
      ``demand_ref`` pointing at one of the DA demand (fraction of demand_avg).
      Caps win over ramps in the core, so a pinned generator follows its pin
      whatever its ramp rates.

    - **Deviation prices**, as unit data (EUR/MWh), like ``pmax``/``pmax_ref``: on
      generators ``rt_price_up``/``rt_price_down`` (generation) and
      ``rt_pump_price_up``/``rt_pump_price_down`` (pumping), on consumers
      ``rt_price_up``/``rt_price_down`` (flexible demand), each with an optional profile
      multiplier ``<column>_ref`` (default ``"const"``, a factor of 1). "Up" is more than
      DA. Every tracked row needs both prices; :func:`~powergama.lp.rt.price_columns`
      (or ``prepare_rt(prices=...)``) writes them from the usual rules.

    The objective drops the operating cost and minimises the priced deviations
    plus load shedding. Storage is settled on its power, generation and pumping,
    not on its filling level. Windows of one timestep only (single-period mode).

    Parameters
    ----------
    xborder_flow_penalty_eur_per_mwh : float
        Penalty per MWh of deviation from day-ahead cross-border flows (c_xborder).
    """

    supports_multi_period = False

    def __init__(self, *, xborder_flow_penalty_eur_per_mwh=0.0):
        self.xborder_flow_penalty_eur_per_mwh = max(0.0, float(xborder_flow_penalty_eur_per_mwh))
        # Set by prepare():
        self.locks = None
        self.ctx = None
        self.debug_path = None
        self.debug_session_id = ""

    def __repr__(self):
        return f"RtExtension(xborder_flow_penalty_eur_per_mwh={self.xborder_flow_penalty_eur_per_mwh})"

    # --- construction ---------------------------------------------------------

    def prepare(self, lp, grid):
        self.locks = rt_locks.load_da_locks(grid)
        self.ctx = RtContext.from_grid(grid, lp._ctx, self.locks)
        rt_diagnostics.initialize_debug_stream(self, grid)

    def add_parameters(self, model, grid, ctx):
        rt_build.add_params(model)

    def add_variables(self, model, grid, ctx):
        rt_build.add_deviation_vars(model)

    def edit_objective_terms(self, model, terms, h):
        # Windows of one timestep only (supports_multi_period is False), so h is 0.
        rt_build.edit_objective_terms(model, self, terms)

    def add_constraints(self, model, grid, ctx):
        rt_build.add_target_tracking(model, grid, ctx, self)
        rt_build.add_flexload_target_tracking(model, grid, ctx, self)
        rt_build.add_pump_target_tracking(model, grid, ctx, self)
        rt_build.add_border_flow_target_tracking(model, grid, ctx, self)

    # --- simulation -----------------------------------------------------------

    def update_timestep(self, lp, timestep):
        # Locks first: they override the limits and demand the core has just written.
        rt_data.apply_da_locks(lp, self, timestep)
        rt_data.update_generator_targets(lp, self, timestep)
        rt_data.update_pump_targets(lp, self, timestep)
        rt_data.update_flexload_targets(lp, self, timestep)
        rt_data.update_deviation_prices(lp, self, timestep)
        rt_data.update_border_flow_targets(lp, self, timestep)

    def on_solve_start(self, lp, timesteps, continue_from_last, results):
        rt_diagnostics.append_solve_start(lp, self, timesteps, continue_from_last, results)

    def after_timestep(self, lp, timestep):
        rt_diagnostics.append_hourly_debug(lp, self, timestep)

    def append_debug_event(self, payload):
        """Append a caller-supplied record to the RT debug stream, if one is configured."""
        rt_diagnostics.append_payload(self, payload)
