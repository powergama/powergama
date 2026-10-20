"""Per-window data for the multi-period formulation.

The counterpart of :mod:`..single_period.data`, and the place the two
formulations are shaped differently. The single-period module rewrites a
model's Params in place each timestep; this one computes a snapshot that gets
baked into a model rebuilt for the window, because the whole window is one LP
and every coefficient in it has to be constant.

That is why the storage value is fixed once per window here, from the filling
level the window starts at, while the single-period path re-prices before every
timestep.

WindowData is the explicit interface between this module and :mod:`.build`:
30 values used to cross that boundary as locals of one 550-line method.
"""

from dataclasses import dataclass, field

from .. import storage_value


@dataclass
class WindowData:
    """Everything the window model is built from. One window, all constants."""

    H: object = None
    ac_lb: object = None
    ac_ub: object = None
    branch_ac_position: object = None
    capacity_h: object = None
    dc_lb: object = None
    dc_ub: object = None
    demand_h: object = None
    dt: object = None
    flex_drain_mw: object = None
    flex_has_floor: object = None
    flex_set: object = None
    flex_storage_cap: object = None
    flex_storage_ini: object = None
    flexload_cost_h: object = None
    gen_cost_h: object = None
    gen_pump_set: object = None
    gen_storage_set: object = None
    inflow_h: object = None
    node_list: object = None
    node_position: object = None
    pmin_h: object = None
    pump_cap_raw: object = None
    pump_cost_h: object = None
    pump_eff_g: object = None
    spill_cap_mw: object = None
    storage_cap: object = None
    storage_ini: object = None
    terminal_storval: object = None
    day_timesteps: list = field(default_factory=list)


def prepare_window(lp, day_timesteps):
    """Build a multi-period Pyomo LP covering all timesteps in *day_timesteps*.

    KEY new constraint (storage continuity):
        varStorage[i,h] = storage_prev
                          + (inflow[h,i] - varGen[i,h]
                             + varPump[i,h]*eff - varSpill[i,h]) * dt
    varStorage is an optimisation variable, so the solver sees the full
    day's inflow/demand profile and can pre-charge/pre-dispatch storage.

    Everything else (power balance, DC flow-angle, branch limits, ramp
    rates) is the same physics as the single-period model,
    now indexed by (entity, hour) instead of (entity) alone.

    Returns (model, pre_data_dict).
    """
    ctx, grid = lp._ctx, lp._grid
    H = len(day_timesteps)
    grid = grid
    dt = float(lp.timeDelta)
    node_list = list(ctx.node_ids)
    branch_ac_list = list(ctx.branch_ac_ids)
    # Positions for the Bbus and DA rows, mapped once rather than scanned per rule
    # call -- the rules below run once per (entity, hour), so a linear scan there is
    # quadratic in the entity count and linear again in the horizon.
    node_position = {n: i for i, n in enumerate(node_list)}
    branch_ac_position = {b: i for i, b in enumerate(branch_ac_list)}
    gen_storage_set = frozenset(lp._idx_generatorsWithStorage)
    gen_pump_set = frozenset(lp._idx_generatorsWithPumping)
    flex_set = frozenset(lp._idx_consumersWithFlexLoad)

    # ── pre-compute storage capacity and pump parameters ──────────
    storage_ini = {int(i): float(lp._storage[i]) for i in lp._idx_generatorsWithStorage}
    storage_cap = {int(i): max(0.0, float(grid.generator.loc[i, "storage_cap"])) for i in lp._idx_generatorsWithStorage}
    spill_cap_mw = {int(i): cap for i, cap in ctx.spill_cap_mw.items()}
    pump_eff_g = {int(i): float(grid.generator.loc[i, "pump_efficiency"]) for i in lp._idx_generatorsWithPumping}
    pump_cap_raw = {int(i): float(grid.generator.loc[i, "pump_cap"]) for i in lp._idx_generatorsWithPumping}

    # ── costs, and the value of energy left in storage ─────────────
    # Per-MWh costs of generating and pumping do not depend on the storage level,
    # storage generators included. Stored energy is valued once, at the end of the
    # window, from the filling level at its start and the time factor at its last
    # timestep -- the single-period path does the same with a window of one.
    gen_cost_h: dict = {}  # (h, i) → €/MWh
    pump_cost_h: dict = {}  # (h, i) → €/MWh pumped
    for i in ctx.gen_ids:
        base_cost = float(grid.generator["fuelcost"].loc[i])
        for hi in range(H):
            gen_cost_h[(hi, i)] = base_cost
    for i in gen_pump_set:
        deadband = float(grid.generator.loc[i, "pump_deadband"])
        for hi in range(H):
            pump_cost_h[(hi, i)] = deadband
    storage_gens = list(lp._idx_generatorsWithStorage)
    values = storage_value.generator_storage_values(
        grid, storage_gens, [storage_ini[i] for i in storage_gens], day_timesteps[-1]
    )
    terminal_storval = {i: float(v) for i, v in zip(storage_gens, values)}

    # ── pre-compute inflow / pmax / pmin per (hour, gen) ─────────
    P_max_base = grid.generator["pmax"]
    P_min_base = grid.generator["pmin"]
    inflow_h: dict = {}  # (hi, i) → inflow MW
    capacity_h: dict = {}  # (hi, i) → installed pmax × availability
    pmin_h: dict = {}  # (hi, i) → pmin lower bound

    for hi, ts in enumerate(day_timesteps):
        for i in ctx.gen_ids:
            inflow_fac = float(grid.generator.loc[i, "inflow_fac"])
            inflow_ref = grid.generator.loc[i, "inflow_ref"]
            pmax_fac = 1.0
            if lp._pmax_ref is not None:
                pr = lp._pmax_ref.iloc[i]
                if isinstance(pr, str) and pr in grid.profiles.columns:
                    pmax_fac = float(grid.profiles.loc[ts, pr])
            cap = max(0.0, float(P_max_base[i]) * pmax_fac)
            infl = cap * inflow_fac * float(grid.profiles.loc[ts, inflow_ref])
            pmin = float(P_min_base[i])
            if lp._pmin_ref is not None:
                pr = lp._pmin_ref.iloc[i]
                if isinstance(pr, str) and pr in grid.profiles.columns:
                    pmin = max(0.0, float(P_max_base[i]) * float(grid.profiles.loc[ts, pr]))
            inflow_h[(hi, i)] = float(infl)
            capacity_h[(hi, i)] = float(cap)
            pmin_h[(hi, i)] = float(pmin)

    # ── pre-compute demand per (hour, consumer) ───────────────────
    demand_h: dict = {}
    for hi, ts in enumerate(day_timesteps):
        for j in ctx.load_ids:
            avg = float(grid.consumer.loc[j, "demand_avg"]) * (1 - float(grid.consumer.loc[j, "flex_fraction"]))
            prof = grid.consumer.loc[j, "demand_ref"]
            d_now = float(grid.profiles.loc[ts, prof]) * avg
            demand_h[(hi, j)] = float(d_now)

    # ── flexible load: reservoir parameters and per-hour value ────
    # Valued like generator storage in this path: the filling level is frozen at
    # window start and only the time-of-year factor varies within the window, so
    # the price stays a constant coefficient in the LP. The single-period path re-prices
    # from the current level every timestep instead.
    flexload_cost_h: dict = {}
    flex_storage_cap: dict = {}  # j → reservoir size, MWh
    flex_storage_ini: dict = {}  # j → level at window start, MWh
    flex_drain_mw: dict = {}  # j → the constant flexible demand it serves, MW
    flex_has_floor: dict = {}  # j → whether the level can be held at or above zero
    for j in flex_set:
        flex_drain_mw[int(j)] = ctx.flex_drain_mw[j]
        flex_storage_cap[int(j)] = ctx.flex_storage_cap[j]
        flex_has_floor[int(j)] = ctx.flex_has_floor[j]
        flex_storage_ini[int(j)] = ctx.flex_storage_seed(j, float(lp._storage_flexload[j]))
        for hi, ts in enumerate(day_timesteps):
            flexload_cost_h[(hi, j)] = float(
                storage_value.flexload_storage_value(lp._grid, j, lp._storage_flexload[j], ts)
            )

    # ── per-hour branch flow bounds: the branch rating ───────────
    # The day-ahead lock windows are RT inputs and never apply here.
    ac_lb: dict = {}  # (hi, b) → lb or None
    ac_ub: dict = {}  # (hi, b) → ub or None
    dc_lb: dict = {}
    dc_ub: dict = {}
    for hi in range(H):
        for b in ctx.branch_ac_ids:
            ac_lb[(hi, b)], ac_ub[(hi, b)] = lp._default_ac_flow_bounds.get(int(b), (None, None))
        for b in ctx.branch_dc_ids:
            dc_lb[(hi, b)], dc_ub[(hi, b)] = lp._default_dc_flow_bounds.get(int(b), (None, None))

    return WindowData(
        H=H,
        ac_lb=ac_lb,
        ac_ub=ac_ub,
        branch_ac_position=branch_ac_position,
        capacity_h=capacity_h,
        dc_lb=dc_lb,
        dc_ub=dc_ub,
        demand_h=demand_h,
        dt=dt,
        flex_drain_mw=flex_drain_mw,
        flex_has_floor=flex_has_floor,
        flex_set=flex_set,
        flex_storage_cap=flex_storage_cap,
        flex_storage_ini=flex_storage_ini,
        flexload_cost_h=flexload_cost_h,
        gen_cost_h=gen_cost_h,
        gen_pump_set=gen_pump_set,
        gen_storage_set=gen_storage_set,
        inflow_h=inflow_h,
        node_list=node_list,
        node_position=node_position,
        pmin_h=pmin_h,
        pump_cap_raw=pump_cap_raw,
        pump_cost_h=pump_cost_h,
        pump_eff_g=pump_eff_g,
        spill_cap_mw=spill_cap_mw,
        storage_cap=storage_cap,
        storage_ini=storage_ini,
        terminal_storval=terminal_storval,
        day_timesteps=list(day_timesteps),
    )
