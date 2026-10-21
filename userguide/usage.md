# PowerGAMA usage

There are three distinct ways to run PowerGAMA
1. Single-period scheduling (default)
    - sequential solving timestep by timestep. All energy is traded in each timestep and storage is updated between each timestep
1. Multi-period schceduling
    - optimising over a planning horizon, typically representing day-ahead scheduling. Storage is optimised within the horizon. Rolling-horizon optimisation.
1. Balancing
    - Balancing market, timestep-by-timestep optimisation, where there is a cost for deviations from a planned schedule.

## Timestep by timestep optimisation

1. Load data
    ```python
    datapath = pathlib.Path("test_data")

    data = powergama.GridData()
    data.readGridData(
        nodes=datapath / "9busmod_nodes.csv",
        ac_branches=datapath / "9busmod_branches.csv",
        dc_branches=None,
        generators=datapath / "9busmod_generators.csv",
        consumers=datapath / "9busmod_consumers.csv",
    )
    data.readProfileData(
        filename=datapath / "9busmod_profiles.csv",
        storagevalue_filling=datapath / "9busmod_profiles_storval_filling.csv",
        storagevalue_time=datapath / "9busmod_profiles_storval_time.csv",
        timerange=timerange,
        timedelta=1.0,
    )
    ```

2. Run simulation
    ```python
    lp = powergama.LpProblem(data)
    res = powergama.Results(data, "results.sqlite")
    lp.solve(res, solver="appsi_highs", solve_args={})
    ```

3. Analyse results
    
    For example:
    ```python
    powergama.plots.plotMap(
        pg_data=data, pg_res=res, nodetype="nodalprice", branchtype="capacity",
        zoom_start=5
    )
    ```

## 2. Day-ahead market (multi-period optimisation)

```python
lp = powergama.LpProblem(
    data,
    objective_mode="multi_period",
    horizon_hours=24,
    commit_hours=24,
)
```

`horizon_hours` is the length of the window that is optimised; `commit_hours` is how
much of each window is kept before the next window starts. Setting `commit_hours`
below `horizon_hours` gives a rolling horizon with lookahead (e.g. 48/24).

This is the same LP as the single-period mode, over a window of `horizon_hours`
timesteps instead of one: `horizon_hours=1` gives the single-period results. Any
solver works, with the same `solve()` options. Losses (`lossmethod` 1 or 2) and
extensions such as `RtExtension` need `horizon_hours=1`.

Storage values — both for generators and for flexible load — are evaluated once per
window from the filling level at the start of it, since the whole window is a single
LP and the value has to enter as a constant coefficient. The single-period mode
re-prices from the current filling level before every time step, being a window of
one. See [model_formulations.md](model_formulations.md) and
[model_description.md](model_description.md).

## 3. Balancing market (deviations from day-ahead)

This simulation mode can be run after a day-ahead simulation to represent the balancing ("real-time") market, where there may be a cost for deviations from
the day-ahead schedule. It is an extension of the single-period mode, switched on by
passing `RtExtension` to the problem. Model components are indexed by hour in the
window, `[·, h]`, and in single-period mode `h` is 0: `lp.model.varGeneration[i, 0]`.

In the example below, `rt_lock_profile` is the day-ahead schedule for the given
generator, as a fraction of its `pmax`.

```python
from powergama.lp.rt import RtExtension

data.profiles["rt_lock_profile"] = ...
data.generator.loc[index_gen, "rt_target_ref"] = "rt_lock_profile"

lp = powergama.LpProblem(
    data,
    extensions=[RtExtension(balancing_fee_eur_per_mwh=100)],
)
```

`RtExtension` also takes `res_curtailment_factor` and `xborder_flow_penalty_eur_per_mwh`.
In the notation of the WILLOW D4.5 annex these are c0 (`balancing_fee_eur_per_mwh`),
1 + δ_curt (`res_curtailment_factor`) and c_xborder (`xborder_flow_penalty_eur_per_mwh`).
A deviation of anything else from its target costs c0 plus its marginal cost when it is
more than the day-ahead schedule, and c0 minus it when it is less.

Storage is settled on its power, not its filling level: the generation of a storage unit
is tracked through `generator.rt_target_ref`, and its pumping through
`generator.rt_pump_target_ref` (a fraction of `pump_cap`). The storage value enters
through the deviation prices. Flexible-load targets go in `consumer.rt_target_ref`.
`generator.rt_storage_target_ref` and the `da_storage_marginalprice_parquet` lock file
were removed, and setting either raises an error. Hard day-ahead locks (pinned generators, fixed demand,
border-flow windows) are read from parquet files named by attributes on the
`GridData` object; see `powergama.lp.rt.locks`. Without the extension, none of
these inputs are read.