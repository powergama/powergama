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
    lp.solve(res, solver="appsi_highs",solver_args={})
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
    objective_mode="daily_24h",
    objective_day_horizon_hours=24,
    objective_day_commit_hours=24,
)
```

## 3. Balancing market (deviations from day-ahead)

This simulation mode can be run after a day-ahead simulation to represent the balancing ("real-time") market, where there may be a cost for devaitions from
the day-ahead schedule.

In the example below, `rt_lock_profile` is the day-ahead schedule for the given
generator.

```python
data.profiles["rt_lock_profile"] = ... 
data.generator.loc[index_gen, "rt_target_ref"] = "rt_lock_profile"

lp = powergama.LpProblem(
    data,
    is_rt=True,
    rt_balancing_fee_eur_per_mwh=100,
)
```