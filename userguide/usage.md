# PowerGAMA usage

PowerGAMA is a lightweight maket model that may be used in different ways for different
types of analyses. A typical use case has been to investigate future power system 
scenarios and their implications on timestep by timestape operations.

Recently, it has been extended to allow more realistic representation of power system 
operations, and there are now three main ways set up PowerGAMA simulations:

- **Timestep-by-timestep optimisation** (section 1) optimises each timestep using
  linearised power flow equations and generator dispatch based on marginal costs. This is
  the default setup and has been the only option in PowerGAMA versions prior to 1.6.
- **Multi-period optimisation with a rolling horizon** (section 2) optimises a window of
  several timesteps at once, for example 24 hours, and keeps part of it before moving on.
  It was introduced to mimic the day-ahead (DA) market, where the dispatch of a whole day
  is cleared at once on forecasts, and storage is scheduled across the day.
- **The real-time (RT) extension** (section 3) reproduces, in a simplified way, how the
  forecast errors of a day-ahead schedule are balanced when the delivery hour comes: it
  starts from the DA schedule and finds the cheapest way to correct it. It is a high-level
  driver for what the intraday and balancing markets do, not a model of their internal
  mechanisms.

## 1. Timestep by timestep optimisation

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
        storagevalue_time=datapath / "9busmod_profiles_storval_time.csv",  # deprecated, see below
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
    powergama.plots.plotMap(pg_data=data, pg_res=res, nodetype="nodalprice", branchtype="capacity", zoom_start=5)
    ```

The storage-value time curves of this example come from a separate file
(`storagevalue_time`), which is deprecated: they can be ordinary profile columns, and
`filename` can be a list of CSV or Parquet profile files (see
[input_data.md](input_data.md)).

### Solving part of the time range from a given state

`solve` can also solve a few consecutive timesteps, starting from the state of an
earlier run instead of the initial state. The state is the storage levels, the
flexible-load reservoir levels and the previous dispatch that ramp limits use
(`SimulationState`). This lets a study re-solve a few timesteps on a modified grid,
for example with a generator or branch out, without simulating everything before them.
`MemoryResults` keeps such short results in memory instead of a database:

```python
state = powergama.SimulationState.from_results(res, timestep=99)  # end of timestep 99

outaged = copy.deepcopy(data)
outaged.generator.loc[3, ["pmax", "pmin"]] = 0.0
out = powergama.MemoryResults(outaged)
powergama.LpProblem(outaged).solve(out, timesteps=range(100, 124), state=state, show_progress=False)
out.loadshed()  # timestep x node, MW
```

`lp.get_state()` gives the state where the last solve ended. Repeated calls of `solve`
on one `LpProblem` keep the solver, so many short solves are cheap.

## 2. Multi-period optimisation with a rolling horizon

```python
lp = powergama.LpProblem(
    data,
    objective_mode="multi_period",
    horizon_hours=24,
    commit_hours=12,
)
```

`horizon_hours` is the length of the window that is optimised; `commit_hours` is how
much of each window is kept before the next window starts. Setting `commit_hours`
below `horizon_hours` gives a rolling horizon with lookahead: with 24/12, each window
sees a day ahead, its first 12 hours are kept, and the next window starts there, from the
storage levels the kept hours left. With `commit_hours = horizon_hours` the windows do
not overlap. This is how a day-ahead market is mimicked: the dispatch of each window is
cleared at once, so storage is scheduled across the day instead of hour by hour.

This is the same LP as the single-period mode, over a window of `horizon_hours`
timesteps instead of one: `horizon_hours=1` gives the single-period results. Any
solver works, with the same `solve()` options. Losses (`lossmethod` 1 or 2) and
extensions such as `RtExtension` need `horizon_hours=1`.

Storage values — both for generators and for flexible load — are evaluated once per
window from the filling level at the start of it, since the whole window is a single
LP and the value has to enter as a constant coefficient. The single-period mode
re-prices from the current filling level before every time step, being a window of
one. Ramp limits (`ramp_up_pu`, `ramp_down_pu`) apply within each window and from one
window to the next. See [model_description.md](model_description.md) and
[model_formulations.md](model_formulations.md).

## 3. The real-time (RT) extension

The RT extension runs after a day-ahead run. It solves timestep by timestep, with
`RtExtension` attached to the problem, and pays for every deviation from the day-ahead
schedule. A function, `prepare_rt`, writes its inputs from the results of the day-ahead
run:

```python
from powergama.lp.rt import RtExtension, prepare_rt

prepared = prepare_rt(res_da, "rt_inputs", ...)   # res_da: a finished day-ahead run
data_rt = powergama.GridData()
data_rt.readGridData(**prepared.grid_files())
data_rt.readProfileData(filename=["profiles_rt.csv", prepared.files["da_solution"]], ...)
lp = powergama.LpProblem(data_rt, extensions=[RtExtension()])
```

How to set it up and what it does are in [Usage_RT.md](Usage_RT.md) and
[model_description_RT.md](model_description_RT.md).
