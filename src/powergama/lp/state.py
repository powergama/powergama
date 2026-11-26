"""The simulation state that makes the single-period problem sequential.

Between timesteps, an :class:`~powergama.lp.LpProblem` carries three things forward:
the storage levels of generators, the reservoir levels of flexible loads, and the
previous dispatch that ramp limits are measured from. :class:`SimulationState` holds
them, so that a solve can start from any point of an earlier run::

    state = SimulationState.from_results(res, timestep=99)   # end of timestep 99
    lp.solve(other_results, timesteps=range(100, 124), state=state)

This is what lets a study re-solve a few timesteps of a long run on a modified grid
(an outage, say) without simulating everything before them.
"""

import dataclasses

import numpy as np
import pandas as pd


@dataclasses.dataclass
class SimulationState:
    """Storage, flexible-load storage and previous dispatch at the end of a timestep.

    storage : pd.Series
        Generator storage levels in MWh, indexed like ``grid.generator``.
    storage_flexload : pd.Series
        Flexible-load reservoir levels in MWh, indexed like ``grid.consumer``.
    gen_prev : pd.Series or None
        Generator output in MW, indexed like ``grid.generator``, that ramp limits in the
        next timestep are measured from. None (or NaN for a generator) means no ramp
        limit in the next timestep, as at the start of a fresh run.
    """

    storage: pd.Series
    storage_flexload: pd.Series
    gen_prev: pd.Series | None = None

    @classmethod
    def from_results(cls, results, timestep):
        """The state at the end of `timestep` in a results database.

        `timestep` is the timestep as stored in the database. Generators without
        storage and consumers without flexible load get level 0.
        """
        grid = results.grid
        db = results.db
        storage = pd.Series(0.0, index=grid.generator.index, dtype=float)
        stores = grid.getIdxGeneratorsWithStorage()
        values = db.getResultStorageFillingAll(timestep=timestep)
        _check_found(values, stores, "storage", timestep)
        storage.iloc[stores] = values

        flex = pd.Series(0.0, index=grid.consumer.index, dtype=float)
        flex_idx = grid.getIdxConsumersWithFlexibleLoad()
        values = db.getResultFlexloadStorageFillingAll(timestep=timestep)
        _check_found(values, flex_idx, "flexible-load storage", timestep)
        flex.iloc[flex_idx] = values

        prev = db.getResultGeneratorPowerAll(timestep=timestep)
        if not prev:
            raise ValueError(f"No generator results for timestep {timestep} in the results database")
        gen_prev = pd.Series(
            [prev.get(i, 0.0) for i in range(len(grid.generator))], index=grid.generator.index, dtype=float
        )
        return cls(storage=storage, storage_flexload=flex, gen_prev=gen_prev)

    def copy(self):
        return SimulationState(
            storage=self.storage.copy(),
            storage_flexload=self.storage_flexload.copy(),
            gen_prev=None if self.gen_prev is None else self.gen_prev.copy(),
        )


def _check_found(values, indices, what, timestep):
    if len(values) != len(indices):
        raise ValueError(
            f"The results database has {len(values)} {what} values for timestep {timestep}, "
            f"the grid {len(indices)} elements with {what}"
        )


def gen_prev_array(gen_prev, generator_index) -> np.ndarray:
    """`gen_prev` as the writable float array LpProblem keeps (NaN = no ramp limit)."""
    if gen_prev is None:
        return np.full(len(generator_index), np.nan)
    return pd.Series(gen_prev, index=generator_index, dtype=float).to_numpy(copy=True)
