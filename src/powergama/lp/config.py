"""Validated constructor arguments for :class:`~powergama.LpProblem`.

Everything here comes from the caller rather than from the grid data, is settled
once at construction and never changes afterwards. Keeping it in a frozen object
is what lets the model builders take an explicit ``cfg`` argument instead of
reaching into ``self`` for a dozen unrelated flags.

Both modes solve the same LP over a window of timesteps (:mod:`.formulation`):
single-period is a window of one timestep, committed one at a time
(:attr:`LpConfig.window_hours`, :attr:`LpConfig.window_commit`).

The unsupported combinations are rejected here rather than at the point of use.
Each of them used to be a silent wrong answer:

- an extension with windows longer than one timestep was silently ignored (see
  :meth:`LpConfig.check_extensions`).
- ``lossmethod != 0`` with windows longer than one timestep reported losses as hard
  zeros. Losses are linearised around the previous timestep's flows, which only
  the first hour of a window has.
"""

import dataclasses
from dataclasses import dataclass

VALID_OBJECTIVE_MODES = ("single_period", "multi_period")


@dataclass(frozen=True)
class LpConfig:
    """Constructor arguments after normalisation and validation.

    Build with :meth:`from_args`, which does the coercion; the constructor itself
    takes already-normalised values.
    """

    lossmethod: int = 0
    penalty_twoway_flow: float = 0.0
    objective_mode: str = "single_period"
    horizon_hours: int = 24
    commit_hours: int = 24
    storage_initial_fill_scale: float = 1.0

    @classmethod
    def from_args(
        cls,
        lossmethod=0,
        penalty_twoway_flow=0,
        objective_mode="single_period",
        horizon_hours=24,
        commit_hours=24,
        storage_initial_fill_scale=1.0,
    ):
        """Normalise and validate the LpProblem constructor arguments."""
        mode = str(objective_mode).strip().lower()
        if mode not in VALID_OBJECTIVE_MODES:
            raise ValueError("objective_mode must be one of: " + ", ".join(VALID_OBJECTIVE_MODES))

        if mode == "multi_period" and int(max(1, int(horizon_hours))) > 1 and lossmethod != 0:
            # Losses are linearised around the flows of the timestep before, which only
            # the first hour of a window has.
            raise NotImplementedError(
                f"lossmethod={lossmethod} is not supported with objective_mode='multi_period' and "
                f"horizon_hours={horizon_hours}: power losses are only modelled in windows of one "
                "timestep. Use lossmethod=0, or objective_mode='single_period' if losses are required."
            )

        return cls(
            lossmethod=lossmethod,
            penalty_twoway_flow=penalty_twoway_flow,
            objective_mode=mode,
            horizon_hours=int(max(1, int(horizon_hours))),
            commit_hours=int(max(1, int(commit_hours))),
            storage_initial_fill_scale=float(max(0.0, min(1.0, storage_initial_fill_scale))),
        )

    @property
    def window_hours(self) -> int:
        """Timesteps in each LP: one in single-period mode, `horizon_hours` in multi-period."""
        return 1 if self.objective_mode == "single_period" else self.horizon_hours

    @property
    def window_commit(self) -> int:
        """Timesteps stored from each window before the next one starts."""
        return 1 if self.objective_mode == "single_period" else min(self.commit_hours, self.horizon_hours)

    def check_extensions(self, extensions) -> None:
        """Reject an extension the chosen window cannot honour.

        Extensions are written for windows of one timestep. Ignoring one in a longer
        window would solve a different problem from the one asked for -- for the RT
        extension, a cost-minimising LP instead of a deviation-minimising one.
        """
        if self.window_hours == 1:
            return
        for ext in extensions:
            if not ext.supports_multi_period:
                raise NotImplementedError(
                    f"{type(ext).__name__} is not supported with objective_mode='multi_period' and "
                    f"horizon_hours={self.horizon_hours}: it only works in windows of one timestep. "
                    "Use objective_mode='single_period'."
                )


# The names LpProblem.__getattr__ resolves against the config, spelled with the
# leading underscore the rest of the code uses. Listed explicitly rather than
# probed with hasattr, so that adding a field to LpConfig cannot silently start
# shadowing an LpProblem attribute of the same name.
CONFIG_ATTRS = frozenset(f.name for f in dataclasses.fields(LpConfig))
