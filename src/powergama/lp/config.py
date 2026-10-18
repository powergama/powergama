"""Validated constructor arguments for :class:`~powergama.LpProblem`.

Everything here comes from the caller rather than from the grid data, is settled
once at construction and never changes afterwards. Keeping it in a frozen object
is what lets the model builders take an explicit ``cfg`` argument instead of
reaching into ``self`` for a dozen unrelated flags.

The unsupported mode combinations are rejected here rather than at the point of
use. Each of them used to be a silent wrong answer:

- an extension with ``multi_period`` was silently ignored: the multi-period
  model has no extension hooks (see :meth:`LpConfig.check_extensions`).
- ``lossmethod != 0`` with ``multi_period`` reported losses as hard zeros.
- ``ramp_daily_reset`` with ``commit_hours != 24`` released the ramp constraint
  at every commit boundary instead of at midnight.
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

        if mode == "multi_period" and lossmethod != 0:
            # Previously only a warning; the day-joint power balance has no loss terms at
            # all and losses were reported as hard-coded zeros.
            raise NotImplementedError(
                f"lossmethod={lossmethod} is not supported with objective_mode='multi_period': "
                "power losses are not modelled in the multi-period LP. Use lossmethod=0, or "
                "objective_mode='single_period' if losses are required."
            )

        return cls(
            lossmethod=lossmethod,
            penalty_twoway_flow=penalty_twoway_flow,
            objective_mode=mode,
            horizon_hours=int(max(1, int(horizon_hours))),
            commit_hours=int(max(1, int(commit_hours))),
            storage_initial_fill_scale=float(max(0.0, min(1.0, storage_initial_fill_scale))),
        )

    def check_extensions(self, extensions) -> None:
        """Reject an extension the chosen formulation cannot honour.

        Extensions hook into the single-period model only. Ignoring one in
        multi-period mode would solve a different problem from the one asked for --
        for the RT extension, a cost-minimising LP instead of a deviation-minimising
        one.
        """
        if self.objective_mode != "multi_period":
            return
        for ext in extensions:
            if not ext.supports_multi_period:
                raise NotImplementedError(
                    f"{type(ext).__name__} is not supported with objective_mode='multi_period': the "
                    "multi-period model has no extension hooks. Use objective_mode='single_period'."
                )

    def check_ramp_daily_reset(self, ramp_daily_reset) -> None:
        """Reject ramp_daily_reset on a commit window that is not a whole day.

        Config alone cannot decide this -- it depends on whether any generator in
        the grid actually sets the flag -- so it is checked separately, once the
        grid column has been read.
        """
        # The two solve paths release the ramp constraint on different clocks: the
        # single-period path at absolute midnight (timestep % 24 == 0), the
        # multi-period path at every commit-window boundary. These only coincide
        # when a window is a whole day, so any other commit length would silently
        # change the meaning of ramp_daily_reset.
        if (
            ramp_daily_reset is not None
            and bool(ramp_daily_reset.any())
            and self.objective_mode == "multi_period"
            and self.commit_hours != 24
        ):
            raise NotImplementedError(
                "ramp_daily_reset=True requires commit_hours=24 in "
                f"objective_mode='multi_period' (got {self.commit_hours}). The "
                "multi-period path releases the ramp constraint at each commit-window "
                "boundary, which only coincides with midnight for whole-day windows."
            )


# The names LpProblem.__getattr__ resolves against the config, spelled with the
# leading underscore the rest of the code uses. Listed explicitly rather than
# probed with hasattr, so that adding a field to LpConfig cannot silently start
# shadowing an LpProblem attribute of the same name.
CONFIG_ATTRS = frozenset(f.name for f in dataclasses.fields(LpConfig))
