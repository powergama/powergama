"""POWERGAMA_HIGHS_* environment variables are applied to HiGHS (issue #45)."""

import pyomo.environ as pyo
import pytest

from powergama.lp import solver as solver_mod


def _solver():
    return solver_mod.create_solver("appsi_highs", pyo.ConcreteModel())


BUILDERS = pytest.mark.parametrize("build", [_solver], ids=["appsi_highs"])


@BUILDERS
def test_valid_values_are_applied(monkeypatch, build):
    monkeypatch.setenv("POWERGAMA_HIGHS_RANDOM_SEED", " 42 ")
    monkeypatch.setenv("POWERGAMA_HIGHS_THREADS", "2")
    monkeypatch.setenv("POWERGAMA_HIGHS_PARALLEL", "Off")

    opts = build().highs_options

    assert opts["random_seed"] == 42
    assert opts["threads"] == 2
    assert opts["parallel"] == "off"


@BUILDERS
def test_invalid_values_warn_and_are_ignored(monkeypatch, build):
    monkeypatch.setenv("POWERGAMA_HIGHS_RANDOM_SEED", "abc")
    monkeypatch.setenv("POWERGAMA_HIGHS_THREADS", "many")
    monkeypatch.setenv("POWERGAMA_HIGHS_PARALLEL", "sometimes")

    with pytest.warns(UserWarning) as record:
        opts = build().highs_options

    messages = " ".join(str(w.message) for w in record)
    assert "POWERGAMA_HIGHS_RANDOM_SEED" in messages
    assert "POWERGAMA_HIGHS_THREADS" in messages
    assert not {"random_seed", "threads", "parallel"} & set(opts)


@BUILDERS
def test_unset_variables_leave_options_alone(monkeypatch, build):
    for name in ("POWERGAMA_HIGHS_RANDOM_SEED", "POWERGAMA_HIGHS_THREADS", "POWERGAMA_HIGHS_PARALLEL"):
        monkeypatch.delenv(name, raising=False)

    opts = build().highs_options

    assert not {"random_seed", "threads", "parallel"} & set(opts)
