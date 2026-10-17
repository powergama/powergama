"""Source-level invariants on how the Pyomo model is built.

These are not simulation properties -- they are structural rules that keep the
model-construction code decoupled from `LpProblem` being the Pyomo model itself.
`LpProblem` holds the model as `self.model` rather than being one, so the two
spellings no longer resolve to the same object at all. The two failure modes are
very different:

- `model._grid` raises AttributeError at construction time, which is the good case.
- `self.p_gen_cost` inside a rule keeps resolving, against the wrong object, and
  produces a quietly wrong model.

An AST check costs nothing and catches both before they can be written.
"""

import ast
import re
import sqlite3
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo
import pytest

import powergama

LP_PACKAGE = Path(__file__).parent.parent / "src" / "powergama" / "lp"
# Every module in the lp package, so the rules stay pinned as construction code is
# split across single_period/, multi_period/ and rt/.
MODULES = sorted(LP_PACKAGE.rglob("*.py"))

# Pyomo components live under these name prefixes: Params, Sets, Vars, Constraints,
# the objective and the dual suffix.
COMPONENT_NAME = re.compile(r"^(p_|s_|var|c[A-Z]|OBJ$|dual$)")

# A rule is a function whose first positional argument is the Pyomo model. Pyomo
# calls these back with the model it is constructing.
MODEL_ARG_NAMES = {"model", "m"}


def _rule_functions(tree):
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.Lambda)):
            continue
        args = [a.arg for a in node.args.args]
        if args and args[0] in MODEL_ARG_NAMES:
            yield node


def _attribute_reads(node, base):
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id == base:
            yield sub


@pytest.fixture(scope="module")
def trees():
    assert MODULES, f"no modules found under {LP_PACKAGE}"
    return [(m, ast.parse(m.read_text(encoding="utf-8"))) for m in MODULES]


def test_rules_read_components_off_the_model_argument(trees):
    """No rule may reach a Pyomo component through `self`.

    This is the silent failure: once the model is an attribute rather than the
    class, `self.p_gen_cost` still resolves and still builds a model -- the wrong
    one -- with nothing raised.
    """
    offenders = [
        f"{path.name}:{sub.lineno} in {fn.name if isinstance(fn, ast.FunctionDef) else '<lambda>'}(): self.{sub.attr}"
        for path, tree in trees
        for fn in _rule_functions(tree)
        for sub in _attribute_reads(fn, "self")
        if COMPONENT_NAME.match(sub.attr)
    ]
    assert not offenders, "Pyomo components must be read off the rule's model argument:\n  " + "\n  ".join(offenders)


def test_rules_read_non_components_off_self(trees):
    """No rule may reach non-component state through the model argument.

    `model._grid` and `model._lossmethod` work only because the model *is* the
    LpProblem. They fail loudly under composition, but a search-and-replace over
    `self.` does not find them -- so pin them here instead.
    """
    offenders = [
        f"{path.name}:{sub.lineno} in {fn.name if isinstance(fn, ast.FunctionDef) else '<lambda>'}(): "
        f"{sub.value.id}.{sub.attr}"
        for path, tree in trees
        for fn in _rule_functions(tree)
        for arg in MODEL_ARG_NAMES
        for sub in _attribute_reads(fn, arg)
        if not COMPONENT_NAME.match(sub.attr)
    ]
    assert not offenders, (
        "Non-component state must be closed over from self, not read off the model:\n  " + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Composition: LpProblem holds a Pyomo model rather than being one.
# ---------------------------------------------------------------------------


def test_lpproblem_is_not_a_pyomo_model(testcase_9bus_data):
    """The model is an attribute, and attribute reads still fall through to it."""
    lp = powergama.LpProblem(testcase_9bus_data)

    assert not isinstance(lp, pyo.ConcreteModel)
    assert isinstance(lp.model, pyo.ConcreteModel)
    # Callers written when LpProblem was the model keep working for reads.
    assert lp.varGeneration is lp.model.varGeneration
    with pytest.raises(AttributeError):
        lp.no_such_attribute


def test_suffix_duals_reach_extraction_on_the_glpk_path(tmp_path, testcase_9bus_data):
    """Nodal prices must survive the Suffix living on the model, not the LpProblem.

    pyo.Suffix only imports duals when it is a component of the object handed to the
    solver. Attached to the LpProblem instead, the solve still succeeds and the Suffix
    is simply never filled. Here that surfaces as a KeyError out of extraction rather
    than as zeroed prices, because _storeResultsAndUpdateStorage indexes the Suffix
    directly -- but that is a property of the extraction code, not a guarantee, so
    this asserts the prices that reach the database rather than the plumbing.
    """
    data = testcase_9bus_data
    lp = powergama.LpProblem(data)
    res = powergama.Results(data, tmp_path / "glpk_duals.sqlite", replace=True)
    lp.solve(res, solver="glpk")

    assert lp.model.component("dual") is not None, "Suffix is not a component of the model"
    assert len(lp.dual) > 0, "no duals were imported"

    with sqlite3.connect(res.db.filename) as con:
        prices = pd.read_sql_query("SELECT nodalprice FROM Res_Nodes", con)["nodalprice"]
    assert (prices.abs() > 1e-9).any(), "every nodal price is zero -- duals were not imported"
