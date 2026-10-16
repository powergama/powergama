"""Source-level invariants on how the Pyomo model is built.

These are not simulation properties -- they are structural rules that keep the
model-construction code decoupled from `LpProblem` being the Pyomo model itself.
`LpProblem` subclasses `pyo.ConcreteModel`, so inside a rule `model is self` and
both spellings resolve to the same object. The distinction only starts to matter
when the model moves into an attribute, and the two failure modes are very
different:

- `model._grid` raises AttributeError at construction time, which is the good case.
- `self.p_gen_cost` inside a rule keeps resolving, against the wrong object, and
  produces a quietly wrong model.

An AST check costs nothing and catches both before they can be written.
"""

import ast
import re
from pathlib import Path

import pytest

MODULE = Path(__file__).parent.parent / "src" / "powergama" / "LpProblemPyomo.py"

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
def tree():
    return ast.parse(MODULE.read_text(encoding="utf-8"))


def test_rules_read_components_off_the_model_argument(tree):
    """No rule may reach a Pyomo component through `self`.

    This is the silent failure: once the model is an attribute rather than the
    class, `self.p_gen_cost` still resolves and still builds a model -- the wrong
    one -- with nothing raised.
    """
    offenders = [
        f"{MODULE.name}:{sub.lineno} in {fn.name if isinstance(fn, ast.FunctionDef) else '<lambda>'}(): self.{sub.attr}"
        for fn in _rule_functions(tree)
        for sub in _attribute_reads(fn, "self")
        if COMPONENT_NAME.match(sub.attr)
    ]
    assert not offenders, "Pyomo components must be read off the rule's model argument:\n  " + "\n  ".join(offenders)


def test_rules_read_non_components_off_self(tree):
    """No rule may reach non-component state through the model argument.

    `model._grid` and `model._lossmethod` work only because the model *is* the
    LpProblem. They fail loudly under composition, but a search-and-replace over
    `self.` does not find them -- so pin them here instead.
    """
    offenders = [
        f"{MODULE.name}:{sub.lineno} in {fn.name if isinstance(fn, ast.FunctionDef) else '<lambda>'}(): "
        f"{sub.value.id}.{sub.attr}"
        for fn in _rule_functions(tree)
        for arg in MODEL_ARG_NAMES
        for sub in _attribute_reads(fn, arg)
        if not COMPONENT_NAME.match(sub.attr)
    ]
    assert not offenders, (
        "Non-component state must be closed over from self, not read off the model:\n  " + "\n  ".join(offenders)
    )
