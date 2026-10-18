"""The core LP code never names the real-time extension (issue #44).

RT plugs into the single-period model through the hooks of ModelExtension. This
pins that from the source side: no module of ``powergama.lp`` outside ``lp/rt/``
may import from the RT package, refer to an RT setting or state, or name an RT
model component. Docstrings and comments may still mention RT; the check is on
code only.
"""

import ast
import re
from pathlib import Path

LP_PACKAGE = Path(__file__).parent.parent / "src" / "powergama" / "lp"
RT_PACKAGE = LP_PACKAGE / "rt"
CORE_MODULES = sorted(p for p in LP_PACKAGE.rglob("*.py") if RT_PACKAGE not in p.parents)

# Identifiers: is_rt, anything rt_-prefixed or _rt-suffixed, and the RT component families.
RT_IDENTIFIER = re.compile(r"(^|_)rt(_|$)|^p_rt_|^varRt|^cRt|RtExtension|RtContext", re.IGNORECASE)
# String constants: component names looked up by string.
RT_STRING = re.compile(r"p_rt_|varRt|cRt[A-Z]")


def _offences(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "rt" or module.startswith("rt.") or ".rt" in f".{module}":
                yield node.lineno, f"imports from {'.' * node.level}{module}"
            for alias in node.names:
                if RT_IDENTIFIER.search(alias.name):
                    yield node.lineno, f"imports {alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if ".lp.rt" in alias.name:
                    yield node.lineno, f"imports {alias.name}"
        elif isinstance(node, ast.Name) and RT_IDENTIFIER.search(node.id):
            yield node.lineno, f"name {node.id}"
        elif isinstance(node, ast.Attribute) and RT_IDENTIFIER.search(node.attr):
            yield node.lineno, f"attribute .{node.attr}"
        elif isinstance(node, ast.arg) and RT_IDENTIFIER.search(node.arg):
            yield node.lineno, f"argument {node.arg}"
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and RT_IDENTIFIER.search(node.name):
            yield node.lineno, f"definition {node.name}"
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and RT_STRING.search(node.value):
            yield node.lineno, f"string {node.value!r}"


def test_core_modules_exist():
    names = {p.relative_to(LP_PACKAGE).as_posix() for p in CORE_MODULES}
    assert {"problem.py", "extension.py", "single_period/build.py", "single_period/data.py"} <= names


def test_core_never_names_rt():
    offenders = [
        f"{path.relative_to(LP_PACKAGE).as_posix()}:{line}: {what}"
        for path in CORE_MODULES
        for line, what in _offences(path)
    ]
    assert not offenders, "RT must reach the core only through ModelExtension hooks:\n  " + "\n  ".join(offenders)


def test_check_catches_an_rt_reference(tmp_path):
    """Guard against the check silently matching nothing."""
    bad = tmp_path / "bad.py"
    bad.write_text(
        "from .rt import data as rt_data\n"
        "def f(lp, cfg):\n"
        "    if cfg.is_rt:\n"
        "        rt_data.update(lp)\n"
        "    return lp.model.component('p_rt_target')\n",
        encoding="utf-8",
    )
    found = {what for _, what in _offences(bad)}
    assert {"imports from .rt", "attribute .is_rt", "name rt_data", "string 'p_rt_target'"} <= found
