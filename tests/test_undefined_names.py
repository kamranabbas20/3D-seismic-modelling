"""Every name the package loads must be bound somewhere in its module.

A page callback that references a helper nobody imported raises only when a
user clicks that page, which is exactly the bug this file exists to stop -
the Streamlit app lost an import in a refactor and the whole test suite
stayed green because no test opens the page.

The check deliberately over-approximates what counts as bound: any name
stored, imported, defined, taken as an argument or caught in an ``except``
anywhere in the file, at any nesting depth, plus the builtins.  That makes
a *shadowed* or out-of-scope name invisible here, and it makes a reported
name certainly undefined - there are no false alarms to teach anyone to
ignore this test.
"""

import ast
import builtins
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "sim3d"
MODULES = sorted(SRC.rglob("*.py"))


def bound_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        match node:
            case ast.Name(ctx=ast.Store() | ast.Del()):
                names.add(node.id)
            case ast.FunctionDef() | ast.AsyncFunctionDef() | ast.ClassDef():
                names.add(node.name)
            case ast.arg():
                names.add(node.arg)
            case ast.alias():
                names.add((node.asname or node.name).split(".")[0])
            case ast.ExceptHandler() if node.name:
                names.add(node.name)
            case ast.Global() | ast.Nonlocal():
                names.update(node.names)
            case ast.MatchAs() | ast.MatchStar() if node.name:
                names.add(node.name)
            case ast.MatchMapping() if node.rest:
                names.add(node.rest)
    return names


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_module_uses_a_name_it_never_binds(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    known = bound_names(tree) | set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
    used = {n.id: n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    missing = {name: line for name, line in used.items() if name not in known}
    assert not missing, "\n".join(
        f"{path.name}:{line}: '{name}' is used but never bound"
        for name, line in sorted(missing.items(), key=lambda kv: kv[1]))
