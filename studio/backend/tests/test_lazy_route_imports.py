# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Source guard for the route imports kept off the boot path.

``python -X importtime -c "import main"`` is the desktop backend's startup, so every import in a
module body is paid on every boot. Two of them were not needed there: ``routes/systemone`` pulled
``core.systemone`` (the Laya checkpoint catalog and the Laya runtime) although only the Decision API
handlers reach it, and ``routes/inference`` / ``routes/models`` pulled
``core.inference.memory_contract`` although only the two memory-estimate handlers price a footprint.
Each now lives in the handler that uses it, and this file is what keeps it there.

A regression here is silent at runtime: the import still succeeds, it just puts the work back on
every boot. Nothing else in the suite would notice, which is why this checks the source, and why it
checks scope the way the interpreter does it: statements in ``tree.body``, and in the ``if``/``try``
blocks beside it, run at import; a statement in a function body does not. Function bodies are the
only boundary skipped -- a class body executes on import too, so an import there counts as module
scope.

``core.inference.studio_tool_loop`` is deliberately NOT in the table even though only
``_proxy_to_external_provider``'s ``_stream`` uses it: four suites replace
``routes.inference.stream_with_studio_tools`` with a stub, so the name has to stay a module
attribute. Measured, deferring it bought 2.6 ms and moved the 43 ms ``core.inference.tools`` import
into ``routes.browser``, which needs it anyway; a seam is worth more than that.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]

# (route module, module kept deferred there, the functions that must import it themselves)
_DEFERRED = (
    (
        "routes/systemone.py",
        "core.systemone",
        (
            "system_one",
            "_decide",
            "_connection_decide",
            "refresh_listed_decision_models",
            "decision_model_objects",
        ),
    ),
    ("routes/inference.py", "core.inference.memory_contract", ("_estimate",)),
    ("routes/models.py", "core.inference.memory_contract", ("_estimate",)),
)


def _tree(rel_path: str) -> ast.Module:
    source = (_BACKEND / rel_path).read_text(encoding="utf-8")
    return ast.parse(source, filename = rel_path)


def _imported_modules(statements) -> set[str]:
    """Every module an ``import`` / ``from ... import`` statement in this body loads.

    A ``from pkg import name`` also names ``pkg.name``: that is how ``from core import systemone``
    and ``import core.systemone`` end up with the same target recorded.
    """
    modules: set[str] = set()
    for statement in statements:
        if isinstance(statement, ast.Import):
            modules.update(alias.name for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom):
            modules.add(statement.module or "")
            modules.update(f"{statement.module}.{alias.name}" for alias in statement.names)
    return modules


def _executed_statements(statements):
    """The statements of this body that run when it runs, without descending into functions."""
    for statement in statements:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        yield statement
        for field in ("body", "orelse", "finalbody"):
            nested = getattr(statement, field, None)
            if isinstance(nested, list):
                yield from _executed_statements(nested)
        for handler in getattr(statement, "handlers", None) or ():
            yield from _executed_statements(handler.body)


def _matches(module: str, loaded: set[str]) -> list[str]:
    """The entries of ``loaded`` that are ``module`` or one of its submodules."""
    return sorted(name for name in loaded if name == module or name.startswith(module + "."))


def _functions_by_name(tree: ast.Module) -> dict[str, list[ast.AST]]:
    """Every function defined in the module, nested ones included, keyed by name."""
    found: dict[str, list[ast.AST]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.setdefault(node.name, []).append(node)
    return found


def _bound_names(function: ast.AST) -> set[str]:
    """Names this function binds itself from the deferred module."""
    bound: set[str] = set()
    wanted = {module for _, module, _ in _DEFERRED}
    for statement in _executed_statements(function.body):
        if isinstance(statement, ast.Import):
            bound.update(
                alias.asname or alias.name.split(".")[0]
                for alias in statement.names
                if _matches(alias.name, wanted)
            )
        elif isinstance(statement, ast.ImportFrom):
            module = statement.module or ""
            if _matches(module, wanted):
                bound.update(alias.asname or alias.name for alias in statement.names)
    return bound


def _loaded_names(function: ast.AST) -> set[str]:
    """Every name the function reads, so an import nothing resolves to can be told apart."""
    return {
        node.id
        for node in ast.walk(function)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


def test_every_owned_route_is_covered():
    """The table is the guard: a file missing from it is a file with no guard left."""
    assert {rel_path for rel_path, _, _ in _DEFERRED} == {
        "routes/systemone.py",
        "routes/inference.py",
        "routes/models.py",
    }


@pytest.mark.parametrize(("rel_path", "module", "functions"), _DEFERRED)
def test_deferred_module_is_not_imported_at_module_scope(rel_path, module, functions):
    tree = _tree(rel_path)
    loaded = _imported_modules(_executed_statements(tree.body))
    offenders = _matches(module, loaded)
    assert not offenders, (
        f"{rel_path} imports {offenders} at module scope again: every boot pays for it, while "
        f"only {sorted(functions)} need it. Move it to the top of those functions."
    )


@pytest.mark.parametrize(("rel_path", "module", "functions"), _DEFERRED)
def test_using_function_imports_the_deferred_module(rel_path, module, functions):
    tree = _tree(rel_path)
    defined = _functions_by_name(tree)
    for name in functions:
        matches = defined.get(name)
        assert matches, f"{rel_path} no longer defines {name}(); update this guard"
        # A 50k-line module reuses short names for nested helpers, so a ``def`` of this name need
        # not be unique: one of the same-named ones is the one that defers, and it has to use what
        # it imports (an import no load resolves to is a botched move, not a deferral).
        deferring = [fn for fn in matches if _bound_names(fn) & _loaded_names(fn)]
        lines = ", ".join(str(fn.lineno) for fn in matches)
        assert deferring, (
            f"{rel_path}: no {name}() (defined at {lines}) imports {module} and uses it, now that "
            f"the module scope no longer does"
        )


def test_the_monkeypatched_tool_loop_seam_stays_a_module_attribute():
    """Why ``core.inference.studio_tool_loop`` is not deferred: four suites patch this name."""
    tree = _tree("routes/inference.py")
    loaded = _imported_modules(_executed_statements(tree.body))
    assert _matches("core.inference.studio_tool_loop", loaded), (
        "routes.inference must import core.inference.studio_tool_loop at module scope: "
        "tests/test_external_hosted_tool_passthrough.py and three other suites monkeypatch "
        "routes.inference.stream_with_studio_tools, which a function-local import would shadow"
    )
