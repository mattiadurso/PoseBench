"""Enforce that the hand-written core modules are fully type-annotated.

This scans source with ``ast`` (no import needed, so wrappers that require a GPU
or cloned ``methods/`` checkouts are still covered) and asserts every function --
including nested and dunder functions -- annotates all parameters (except
``self``/``cls``/``*args``/``**kwargs``) and its return type. It guards against
the annotation sweep silently regressing.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Hand-written modules expected to stay fully annotated. Vendored third party
# (methods/, benchmarks_2D/imc/image-matching-benchmark/) is intentionally excluded.
CORE_MODULES = [
    "common/serialization.py",
    "common/geometry.py",
    "matchers/mnn.py",
    "wrappers/wrapper.py",
    "benchmarks_2D/utils_benchmark.py",
    "benchmarks_3D/utils_benchmark_pose.py",
]
WRAPPER_MODULES = sorted(
    str(p.relative_to(REPO_ROOT)) for p in (REPO_ROOT / "wrappers").glob("*.py")
)
ALL_MODULES = CORE_MODULES + WRAPPER_MODULES

_SKIP_PARAMS = {"self", "cls"}


def _unannotated(func: ast.FunctionDef) -> list[str]:
    """Return human-readable findings for a function missing annotations."""
    findings = []
    args = func.args
    params = args.posonlyargs + args.args + args.kwonlyargs
    for a in params:
        if a.arg in _SKIP_PARAMS:
            continue
        if a.annotation is None:
            findings.append(f"param '{a.arg}'")
    # *args / **kwargs are optional to annotate; only flag positional/keyword params.
    if func.returns is None:
        findings.append("return")
    return findings


@pytest.mark.parametrize("module", ALL_MODULES)
def test_module_fully_annotated(module):
    """Every function in a core module annotates its params and return type."""
    tree = ast.parse((REPO_ROOT / module).read_text())
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            missing = _unannotated(node)
            if missing:
                problems.append(f"  L{node.lineno} {node.name}: {', '.join(missing)}")
    assert not problems, f"{module} has unannotated functions:\n" + "\n".join(problems)
