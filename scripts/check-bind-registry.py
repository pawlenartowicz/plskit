#!/usr/bin/env python3
"""Check the plskit-bind registry against the Python signatures.

Every wrapper exposes the same functions with the same argument
names. The plskit-bind registry drives R and Julia, so it must equal the
Python surface: the same functions, and per function the same parameters
in the same order, with the same keyword-only boundary and defaults.
`_api.py` is read with `ast`, so no import (and no numpy) is needed.

Usage:
    cargo run -p plskit-bind --release --bin plskit-bind-dump > registry.json
    python3 scripts/check-bind-registry.py registry.json
"""

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "plskit-py" / "python" / "plskit"


def python_signatures() -> dict[str, list[tuple]]:
    """name -> [(param, keyword_only, required, default)] for public defs."""
    tree = ast.parse((PKG / "_api.py").read_text(encoding="utf-8"))
    sigs = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
            continue
        a = node.args
        positional = a.posonlyargs + a.args
        pos_defaults = [None] * (len(positional) - len(a.defaults)) + list(a.defaults)
        params = []
        for arg, d in zip(positional, pos_defaults):
            params.append((arg.arg, False, d is None, None if d is None else ast.literal_eval(d)))
        for arg, d in zip(a.kwonlyargs, a.kw_defaults):
            params.append((arg.arg, True, d is None, None if d is None else ast.literal_eval(d)))
        sigs[node.name] = params
    return sigs


def exported_names() -> set[str]:
    tree = ast.parse((PKG / "__init__.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            return {e.value for e in node.value.elts if isinstance(e, ast.Constant)}
    return set()


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    dump = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    sigs = python_signatures()
    python_fns = exported_names() & set(sigs)
    registry = {f["name"]: f for f in dump["functions"]}

    problems = []
    for name in sorted(python_fns - set(registry)):
        problems.append(f"{name}: exported by Python, missing from the registry")
    for name in sorted(set(registry) - python_fns):
        problems.append(f"{name}: in the registry, not exported by Python")
    for name in sorted(python_fns & set(registry)):
        want = sigs[name]
        got = [
            (p["name"], p["keyword_only"], p["required"], None if p["required"] else p["default"])
            for p in registry[name]["params"]
        ]
        if got != want:
            problems.append(f"{name}:\n    python   {want}\n    registry {got}")

    if problems:
        print("== plskit-bind registry drift ==")
        for p in problems:
            print("  " + p)
        return 1
    print(f"== plskit-bind registry: {len(registry)} functions match Python ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
