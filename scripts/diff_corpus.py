#!/usr/bin/env python3
"""Compare two plskit corpus trees for numerical similarity.

Usage:
    python3 scripts/diff_corpus.py REFERENCE CANDIDATE [--quiet]
    python3 scripts/diff_corpus.py --self-test

REFERENCE is normally the committed `testdata/`, CANDIDATE a corpus generated
into a scratch root (`cargo run --release -p plskit-testdata-gen --
--testdata-root <scratch>`). The two trees are "similar" when:

* both hold the same `.npz` files under `inputs/` and `outputs/`, and their
  `manifest.json` files list the same cases with the same function, paths
  and kwargs (hashes are not compared: they are what changes);
* every file holds the same keys;
* every field has the same shape and dtype, and float fields have the same
  NaN and infinity masks (infinities of the same sign);
* integer, boolean and string (byte) fields are exactly equal;
* every finite float differs by at most 1e-8 absolute or 1e-6 relative,
  whichever is looser.

For every file and field it prints "bit-identical" or the largest absolute,
relative and ulp difference, so the report can justify a regeneration.
`--quiet` omits the bit-identical fields. Exit status: 0 similar, 1 not
similar, 2 usage error. Needs numpy. A dev tool, not a CI gate.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np

ABS_TOL = 1e-8
REL_TOL = 1e-6
SIGN_MASK = np.int64(0x7FFFFFFFFFFFFFFF)


def npz_files(root: Path) -> set[str]:
    return {
        p.relative_to(root).as_posix()
        for sub in ("inputs", "outputs")
        if (root / sub).is_dir()
        for p in (root / sub).rglob("*.npz")
    }


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def manifest_cases(root: Path) -> dict[str, dict]:
    manifest = json.loads((root / "manifest.json").read_text())
    return {
        c["name"]: {k: c.get(k) for k in ("function", "inputs", "outputs", "kwargs")}
        for c in manifest["cases"]
    }


def ordered(a: np.ndarray) -> np.ndarray:
    """float64 bit patterns mapped to integers that are monotone in the float
    value (+0.0 and -0.0 both map to 0), so a difference of two images counts
    the ulps between the floats."""
    i = np.ascontiguousarray(a, dtype=np.float64).view(np.int64)
    return np.where(i < 0, -(i & SIGN_MASK), i)


def compare_field(a: np.ndarray, b: np.ndarray) -> tuple[bool, str]:
    """(similar, message) for one field."""
    if a.dtype != b.dtype:
        return False, f"dtype {a.dtype} vs {b.dtype}"
    if a.shape != b.shape:
        return False, f"shape {a.shape} vs {b.shape}"
    if a.dtype.kind != "f":
        if np.array_equal(a, b):
            return True, "bit-identical"
        return False, "exact (integer, boolean or string) field differs"
    if a.tobytes() == b.tobytes():
        return True, "bit-identical"
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    if not np.array_equal(nan_a, nan_b):
        return False, "NaN mask differs"
    inf_a, inf_b = np.isinf(a), np.isinf(b)
    if not np.array_equal(inf_a, inf_b) or not np.array_equal(a[inf_a], b[inf_b]):
        return False, "infinity mask differs"
    finite = ~(nan_a | inf_a)
    x = a[finite].astype(np.float64)
    y = b[finite].astype(np.float64)
    if x.size == 0:
        return True, "equal values (NaN payload bits differ)"
    diff = np.abs(x - y)
    scale = np.maximum(np.abs(x), np.abs(y))
    rel = np.divide(diff, scale, out=np.zeros_like(diff), where=scale > 0)
    ulp = np.abs(ordered(x).astype(np.float64) - ordered(y).astype(np.float64))
    ok = bool(np.all(diff <= np.maximum(ABS_TOL, REL_TOL * scale)))
    return ok, f"max abs {diff.max():.3e}, max rel {rel.max():.3e}, max ulp {ulp.max():.0f}"


def diff_trees(
    ref: Path, cand: Path, quiet: bool = False, out: Callable[[str], None] = print
) -> bool:
    similar = True

    def report(ok: bool, line: str, identical: bool = False) -> None:
        nonlocal similar
        similar = similar and ok
        if not (quiet and identical):
            out(("ok   " if ok else "FAIL ") + line)

    ma, mb = manifest_cases(ref), manifest_cases(cand)
    for name in sorted(set(ma) | set(mb)):
        if name not in mb:
            report(False, f"manifest: case {name} missing in candidate")
        elif name not in ma:
            report(False, f"manifest: case {name} missing in reference")
        elif ma[name] != mb[name]:
            report(False, f"manifest: case {name} differs in function, paths or kwargs")
    fa, fb = npz_files(ref), npz_files(cand)
    for f in sorted(fa - fb):
        report(False, f"{f}: missing in candidate")
    for f in sorted(fb - fa):
        report(False, f"{f}: missing in reference")
    for f in sorted(fa & fb):
        a, b = load_npz(ref / f), load_npz(cand / f)
        if set(a) != set(b):
            report(False, f"{f}: keys differ: {sorted(set(a) ^ set(b))}")
            continue
        for key in sorted(a):
            ok, msg = compare_field(a[key], b[key])
            report(ok, f"{f} :: {key}: {msg}", identical=(msg == "bit-identical"))
    out("SIMILAR" if similar else "NOT SIMILAR")
    return similar


def self_test() -> int:
    def write_tree(
        root: Path, fields: dict[str, np.ndarray], extra: bool = False, kwargs=None
    ) -> None:
        (root / "inputs").mkdir(parents=True)
        (root / "outputs" / "f").mkdir(parents=True)
        np.savez(root / "inputs" / "c.npz", X=np.arange(6.0).reshape(3, 2))
        np.savez(root / "outputs" / "f" / "c.npz", **fields)
        if extra:
            np.savez(root / "outputs" / "f" / "extra.npz", v=np.zeros(1))
        manifest = {
            "schema_version": 2,
            "producing_version": "0",
            "cases": [
                {
                    "name": "c",
                    "function": "f",
                    "inputs": "inputs/c.npz",
                    "outputs": "outputs/f/c.npz",
                    "kwargs": kwargs if kwargs is not None else {"k": 1},
                    "hashes": {"inputs_sha256": "", "outputs_sha256": ""},
                }
            ],
        }
        (root / "manifest.json").write_text(json.dumps(manifest))

    base = {
        "v": np.array([1e-3, -2.0, 1e3, np.nan]),
        "n": np.array(3, dtype=np.int64),
        "s": np.frombuffer(b"split_exact", dtype=np.uint8).copy(),
    }

    def variant(**changes: np.ndarray) -> dict[str, np.ndarray]:
        d = {k: v.copy() for k, v in base.items()}
        d.update(changes)
        return d

    v = base["v"]
    cases = [
        ("identical", variant(), False, None, True),
        ("abs 5e-9 near 1e-3", variant(v=np.array([1e-3 + 5e-9, -2.0, 1e3, np.nan])), False, None, True),
        ("abs 1e-7 near 1e-3", variant(v=np.array([1e-3 + 1e-7, -2.0, 1e3, np.nan])), False, None, False),
        ("rel 5e-7 on 1e3", variant(v=np.array([1e-3, -2.0, 1e3 * (1 + 5e-7), np.nan])), False, None, True),
        ("rel 1e-5 on 1e3", variant(v=np.array([1e-3, -2.0, 1e3 * (1 + 1e-5), np.nan])), False, None, False),
        ("NaN moved", variant(v=np.array([np.nan, -2.0, 1e3, v[0]])), False, None, False),
        ("integer field", variant(n=np.array(4, dtype=np.int64)), False, None, False),
        ("string field", variant(s=np.frombuffer(b"split_nb", dtype=np.uint8).copy()), False, None, False),
        ("dtype", variant(n=np.array(3.0)), False, None, False),
        ("shape", variant(v=np.array([1e-3, -2.0, 1e3])), False, None, False),
        ("missing key", {k: x for k, x in base.items() if k != "n"}, False, None, False),
        ("extra file", variant(), True, None, False),
        ("manifest kwargs", variant(), False, {"k": 2}, False),
    ]
    failures = 0
    for name, fields, extra, kwargs, expect in cases:
        with tempfile.TemporaryDirectory() as tmp:
            ref, cand = Path(tmp) / "ref", Path(tmp) / "cand"
            write_tree(ref, base)
            write_tree(cand, fields, extra=extra, kwargs=kwargs)
            got = diff_trees(ref, cand, quiet=True, out=lambda _line: None)
        if got != expect:
            failures += 1
            print(f"self-test FAIL: {name}: similar={got}, expected {expect}")
    print("self-test OK" if failures == 0 else f"self-test: {failures} failure(s)")
    return 0 if failures == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare two plskit corpus trees.")
    ap.add_argument("reference", nargs="?", type=Path)
    ap.add_argument("candidate", nargs="?", type=Path)
    ap.add_argument("--quiet", action="store_true", help="omit bit-identical fields")
    ap.add_argument("--self-test", action="store_true", help="run the built-in checks")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.reference is None or args.candidate is None:
        ap.print_usage(sys.stderr)
        return 2
    return 0 if diff_trees(args.reference, args.candidate, quiet=args.quiet) else 1


if __name__ == "__main__":
    sys.exit(main())
