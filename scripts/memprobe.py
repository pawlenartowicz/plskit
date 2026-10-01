#!/usr/bin/env python3
"""Peak-RSS and wall-time probe for plskit resampling calls.

Usage:
    python3 scripts/memprobe.py CALL N P B PAR [--k K] [--n-folds F]
        one probe; PAR is 0 (serial) or 1 (parallel)
    python3 scripts/memprobe.py --table [--out FILE] [--k K] [--n-folds F]
        the reference shapes, serial and parallel
    python3 scripts/memprobe.py --list
        the CALL names

--k sets the component count of every PLS1 call (default 2); `pls3` keeps
k = 1 and `findk` k_max = 4, and the `k` column reports what each call ran
with. --n-folds sets raw_perm's fold count (default: the method's own).

Each probe of --table runs in a fresh interpreter, so its peak belongs to that
call alone. "delta MB" is the peak resident set size above the baseline taken
after the inputs are built, which is the memory the call itself added. Probe
the extension built from the current tree (`pip install . -v` from the
workspace root). A dev tool, not a CI gate.
"""

from __future__ import annotations

import argparse
import resource
import subprocess
import sys
import time
from pathlib import Path

# (call, n, p, B) for --table.
TABLE = [
    ("perm", 100, 20000, 1000),
    ("rotstab", 100, 100000, 200),
    ("pls3", 100, 100000, 0),
    ("ci", 200, 500, 200),
    ("perm_w", 100, 20000, 1000),
    ("raw_perm", 60, 3000, 200),
    ("split_exact", 200, 500, 1000),
]
HEADER = (
    "| call | n | p | B | k | mode | data MB | delta MB | seconds |\n"
    "|---|---|---|---|---|---|---|---|---|"
)
# Calls whose component count does not follow --k.
FIXED_K = {"pls3": 1, "findk": 4}


def peak_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS and KiB on Linux.
    return r / 2**20 if sys.platform == "darwin" else r / 2**10


def calls(plskit, X, y, Y, w, b: int, dp: bool, k: int = 2, n_folds: int | None = None) -> dict:
    rp_args = {"n_perm": b} if n_folds is None else {"n_perm": b, "n_folds": n_folds}
    return {
        "fit": lambda: plskit.pls1_fit(X, y, k=k),
        "perm": lambda: plskit.pls1_perm_null(X, y, k=k, n_perm=b, seed=1, disable_parallelism=dp),
        "perm_w": lambda: plskit.pls1_perm_null(
            X, y, k=k, n_perm=b, seed=1, disable_parallelism=dp, weights=w
        ),
        "perm_mat": lambda: plskit.pls1_perm_null(
            X, y, k=k, n_perm=b, seed=1, disable_parallelism=dp, return_perm_matrix=True
        ),
        "rotstab": lambda: plskit.pls1_rotation_stability(
            X, y, k=k, n_boot=b, seed=1, disable_parallelism=dp
        ),
        "ci": lambda: plskit.pls1_confirmatory_test(
            X, y, k=k, test_method="split_exact", ci=True, n_boot=b, seed=1, disable_parallelism=dp
        ),
        "split_exact": lambda: plskit.pls1_confirmatory_test(
            X, y, k=k, test_method="split_exact", args={"n_perm": b}, seed=1, disable_parallelism=dp
        ),
        "raw_perm": lambda: plskit.pls1_confirmatory_test(
            X, y, k=k, test_method="raw_perm", args=rp_args, seed=1, disable_parallelism=dp
        ),
        "raw_perm_w": lambda: plskit.pls1_confirmatory_test(
            X, y, k=k, test_method="raw_perm", args=rp_args, seed=1, disable_parallelism=dp,
            weights=w,
        ),
        "findk": lambda: plskit.pls1_find_k_optimal(X, y, k_max=4, seed=1, disable_parallelism=dp),
        "pls3": lambda: plskit.pls3_confirmatory_test(
            X, Y, k=1, test_method="split_exact", seed=1, disable_parallelism=dp
        ),
    }


def probe(call: str, n: int, p: int, b: int, par: bool, k: int, n_folds: int | None) -> str:
    import numpy as np
    import plskit

    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, p))
    y = X[:, :5].sum(axis=1) + rng.standard_normal(n)
    Y = np.column_stack(
        [y, X[:, 5:8].sum(axis=1) + rng.standard_normal(n), rng.standard_normal(n)]
    )
    w = 0.5 + (np.arange(n) % 5) * 0.25
    fn = calls(plskit, X, y, Y, w, b, not par, k, n_folds)[call]
    base = peak_mb()
    t0 = time.perf_counter()
    fn()
    dt = time.perf_counter() - t0
    mode = "parallel" if par else "serial"
    return (
        f"| {call} | {n} | {p} | {b} | {FIXED_K.get(call, k)} | {mode} "
        f"| {X.nbytes / 2**20:.1f} | {peak_mb() - base:.1f} | {dt:.2f} |"
    )


def table(out: Path | None, k: int, n_folds: int | None) -> int:
    print(HEADER, flush=True)
    rows = [HEADER]
    extra = ["--k", str(k)] + ([] if n_folds is None else ["--n-folds", str(n_folds)])
    for call, n, p, b in TABLE:
        for par in (0, 1):
            r = subprocess.run(
                [sys.executable, __file__, call, str(n), str(p), str(b), str(par), *extra],
                capture_output=True,
                text=True,
                check=False,
            )
            if r.returncode != 0:
                sys.stderr.write(r.stderr)
                return 1
            rows.append(r.stdout.strip())
            print(rows[-1], flush=True)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(rows) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Peak-RSS probe for plskit calls.")
    ap.add_argument("probe", nargs="*", help="CALL N P B PAR")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--k", type=int, default=2, help="components of every PLS1 call")
    ap.add_argument("--n-folds", type=int, default=None, help="raw_perm fold count")
    args = ap.parse_args()
    if args.list:
        print("\n".join(calls(None, None, None, None, None, 0, True)))
        return 0
    if args.table:
        return table(args.out, args.k, args.n_folds)
    if len(args.probe) != 5:
        ap.print_usage(sys.stderr)
        return 2
    call, n, p, b, par = args.probe
    print(probe(call, int(n), int(p), int(b), par == "1", args.k, args.n_folds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
