#!/usr/bin/env python3
"""Fit speed, single core: every routine in cases.GROUPS over common.FIT_GRID.

Writes one row per routine and shape to --out. Every routine is checked
against the first plskit routine of its group (tolerance cases.TOL).
"""
import argparse
import sys
from datetime import datetime, timezone

import plskit

from cases import GROUPS, TOL, data, max_diff
from common import FIT_GRID, require_single_core, time_call, write_csv

FIELDS = ["bench", "lib", "version", "routine", "n", "p", "k", "wall_ms", "cpu_ms", "threads", "when"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="CSV output path")
    args = ap.parse_args()
    require_single_core()
    when = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows, bad = [], False
    for name, group in GROUPS.items():
        for n, p, k in FIT_GRID:
            d = data(name, n, p)
            ref = None
            for r in group.plskit + group.refs:
                if not r.runs(n, p):
                    continue
                label = f"{name}/{r.lib}/{r.routine}"
                try:
                    out = r.out(r.fit(d, k), d, k)
                except plskit.PlsKitError as e:
                    # pre_standardized=True refuses a fit that keeps fewer than k components
                    # (n=10k, p=10, k=10 stops at 7); the cell stays empty.
                    print(f"{n:>6} {p:>6} {k:>3}  {label:<40} not timed: {str(e)[:100]}", flush=True)
                    continue
                if ref is None:
                    ref = out  # the group's first plskit routine
                diff = max_diff(ref, out, group.sign_by)
                if diff > TOL:
                    bad = True
                    print(f"MISMATCH {label} n={n} p={p} k={k}: {diff:.2e}", flush=True)
                wall, cpu = time_call(lambda: r.fit(d, k))
                rows.append(dict(bench=name, lib=r.lib, version=r.version, routine=r.routine,
                                 n=n, p=p, k=k, wall_ms=f"{wall:.6g}", cpu_ms=f"{cpu:.6g}",
                                 threads=1, when=when))
                print(f"{n:>6} {p:>6} {k:>3}  {label:<40} wall {wall:10.3f} ms"
                      f"  cpu {cpu:10.3f} ms  diff {diff:.1e}", flush=True)
    write_csv(args.out, FIELDS, rows)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
