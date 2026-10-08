#!/usr/bin/env python3
"""Compare every plskit fit in cases.GROUPS with its outside references.

Prints the largest difference per comparison; --save also writes
results/validation.csv. Exit code 1 when a difference exceeds cases.TOL.
"""
import argparse
import sys

from cases import GROUPS, TOL, data, max_diff
from common import ROOT, write_csv

# n x p: small, wide, tall
SHAPES = [(50, 10), (30, 100), (2_000, 20)]
KS = (1, 3)
FIELDS = ["group", "routine", "version", "ref_lib", "ref_version", "ref_routine",
          "n", "p", "k", "compared", "max_diff", "ok"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--save", action="store_true", help="also write results/validation.csv")
    args = ap.parse_args()
    rows = []
    for name, group in GROUPS.items():
        for n, p in SHAPES:
            d = data(name, n, p)
            for k in KS:
                refs = [(r, r.out(r.fit(d, k), d, k)) for r in group.refs]
                for mine in group.plskit:
                    got = mine.out(mine.fit(d, k), d, k)
                    for r, ref in refs:
                        diff = max_diff(got, ref, group.sign_by)
                        ok = diff <= TOL
                        rows.append(dict(group=name, routine=mine.routine, version=mine.version,
                                         ref_lib=r.lib, ref_version=r.version, ref_routine=r.routine,
                                         n=n, p=p, k=k, compared=" ".join(got),
                                         max_diff=f"{diff:.3g}", ok=ok))
                        print(f"{name:<14} {mine.routine:<19} vs {r.lib + '/' + r.routine:<22}"
                              f" n={n:<5} p={p:<4} k={k}  {diff:8.1e}  {'ok' if ok else 'MISMATCH'}",
                              flush=True)
    bad = sum(not r["ok"] for r in rows)
    print(f"{len(rows) - bad} of {len(rows)} comparisons within {TOL:g}")
    if args.save:
        write_csv(ROOT / "results" / "validation.csv", FIELDS, rows)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
