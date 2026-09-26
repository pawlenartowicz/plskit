#!/usr/bin/env python3
"""PLS1 fit speed: plskit vs scikit-learn vs ikpls (Improved Kernel PLS).

A development tool, not a CI gate. Run it before and after a kernel change
and compare the two CSVs with --compare.

Requirements: plskit built into the active environment (`pip install . -v`
from plskit/), plus `pip install scikit-learn ikpls`.

Default grid: four representative shapes (n x p, k: 1000 x 10000 at
k=1 and k=10, 10000 x 1000 at k=10,
10000 x 10 at k=10). --full runs n in {100, 1000, 10000} x p in
{10, 100, 1000, 10000} x k in {1, 5, 10}, skipping n*p > 2e7.
ikpls Algorithm 2 forms the p x p Gram X'X and runs only for p <= 1000.
Every library is checked for agreement with plskit's predictions; a row
that disagrees by more than 1e-8 is flagged and the exit code is 1.
"""
import argparse
import csv
import platform
import sys
import time

import numpy as np
import plskit
import sklearn
from sklearn.cross_decomposition import PLSRegression

try:
    from ikpls.numpy import PLS as IKPLS
except ImportError:  # older ikpls layouts
    from ikpls.numpy_ikpls import PLS as IKPLS

DEFAULT_GRID = [(1_000, 10_000, 1), (1_000, 10_000, 10), (10_000, 1_000, 10), (10_000, 10, 10)]
FULL_NS = [100, 1_000, 10_000]
FULL_PS = [10, 100, 1_000, 10_000]
FULL_KS = [1, 5, 10]
MIN_BATCH_S = 0.2  # each timed batch runs at least this long
REPEATS = 5
PRED_TOL = 1e-8


def time_call(fn):
    """Median seconds per call over REPEATS batches of at least MIN_BATCH_S."""
    fn()  # warm-up
    number = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
        dt = time.perf_counter() - t0
        if dt >= MIN_BATCH_S or number >= 1_000_000:
            break
        number *= 2 if dt == 0 else max(2, int(MIN_BATCH_S / dt * 1.2))
    samples = [dt / number]
    for _ in range(REPEATS - 1):
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
        samples.append((time.perf_counter() - t0) / number)
    return float(np.median(samples))


def grid(full):
    if not full:
        return DEFAULT_GRID
    return [(n, p, k) for n in FULL_NS for p in FULL_PS for k in FULL_KS
            if n * p <= 20_000_000 and k <= min(n, p)]


def run(full, out_csv):
    print(f"python {sys.version.split()[0]}, numpy {np.__version__}, "
          f"sklearn {sklearn.__version__}, plskit {plskit.__version__}, "
          f"{platform.machine()}")
    print(f"{'n':>6} {'p':>6} {'k':>3} {'plskit':>9} {'sklearn':>9} "
          f"{'ikpls1':>9} {'ikpls2':>9}   (ms)  max|dpred| vs plskit")
    rng = np.random.default_rng(0)
    rows, bad = [], False
    data = {}
    for n, p, k in grid(full):
        if (n, p) not in data:
            X = rng.standard_normal((n, p))
            y = X[:, : min(3, p)].sum(axis=1) + rng.standard_normal(n)
            data[(n, p)] = (X, y)
        X, y = data[(n, p)]
        fits = {
            "plskit": lambda: plskit.pls1_fit(X, y, k=k),
            "sklearn": lambda: PLSRegression(n_components=k, scale=True).fit(X, y),
            "ikpls1": lambda: IKPLS(algorithm=1).fit(X, y, k),
        }
        if p <= 1_000:
            fits["ikpls2"] = lambda: IKPLS(algorithm=2).fit(X, y, k)
        ref = plskit.pls1_predict(fits["plskit"](), X)
        diffs = [np.max(np.abs(ref - fits["sklearn"]().predict(X).ravel()))]
        for name in ("ikpls1", "ikpls2"):
            if name in fits:
                pred = np.asarray(fits[name]().predict(X, n_components=k)).ravel()
                diffs.append(np.max(np.abs(ref - pred)))
        dpred = float(max(diffs))
        row = {"n": n, "p": p, "k": k}
        for name in ("plskit", "sklearn", "ikpls1", "ikpls2"):
            row[f"{name}_ms"] = time_call(fits[name]) * 1e3 if name in fits else float("nan")
        row["max_abs_pred_diff"] = dpred
        rows.append(row)
        flag = "" if dpred <= PRED_TOL else "  MISMATCH"
        bad |= dpred > PRED_TOL
        print(f"{n:>6} {p:>6} {k:>3} {row['plskit_ms']:>9.3f} {row['sklearn_ms']:>9.3f} "
              f"{row['ikpls1_ms']:>9.3f} {row['ikpls2_ms']:>9.3f}         {dpred:.1e}{flag}",
              flush=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv}")
    return 1 if bad else 0


def compare(before_csv, after_csv):
    def load(path):
        with open(path, newline="") as f:
            return {(int(r["n"]), int(r["p"]), int(r["k"])): r for r in csv.DictReader(f)}
    before, after = load(before_csv), load(after_csv)
    print(f"{'n':>6} {'p':>6} {'k':>3} {'before ms':>10} {'after ms':>10} {'speedup':>8} "
          f"{'sklearn ms':>11} {'ikpls1 ms':>10}")
    for key in sorted(before.keys() & after.keys()):
        b, a = float(before[key]["plskit_ms"]), float(after[key]["plskit_ms"])
        print(f"{key[0]:>6} {key[1]:>6} {key[2]:>3} {b:>10.3f} {a:>10.3f} {b / a:>7.2f}x "
              f"{float(after[key]['sklearn_ms']):>11.3f} {float(after[key]['ikpls1_ms']):>10.3f}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--full", action="store_true", help="run the full n x p x k grid")
    ap.add_argument("--out", default="bench_pls1_fit.csv", help="CSV output path")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                    help="print plskit speedups between two CSVs and exit")
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare)
    return run(args.full, args.out)


if __name__ == "__main__":
    sys.exit(main())
