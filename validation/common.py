"""Shared helpers: host id, CSV output, single-core guard, timing, data."""
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

# Fit grid: n x p x k, skipping n*p > 2e7. Mirrors fit_grid in rust/src/main.rs (change together).
FIT_GRID = [(n, p, k) for n in (100, 1_000, 10_000) for p in (10, 100, 1_000, 10_000, 100_000)
            for k in (1, 5, 10) if n * p <= 20_000_000 and k <= min(n, p)]

MIN_BATCH_S = 0.2  # each timed batch runs at least this long
REPEATS = 5


def host():
    return os.environ.get("BENCH_HOST") or os.uname().nodename.split(".")[0].lower()


def require_single_core():
    if os.environ.get("BENCH_SINGLE_CORE") != "1":
        sys.exit("source single.env first (or use ./run.sh): runs must be single-core")


def write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def time_call(fn):
    """(wall_ms, cpu_ms) per call: medians over REPEATS batches of >= MIN_BATCH_S.
    CPU time is the number to trust on a loaded host."""
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
    walls, cpus = [], []
    for _ in range(REPEATS):
        w0, c0 = time.perf_counter(), time.process_time()
        for _ in range(number):
            fn()
        walls.append((time.perf_counter() - w0) / number)
        cpus.append((time.process_time() - c0) / number)
    return float(np.median(walls)) * 1e3, float(np.median(cpus)) * 1e3


def fit_data(n, p, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = X[:, : min(3, p)].sum(axis=1) + rng.standard_normal(n)
    return X, y
