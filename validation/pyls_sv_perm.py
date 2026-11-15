"""plskit `sv_perm` against pyls `behavioral_pls`: statistic and p-value (Monte Carlo).

Standalone, because cases.py / validate.py compare arrays at 1e-8 and this p-value check is statistical.
Run from validation/: .venv/bin/python pyls_sv_perm.py   (exit 1 on any failed check)
"""
import sys

import numpy as np

if not hasattr(np, "row_stack"):  # NumPy removed the alias of vstack; pyls still calls it
    np.row_stack = np.vstack
import pyls  # noqa: E402
from plskit import pls1_confirmatory_test, pls3_confirmatory_test

N, P, B = 60, 30, 10_000
STRENGTHS = [0.05, 0.08, 0.11, 0.14, 0.17, 0.20, 0.25, 0.30]  # planted LV1 strength, tried in order
P_LO, P_HI = 0.01, 0.5
STAT_RTOL = 1e-8


def make(seed, q, strength):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(N)
    X = rng.standard_normal((N, P)) + strength * np.outer(z, rng.standard_normal(P))
    Y = rng.standard_normal((N, q)) + strength * np.outer(z, rng.standard_normal(q))
    return X, Y


def run_pyls(X, Y, rotate, seed):
    r = pyls.behavioral_pls(X, Y, n_perm=B, n_boot=0, n_split=0, test_split=0,
                            rotate=rotate, seed=seed, verbose=False)
    return float(np.ravel(r.singvals)[0]), float(r.permres.pvals[0])


def run_plskit(X, Y, rotate, seed):
    if Y.shape[1] == 1:
        r = pls1_confirmatory_test(X, Y[:, 0], k=1, test_method="sv_perm", args={"n_perm": B}, seed=seed)
    else:
        r = pls3_confirmatory_test(X, Y, k=1, test_method="sv_perm", args={"n_perm": B, "rotate": rotate}, seed=seed)
    return r.statistic, r.pvalue


def pick(q, rotates, count=3):
    """First `count` datasets whose pyls p lies in (P_LO, P_HI) for every rotate setting."""
    out = []
    for i, strength in enumerate(STRENGTHS * 4):
        seed = 1000 * q + i
        X, Y = make(seed, q, strength)
        res = {rot: run_pyls(X, Y, rot, seed) for rot in rotates}
        if all(P_LO < p < P_HI for _, p in res.values()):
            out.append((seed, strength, X, Y, res))
            if len(out) == count:
                return out
    sys.exit(f"q={q}: only {len(out)} of {count} datasets had pyls p in ({P_LO}, {P_HI})")


def main():
    checks = failures = 0
    worst_stat = 0.0
    for q, rotates in ((4, (False, True)), (1, (False,))):
        for seed, strength, X, Y, res in pick(q, rotates):
            for rot in rotates:
                sv, p_py = res[rot]
                stat, p_pl = run_plskit(X, Y, rot, seed)
                rel = abs(stat - N * sv) / stat
                pbar = (p_py + p_pl) / 2
                tol = 4 * np.sqrt(2 * pbar * (1 - pbar) / B)
                ok_stat, ok_p = rel <= STAT_RTOL, abs(p_pl - p_py) <= tol
                checks += 1
                failures += not (ok_stat and ok_p)
                worst_stat = max(worst_stat, rel)
                print(f"PLS{'3' if q > 1 else '1'} q={q} seed={seed} strength={strength:.2f} rotate={rot}: "
                      f"stat rel diff {rel:.2e} {'ok' if ok_stat else 'FAIL'}; "
                      f"p plskit {p_pl:.4f} pyls {p_py:.4f} |diff| {abs(p_pl - p_py):.4f} tol {tol:.4f} "
                      f"{'ok' if ok_p else 'FAIL'}", flush=True)
    print(f"{checks} checks (dataset x setting, statistic and p each), {failures} failed; "
          f"worst statistic rel diff {worst_stat:.2e}")
    # PLS3: 3 datasets x 2 rotate settings; PLS1: 3 datasets.
    if checks != 9:
        print(f"expected 9 checks, ran {checks}")
        sys.exit(1)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
