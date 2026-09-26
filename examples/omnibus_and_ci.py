"""End-to-end example: PLS1 fit + every omnibus method + bootstrap CI bundle.

Run from the public monorepo root (`plskit/`) after building the Python
wrapper:

    maturin develop --release
    python examples/omnibus_and_ci.py

This is the extended companion to `fit_and_ci.py`. Where that example
demonstrates a single recommended test + CI, this one walks the full
omnibus method axis on the same data so the methods can be compared
side by side, including the score/split_exact complementarity
(`_docs/python/api.md` §3.1).

Pipeline:

  1. Generate synthetic data with a known PLS1 signal in the first two
     of D=8 columns (everything else is noise). y depends on X through a
     single direction, so the signal is concentrated.
  2. Fit PLS1 at K=1 (the recommended omnibus K, `_docs/python/api.md`
     §3.1 "default `k=1`"; this design has no reason to fix K higher).
  3. Run every omnibus method on the same X, y, k, seed and tabulate
     p-value + statistic + per-method workload knobs:
       * split_exact: permutation-calibrated split-half test that holds
                      its level on any design; recommended at k=1
       * split_nb:    same held-out-correlation statistic, calibrated by
                      the Fisher-z t approximation instead of permutation
                      (faster; auto-gated on small or low-rank designs)
       * score:       closed-form Welch-Satterthwaite on ||X'y||^2
                      (anisotropy-aware, K-free)
       * e:           universal inference (split-LR e-value,
                      calibration-free, non-asymptotic α bound)
     Reporting `score` and `split_exact` side-by-side is the canonical
     pattern (`_docs/python/api.md` §3.1): they complement rather than
     replace each other. `score` wins on signal diffuse across many
     directions, `split_exact` on signal concentrated in a few, as here.
     `split_exact`'s p-value is floored at 1/(n_perm+1); the closed-form
     and asymptotic p-values are not, so compare them against α, not
     against each other.
     `raw_perm` is legacy (`_docs/concepts/PLS1/fit-and-predict.md`) and
     is omitted here.
  4. Re-run the recommended test (`split_exact`) with `ci=True` to attach
     a bootstrap CI bundle, then print:
       * holdout-correlation CI (composite generalization test, r-scale,
         asymmetric Fisher-z bounds in (-1, 1)),
       * per-variable leverage CIs (variable importance; a normal-theory
         bootstrap interval around the full-data leverage, which is in
         [0, 1] and so are both bounds: signal coords clear zero, noise
         coords reach it and read exactly 0),
       * per-variable subsampling z for β (β_ref[j] over a subsampling
         SE corrected for sampling without replacement and for the
         shrinkage of subsample fits; at k=1 roughly N(0, 1) when the
         population coefficient is 0, so noise coords mostly stay
         below 1.96 and signal coords land far above it; does not grow
         with n_boot; see `_docs/python/results.md`),
       * per-coordinate β CIs (regression-style diagnostic; see
         `_docs/python/results.md` caveats; the midpoint is *not* guaranteed to
         bracket the full-data β_ref under PLS shrinkage on small m),
       * engine diagnostic counters.

The CI bundle uses an independent resampling pass (subsamples for β and
holdout correlation, bootstrap resamples for leverage) on a child-seed
branch of the user-facing seed, so the omnibus and CI streams don't
interfere.
"""

from __future__ import annotations

import numpy as np

import plskit


# args required by each omnibus method (`_docs/python/api.md` §3.1,
# "args by method").
# `score` and `e` take no method-specific kwargs.
_OMNIBUS_ARGS: dict[str, dict] = {
    "split_exact": {"n_perm": 500, "n_splits": 50},
    "split_nb":    {"n_splits": 50},
    "score":       {},
    "e":           {},
}


def synth(n: int = 200, d: int = 8, snr: float = 4.0, seed: int = 42):
    """Two-signal-variable + (d-2) noise-variable PLS1 design."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    beta_signal = np.zeros(d)
    beta_signal[:2] = 1.0
    y = X @ beta_signal * snr + rng.standard_normal(n)
    return X, y


def fmt_ci(name: str, ci) -> str:
    return (
        f"  {name:<14} point={ci.point:+.4f}  "
        f"CI=[{ci.lower:+.4f}, {ci.upper:+.4f}]  sd={ci.sd:.4f}"
    )


def fmt_workload(r) -> str:
    """One-line description of the engine workload that produced `r`."""
    parts = []
    if r.n_perm is not None:
        parts.append(f"n_perm={r.n_perm}")
    if r.n_splits is not None:
        parts.append(f"n_splits={r.n_splits}")
    return ", ".join(parts) if parts else "closed-form"


def main() -> None:
    X, y = synth(n=200, d=8, snr=4.0, seed=42)

    # 1. Fit ----------------------------------------------------------------
    model = plskit.pls1_fit(X, y, k=1)
    print(f"PLS1 fit: K={model.k_used}, ‖β‖={np.linalg.norm(model.beta):.4f}")

    # 2. Omnibus tests: full method axis, shared seed ------------------------
    print("\nomnibus tests (same X, y, k=1, seed=2026):")
    print(f"  {'method':<11}  {'p-value':>10}  {'statistic':>10}  {'workload'}")
    omni: dict[str, plskit.ConfirmatoryTestResult] = {}
    for method, args in _OMNIBUS_ARGS.items():
        omni[method] = plskit.pls1_confirmatory_test(
            X, y, k=1,
            method=method, args=args or None,
            seed=2026,
        )
    for method, r in omni.items():
        print(
            f"  {method:<11}  {r.pvalue:>10.4g}  {r.statistic:>10.4f}  "
            f"{fmt_workload(r)}"
        )
    print(
        "  (split_exact's p-value is floored at 1/(n_perm+1); legacy"
        " raw_perm is omitted.)"
    )
    print(
        "  score and split_exact complement each other: score wins on"
        " diffuse signal,\n  split_exact on concentrated signal (this"
        " design); report both."
    )

    # 3. Recommended test + bootstrap CI bundle ------------------------------
    r = plskit.pls1_confirmatory_test(
        X, y, k=1,
        method="split_exact",
        args={"n_perm": 500, "n_splits": 50},
        ci=True,
        n_boot=500,
        m_rate=0.7,
        level=0.95,
        max_failure_rate=0.0,   # strict: error if any resample fails
        seed=2026,
    )

    print("\nsplit_exact (recommended) + ci=True:")
    print(f"  p={r.pvalue:.4g}, statistic={r.statistic:.4f}")
    print(
        f"  CI level={r.ci.level}, n_boot={r.ci.n_boot}, "
        f"m={r.ci.m} (m_rate={r.ci.m_rate})"
    )

    print("\nholdout-correlation CI (composite generalization test, r-scale):")
    print(fmt_ci("holdout_corr", r.ci.holdout_corr))

    # Per-variable leverage CIs: normal-theory bootstrap interval around the
    # full-data leverage, clamped to [0, 1]. Noise lower bounds sit at the
    # boundary, 0.
    print("\nper-variable leverage CI (variable importance, bootstrap):")
    print(f"  {'':<7} {'CI lower':>9}  {'CI upper':>9}  {'SE':>7}")
    for j in range(len(model.beta)):
        flag = "  signal" if j < 2 else ""
        print(
            f"  β[{j}]  {r.ci.leverage_ci_lower[j]:+9.4f}  "
            f"{r.ci.leverage_ci_upper[j]:+9.4f}  "
            f"{r.ci.leverage_se[j]:7.4f}{flag}"
        )

    print(
        "\nper-variable subsampling z for β (signed; at k=1, |z| > 1.96"
        " ≈ 5% two-sided per variable, no multiplicity correction):"
    )
    for j, z in enumerate(r.ci.beta_sign_z_signed):
        flag = "  signal" if j < 2 else ""
        print(f"  β[{j}] z={z:+7.2f}{flag}")

    # Per-coordinate β CIs: a regression-style diagnostic (PLS1 only).
    # The calibrated readouts are `holdout_corr` and, at k=1, `beta_sign_z`;
    # the CI below omits the SE corrections that `beta_sign_z` applies.
    # Centered-scaled CIs may be biased by PLS shrinkage on small m; see
    # `_docs/python/results.md` caveats; the midpoint is *not* guaranteed to bracket
    # the full-data β_ref.
    print("\nper-coordinate β CI (level=0.95, on the same scale as β_ref):")
    print(f"  {'':<7} {'β_ref':>9}  {'CI lower':>9}  {'CI upper':>9}  {'SE':>7}")
    for j in range(len(model.beta)):
        flag = "  signal" if j < 2 else ""
        print(
            f"  β[{j}]  {model.beta[j]:+9.4f}  "
            f"{r.ci.beta_ci_lower[j]:+9.4f}  {r.ci.beta_ci_upper[j]:+9.4f}  "
            f"{r.ci.beta_se[j]:7.4f}{flag}"
        )

    print(
        f"\ndiagnostics: n_boot={r.ci.n_boot}, "
        f"n_boot_finite={r.ci.n_boot_finite}, "
        f"n_boot_finite_holdout_corr={r.ci.n_boot_finite_holdout_corr}"
    )


if __name__ == "__main__":
    main()
