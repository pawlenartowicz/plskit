"""End-to-end example: PLS1 fit + every omnibus method + bootstrap CI bundle.

Run from the public monorepo root (`plskit/`) after building the Python
wrapper:

    maturin develop --release
    python examples/omnibus_and_ci.py

This is the extended companion to `fit_and_ci.py`. Where that example
demonstrates a single canonical test + CI, this one walks the full
omnibus method axis on the same data so the legacy/canonical contrast
and the score/split_nb complementarity (SPECS.md §2.1) are visible at a
glance.

Pipeline:

  1. Generate synthetic data with a known PLS1 signal in the first two
     of D=8 columns (everything else is noise).
  2. Fit PLS1 at K=2.
  3. Run every canonical omnibus method on the same X, y, k, seed and
     tabulate p-value + statistic + per-method workload knobs:
       * split_nb   — canonical NB-Wald split test (concentrated signal)
       * split_perm — canonical split-permutation variant
       * score      — closed-form Welch-Satterthwaite (diffuse signal,
                      anisotropy-aware, K-free)
       * e          — universal inference (calibration-free, exact α)
     Reporting `score` and `split_nb` side-by-side is the canonical
     pattern (SPECS.md §2.1 — they complement rather than replace).
     The legacy `raw_perm` test is omitted here.
  4. Re-run the canonical test (`split_nb`) with `ci=True` to attach a
     bootstrap CI bundle, then print:
       * holdout-correlation CI (composite generalization test, r-scale,
         asymmetric Fisher-z bounds in (-1, 1)),
       * per-variable leverage CIs (variable-importance with proper
         shrinkage; signal coords should clear zero, noise should not),
       * per-variable sign-stability z (canonical hypothesis test,
         folded — large |z| ⇒ stable sign across resamples),
       * per-coordinate β CIs (regression-style diagnostic — see
         RESULTS_FORMAT.md caveats; the midpoint is *not* guaranteed to
         bracket the full-data β_ref under PLS shrinkage on small m),
       * engine diagnostic counters.

The CI bundle uses an independent subsampling pass on a child-seed
branch of the user-facing seed, so the omnibus and CI streams don't
interfere.
"""

from __future__ import annotations

import numpy as np

import plskit


# args required by each omnibus method (SPECS.md §2.1, "args by method").
# `score` and `e` take no method-specific kwargs.
_OMNIBUS_ARGS: dict[str, dict] = {
    "split_nb":   {"n_splits": 50},
    "split_perm": {"n_perm": 500, "n_splits": 50},
    "score":      {},
    "e":          {},
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
    model = plskit.pls1_fit(X, y, k=2)
    print(f"PLS1 fit: K={model.k_used}, ‖β‖={np.linalg.norm(model.beta):.4f}")

    # 2. Omnibus tests — full method axis, shared seed -----------------------
    print("\nomnibus tests (same X, y, k=2, seed=2026):")
    print(f"  {'method':<11}  {'p-value':>10}  {'statistic':>10}  {'workload'}")
    omni: dict[str, plskit.ConfirmatoryTestResult] = {}
    for method, args in _OMNIBUS_ARGS.items():
        omni[method] = plskit.pls1_confirmatory_test(
            X, y, k=2,
            method=method, args=args or None,
            seed=2026,
        )
    for method, r in omni.items():
        print(
            f"  {method:<11}  {r.pvalue:>10.4g}  {r.statistic:>10.4f}  "
            f"{fmt_workload(r)}"
        )
    print(
        "  (split_nb / split_perm / score / e are the canonical"
        " modern-inference family; legacy raw_perm is omitted here.)"
    )

    # 3. Canonical test + bootstrap CI bundle --------------------------------
    r = plskit.pls1_confirmatory_test(
        X, y, k=2,
        method="split_nb",
        args={"n_splits": 50},
        ci=True,
        n_boot=500,
        m_rate=0.7,
        level=0.95,
        max_failure_rate=0.0,   # strict — error if any resample fails
        seed=2026,
    )

    print("\nsplit_nb (canonical) + ci=True:")
    print(f"  p={r.pvalue:.4g}, statistic={r.statistic:.4f}")
    print(
        f"  CI level={r.ci.level}, n_boot={r.ci.n_boot}, "
        f"m={r.ci.m} (m_rate={r.ci.m_rate})"
    )

    print("\nholdout-correlation CI (composite generalization test, r-scale):")
    print(fmt_ci("holdout_corr", r.ci.holdout_corr))

    # Per-variable leverage CIs: centered-scaled importance score, the
    # canonical variable-importance complement to beta_sign_z.
    print("\nper-variable leverage CI (variable importance, centered-scaled):")
    print(f"  {'':<7} {'CI lower':>9}  {'CI upper':>9}  {'SE':>7}")
    for j in range(len(model.beta)):
        flag = "  signal" if j < 2 else ""
        print(
            f"  β[{j}]  {r.ci.leverage_ci_lower[j]:+9.4f}  "
            f"{r.ci.leverage_ci_upper[j]:+9.4f}  "
            f"{r.ci.leverage_se[j]:7.4f}{flag}"
        )

    print("\nper-variable sign-stability z (folded; canonical hypothesis test):")
    for j, z in enumerate(r.ci.beta_sign_z):
        flag = "  signal" if j < 2 else ""
        print(f"  β[{j}] z={z:+.2f}{flag}")

    # Per-coordinate β CIs: a regression-style diagnostic (PLS1 only).
    # Canonical inference is `beta_sign_z` + `leverage_ci_*` + `holdout_corr`.
    # Centered-scaled CIs may be biased by PLS shrinkage on small m — see
    # RESULTS_FORMAT.md caveats; the midpoint is *not* guaranteed to bracket
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
