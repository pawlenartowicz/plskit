"""End-to-end example: PLS1 fit + confirmatory test with CI.

Run from the public monorepo root (`plskit/`) after building the Python
wrapper:

    maturin develop --release
    python examples/fit_and_ci.py

Demonstrates the recommended pipeline (`_docs/python/api.md` §3.1):

  1. Generate synthetic data with a known PLS1 signal in the first two
     of D=8 columns (everything else is noise).
  2. Fit PLS1 at K=1 (the recommended omnibus K; y depends on X through
     a single direction here, so there is no reason to fix K higher).
  3. Run `pls1_confirmatory_test(k=1, test_method="split_exact", ci=True)` to
     get both the omnibus p-value AND the rotation-invariant CI bundle.
     `split_exact` is the recommended method: a split-half test
     calibrated by permutation, so it holds its level on any design.
  4. Print the headline test result, the holdout-correlation CI,
     the per-variable subsampling z for β, the per-coordinate β
     CIs (regression-style diagnostic; see the caveats under "Per-coordinate
     β CIs" in `_docs/python/results.md`),
     and the engine's diagnostic counters.

The CI bundle is computed in an independent subsampling pass that uses
its own child-seed branch; you get one user-facing seed but two
non-interfering RNG streams.
"""

from __future__ import annotations

import numpy as np

import plskit


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


def main() -> None:
    X, y = synth(n=200, d=8, snr=4.0, seed=42)

    # 1. Fit
    model = plskit.pls1_fit(X, y, k=1)
    print(f"PLS1 fit: K={model.k_used}, ‖β‖={np.linalg.norm(model.beta):.4f}")

    # 2. Confirmatory test + CI
    r = plskit.pls1_confirmatory_test(
        X, y, k=1,
        test_method="split_exact",
        args={"n_perm": 500, "n_splits": 50},
        ci=True,
        n_boot=500,
        m_rate=0.7,
        level=0.95,
        max_failure_rate=0.0,   # strict: error if any resample fails
        seed=2026,
    )

    print(f"\nsplit_exact omnibus test: p={r.pvalue:.4g}, statistic={r.statistic:.4f}")

    print("\nholdout-correlation CI (level=0.95):")
    print(fmt_ci("holdout_corr", r.ci.holdout_corr))

    # Subsampling z for β: β_ref[j] over a subsampling SE corrected for
    # sampling without replacement and for the shrinkage of subsample fits.
    # At k=1 it is roughly N(0, 1) for a variable whose population PLS
    # coefficient is 0, so noise coords mostly stay below 1.96 (about 1 in
    # 20 exceeds it by chance; no multiplicity correction) and it does not
    # grow with n_boot.
    print("\nper-variable subsampling z for β (signed; |z| > 1.96 ≈ 5% two-sided):")
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
