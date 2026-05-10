"""Integration tests for pls1_confirmatory_test(ci=True)."""
import os

import numpy as np
import pytest

from plskit import (
    CIScalar,
    ConfirmatoryCI,
    ConfirmatoryTestResult,
    PlsKitError,
    pls1_confirmatory_test,
    pls1_fit,
)


def _synth(n=100, d=6, snr=4.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    beta = np.array([1.0] * 2 + [0.0] * (d - 2))
    y = x @ beta * snr + rng.standard_normal(n)
    return x, y


def test_ci_false_returns_no_ci_field():
    x, y = _synth()
    r = pls1_confirmatory_test(x, y, k=2, method="split_nb", seed=7)
    assert r.ci is None


def test_ci_true_populates_ci():
    x, y = _synth()
    r = pls1_confirmatory_test(
        x, y, k=2, method="split_nb",
        ci=True, n_boot=200, m_rate=0.7, level=0.95, seed=7,
    )
    assert isinstance(r.ci, ConfirmatoryCI)
    assert r.ci.n_boot == 200
    assert r.ci.m == 26  # ceil(100^0.7)
    assert r.ci.m_rate == pytest.approx(0.7)
    assert r.ci.level == pytest.approx(0.95)
    assert r.ci.beta_sign_z.shape == (6,)
    assert r.ci.leverage_ci_lower.shape == (6,)
    assert r.ci.leverage_ci_upper.shape == (6,)
    assert r.ci.leverage_se.shape == (6,)
    assert isinstance(r.ci.holdout_corr, CIScalar)


def test_ci_signal_variables_have_higher_sign_z_than_noise():
    x, y = _synth(n=200, d=8, snr=6.0, seed=11)
    r = pls1_confirmatory_test(
        x, y, k=2, method="split_nb",
        ci=True, n_boot=400, m_rate=0.7, seed=7,
    )
    sig = r.ci.beta_sign_z[:2]
    noi = r.ci.beta_sign_z[2:]
    # Signal-bearing variables should have larger |z| than noise on average.
    assert np.mean(np.abs(sig)) > np.mean(np.abs(noi))


def test_ci_seed_is_reproducible():
    x, y = _synth(n=120, d=6, snr=4.0, seed=42)
    r1 = pls1_confirmatory_test(x, y, k=2, method="split_nb",
                                ci=True, n_boot=200, seed=99)
    r2 = pls1_confirmatory_test(x, y, k=2, method="split_nb",
                                ci=True, n_boot=200, seed=99)
    np.testing.assert_array_equal(r1.ci.beta_sign_z, r2.ci.beta_sign_z)
    np.testing.assert_array_equal(r1.ci.leverage_ci_lower, r2.ci.leverage_ci_lower)
    assert r1.ci.holdout_corr == r2.ci.holdout_corr


@pytest.mark.parametrize("bad_m_rate", [0.4, 0.5, 0.95, 1.0])
def test_ci_rejects_out_of_range_m_rate(bad_m_rate):
    x, y = _synth()
    with pytest.raises(PlsKitError) as excinfo:
        pls1_confirmatory_test(x, y, k=2, method="split_nb",
                               ci=True, n_boot=200, m_rate=bad_m_rate, seed=7)
    assert excinfo.value.code == "invalid_argument"


def test_ci_rejects_low_n_boot():
    x, y = _synth()
    with pytest.raises(PlsKitError) as excinfo:
        pls1_confirmatory_test(x, y, k=2, method="split_nb",
                               ci=True, n_boot=50, seed=7)
    assert excinfo.value.code == "invalid_argument"


@pytest.mark.parametrize("bad_level", [0.49, 0.991, 1.0])
def test_ci_rejects_out_of_range_level(bad_level):
    x, y = _synth()
    with pytest.raises(PlsKitError) as excinfo:
        pls1_confirmatory_test(x, y, k=2, method="split_nb",
                               ci=True, n_boot=200, level=bad_level, seed=7)
    assert excinfo.value.code == "invalid_argument"


def test_ci_rejects_m_less_than_k_plus_2():
    # n=20, k=4, m_rate=0.51 → m = ceil(20^0.51) = 5 < k+2 = 6
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 6))
    y = rng.standard_normal(20)
    with pytest.raises(PlsKitError) as excinfo:
        pls1_confirmatory_test(x, y, k=4, method="split_nb",
                               ci=True, n_boot=200, m_rate=0.51, seed=7)
    assert excinfo.value.code == "invalid_argument"


def test_ci_exposes_signed_beta_sign_z():
    x, y = _synth(n=200, d=8, snr=6.0, seed=11)
    r = pls1_confirmatory_test(
        x, y, k=2, method="split_nb",
        ci=True, n_boot=400, m_rate=0.7, seed=7,
    )
    assert r.ci.beta_sign_z_signed.shape == (8,)
    # Magnitudes match folded form.
    np.testing.assert_allclose(
        np.abs(r.ci.beta_sign_z_signed),
        np.abs(r.ci.beta_sign_z),
        atol=1e-12,
    )


def test_ci_point_estimates_match_full_data_invariants():
    """Cross-language tripwire for the .point invariants on holdout_corr."""
    x, y = _synth(n=120, d=6, snr=4.0, seed=42)
    r = pls1_confirmatory_test(
        x, y, k=2, method="split_nb",
        ci=True, n_boot=200, m_rate=0.7, level=0.95, seed=7,
    )
    assert np.isfinite(r.ci.holdout_corr.point)
    assert -1.0 <= r.ci.holdout_corr.point <= 1.0


def test_ci_beta_arrays_have_expected_shape():
    x, y = _synth(n=200, d=8, snr=4.0, seed=42)
    r = pls1_confirmatory_test(
        x, y, k=2, method="split_nb",
        ci=True, n_boot=200, m_rate=0.7, level=0.95, seed=7,
    )
    assert r.ci.beta_ci_lower.shape == (8,)
    assert r.ci.beta_ci_upper.shape == (8,)
    assert r.ci.beta_se.shape == (8,)


def test_ci_beta_lower_le_upper():
    x, y = _synth(n=120, d=6, snr=4.0, seed=11)
    r = pls1_confirmatory_test(
        x, y, k=2, method="split_nb",
        ci=True, n_boot=200, m_rate=0.7, seed=7,
    )
    assert np.all(r.ci.beta_ci_lower <= r.ci.beta_ci_upper)


def test_ci_beta_se_nonnegative():
    x, y = _synth(n=120, d=6, snr=4.0, seed=11)
    r = pls1_confirmatory_test(
        x, y, k=2, method="split_nb",
        ci=True, n_boot=200, m_rate=0.7, seed=7,
    )
    assert np.all(r.ci.beta_se >= 0.0)


def _signal_vs_noise_fixture():
    """The canonical signal-vs-noise dataset from the spec test grid."""
    rng = np.random.default_rng(42)
    n, d = 200, 8
    x = rng.standard_normal((n, d))
    beta_true = np.zeros(d)
    beta_true[0] = 3.0
    beta_true[1] = 3.0
    eps = rng.standard_normal(n)
    y = x @ beta_true + eps
    return x, y


def test_ci_beta_signal_far_from_zero_noise_close_to_zero():
    """β CIs separate signal (β_true=3) from noise (β_true=0).

    Per caveat #1 (shrinkage bias on small m), the CI midpoint may sit
    well off β_ref, so we assert separation rather than coverage of β_ref:
    signal coords have CIs that strictly exclude zero on the positive side,
    and noise coords have CIs that lie within a small band around zero.
    """
    x, y = _signal_vs_noise_fixture()
    r = pls1_confirmatory_test(
        x, y, k=2, method="split_nb",
        ci=True, n_boot=500, m_rate=0.7, level=0.95, seed=42,
    )
    lo = r.ci.beta_ci_lower
    hi = r.ci.beta_ci_upper

    assert np.all(lo[:2] > 0.0), f"signal CIs not strictly positive: lo={lo[:2]}"
    assert np.all(hi[2:] < 1.0) and np.all(lo[2:] > -1.0), (
        f"noise CIs left a small band around 0: lo={lo[2:]} hi={hi[2:]}"
    )
    # Signal CIs sit much further from zero than noise CIs (≥ 10× by midpoint).
    signal_mid = np.abs(0.5 * (lo[:2] + hi[:2]))
    noise_mid_max = float(np.max(np.abs(0.5 * (lo[2:] + hi[2:]))))
    assert np.all(signal_mid >= 10.0 * noise_mid_max), (
        f"signal/noise midpoint ratio too small: signal={signal_mid} noise_max={noise_mid_max}"
    )


def test_ci_beta_reproducible_with_seed():
    x, y = _signal_vs_noise_fixture()
    r1 = pls1_confirmatory_test(
        x, y, k=2, method="split_nb",
        ci=True, n_boot=300, seed=42,
    )
    r2 = pls1_confirmatory_test(
        x, y, k=2, method="split_nb",
        ci=True, n_boot=300, seed=42,
    )
    np.testing.assert_array_equal(r1.ci.beta_ci_lower, r2.ci.beta_ci_lower)
    np.testing.assert_array_equal(r1.ci.beta_ci_upper, r2.ci.beta_ci_upper)
    np.testing.assert_array_equal(r1.ci.beta_se, r2.ci.beta_se)


def test_ci_beta_brackets_full_data_beta_ref():
    """β_ref must lie inside (or at the boundary of) its β CI on the canonical
    fixture. With β_b properly back-projected to the same scale as β_ref, the
    centered-scaled CI is centered on β_ref plus a bias term that vanishes
    when β_b is unbiased for β_ref — true here since both estimators target
    the same population β at K=2 with 2 signal coords.
    """
    x, y = _signal_vs_noise_fixture()
    r = pls1_confirmatory_test(
        x, y, k=2, method="split_nb",
        ci=True, n_boot=500, m_rate=0.7, level=0.95, seed=42,
    )
    fit = pls1_fit(x, y, k=2)
    beta_ref = fit.beta
    inside = (r.ci.beta_ci_lower <= beta_ref) & (beta_ref <= r.ci.beta_ci_upper)
    # Allow ≤ 1 miss across all D=8 coords as shrinkage-bias headroom.
    assert int(inside.sum()) >= len(beta_ref) - 1, (
        f"β_ref outside CI on too many coords: inside={inside}, "
        f"lo={r.ci.beta_ci_lower}, ref={beta_ref}, hi={r.ci.beta_ci_upper}"
    )


def test_ci_diagnostic_fields_present_and_consistent():
    """n_boot_finite_holdout_corr ≤ n_boot_finite ≤ n_boot is a contract invariant."""
    x, y = _synth(n=120, d=6, snr=4.0, seed=42)
    r = pls1_confirmatory_test(
        x, y, k=2, method="split_nb",
        ci=True, n_boot=200, m_rate=0.7, seed=7,
    )
    assert isinstance(r.ci.n_boot_finite, int)
    assert isinstance(r.ci.n_boot_finite_holdout_corr, int)
    assert r.ci.n_boot_finite <= r.ci.n_boot
    assert r.ci.n_boot_finite_holdout_corr <= r.ci.n_boot_finite
    # Canonical synthetic input under default threshold should produce zero failures.
    assert r.ci.n_boot_finite == r.ci.n_boot
    assert r.ci.n_boot_finite_holdout_corr == r.ci.n_boot


def test_ci_strict_mode_passes_on_clean_synthetic_input():
    """max_failure_rate=0.0 must not error on the canonical n=100, d=6, k=2 input."""
    x, y = _synth(n=100, d=6, snr=4.0, seed=42)
    r = pls1_confirmatory_test(
        x, y, k=2, method="split_nb",
        ci=True, n_boot=200, m_rate=0.7, seed=7,
        max_failure_rate=0.0,
    )
    assert r.ci.n_boot_finite == r.ci.n_boot
    assert r.ci.n_boot_finite_holdout_corr == r.ci.n_boot


def test_ci_max_failure_rate_validated():
    """Range check: max_failure_rate ∈ [0.0, 1.0]."""
    x, y = _synth()
    for bad in (-0.01, 1.01, 2.0):
        with pytest.raises(PlsKitError) as excinfo:
            pls1_confirmatory_test(
                x, y, k=2, method="split_nb",
                ci=True, n_boot=200, seed=7,
                max_failure_rate=bad,
            )
        assert excinfo.value.code == "invalid_argument", (
            f"max_failure_rate={bad}: expected invalid_argument, got {excinfo.value.code}"
        )


@pytest.mark.skipif(
    not os.getenv("PLSKIT_SLOW_TESTS"),
    reason="slow Phase-2 sanity check; set PLSKIT_SLOW_TESTS=1 to run",
)
@pytest.mark.xfail(
    strict=False,
    reason=(
        "Half-normal premise does not hold for subsample-bootstrap β under "
        "H0: bootstrap β_b is centered at the full-data β_ref (which is "
        "nonzero by sampling noise), not at the population β=0, so p̂_pos "
        "is biased toward sign(β_ref). Documented as a Phase-2 follow-up; "
        "keep this test as a tripwire if the centering changes."
    ),
)
def test_beta_sign_z_is_half_normal_under_null_beta():
    """Sanity check: under H0 (β=0), |beta_sign_z| ~ half-normal.

    Under the null where y is independent of every column of X, the bootstrap
    fraction p̂_pos[j] of resamples with β_b[j] > 0 should center on 0.5
    (the spec premise). The naive sign-z (2·p̂_pos - 1)·√n_boot would then be
    N(0, 1), so the folded |z| should be half-normal with mean √(2/π) ≈ 0.7979
    and sd √(1 - 2/π) ≈ 0.6028.

    Empirically this fails: subsample-bootstrap β_b is centered at β_ref
    (the full-data point estimate, nonzero by sampling noise under H0), not
    at the population β=0, so p̂_pos clusters near I(β_ref > 0). Pooled mean
    runs ~5–9 vs. target 0.798. Marked xfail as a Phase-2 follow-up (per
    spec: "if it fails, treat as a follow-up rather than a blocker for the
    trim"). The test stays in place as a tripwire if/when the centering or
    folding rule changes.
    """
    n_replications = 30
    n, d = 80, 6
    target_mean = np.sqrt(2.0 / np.pi)            # ≈ 0.7979
    target_sd = np.sqrt(1.0 - 2.0 / np.pi)        # ≈ 0.6028
    # Tolerances are loose because n_boot=300 is finite and p̂_pos lives on a
    # discrete grid of size n_boot+1, which biases moments of the naive sign-z
    # toward slight discretization noise. 0.15 is comfortable for ~180 samples.
    TOL_MEAN = 0.15
    TOL_SD = 0.15

    rng_master = np.random.default_rng(20240501)
    pooled_abs_z = []
    for rep in range(n_replications):
        # Independent dataset per replication; deterministic via master RNG.
        rep_seed = int(rng_master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(rep_seed)
        x = rng.standard_normal((n, d))
        y = rng.standard_normal(n)  # β = 0: y independent of x
        r = pls1_confirmatory_test(
            x, y, k=2, method="split_nb",
            args={"n_splits": 30},
            ci=True, n_boot=300, m_rate=0.7, level=0.95,
            max_failure_rate=0.05,
            seed=rep_seed,
        )
        pooled_abs_z.append(np.abs(r.ci.beta_sign_z))

    pooled = np.concatenate(pooled_abs_z)
    # Filter out any non-finite entries defensively (shouldn't happen, but a
    # bootstrap with n_boot_finite < n_boot could in principle produce nans).
    pooled = pooled[np.isfinite(pooled)]
    assert pooled.size >= n_replications * d - 5, (
        f"too many non-finite |z| values: kept {pooled.size} of "
        f"{n_replications * d}"
    )

    emp_mean = float(np.mean(pooled))
    emp_sd = float(np.std(pooled, ddof=1))

    msg = (
        f"\nempirical mean = {emp_mean:.4f} (target {target_mean:.4f}, "
        f"|Δ|={abs(emp_mean - target_mean):.4f}, tol {TOL_MEAN})"
        f"\nempirical sd   = {emp_sd:.4f} (target {target_sd:.4f}, "
        f"|Δ|={abs(emp_sd - target_sd):.4f}, tol {TOL_SD})"
        f"\npooled n = {pooled.size}"
    )
    assert abs(emp_mean - target_mean) < TOL_MEAN, msg
    assert abs(emp_sd - target_sd) < TOL_SD, msg
