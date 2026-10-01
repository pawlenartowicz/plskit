"""Observation weights through the Python surface.

One test per wrapper-seam contract: weight errors map to typed exceptions,
weights reach the engines that have no weighted corpus fixture, uniform
weights are invisible to inference, the resampling skip-rate guard, and the
full-rank weighted fit is weighted least squares. Weighted numerics are owned
by the Rust tests and the weighted corpus fixtures.
"""
import numpy as np
import pytest

import plskit


# ── Weight errors map to typed exceptions ────────────────────────────────

_WEIGHT_ERRORS = {
    # case: (weights for n = 40, exception type, code, reason or None)
    "negative": (np.r_[-0.1, np.ones(39)], plskit.PlsKitInvalidWeights,
                 "invalid_weights", "negative"),
    "all_zero": (np.zeros(40), plskit.PlsKitInvalidWeights,
                 "invalid_weights", "all_zero"),
    # FitOpts::check_n_eff, which the seam sets through FitOpts::default().
    "insufficient_effective_n": (np.r_[1.0, np.full(39, 1e-6)],
                                 plskit.PlsKitInvalidWeights,
                                 "invalid_weights", "insufficient_effective_n"),
    "length_mismatch": (np.ones(39), plskit.PlsKitInvalidWeights,
                        "invalid_weights", "length_mismatch"),
    # A non-finite weight is a non-finite input, not a weights error.
    "nan": (np.r_[np.nan, np.ones(39)], plskit.PlsKitError,
            "non_finite_input", None),
}


@pytest.mark.parametrize("case", sorted(_WEIGHT_ERRORS))
def test_weight_errors_map_to_typed_exceptions(case):
    w, exc, code, reason = _WEIGHT_ERRORS[case]
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(40, 5)), rng.normal(size=40)
    with pytest.raises(exc) as ei:
        plskit.pls1_fit(X, y, k=2, weights=w)
    assert ei.value.code == code
    if reason is not None:
        assert ei.value.reason == reason


# ── Weights reach the engine ─────────────────────────────────────────────


def _find_k_data():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 6))
    return X, X[:, 0] + 0.3 * X[:, 1] + 0.5 * rng.normal(size=80)


def _find_k_optimal_bic():
    X, y = _find_k_data()
    w = np.ones(80)
    w[:20] = 10.0
    return plskit.pls1_find_k_optimal(
        X, y, k_max=4, selector="bic", weights=w, seed=0), 80


def _find_k_sequence():
    X, y = _find_k_data()
    w = np.random.default_rng(4).uniform(0.5, 2.0, size=80)
    return plskit.pls1_find_k_sequence(X, y, k_max=4, weights=w, seed=0), 80


def _rotation_stability():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(60, 5))
    y = X[:, 0] + 0.2 * rng.normal(size=60)
    w = rng.uniform(0.5, 2.0, size=60)
    return plskit.pls1_rotation_stability(
        X, y, k=2, weights=w, n_boot=200, seed=0), 60


_WEIGHTED_CALLS = {
    "find_k_optimal_bic": _find_k_optimal_bic,
    "find_k_sequence": _find_k_sequence,
    "rotation_stability": _rotation_stability,
}


@pytest.mark.parametrize("entry", sorted(_WEIGHTED_CALLS))
def test_weights_reach_the_engine(entry):
    # Dropped weights would report n_eff == n.
    r, n = _WEIGHTED_CALLS[entry]()
    assert r.n_eff < n


# ── Uniform weights are invisible to inference ───────────────────────────


def _perm_null(w):
    rng = np.random.default_rng(8)
    X = rng.normal(size=(60, 4))
    y = X[:, 0] + rng.normal(size=60)
    return plskit.pls1_perm_null(X, y, k=2, n_perm=200, weights=w, seed=0)


def _confirmatory_test(w):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5))
    y = X[:, 0] + 0.5 * rng.normal(size=60)
    return plskit.pls1_confirmatory_test(
        X, y, k=2, method="raw_perm", args={"n_perm": 200}, weights=w, seed=42)


def _find_k_optimal(w):
    X, y = _find_k_data()
    return plskit.pls1_find_k_optimal(X, y, k_max=4, weights=w, seed=0)


_UNIFORM = {
    # entry: (call, n, fields that must be bit-identical)
    "perm_null": (_perm_null, 60,
                  ("beta_ref", "beta_perm_mean", "beta_perm_sd", "beta_perm_z")),
    "confirmatory_test": (_confirmatory_test, 60, ("pvalue", "statistic")),
    "find_k_optimal": (_find_k_optimal, 80,
                       ("k_star", "cv_scores", "cv_scores_se")),
}


@pytest.mark.parametrize("entry", sorted(_UNIFORM))
def test_uniform_weights_are_invisible(entry):
    call, n, fields = _UNIFORM[entry]
    r_w, r_n = call(np.ones(n)), call(None)
    for f in fields:
        a, b = getattr(r_w, f), getattr(r_n, f)
        assert (a == b) if isinstance(a, dict) else np.array_equal(a, b), f
    assert r_w.n_eff == r_n.n_eff == n


# ── Resampling skip-rate guard ───────────────────────────────────────────


def _pathological():
    # Five rows hold almost all the weight, so most subsamples fail the
    # n_eff_sub >= k + 1 check.
    rng = np.random.default_rng(13)
    X = rng.normal(size=(60, 5))
    y = X[:, 0] + 0.2 * rng.normal(size=60)
    w = np.full(60, 1e-8)
    w[:5] = 1.0
    return X, y, w


def test_rotation_stability_skip_rate_guard_fires():
    X, y, w = _pathological()
    with pytest.raises(plskit.PlsKitResamplingDegenerate) as ei:
        plskit.pls1_rotation_stability(X, y, k=3, weights=w, n_boot=500, seed=0)
    err = ei.value
    assert err.threshold == 0.01
    assert err.skipped > 0
    assert err.total == 500
    assert err.skip_rate > 0.01


def test_rotation_stability_max_skip_rate_one_returns_truncated():
    X, y, w = _pathological()
    r = plskit.pls1_rotation_stability(
        X, y, k=3, weights=w, n_boot=500, seed=0, max_skip_rate=1.0)
    assert r.n_boot_finite < r.n_boot


def test_confirmatory_ci_skip_rate_guard_fires():
    X, y, w = _pathological()
    with pytest.raises(plskit.PlsKitResamplingDegenerate):
        plskit.pls1_confirmatory_test(
            X, y, k=3, method="raw_perm", args={"n_perm": 200},
            ci=True, weights=w, n_boot=500, seed=0,
        )


# ── Full-rank weighted PLS1 is weighted least squares ────────────────────


def test_full_rank_weighted_pls1_is_wls():
    # At k = p the PLS1 components span the column space of X, so beta and
    # the intercept are the WLS solution (closed form via lstsq on sqrt(w)-
    # scaled rows). k = p - 1 misses by ~6e-4 and the unweighted fit by ~2e-2.
    rng = np.random.default_rng(11)
    n, p = 30, 4
    X = rng.normal(size=(n, p))
    y = X @ rng.normal(size=p) + 0.2 * rng.normal(size=n)
    w = rng.uniform(0.5, 2.0, size=n)
    m = plskit.pls1_fit(X, y, k=p, weights=w)
    sw = np.sqrt(w)
    A = np.column_stack([np.ones(n), X]) * sw[:, None]
    params = np.linalg.lstsq(A, y * sw, rcond=None)[0]
    np.testing.assert_allclose(m.beta, params[1:], atol=1e-10)
    np.testing.assert_allclose(m.intercept, params[0], atol=1e-10)
