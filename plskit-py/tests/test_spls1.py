"""spls1 sparse-PLS1 family: bit-parity, sparsity, tuning modes, errors."""

import numpy as np
import pytest

import plskit
from plskit import PlsKitError


def synth(n, d, k_signal, snr, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = X[:, :k_signal].sum(axis=1) * snr + rng.standard_normal(n)
    return X, y


def test_spls1_fit_dense_endpoint_bit_parity():
    X, y = synth(50, 8, 2, 4.0, 1)
    dense = plskit.pls1_fit(X, y, k=3)
    sparse = plskit.spls1_fit(X, y, 3, 8)
    # keep = n_features must be bit-identical to the dense fit.
    assert np.array_equal(dense.coef, sparse.coef)
    assert np.array_equal(dense.beta, sparse.beta)
    assert dense.intercept == sparse.intercept
    assert dense.k_used == sparse.k_used
    assert sparse.keep == 8
    assert dense.keep is None


def test_spls1_fit_exact_nonzero_count():
    X, y = synth(50, 10, 3, 4.0, 2)
    for keep in (1, 3, 7):
        r = plskit.spls1_fit(X, y, 3, keep)
        for a in range(r.k_used):
            assert np.count_nonzero(r.W[:, a]) == keep


def test_spls1_fit_predict_roundtrip():
    X, y = synth(60, 10, 2, 4.0, 3)
    r = plskit.spls1_fit(X, y, 2, 4)
    yhat = plskit.pls1_predict(r, X)
    assert yhat.shape == (60,)
    assert np.isfinite(yhat).all()


def test_spls1_fit_rejects_bad_keep():
    X, y = synth(30, 5, 2, 4.0, 4)
    with pytest.raises(PlsKitError) as ei:
        plskit.spls1_fit(X, y, 2, 0)
    assert ei.value.code == "invalid_argument"
    with pytest.raises(PlsKitError) as ei:
        plskit.spls1_fit(X, y, 2, 6)
    assert ei.value.code == "invalid_argument"


def test_spls1_find_keep_optimal_grid_and_selection():
    X, y = synth(80, 6, 2, 5.0, 7)
    r = plskit.spls1_find_keep_optimal(X, y, 1, seed=7)
    assert r.keep_grid == [1, 2, 4, 6]
    assert r.k == 1
    assert r.keep_star in r.keep_grid
    # 1-SE parsimony invariant against the returned maps.
    best = max((v, k) for k, v in r.cv_scores.items() if np.isfinite(v))[1]
    threshold = r.cv_scores[best] - r.cv_scores_se[best]
    expected = min(k for k, v in r.cv_scores.items()
                   if np.isfinite(v) and v >= threshold)
    assert r.keep_star == expected


def test_spls1_find_keep_optimal_rejects_unknown_args():
    X, y = synth(40, 5, 1, 4.0, 8)
    with pytest.raises(PlsKitError) as ei:
        plskit.spls1_find_keep_optimal(X, y, 1, args={"bogus": 1})
    assert ei.value.code == "invalid_args"


def test_spls1_find_k_optimal_dense_endpoint_matches():
    X, y = synth(80, 5, 1, 5.0, 9)
    dense = plskit.pls1_find_k_optimal(X, y, 4, seed=7)
    sparse = plskit.spls1_find_k_optimal(X, y, 4, 5, seed=7)
    assert dense.k_star == sparse.k_star
    assert dense.cv_scores == sparse.cv_scores


def test_spls1_find_k_optimal_runs_sparse():
    X, y = synth(80, 6, 2, 5.0, 10)
    r = plskit.spls1_find_k_optimal(X, y, 4, 2, seed=7)
    assert 1 <= r.k_star <= 4
    assert r.selector == "r2_se"


def test_spls1_find_k_sequence_dense_endpoint_matches():
    X, y = synth(80, 5, 1, 5.0, 11)
    dense = plskit.pls1_find_k_sequence(X, y, 4, seed=7, args={"n_splits": 30})
    sparse = plskit.spls1_find_k_sequence(X, y, 4, 5, seed=7, args={"n_splits": 30})
    assert dense.k_star == sparse.k_star
    np.testing.assert_array_equal(dense.pvalues, sparse.pvalues)


def test_spls1_find_k_sequence_runs_sparse():
    X, y = synth(80, 6, 2, 5.0, 12)
    r = plskit.spls1_find_k_sequence(X, y, 4, 2, seed=7, args={"n_splits": 30})
    assert r.pvalues.shape == (4,)
    assert r.test_method == "split_nb"


def test_spls1_fit_weights_smoke():
    X, y = synth(50, 8, 2, 4.0, 13)
    w = np.linspace(0.5, 2.0, 50)
    r = plskit.spls1_fit(X, y, 2, 3, weights=w)
    assert np.all(np.isfinite(r.beta))
    assert r.keep == 3


def test_spls1_fit_negative_weights_raise():
    X, y = synth(50, 8, 2, 4.0, 13)
    w = np.ones(50)
    w[0] = -1.0
    with pytest.raises(plskit.PlsKitInvalidWeights):
        plskit.spls1_fit(X, y, 2, 3, weights=w)
