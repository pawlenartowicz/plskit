import numpy as np
import pytest
import plskit


def _setup():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 6))
    y = X[:, 0] + 0.3 * X[:, 1] + 0.5 * rng.normal(size=80)
    return X, y


def test_find_k_optimal_uniform_weights_invariance():
    X, y = _setup()
    r_w = plskit.pls1_find_k_optimal(X, y, k_max=4, weights=np.ones(80), seed=0)
    r_n = plskit.pls1_find_k_optimal(X, y, k_max=4, seed=0)
    assert r_w.k_star == r_n.k_star
    assert r_w.n_eff == pytest.approx(80.0)


def test_find_k_optimal_with_weights_runs():
    X, y = _setup()
    w = np.random.default_rng(3).uniform(0.5, 2.0, size=80)
    r = plskit.pls1_find_k_optimal(X, y, k_max=4, weights=w, seed=0)
    assert 1 <= r.k_star <= 4
    assert 0 < r.n_eff <= 80


def test_find_k_sequence_with_weights_runs():
    X, y = _setup()
    w = np.random.default_rng(4).uniform(0.5, 2.0, size=80)
    r = plskit.pls1_find_k_sequence(X, y, k_max=4, weights=w, seed=0)
    assert 0 <= r.k_star <= 4  # k_star may be 0 if no rejection
    assert 0 < r.n_eff <= 80


def test_find_k_optimal_bic_uses_n_eff():
    X, y = _setup()
    w = np.ones(80); w[:20] = 10.0
    r = plskit.pls1_find_k_optimal(X, y, k_max=4, selector="bic", weights=w, seed=0)
    assert 1 <= r.k_star <= 4
    assert r.n_eff < 80  # n_eff smaller than n due to weight concentration
