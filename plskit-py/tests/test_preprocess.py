import numpy as np
import pytest
import plskit


def _data(n=20, p=4):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    w = rng.uniform(0.5, 2.0, size=n)
    return X, y, w


def test_preprocess_all_none_returns_empty():
    r = plskit.preprocess()
    assert r.X_std is None
    assert r.Y_std is None
    assert r.weights_normalized is None
    assert r.n_eff is None


def test_preprocess_full_inputs():
    X, y, w = _data()
    r = plskit.preprocess(X=X, Y=y, weights=w)
    assert r.X_std.shape == X.shape
    assert r.Y_std.shape == y.shape
    assert r.weights_normalized.shape == (X.shape[0],)
    np.testing.assert_allclose(r.weights_normalized.mean(), 1.0, atol=1e-12)
    assert 0 < r.n_eff <= X.shape[0]


def test_preprocess_2d_y_round_trips_2d():
    X, y, w = _data()
    Y = y.reshape(-1, 1)
    r = plskit.preprocess(X=X, Y=Y, weights=w)
    assert r.Y_std.shape == Y.shape


def test_preprocess_negative_weight_raises():
    X, y, w = _data()
    w[0] = -1.0
    with pytest.raises(plskit.PlsKitError):
        plskit.preprocess(X=X, Y=y, weights=w)


def test_preprocess_shape_mismatch_raises():
    X, y, _ = _data()
    with pytest.raises(plskit.PlsKitError):
        plskit.preprocess(X=X, Y=y[:-1])


def test_preprocess_cache_pattern_round_trip():
    """Spec §5.5: cache-pattern parity. Helper output + pre_standardized=True
    must reproduce the from-raw fit. The standardized-space coef is identical
    in both paths; beta and intercept differ because the cached fit operates in
    standardized space (beta == coef, intercept == 0) while the raw fit
    back-projects to the original scale."""
    X, y, w = _data()
    pre = plskit.preprocess(X=X, Y=y, weights=w)
    m_cached = plskit.pls1_fit(
        pre.X_std, pre.Y_std, k=2,
        weights=pre.weights_normalized,
        pre_standardized=True,
    )
    m_raw = plskit.pls1_fit(X, y, k=2, weights=w)
    # coef is the same in both: both operate in standardized space for NIPALS.
    np.testing.assert_allclose(m_cached.coef, m_raw.coef, atol=1e-10)
    # Sanity: cached fit has no intercept (pre_standardized path).
    assert m_cached.intercept == 0.0
