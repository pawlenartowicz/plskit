import numpy as np
import pytest
import plskit


def _data():
    rng = np.random.default_rng(0)
    return rng.normal(size=(40, 5)), rng.normal(size=40)


def test_weights_none_baseline():
    X, y = _data()
    m = plskit.pls1_fit(X, y, k=2)
    assert m.weights is None
    assert m.n_eff == pytest.approx(40.0)


def test_weights_uniform_constant_invariance():
    X, y = _data()
    w = np.full(40, 7.5)
    m_w = plskit.pls1_fit(X, y, k=2, weights=w)
    m_n = plskit.pls1_fit(X, y, k=2)
    np.testing.assert_allclose(m_w.beta, m_n.beta, atol=1e-12)
    np.testing.assert_allclose(m_w.intercept, m_n.intercept, atol=1e-12)


def test_weights_negative_raises():
    X, y = _data()
    w = np.ones(40); w[0] = -0.1
    with pytest.raises(plskit.PlsKitInvalidWeights) as exc_info:
        plskit.pls1_fit(X, y, k=2, weights=w)
    assert exc_info.value.reason == "negative"


def test_weights_all_zero_raises():
    X, y = _data()
    w = np.zeros(40)
    with pytest.raises(plskit.PlsKitInvalidWeights) as exc_info:
        plskit.pls1_fit(X, y, k=2, weights=w)
    assert exc_info.value.reason == "all_zero"


def test_weights_insufficient_n_eff_raises():
    X, y = _data()
    w = np.full(40, 1e-6); w[0] = 1.0
    with pytest.raises(plskit.PlsKitInvalidWeights) as exc_info:
        plskit.pls1_fit(X, y, k=2, weights=w)
    assert exc_info.value.reason == "insufficient_effective_n"


def test_weights_length_mismatch_raises():
    X, y = _data()
    w = np.ones(39)
    with pytest.raises(plskit.PlsKitError):
        plskit.pls1_fit(X, y, k=2, weights=w)


def test_weights_nan_raises():
    X, y = _data()
    w = np.ones(40); w[0] = np.nan
    with pytest.raises(plskit.PlsKitError):
        plskit.pls1_fit(X, y, k=2, weights=w)
