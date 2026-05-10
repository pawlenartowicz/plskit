import numpy as np
import pytest

sm_api = pytest.importorskip("statsmodels.api")
import plskit


def test_full_rank_pls1_matches_statsmodels_wls():
    rng = np.random.default_rng(11)
    n, p = 30, 4
    X = rng.normal(size=(n, p))
    y = X @ rng.normal(size=p) + 0.2 * rng.normal(size=n)
    w = rng.uniform(0.5, 2.0, size=n)

    m = plskit.pls1_fit(X, y, k=p, weights=w)

    Xc = sm_api.add_constant(X)
    res = sm_api.WLS(y, Xc, weights=w).fit()
    expected_intercept, expected_beta = res.params[0], res.params[1:]

    np.testing.assert_allclose(m.beta, expected_beta, atol=1e-10)
    np.testing.assert_allclose(m.intercept, expected_intercept, atol=1e-10)
