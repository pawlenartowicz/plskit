"""gram= on the held-out tests and pls1_perm_alpha: the GramMatrixResult or
an array with n_features, in place of X."""
import numpy as np
import pytest

import plskit
from plskit import PlsKitError, gram_matrix


def _data(n=30, p=80, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)) * np.arange(1, p + 1) / 10 + 2.0
    y = X[:, 0] / 0.1 + 0.5 * X[:, 1] / 0.2 + rng.standard_normal(n)
    return X, y


EXACT = {"k": 1, "test_method": "split_exact", "args": {"n_perm": 19, "n_splits": 8}, "seed": 3}


def test_gram_call_equals_x_call():
    X, y = _data()
    g = gram_matrix(X, pre_standardized=True)
    a = plskit.pls1_confirmatory_test(X, y, pre_standardized=True, **EXACT)
    b = plskit.pls1_confirmatory_test(gram=g, y=y, **EXACT)
    c = plskit.pls1_confirmatory_test(
        gram=g.gram, n_features=g.n_features, pre_standardized=True, y=y, **EXACT
    )
    for r in (b, c):
        assert r.test_method == a.test_method
        np.testing.assert_allclose(r.statistic, a.statistic, rtol=1e-12, atol=0)
        np.testing.assert_allclose(r.pvalue, a.pvalue, rtol=1e-12, atol=0)


def test_pls3_and_perm_alpha_take_both_forms():
    X, y = _data()
    Y = np.column_stack([y, X[:, 2], X[:, 3] + y])
    g = gram_matrix(X)
    a = plskit.pls3_confirmatory_test(gram=g, Y=Y, **EXACT)
    b = plskit.pls3_confirmatory_test(gram=g.gram, n_features=80, Y=Y, **EXACT)
    assert (a.statistic, a.pvalue) == (b.statistic, b.pvalue)
    s = g.subset(list(range(25)))
    r = plskit.pls1_perm_alpha(s, Y[:25], k=3)
    assert r.alpha.shape == (25, 3) and (r.k_used <= 3).all()
    with pytest.raises(PlsKitError) as e:
        plskit.pls1_perm_alpha(s.gram, Y[:25], k=3, n_features=80)
    assert e.value.code == "invalid_input"
