import numpy as np
import pytest
import plskit


def _data(n=60, d=5, snr=3.0, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = X[:, 0] * snr + rng.normal(size=n)
    return X, y


_SMALL_ARGS_BY_METHOD = {
    "raw_perm": {"n_perm": 100},
    "split_nb": {"n_splits": 20},
    "split_perm": {"n_perm": 100, "n_splits": 20},
    "score": {},
    "e": {},
}


@pytest.mark.parametrize("method", ["raw_perm", "split_nb", "split_perm", "score", "e"])
def test_confirmatory_methods_run(method):
    X, y = _data()
    r = plskit.pls1_confirmatory_test(
        X, y, k=1, method=method,
        args=_SMALL_ARGS_BY_METHOD[method], seed=7,
    )
    assert isinstance(r, plskit.ConfirmatoryTestResult)
    assert r.method == method
    assert r.k == 1
    assert 0.0 <= r.pvalue <= 1.0


def test_confirmatory_at_param_no_longer_accepted():
    X, y = _data()
    with pytest.raises(TypeError):
        plskit.pls1_confirmatory_test(X, y, k=1, method="split_nb", at="fitted_k")


def test_confirmatory_score_n_perm_field_is_none():
    X, y = _data()
    r = plskit.pls1_confirmatory_test(X, y, k=1, method="score", seed=7)
    assert r.n_perm is None
    assert r.n_splits is None
