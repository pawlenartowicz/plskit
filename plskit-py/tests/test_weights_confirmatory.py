import numpy as np
import pytest
import plskit


def test_confirmatory_weights_uniform_invariance():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5))
    y = X[:, 0] + 0.5 * rng.normal(size=60)
    r_w = plskit.pls1_confirmatory_test(
        X, y, k=2, method="raw_perm", args={"n_perm": 200}, weights=np.ones(60), seed=42,
    )
    r_n = plskit.pls1_confirmatory_test(
        X, y, k=2, method="raw_perm", args={"n_perm": 200}, seed=42,
    )
    assert r_w.pvalue == r_n.pvalue
    assert r_w.n_eff == pytest.approx(60.0)
    assert r_n.n_eff == pytest.approx(60.0)


@pytest.mark.parametrize("method", ["raw_perm", "split_nb", "split_exact", "score", "e"])
def test_confirmatory_each_method_accepts_weights(method):
    rng = np.random.default_rng(1)
    # d = 6, not 4: the split_nb auto-gate reroutes any X with four columns or
    # fewer, and this cell is meant to exercise split_nb's own weighted path.
    X = rng.normal(size=(80, 6))
    y = X[:, 0] + rng.normal(size=80)
    w = rng.uniform(0.5, 2.0, size=80)
    r = plskit.pls1_confirmatory_test(X, y, k=2, method=method, weights=w, seed=7)
    assert 0.0 <= r.pvalue <= 1.0
    assert 0 < r.n_eff <= 80
