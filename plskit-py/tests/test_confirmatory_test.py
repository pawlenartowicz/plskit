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
    "split_perm_nr": {"n_perm": 100, "n_splits": 20},
    "split_perm": {"n_perm": 100, "n_splits": 20},
    "score": {},
    "e": {},
}


@pytest.mark.parametrize("method", ["raw_perm", "split_nb", "split_perm_nr", "split_perm", "score", "e"])
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


def test_split_perm_nr_guard_rejects_k_not_one():
    X, y = _data()
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.pls1_confirmatory_test(
            X, y, k=2, method="split_perm_nr", args={"n_perm": 100, "n_splits": 20}, seed=7,
        )
    assert "split_perm" in str(ei.value)


def test_split_perm_nr_guard_rejects_weighted_input():
    X, y = _data()
    w = np.where(np.arange(X.shape[0]) % 2 == 0, 1.5, 0.5)
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.pls1_confirmatory_test(
            X, y, k=1, method="split_perm_nr", args={"n_perm": 100, "n_splits": 20},
            weights=w, seed=7,
        )
    assert "split_perm" in str(ei.value)


def test_split_perm_nr_succeeds_unweighted_k_one():
    X, y = _data()
    r = plskit.pls1_confirmatory_test(
        X, y, k=1, method="split_perm_nr", args={"n_perm": 100, "n_splits": 20}, seed=7,
    )
    assert isinstance(r, plskit.ConfirmatoryTestResult)


def test_rho_hat_populated_for_split_nb_unweighted_only():
    X, y = _data()

    r_nb = plskit.pls1_confirmatory_test(
        X, y, k=1, method="split_nb", args={"n_splits": 20}, seed=7,
    )
    assert isinstance(r_nb.rho_hat, float)
    assert 0.0 <= r_nb.rho_hat <= 1.0

    w = np.where(np.arange(X.shape[0]) % 2 == 0, 1.5, 0.5)
    r_nb_weighted = plskit.pls1_confirmatory_test(
        X, y, k=1, method="split_nb", args={"n_splits": 20}, weights=w, seed=7,
    )
    assert r_nb_weighted.rho_hat is None

    r_perm_nr = plskit.pls1_confirmatory_test(
        X, y, k=1, method="split_perm_nr", args={"n_perm": 100, "n_splits": 20}, seed=7,
    )
    assert r_perm_nr.rho_hat is None

    r_perm = plskit.pls1_confirmatory_test(
        X, y, k=1, method="split_perm", args={"n_perm": 100, "n_splits": 20}, seed=7,
    )
    assert r_perm.rho_hat is None


def test_split_perm_nr_no_sequential_variant():
    X, y = _data()
    with pytest.raises(plskit.PlsKitError):
        plskit.pls1_find_k_sequence(
            X, y, k_max=3, test_method="split_perm_nr", seed=7,  # type: ignore[arg-type]
        )
    with pytest.raises(plskit.PlsKitError):
        plskit.pls1_find_k_optimal(
            X, y, k_max=3, diagnostic="split_perm_nr", seed=7,  # type: ignore[arg-type]
        )


def test_split_perm_nr_smoke():
    X, y = _data()
    n_perm = 100
    r = plskit.pls1_confirmatory_test(
        X, y, k=1, method="split_perm_nr", args={"n_perm": n_perm, "n_splits": 20}, seed=7,
    )
    assert 0.0 < r.pvalue <= 1.0
    assert isinstance(r.statistic, float)
    assert r.pvalue >= 1.0 / (n_perm + 1)
