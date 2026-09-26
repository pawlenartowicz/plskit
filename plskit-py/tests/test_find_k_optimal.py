import numpy as np
import pytest
import plskit


def _data(n=80, d=6, k_signal=2, snr=4.0, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    beta = np.zeros(d); beta[:k_signal] = 1.0
    y = X @ beta * snr + rng.normal(size=n)
    return X, y


def test_optimal_r2_se_returns_cv_scores_and_se():
    X, y = _data()
    r = plskit.pls1_find_k_optimal(X, y, k_max=4, selector="r2_se",
                                    args={"n_folds": 5}, seed=7)
    assert isinstance(r, plskit.FindKOptimalResult)
    assert r.selector == "r2_se"
    assert r.cv_scores is not None
    assert r.cv_scores_se is not None
    assert r.bic_scores is None
    assert r.pvalues is None
    assert r.diagnostic is None


def test_optimal_r2_max_returns_cv_scores_no_se():
    X, y = _data()
    r = plskit.pls1_find_k_optimal(X, y, k_max=4, selector="r2_max",
                                    args={"n_folds": 5}, seed=7)
    assert r.selector == "r2_max"
    assert r.cv_scores is not None
    assert r.cv_scores_se is None
    assert r.bic_scores is None


@pytest.mark.parametrize("selector", ["r2_se", "r2_max"])
def test_optimal_rejects_leave_one_out_at_n_eq_2(selector):
    """n = 2: the CV layer's own n_folds cap/floor (max(2, n - 2)) pushes
    the effective fold count back up to n, leave-one-out (every validation
    fold a single row). The same degeneracy raw_perm rejects."""
    X, y = _data(n=2, d=3, k_signal=1, seed=6)
    with pytest.raises(plskit.PlsKitError) as exc_info:
        plskit.pls1_find_k_optimal(X, y, k_max=1, selector=selector, seed=7)
    assert exc_info.value.code == "invalid_argument"


def test_optimal_bic_returns_bic_scores_only():
    X, y = _data()
    r = plskit.pls1_find_k_optimal(X, y, k_max=4, selector="bic", seed=7)
    assert r.selector == "bic"
    assert r.bic_scores is not None
    assert r.cv_scores is None
    assert r.cv_scores_se is None


def test_optimal_bic_rejects_n_folds():
    X, y = _data()
    with pytest.raises(plskit.PlsKitError):
        plskit.pls1_find_k_optimal(X, y, k_max=4, selector="bic",
                                     args={"n_folds": 5}, seed=7)


def test_optimal_with_diagnostic_returns_pvalues():
    X, y = _data()
    r = plskit.pls1_find_k_optimal(
        X, y, k_max=4, selector="r2_se",
        diagnostic="split_nb",
        args={"n_folds": 5, "n_splits": 30},
        seed=7,
    )
    assert r.pvalues is not None
    assert r.pvalues.shape == (r.k_star,)
    assert r.diagnostic == "split_nb"


def test_optimal_diagnostic_score_rejected():
    X, y = _data()
    with pytest.raises(plskit.PlsKitError):
        plskit.pls1_find_k_optimal(
            X, y, k_max=4, selector="r2_se",
            diagnostic="score", seed=7,
        )


def test_optimal_n_splits_without_diagnostic_rejected():
    X, y = _data()
    with pytest.raises(plskit.PlsKitError):
        plskit.pls1_find_k_optimal(
            X, y, k_max=4, selector="r2_se",
            args={"n_splits": 30}, seed=7,
        )
