import numpy as np
import pytest

import plskit


def _data(n=80, p=6, q=3, snr=3.0, seed=1):
    """(X, Y) sharing one latent factor. snr=0.0 gives independent blocks."""
    rng = np.random.default_rng(seed)
    f = rng.normal(size=n)
    X = rng.normal(size=(n, p))
    X[:, 0] += snr * f
    X[:, 1] += snr * f
    Y = rng.normal(size=(n, q))
    Y[:, 0] += snr * f
    return X, Y


def test_fit_returns_PLS3Result_with_expected_shapes():
    X, Y = _data()
    m = plskit.pls3_fit(X, Y, k=2)
    assert isinstance(m, plskit.PLS3Result)
    assert m.U.shape == (6, 2)
    assert m.V.shape == (3, 2)
    assert m.singular_values.shape == (2,)
    assert m.x_scores.shape == (80, 2)
    assert m.y_scores.shape == (80, 2)
    assert m.k_used == 2
    assert m.pre_standardized_X is False
    assert m.pre_standardized_Y is False
    # PLS3 is symmetric — there is no predict and no coefficient vector.
    assert not hasattr(m, "beta")
    assert not hasattr(m, "coef")


def test_plssvd_fit_alias_matches_pls3_fit():
    X, Y = _data()
    a = plskit.pls3_fit(X, Y, k=2)
    b = plskit.plssvd_fit(X, Y, k=2)
    assert np.array_equal(a.U, b.U)
    assert np.array_equal(a.singular_values, b.singular_values)


def test_fit_1d_Y_is_rejected_with_a_clear_message():
    X, Y = _data()
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.pls3_fit(X, Y[:, 0], k=1)
    assert ei.value.code == "invalid_argument"


def test_fit_weights_are_rejected():
    X, Y = _data()
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.pls3_fit(X, Y, k=1, weights=np.ones(80))
    assert ei.value.code == "invalid_argument"


def test_fit_k_above_min_p_q_raises_k_exceeds_max():
    X, Y = _data()
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.pls3_fit(X, Y, k=4)
    assert ei.value.code == "k_exceeds_max"


def test_transform_on_training_data_reproduces_in_sample_scores():
    X, Y = _data()
    m = plskit.pls3_fit(X, Y, k=2)
    s = plskit.pls3_transform(m, X, Y, which="both")
    assert isinstance(s, plskit.PLS3Scores)
    np.testing.assert_allclose(s.x_scores, m.x_scores, atol=1e-10)
    np.testing.assert_allclose(s.y_scores, m.y_scores, atol=1e-10)


def test_transform_which_selects_blocks():
    X, Y = _data()
    m = plskit.pls3_fit(X, Y, k=2)
    s = plskit.pls3_transform(m, X, None, which="x_scores")
    assert s.x_scores is not None and s.y_scores is None
    s = plskit.pls3_transform(m, None, Y, which="y_scores")
    assert s.x_scores is None and s.y_scores is not None


def test_transform_unknown_which_raises_invalid_args():
    X, Y = _data()
    m = plskit.pls3_fit(X, Y, k=2)
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.pls3_transform(m, X, Y, which="scores")
    assert ei.value.code == "invalid_args"


def test_plssvd_transform_alias_matches():
    X, Y = _data()
    m = plskit.pls3_fit(X, Y, k=2)
    a = plskit.pls3_transform(m, X, None, which="x_scores")
    b = plskit.plssvd_transform(m, X, None, which="x_scores")
    assert np.array_equal(a.x_scores, b.x_scores)


def test_confirmatory_test_rejects_a_strong_shared_factor():
    X, Y = _data(snr=3.0, seed=2)
    r = plskit.pls3_confirmatory_test(
        X, Y, k=1, method="split_exact", args={"n_perm": 199, "n_splits": 10}, seed=42
    )
    assert isinstance(r, plskit.ConfirmatoryTestResult)
    assert r.method == "split_exact"
    assert r.k == 1
    assert r.n_perm == 199
    assert r.n_splits == 10
    assert r.seed == 42
    assert r.rho_hat is None
    assert r.stable_rank is None
    assert r.ci is None
    assert r.pvalue <= 0.01


def test_confirmatory_test_does_not_reject_independent_blocks():
    X, Y = _data(snr=0.0, seed=9)
    r = plskit.pls3_confirmatory_test(
        X, Y, k=1, method="split_exact", args={"n_perm": 199, "n_splits": 10}, seed=42
    )
    assert r.pvalue > 0.05


def test_confirmatory_test_same_seed_is_reproducible():
    X, Y = _data(snr=2.0, seed=3)
    kw = dict(method="split_exact", args={"n_perm": 99, "n_splits": 8}, seed=5)
    a = plskit.pls3_confirmatory_test(X, Y, k=1, **kw)
    b = plskit.pls3_confirmatory_test(X, Y, k=1, **kw)
    assert a.pvalue == b.pvalue
    assert a.statistic == b.statistic


def test_confirmatory_test_seed_none_is_recorded_and_replayable():
    X, Y = _data(snr=2.0, seed=3)
    a = plskit.pls3_confirmatory_test(
        X, Y, k=1, method="split_exact", args={"n_perm": 49, "n_splits": 6}
    )
    b = plskit.pls3_confirmatory_test(
        X, Y, k=1, method="split_exact", args={"n_perm": 49, "n_splits": 6}, seed=a.seed
    )
    assert a.pvalue == b.pvalue


@pytest.mark.parametrize("method", ["raw_perm", "score", "e"])
def test_confirmatory_test_rejects_other_methods(method):
    X, Y = _data()
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.pls3_confirmatory_test(X, Y, k=1, method=method, seed=1)
    assert ei.value.code == "invalid_args"


def test_confirmatory_test_unknown_arg_key_is_rejected():
    X, Y = _data()
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.pls3_confirmatory_test(
            X, Y, k=1, method="split_exact", args={"n_folds": 5}, seed=1
        )
    assert ei.value.code == "invalid_args"


def test_confirmatory_test_k_above_one_is_rejected():
    X, Y = _data()
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.pls3_confirmatory_test(
            X, Y, k=2, method="split_exact", args={"n_perm": 49, "n_splits": 6}, seed=1
        )
    assert ei.value.code == "invalid_argument"


def test_split_nb_runs_and_reports_its_own_method():
    # p=6 and n=80 clear the gate's column, n_eff and stable-rank floors.
    X, Y = _data(snr=3.0, seed=2)
    r = plskit.pls3_confirmatory_test(
        X, Y, k=1, method="split_nb", args={"n_splits": 10}, seed=42
    )
    assert r.method == "split_nb"
    assert r.n_perm is None
    assert r.n_splits == 10
    assert r.stable_rank is not None
    assert r.rho_hat is not None
    assert r.ci is None
    assert 0.0 < r.pvalue <= 1.0
    assert r.pvalue <= 0.01


def test_split_nb_statistic_matches_split_exact_at_the_same_seed():
    X, Y = _data(snr=2.0, seed=3)
    nb = plskit.pls3_confirmatory_test(
        X, Y, k=1, method="split_nb", args={"n_splits": 8}, seed=5
    )
    ex = plskit.pls3_confirmatory_test(
        X, Y, k=1, method="split_exact", args={"n_perm": 99, "n_splits": 8}, seed=5
    )
    assert nb.statistic == ex.statistic


def test_split_nb_gate_reroutes_and_warns():
    # 3 columns trips the gate's column precheck.
    X, Y = _data(n=80, p=3, q=3, snr=3.0, seed=2)
    with pytest.warns(UserWarning):
        r = plskit.pls3_confirmatory_test(
            X, Y, k=1, method="split_nb", args={"n_splits": 6}, seed=42
        )
    assert r.method == "split_exact"
    assert r.n_perm == 1000
    assert r.n_splits == 6


def test_split_nb_force_overrides_the_gate():
    X, Y = _data(n=80, p=3, q=3, snr=3.0, seed=2)
    r = plskit.pls3_confirmatory_test(
        X, Y, k=1, method="split_nb", args={"n_splits": 6, "force": True}, seed=42
    )
    assert r.method == "split_nb"
    assert r.n_perm is None


def test_split_nb_unknown_arg_key_is_rejected():
    X, Y = _data()
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.pls3_confirmatory_test(
            X, Y, k=1, method="split_nb", args={"n_perm": 100}, seed=1
        )
    assert ei.value.code == "invalid_args"
