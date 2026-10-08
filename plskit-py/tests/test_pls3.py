import dataclasses
import warnings

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
    assert m.keep_X is None and m.keep_Y is None
    assert m.converged is None and m.n_iter is None


def test_plssvd_aliases_match_pls3():
    X, Y = _data()
    a = plskit.pls3_fit(X, Y, k=2)
    b = plskit.plssvd_fit(X, Y, k=2)
    assert np.array_equal(a.U, b.U)
    assert np.array_equal(a.singular_values, b.singular_values)
    ta = plskit.pls3_transform(a, X, None, which="x_scores")
    tb = plskit.plssvd_transform(a, X, None, which="x_scores")
    assert np.array_equal(ta.x_scores, tb.x_scores)


def test_fit_1d_Y_is_rejected_with_a_clear_message():
    X, Y = _data()
    with pytest.raises(plskit.PlsKitError, match="use pls1_fit") as ei:
        plskit.pls3_fit(X, Y[:, 0], k=1)
    assert ei.value.code == "invalid_argument"


def test_fit_weights_are_rejected():
    X, Y = _data()
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.pls3_fit(X, Y, k=1, weights=np.ones(80))
    assert ei.value.code == "invalid_argument"


def test_transform_which_selects_blocks():
    X, Y = _data()
    m = plskit.pls3_fit(X, Y, k=2)
    s = plskit.pls3_transform(m, X, None, which="x_scores")
    assert isinstance(s, plskit.PLS3Scores)
    assert s.x_scores is not None and s.y_scores is None
    s = plskit.pls3_transform(m, None, Y, which="y_scores")
    assert s.x_scores is None and s.y_scores is not None


def test_transform_unknown_which_raises_invalid_args():
    X, Y = _data()
    m = plskit.pls3_fit(X, Y, k=2)
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.pls3_transform(m, X, Y, which="scores")
    assert ei.value.code == "invalid_args"


@dataclasses.dataclass
class _Foreign:
    """A dataclass that is not a plskit result. `dataclasses.replace`
    refuses it, because of the `init=False` field."""
    a: int = 0
    b: int = dataclasses.field(default=0, init=False)


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda m1, m3, X: plskit.pls1_predict(m3, X),
         "model must be a PLS1Result, got a PLS3Result"),
        (lambda m1, m3, X: plskit.pls1_predict(None, X),
         "pls1_predict() missing required argument 'model'"),
        (lambda m1, m3, X: plskit.pls3_transform(m1, X),
         "model must be a PLS3Result, got a PLS1Result"),
        (lambda m1, m3, X: plskit.plssvd_transform({"U": X}, X),
         "model is missing field 'V'"),
        (lambda m1, m3, X: plskit.pls1_predict(X, X),
         "model must be a PLS1Result, got a matrix"),
        (lambda m1, m3, X: plskit.pls1_predict(1, X),
         "model must be a PLS1Result, got 1"),
        (lambda m1, m3, X: plskit.pls1_predict(_Foreign(), X),
         "model must be a PLS1Result, got a _Foreign"),
    ],
    ids=["predict_pls3", "predict_none", "transform_pls1", "plssvd_transform_dict",
         "predict_ndarray", "predict_int", "predict_foreign_dataclass"],
)
def test_a_model_of_the_wrong_type_is_invalid_argument(call, message):
    """A model argument of the wrong type raises `invalid_argument` naming
    the argument, not the `AttributeError` its fields would."""
    X, Y = _data()
    m1, m3 = plskit.pls1_fit(X, Y[:, 0], k=1), plskit.pls3_fit(X, Y, k=1)
    with pytest.raises(plskit.PlsKitError) as ei:
        call(m1, m3, X)
    assert ei.value.code == "invalid_argument"
    assert str(ei.value) == message


@pytest.mark.parametrize(
    "call",
    [
        lambda m1, m3, X: plskit.pls1_predict(dataclasses.replace(m1, T=m1.T.tolist()), X),
        lambda m1, m3, X: plskit.pls3_transform(dataclasses.replace(m3, U=m3.U.tolist()), X),
    ],
    ids=["pls1_predict", "pls3_transform"],
)
def test_a_model_array_field_given_as_a_nested_list_is_invalid_argument(call):
    X, Y = _data()
    m1, m3 = plskit.pls1_fit(X, Y[:, 0], k=1), plskit.pls3_fit(X, Y, k=1)
    with pytest.raises(plskit.PlsKitError) as ei:
        call(m1, m3, X)
    assert ei.value.code == "invalid_argument"


def test_a_plain_dict_of_a_models_fields_is_accepted_as_the_model():
    X, Y = _data()
    m = plskit.pls3_fit(X, Y, k=2)
    from_dict = plskit.pls3_transform(vars(m), X, Y)
    assert np.array_equal(from_dict.x_scores, plskit.pls3_transform(m, X, Y).x_scores)


def test_confirmatory_test_split_exact_marshals_its_fields():
    X, Y = _data(snr=3.0, seed=2)
    r = plskit.pls3_confirmatory_test(
        X, Y, k=1, test_method="split_exact", args={"n_perm": 199, "n_splits": 10}, seed=42
    )
    assert isinstance(r, plskit.ConfirmatoryTestResult)
    assert r.test_method == "split_exact"
    assert r.k == 1
    assert r.n_perm == 199
    assert r.n_splits == 10
    assert r.seed == 42
    assert r.rho_hat is None
    assert r.stable_rank is None
    assert r.ci is None


def test_confirmatory_test_seed_none_is_recorded_and_replayable():
    X, Y = _data(snr=2.0, seed=3)
    a = plskit.pls3_confirmatory_test(
        X, Y, k=1, test_method="split_exact", args={"n_perm": 49, "n_splits": 6}
    )
    b = plskit.pls3_confirmatory_test(
        X, Y, k=1, test_method="split_exact", args={"n_perm": 49, "n_splits": 6}, seed=a.seed
    )
    assert a.pvalue == b.pvalue


@pytest.mark.parametrize("method", ["raw_perm", "score", "e"])
def test_confirmatory_test_rejects_other_methods(method):
    X, Y = _data()
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.pls3_confirmatory_test(X, Y, k=1, test_method=method, seed=1)
    assert ei.value.code == "invalid_args"


@pytest.mark.parametrize(
    "method, key", [("split_exact", "n_folds"), ("split_nb", "n_perm")]
)
def test_confirmatory_test_unknown_arg_key_is_rejected(method, key):
    X, Y = _data()
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.pls3_confirmatory_test(
            X, Y, k=1, test_method=method, args={key: 5}, seed=1
        )
    assert ei.value.code == "invalid_args"


def test_split_nb_runs_and_reports_its_own_method():
    # p=6 and n=80 clear the gate's column, n_eff and stable-rank floors.
    X, Y = _data(snr=3.0, seed=2)
    r = plskit.pls3_confirmatory_test(
        X, Y, k=1, test_method="split_nb", args={"n_splits": 10}, seed=42
    )
    assert r.test_method == "split_nb"
    assert r.n_perm is None
    assert r.n_splits == 10
    assert r.stable_rank is not None
    assert r.rho_hat is not None
    assert r.ci is None


def test_split_nb_gate_reroutes_and_warns():
    # 3 columns trips the gate's column precheck.
    X, Y = _data(n=80, p=3, q=3, snr=3.0, seed=2)
    with pytest.warns(UserWarning):
        r = plskit.pls3_confirmatory_test(
            X, Y, k=1, test_method="split_nb", args={"n_splits": 6}, seed=42
        )
    assert r.test_method == "split_exact"
    assert r.n_perm == 1000
    assert r.n_splits == 6


def _orthogonal_Y(n=60, p=6, q=3, seed=4):
    # Y residualized on [1, X]: orthogonal to the standardized columns of X
    # up to rounding, so X'Y is rounding noise, sigma_1 included.
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    E = rng.normal(size=(n, q))
    A = np.column_stack([np.ones(n), X])
    Y = E - A @ np.linalg.lstsq(A, E, rcond=None)[0]
    return X, Y


def test_orthogonal_Y_keeps_no_component():
    X, Y = _orthogonal_Y()
    m = plskit.pls3_fit(X, Y, k=1)
    assert m.k_used == 0
    assert m.U.shape == (6, 0)
    assert m.V.shape == (3, 0)
    assert m.singular_values.shape == (0,)
    assert m.x_scores.shape == (60, 0)
    # Same policy as PLS1 on each column of Y.
    assert plskit.pls1_fit(X, Y[:, 0], k=1).k_used == 0
    # And the zero model still transforms, to empty score matrices.
    s = plskit.pls3_transform(m, X_new=X, Y_new=Y)
    assert s.x_scores.shape == (60, 0)
    assert s.y_scores.shape == (60, 0)


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda X, Y: plskit.pls3_fit(X, Y, k=-1), "k must be a non-negative whole number"),
        (lambda X, Y: plskit.pls3_fit(X, Y, pre_standardized_Y=1), "pre_standardized_Y must be a bool"),
        (lambda X, Y: plskit.spls3_fit(X, Y, 1, 2.5, 2), "keep_X must be a non-negative whole number"),
        (lambda X, Y: plskit.spls3_fit(X, Y, 1, 2, 2, tol=2**64),
         r"tol is outside the 64-bit integer range \[-2\^63, 2\^64\)"),
        (lambda X, Y: plskit.pls3_confirmatory_test(X, Y, test_method="split_exact", seed=-1),
         "seed must be a whole number"),
        (lambda X, Y: plskit.pls3_transform(plskit.pls3_fit(X, Y), X, which=1),
         "which must be a string"),
    ],
)
def test_pls3_family_rejects_unusable_top_level_values(call, message):
    X, Y = _data()
    with pytest.raises(plskit.PlsKitError, match=message) as ei:
        call(X, Y)
    assert ei.value.code == "invalid_argument"


def test_auto_is_the_default_and_resolves_per_design():
    X, Y = _data(n=80, p=6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        small = plskit.pls3_confirmatory_test(
            X, Y, args={"n_perm": 49, "n_splits": 6}, seed=3
        )
    assert small.test_method == "split_exact"
    assert small.n_perm == 49

    X, Y = _data(n=300, p=10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        large = plskit.pls3_confirmatory_test(
            X, Y, args={"n_perm": 49, "n_splits": 6}, seed=3
        )
    assert large.test_method == "split_nb"
    assert large.n_perm is None
    assert large.n_splits == 6
    assert large.stable_rank is not None


def test_auto_args_take_n_perm_and_n_splits_only():
    X, Y = _data()
    with pytest.raises(plskit.PlsKitError, match="does not accept arg 'force'") as ei:
        plskit.pls3_confirmatory_test(
            X, Y, test_method="auto", args={"force": True}, seed=1
        )
    assert ei.value.code == "invalid_args"


def test_unknown_method_names_are_listed_as_a_python_list():
    X, Y = _data()
    m = plskit.pls3_fit(X, Y, k=2)
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.pls3_transform(m, X, None, which="bad")
    assert ei.value.code == "invalid_args"
    assert str(ei.value) == (
        "unknown which: bad; allowed: ['x_scores', 'y_scores', 'both']"
    )
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.pls3_confirmatory_test(X, Y, k=1, test_method="raw_perm", seed=7)
    assert ei.value.code == "invalid_args"
    assert str(ei.value) == (
        "test_method='raw_perm' is not available for pls3_confirmatory_test; "
        "allowed: ['split_exact', 'split_nb', 'auto']"
    )
