"""Wiring tests for `covariates` and `null_y`. The numbers are tested in the
engine; these check that the arguments reach it and that the results carry
the new fields."""

import numpy as np
import pytest

import plskit
from plskit import PlsKitError


def _data(n=60, d=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    z1 = X[:, 0] + rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    # Two real covariates, then a constant and a multiple of z1 (both aliased).
    Z = np.column_stack([z1, z2, np.full(n, 3.0), 2.0 * z1])
    y = X[:, :3] @ np.array([1.0, -0.7, 0.5]) + z2 + 0.5 * rng.standard_normal(n)
    return X, y, Z


def _code(excinfo):
    return excinfo.value.code


def _m_z(Z, V):
    """Residuals of the columns of `V` on [1, Z] by least squares."""
    A = np.column_stack([np.ones(Z.shape[0]), Z])
    return V - A @ np.linalg.lstsq(A, V, rcond=None)[0]


def test_fit_records_covariates():
    X, y, Z = _data()
    m = plskit.pls1_fit(X, y, 2, covariates=Z)
    assert m.n_covariates == 2
    assert m.covariate_coef.shape == (4,)
    assert m.covariate_coef[2] == 0.0 and m.covariate_coef[3] == 0.0
    plain = plskit.pls1_fit(X, y, 2)
    assert plain.covariate_coef is None and plain.n_covariates is None
    s = plskit.spls1_fit(X, y, 2, 4, covariates=Z)
    assert s.n_covariates == 2


def test_predict_with_covariates_round_trip():
    X, y, Z = _data()
    m = plskit.pls1_fit(X, y, 2, covariates=Z)
    got = plskit.pls1_predict(m, X, covariates=Z)
    want = m.intercept + X @ m.beta + Z @ m.covariate_coef
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)
    with pytest.raises(PlsKitError) as e:
        plskit.pls1_predict(m, X)
    assert _code(e) == "invalid_argument"
    with pytest.raises(PlsKitError) as e:
        plskit.pls1_predict(plskit.pls1_fit(X, y, 2), X, covariates=Z)
    assert _code(e) == "invalid_argument"


def test_fit_k_optimal_passes_covariates():
    X, y, Z = _data()
    m = plskit.pls1_fit(X, y, "optimal", k_max=3, covariates=Z, seed=1)
    assert m.selection_result.n_covariates == 2
    assert m.n_covariates == 2
    s = plskit.pls1_fit(X, y, "sequence", k_max=2, covariates=Z, seed=1,
                        find_k_args={"test_method": "split_exact",
                                     "args": {"n_perm": 100}})
    assert s.selection_result.n_covariates == 2


def test_find_k_functions_pass_covariates():
    X, y, Z = _data()
    seq = {"test_method": "split_exact", "args": {"n_perm": 100}, "seed": 1}
    assert plskit.pls1_find_k_optimal(X, y, 3, covariates=Z, seed=1).n_covariates == 2
    assert plskit.spls1_find_k_optimal(X, y, 3, 4, covariates=Z, seed=1).n_covariates == 2
    assert plskit.pls1_find_k_sequence(X, y, 2, covariates=Z, **seq).n_covariates == 2
    assert plskit.spls1_find_k_sequence(X, y, 2, 4, covariates=Z, **seq).n_covariates == 2
    assert plskit.spls1_find_keep_optimal(X, y, 1, covariates=Z, seed=1).n_covariates == 2
    assert plskit.pls1_find_k_optimal(X, y, 3, seed=1).n_covariates is None


def test_tests_record_covariates():
    X, y, Z = _data()
    r = plskit.pls1_confirmatory_test(X, y, 1, test_method="split_exact",
                                      args={"n_perm": 100}, covariates=Z, seed=1)
    assert r.n_covariates == 2
    assert plskit.pls1_confirmatory_test(
        X, y, 1, test_method="split_exact", args={"n_perm": 100}, seed=1
    ).n_covariates is None
    p = plskit.pls1_perm_null(X, y, 1, n_perm=100, covariates=Z, seed=1)
    assert p.n_covariates == 2
    assert plskit.pls1_perm_null(X, y, 1, n_perm=100, seed=1).n_covariates is None


def test_factors_rotate_profiles_need_covariates_for_a_covariate_model():
    X, y, Z = _data()
    m = plskit.pls1_fit(X, y, 2, covariates=Z)
    plain = plskit.pls1_fit(X, y, 2)
    f = plskit.factors(m, X, 2, covariates=Z)
    assert f.covariate_adjusted
    assert not plskit.factors(plain, X, 2).covariate_adjusted
    calls = [
        lambda: plskit.factors(m, X, 2),
        lambda: plskit.rotate(f, X),
        lambda: plskit.profiles(m, X, 2),
        lambda: plskit.profiles_search(m, X, y),
        lambda: plskit.factors(plain, X, 2, covariates=Z),
        lambda: plskit.rotate(plskit.factors(plain, X, 2), X, covariates=Z),
        lambda: plskit.rotate(f.W, covariates=Z),
        lambda: plskit.rotate(f, covariates=Z),
    ]
    for c in calls:
        with pytest.raises(PlsKitError) as e:
            c()
        assert _code(e) == "invalid_argument", e.value
    # A raw coefficient vector takes covariates or not, as the caller says.
    assert plskit.factors(m.coef, X, 2, covariates=Z).covariate_adjusted
    assert not plskit.factors(m.coef, X, 2).covariate_adjusted


def test_rotate_target_with_covariates():
    X, y, Z = _data()
    m = plskit.pls1_fit(X, y, 2, covariates=Z)
    f = plskit.factors(m, X, 2, covariates=Z)
    Xs = plskit.preprocess(X).X_std
    L = _m_z(Z, Xs @ f.W)
    got = plskit.rotate(f, X, covariates=Z)
    want = plskit.rotate(f, L=L)
    np.testing.assert_allclose(got.F, want.F, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(got.rotation_spec.R, want.rotation_spec.R,
                               rtol=1e-10, atol=1e-12)
    assert got.covariate_adjusted
    # The array overload adjusts its target the same way.
    a = plskit.rotate(f.W, X, covariates=Z)
    b = plskit.rotate(f.W, L=L)
    np.testing.assert_allclose(a.W_rot, b.W_rot, rtol=1e-10, atol=1e-12)


def test_null_y_runs_and_refuses_n_perm():
    X, y, _ = _data()
    rng = np.random.default_rng(5)
    Ynull = np.column_stack([rng.permutation(y) for _ in range(50)])
    r = plskit.pls1_confirmatory_test(X, y, 1, test_method="split_exact",
                                      null_y=Ynull, seed=1)
    assert r.n_perm == 50
    with pytest.raises(PlsKitError) as e:
        plskit.pls1_confirmatory_test(X, y, 1, test_method="split_exact",
                                      null_y=Ynull, args={"n_perm": 100}, seed=1)
    assert _code(e) == "invalid_args"


def test_error_codes():
    X, y, _ = _data(n=20)
    Z = np.random.default_rng(3).standard_normal((20, 17))
    with pytest.raises(PlsKitError) as e:
        plskit.pls1_fit(X, y, 2, covariates=Z)
    assert _code(e) == "insufficient_residual_df"
