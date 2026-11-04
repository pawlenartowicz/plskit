"""Wiring and oracle tests for factors, pca_basis, profiles_find_n, profiles
and profiles_search."""

import dataclasses
from types import MappingProxyType

import numpy as np
import pytest

import plskit
from plskit import PlsKitError


def _data(n=60, d=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = X[:, :3] @ np.array([1.0, -0.7, 0.5]) + 0.5 * rng.standard_normal(n)
    return X, y


def _fit():
    return plskit.pls1_fit(*_data(), k=2)


def _setup(n_factors=3):
    X, y = _data()
    m = plskit.pls1_fit(X, y, k=2)
    Xs = plskit.preprocess(X).X_std
    f = plskit.factors(m, X, n_factors)
    return X, y, m, Xs, f


def test_spec_snippets_run_as_written():
    X, y = _data()
    m = plskit.pls1_fit(X, y, k=2)
    coef = m.coef
    Xs = plskit.preprocess(X).X_std
    Y = X[:, :3] + np.random.default_rng(2).standard_normal((len(X), 3))
    m3 = plskit.pls3_fit(X, Y, k=3)
    W = np.random.default_rng(3).standard_normal((12, 3))
    L = np.random.default_rng(4).standard_normal((len(X), 3))

    f = plskit.factors(m, X, 3)
    f = plskit.factors(coef, X, 3)
    f = plskit.factors(coef, Xs, 3, pre_standardized=True)
    r = plskit.profiles(m, X, 3)
    r = plskit.profiles(coef, X, 3)
    r = plskit.profiles(coef, Xs, 3, pre_standardized=True)
    f = plskit.rotate(f, X)
    r3 = plskit.rotate(m3, X)
    r3 = plskit.rotate(m3, L=m3.V)
    W_rot = plskit.rotate(W, L=L).W_rot
    assert W_rot.shape == (12, 3) and r3.rotation_spec is not None


def test_numpy_oracle_split_and_projection():
    X, _, m, Xs, f0 = _setup(3)
    n = 3
    V = np.linalg.svd(Xs, full_matrices=False)[2].T
    coef = m.coef
    K = V.shape[1]
    c = V.T @ coef

    # Cuts: block j ends at the smallest e with S(e) >= j/n, clamped.
    S = np.concatenate([[0.0], np.cumsum(c**2) / np.sum(c**2)])
    cuts, start = [0], 0
    for j in range(1, n):
        e = int(np.argmax(S >= j / n))
        e = min(max(e, start + 1), K - (n - j))
        cuts.append(e)
        start = e
    cuts.append(K)
    pieces = np.stack(
        [V[:, a:b] @ c[a:b] for a, b in zip(cuts[:-1], cuts[1:])], axis=1
    )
    Wf = pieces / np.linalg.norm(pieces, axis=0)

    np.testing.assert_allclose(f0.F, pieces, atol=1e-12)
    np.testing.assert_allclose(np.abs(f0.W), np.abs(Wf), atol=1e-12)

    L = Xs @ f0.W
    R = plskit.rotate(f0.W, L=L).spec.R
    W_rot = f0.W @ R
    a = W_rot.T @ coef
    W_rot = W_rot * np.where(a < 0, -1.0, 1.0)
    a = np.abs(a)
    order = np.argsort(-a, kind="stable")
    W_rot, a = W_rot[:, order], a[order]
    F = W_rot * a

    f = plskit.rotate(f0, X)
    np.testing.assert_allclose(f.W, W_rot, atol=1e-12)
    np.testing.assert_allclose(f.size, a, atol=1e-12)
    np.testing.assert_allclose(f.F, F, atol=1e-12)
    np.testing.assert_allclose(f.F.sum(axis=1), coef, atol=1e-12)
    np.testing.assert_allclose(f.W, f0.W @ f.rotation_spec.R, atol=1e-12)


def test_pre_standardized_overrides_the_result():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((60, 6)) * np.array([1, 5, 0.2, 3, 1, 8]) + 2.0
    y = X[:, 0] + rng.standard_normal(60)
    m = plskit.pls1_fit(X, y, k=2)
    default = plskit.profiles(m, X, 2)
    r = plskit.profiles(m, X, 2, pre_standardized=True)
    assert r.factors.pre_standardized is True
    assert default.factors.pre_standardized is False
    np.testing.assert_allclose(r.scores, X @ r.factors.F, atol=1e-12)
    assert not np.allclose(r.scores, default.scores)


def test_vector_form_has_no_model():
    X, y = _data()
    m = plskit.pls1_fit(X, y, k=2)
    r = plskit.profiles(m.coef, X, 3)
    assert r.model is None
    np.testing.assert_allclose(r.factors.F.sum(axis=1), m.coef, atol=1e-12)
    ref = plskit.profiles(m, X, 3)
    np.testing.assert_allclose(r.scores, ref.scores, atol=1e-12)


def test_list_coef_is_read_and_scalar_refused():
    X, _ = _data(d=3)
    coef = [1, -2, 3]
    f = plskit.factors(coef, X, 2)
    g = plskit.factors(np.array([1.0, -2.0, 3.0]), X, 2)
    assert np.array_equal(f.F, g.F)
    assert f.F.shape == (3, 2)
    with pytest.raises(PlsKitError) as e:
        plskit.factors(5, X, 2)
    assert e.value.code == "invalid_argument"


def test_section_2_snippet_runs_as_written():
    X, y = _data()
    m = plskit.pls1_fit(X, y, k="optimal", k_max=5)
    f = plskit.factors(m, X, n=3)
    f = plskit.rotate(f, X)
    np.testing.assert_allclose(f.F.sum(axis=1), m.coef, atol=1e-12)


def test_profiles_returns_nested_dataclasses():
    X, y = _data()
    m = plskit.pls1_fit(X, y, k=2)
    r = plskit.profiles(m, X, 3)
    assert type(r) is plskit.ProfilesResult
    assert type(r.model) is plskit.PLS1Result
    assert type(r.factors) is plskit.FactorsResult
    assert isinstance(r.factors.rotation_spec.args, MappingProxyType)
    with pytest.raises(TypeError):
        r.factors.rotation_spec.args["max_iter"] = 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.scores = None
    np.testing.assert_allclose(r.factors.F.sum(axis=1), r.model.coef, atol=1e-12)


def test_profiles_search_returns_its_own_result():
    X, y = _data()
    m = plskit.pls1_fit(X, y, k=2)
    r = plskit.profiles_search(m, X, y, seed=1)
    assert type(r) is plskit.ProfilesSearchResult
    assert type(r.model) is plskit.PLS1Result
    assert type(r.n_search) is plskit.ProfilesFindNResult
    assert r.seed == 1 and r.n_search.seed == 1


def _weighted_fit():
    X, y = _data()
    return plskit.pls1_fit(X, y, k=2, weights=np.linspace(1, 2, len(y))), X


def _pls3():
    X, _ = _data()
    return plskit.pls3_fit(X, X[:, :3] + 1.0, k=1), X


@pytest.mark.parametrize(
    "call, code",
    [
        (lambda: plskit.factors(np.ones(3), np.ones((6, 4)), 1), "shape_mismatch"),
        (lambda: plskit.factors(np.ones(4), np.array([[1.0, 0, 0, 0]] * 6), 1),
         "invalid_input"),
        (lambda: plskit.factors(np.array([1.0, np.nan, 1, 1]), _data(d=4)[0], 1),
         "invalid_input"),
        (lambda: plskit.factors(np.zeros(4), _data(d=4)[0], 1), "invalid_input"),
        (lambda: plskit.factors(np.ones(4), _data(d=4)[0], 9), "invalid_input"),
        (lambda: plskit.factors(_weighted_fit()[0], _weighted_fit()[1], 2),
         "invalid_input"),
        (lambda: plskit.factors(_pls3()[0], _pls3()[1], 1), "invalid_argument"),
        (lambda: plskit.pca_basis(np.full((5, 3), np.nan)), "invalid_input"),
        (lambda: plskit.profiles(_fit(), _data()[0], "search"), "invalid_argument"),
        (lambda: plskit.profiles(_fit(), _data()[0], 2, rotation="oblimin"),
         "rotation_method_not_implemented"),
        (lambda: plskit.profiles_find_n(*_data(), 1, 0), "invalid_input"),
    ],
    ids=["length", "rank_one_n_range_free", "nan_coef", "zero_coef", "n_range",
         "weighted", "pls3", "pca_nan", "n_string", "rotation", "n_max"],
)
def test_factors_error_codes(call, code):
    with pytest.raises(PlsKitError) as e:
        call()
    assert e.value.code == code


def test_already_rotated_code():
    *_, f = _setup(2)
    g = plskit.rotate(f, L=np.random.default_rng(0).standard_normal((60, 2)))
    with pytest.raises(PlsKitError) as e:
        plskit.rotate(g, L=np.eye(2))
    assert e.value.code == "already_rotated"


def test_n_grid_is_a_list():
    X, y = _data()
    r = plskit.profiles_find_n(X, y, 2, 4, seed=3)
    assert isinstance(r.n_grid, list)
    assert all(isinstance(v, int) for v in r.n_grid)
