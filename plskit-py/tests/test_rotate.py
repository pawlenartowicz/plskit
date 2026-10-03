"""Tests for plskit.rotate (varimax)."""

from __future__ import annotations

import numpy as np
import pytest

import plskit


def _data(n=80, d=6, k_signal=3, snr=4.0, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    beta = np.zeros(d); beta[:k_signal] = 1.0
    y = X @ beta * snr + rng.normal(size=n)
    return X, y


# ── ndarray-in dispatch ────────────────────────────────────────


@pytest.mark.parametrize(
    "args",
    [None, {}, {"max_iter": None, "tol": None, "kaiser_normalize": None}],
    ids=["no_args", "empty", "keys_set_to_None"],
)
def test_rotate_records_the_engine_defaults(args):
    """Omitted keys, and keys set to None, take the engine's own
    `VarimaxArgs::default()`, and `spec.args` records the resolved, typed
    values (as plskit-bind does), never the caller's dict."""
    W = np.random.default_rng(4).normal(size=(20, 2))
    r = plskit.rotate(W, method="varimax", args=args)
    assert dict(r.spec.args) == {"max_iter": 50, "tol": 1e-8, "kaiser_normalize": True}
    assert type(r.spec.args["max_iter"]) is int


@pytest.mark.parametrize("max_iter", [2, 2.0])
def test_rotate_array_max_iter_reaches_the_engine(max_iter):
    # A whole float is recorded as an int.
    W = np.random.default_rng(5).normal(size=(30, 4))
    assert plskit.rotate(W, method="varimax").spec.sweeps > 2
    r = plskit.rotate(W, method="varimax", args={"max_iter": max_iter})
    assert r.spec.sweeps == 2
    assert r.spec.args["max_iter"] == 2 and type(r.spec.args["max_iter"]) is int


def test_rotate_array_tol_reaches_the_engine():
    # With the default tol the second pass leaves R ~3e-5 from I; atol 1e-6
    # holds only with tol=1e-12.
    W = np.random.default_rng(6).normal(size=(40, 3))
    r1 = plskit.rotate(W, method="varimax", args={"tol": 1e-12})
    r2 = plskit.rotate(r1.W_rot, method="varimax", args={"tol": 1e-12})
    np.testing.assert_allclose(r2.spec.R, np.eye(3), atol=1e-6)


# ── Args validation ────────────────────────────────────────────


_W = np.random.default_rng(7).normal(size=(10, 3))


@pytest.mark.parametrize(
    "W, kwargs, code",
    [
        (_W, {"method": "promax"}, "rotation_method_not_implemented"),
        (_W, {"method": "varimax", "args": {"bogus_key": 1}}, "invalid_args"),
        (_W, {"method": "varimax", "args": {"max_iter": "not_an_int"}}, "invalid_args"),
        (_W, {"method": "varimax", "args": {"max_iter": -1}}, "invalid_args"),
        (_W, {"method": "varimax", "args": {"tol": "x"}}, "invalid_args"),
        (_W, {"method": 1}, "invalid_argument"),
        (_W, {"method": "varimax", "args": [1]}, "invalid_argument"),
        (np.zeros((10, 0)), {"method": "varimax"}, "invalid_input"),
        (_W, {"method": "varimax",
              "L": np.random.default_rng(11).normal(size=(40, 2))}, "shape_mismatch"),
    ],
    ids=["unknown_method", "unknown_args_key", "args_wrong_type", "args_negative",
         "args_tol_str", "method_not_str", "args_not_dict", "K0", "L_shape_mismatch"],
)
def test_rotate_rejects_bad_input(W, kwargs, code):
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.rotate(W, **kwargs)
    assert ei.value.code == code


# ── Spec immutability ────────────────────────────────────────


def test_rotation_spec_args_is_immutable():
    W = np.random.default_rng(13).normal(size=(15, 3))
    r = plskit.rotate(W, method="varimax")
    with pytest.raises(TypeError):
        r.spec.args["max_iter"] = 99


# ── Model-in dispatch ────────────────────────────────────────


def test_rotate_model_rotates_scores_and_loadings_and_keeps_coef():
    X, y = _data()
    m = plskit.pls1_fit(X, y, k=3)
    m2 = plskit.rotate(m, method="varimax")
    assert isinstance(m2, plskit.PLS1Result)
    assert m.rotation_spec is None  # the input model is untouched
    assert m2.rotation_spec.method == "varimax"
    R = m2.rotation_spec.R
    np.testing.assert_allclose(m2.W, m.W @ R, atol=1e-12)
    np.testing.assert_allclose(m2.T, m.T @ R, atol=1e-12)
    np.testing.assert_allclose(m2.P, m.P @ R, atol=1e-12)
    # Q rotates by R.T, so the fitted signal T @ Q is unchanged.
    np.testing.assert_allclose(m2.T @ m2.Q, m.T @ m.Q, atol=1e-10)
    np.testing.assert_allclose(m2.coef, m.coef, atol=1e-15)
    np.testing.assert_allclose(m2.beta, m.beta, atol=1e-15)
    assert m2.intercept == m.intercept


def test_rotate_model_already_rotated_raises():
    X, y = _data()
    m = plskit.pls1_fit(X, y, k=3)
    m2 = plskit.rotate(m, method="varimax")
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.rotate(m2, method="varimax")
    assert ei.value.code == "already_rotated"


def test_rotate_sparse_model_keeps_keep():
    X, y = _data()
    m = plskit.spls1_fit(X, y, k=2, keep=3)
    m2 = plskit.rotate(m, method="varimax")
    assert m2.keep == m.keep == 3


def test_rotate_optimal_k_model_keeps_selection_result():
    X, y = _data()
    m = plskit.pls1_fit(X, y, k="optimal", k_max=4)
    m2 = plskit.rotate(m, method="varimax")
    assert m2.selection_result is m.selection_result
    assert m2.selection_result is not None


# ── Bad input ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "first, message",
    [
        ("pls3", "rotate() first arg must be a PLS1Result, got a PLS3Result"),
        (None, "rotate() first arg must be a PLS1Result or a matrix, got None"),
        ("not a model or array", 'W must be a numeric matrix, got the string "not a model or array"'),
        (np.ones(3), "W must be 2-D, got 1-D"),
    ],
    ids=["pls3_result", "none", "string", "vector"],
)
def test_rotate_wrong_first_arg_is_invalid_argument(first, message):
    """A first argument that is neither a PLS1Result nor a matrix raises
    `invalid_argument`, as plskit-bind (R) does, not a bare TypeError."""
    if isinstance(first, str) and first == "pls3":
        X, _ = _data()
        first = plskit.pls3_fit(X, X[:, :3] + 1.0, k=1)
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.rotate(first, method="varimax")
    assert ei.value.code == "invalid_argument"
    assert str(ei.value) == message


def test_rotate_reads_a_nested_list_w_like_any_array_argument():
    W = np.random.default_rng(13).normal(size=(15, 3))
    got = plskit.rotate(W.tolist(), method="varimax")
    ref = plskit.rotate(W, method="varimax")
    assert np.array_equal(got.W_rot, ref.W_rot)
    assert np.array_equal(got.spec.R, ref.spec.R)
