"""Tests for plskit.rotate (varimax)."""

from __future__ import annotations

import dataclasses
from fractions import Fraction

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
        (_W, {"method": "varimax", "args": {1: 2}}, "invalid_argument"),
        (_W, {"method": "varimax", "args": {"tol": Fraction(1, 1000)}}, "invalid_argument"),
        (_W, {"method": "varimax", "args": {"max_iter": object()}}, "invalid_argument"),
        (np.zeros((10, 0)), {"method": "varimax"}, "invalid_input"),
        (_W, {"method": "varimax",
              "L": np.random.default_rng(11).normal(size=(40, 2))}, "shape_mismatch"),
    ],
    ids=["unknown_method", "unknown_args_key", "args_wrong_type", "args_negative",
         "args_tol_str", "method_not_str", "args_not_dict", "args_non_string_key",
         "args_fraction", "args_object", "K0", "L_shape_mismatch"],
)
def test_rotate_rejects_bad_input(W, kwargs, code):
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.rotate(W, **kwargs)
    assert ei.value.code == code


def test_rotate_rejects_a_list_that_contains_itself():
    lst = []
    lst.append(lst)
    with pytest.raises(plskit.PlsKitError, match="is nested too deeply") as ei:
        plskit.rotate(_W, args={"x": lst})
    assert ei.value.code == "invalid_argument"


def test_rotate_rejects_a_dict_that_contains_itself():
    d = {}
    d["a"] = d
    with pytest.raises(plskit.PlsKitError, match="is nested too deeply") as ei:
        plskit.rotate(_W, args=d)
    assert ei.value.code == "invalid_argument"


# ── Spec immutability ────────────────────────────────────────


def test_rotation_spec_args_is_immutable():
    W = np.random.default_rng(13).normal(size=(15, 3))
    r = plskit.rotate(W, method="varimax")
    with pytest.raises(TypeError):
        r.spec.args["max_iter"] = 99


# ── Bad input ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "first, message",
    [
        (None, "rotate() missing required argument 'model_or_W'"),
        ("not a model or array",
         'model_or_W must be a numeric matrix, got the string "not a model or array"'),
        (np.ones(3), "rotate() first arg must be a FactorsResult, a PLS3Result or a matrix, got a vector"),
    ],
    ids=["none", "string", "vector"],
)
def test_rotate_wrong_first_arg_is_invalid_argument(first, message):
    """A first argument that is neither a FactorsResult nor a matrix raises
    `invalid_argument`, not a bare TypeError."""
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


# ── FactorsResult, PLS3Result, PLS1Result ─────────────────────


def _factors(n=3):
    X, y = _data()
    m = plskit.pls1_fit(X, y, k=2)
    return X, m, plskit.factors(m, X, n)


def test_rotate_x_equals_row_scores_target():
    X, _, f = _factors()
    Xs = plskit.preprocess(X).X_std
    a = plskit.rotate(f, X)
    b = plskit.rotate(f, L=Xs @ f.W)
    # Not bitwise: numpy's matmul (Accelerate on macOS) and the engine's round L differently.
    np.testing.assert_allclose(a.F, b.F, rtol=0, atol=1e-12)
    np.testing.assert_allclose(a.rotation_spec.R, b.rotation_spec.R, rtol=0, atol=1e-12)


def test_rotate_pls3_returns_pls3result():
    X, _ = _data()
    Y = X[:, :3] + np.random.default_rng(2).normal(size=(80, 3))
    m3 = plskit.pls3_fit(X, Y, k=3)
    r = plskit.rotate(m3, X)
    assert type(r) is plskit.PLS3Result
    assert r.rotation_spec is not None
    assert m3.rotation_spec is None
    r2 = plskit.rotate(m3, L=m3.V)
    assert type(r2) is plskit.PLS3Result


def test_sparse_pls3_warns():
    X, _ = _data()
    Y = X[:, :3] + np.random.default_rng(2).normal(size=(80, 3))
    m3 = plskit.spls3_fit(X, Y, k=2, keep_X=3, keep_Y=2)
    with pytest.warns(UserWarning, match="spreads its zeros"):
        plskit.rotate(m3, X)


def test_rotate_refuses_pls1_result():
    X, m, _ = _factors()
    with pytest.raises(plskit.PlsKitError, match="does not take a PLS1Result") as ei:
        plskit.rotate(m, X)
    assert ei.value.code == "invalid_argument"


def test_x_and_l_together_is_invalid_argument():
    X, _, f = _factors()
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.rotate(f, X, L=X[:, :3])
    assert ei.value.code == "invalid_argument"


def test_untagged_record_is_invalid_argument():
    X, _ = _data()
    with pytest.raises(plskit.PlsKitError, match="got a record") as ei:
        plskit.rotate({"F": 1}, X)
    assert ei.value.code == "invalid_argument"
