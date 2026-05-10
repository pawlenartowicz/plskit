"""Tests for plskit.rotate (varimax, v0.1.1)."""

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


def test_rotate_array_w_at_r_equals_w_rot():
    W = np.random.default_rng(1).normal(size=(25, 3))
    r = plskit.rotate(W, method="varimax")
    np.testing.assert_allclose(W @ r.spec.R, r.W_rot, atol=1e-12)


def test_rotate_array_R_is_orthogonal():
    W = np.random.default_rng(2).normal(size=(30, 4))
    r = plskit.rotate(W, method="varimax")
    np.testing.assert_allclose(r.spec.R.T @ r.spec.R, np.eye(4), atol=1e-10)


def test_rotate_array_K1_is_noop():
    W = np.random.default_rng(3).normal(size=(20, 1))
    r = plskit.rotate(W, method="varimax")
    np.testing.assert_allclose(r.spec.R, np.eye(1), atol=1e-15)
    np.testing.assert_allclose(r.W_rot, W, atol=1e-15)
    assert r.spec.sweeps == 0


def test_rotate_array_idempotent_on_converged_solution():
    W = np.random.default_rng(6).normal(size=(40, 3))
    r1 = plskit.rotate(W, method="varimax", args={"tol": 1e-12})
    r2 = plskit.rotate(r1.W_rot, method="varimax", args={"tol": 1e-12})
    np.testing.assert_allclose(r2.spec.R, np.eye(3), atol=1e-6)


# ── Args validation ────────────────────────────────────────────


def test_rotate_unknown_method_raises():
    W = np.random.default_rng(7).normal(size=(10, 3))
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.rotate(W, method="promax")
    assert ei.value.code == "rotation_method_not_implemented"


def test_rotate_unknown_args_key_raises():
    W = np.random.default_rng(8).normal(size=(10, 3))
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.rotate(W, method="varimax", args={"bogus_key": 1})
    assert ei.value.code == "invalid_args"


def test_rotate_args_wrong_type_raises():
    W = np.random.default_rng(9).normal(size=(10, 3))
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.rotate(W, method="varimax", args={"max_iter": "not_an_int"})
    assert ei.value.code == "invalid_args"


def test_rotate_K0_raises():
    W = np.zeros((10, 0))
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.rotate(W, method="varimax")
    assert ei.value.code == "invalid_input"


def test_rotate_L_shape_mismatch_raises():
    W = np.random.default_rng(10).normal(size=(10, 3))
    L = np.random.default_rng(11).normal(size=(40, 2))  # 2 ≠ 3
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.rotate(W, method="varimax", L=L)
    assert ei.value.code == "shape_mismatch"


def test_rotate_non_finite_W_raises():
    W = np.random.default_rng(12).normal(size=(10, 3))
    W[0, 0] = np.nan
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.rotate(W, method="varimax")
    assert ei.value.code == "invalid_input"


# ── Spec immutability ────────────────────────────────────────


def test_rotation_spec_args_is_immutable():
    W = np.random.default_rng(13).normal(size=(15, 3))
    r = plskit.rotate(W, method="varimax")
    with pytest.raises(TypeError):
        r.spec.args["max_iter"] = 99


# ── Model-in dispatch ────────────────────────────────────────


def test_rotate_model_returns_PLS1Result_with_rotation_spec():
    X, y = _data()
    m = plskit.pls1_fit(X, y, k=3)
    m2 = plskit.rotate(m, method="varimax")
    assert isinstance(m2, plskit.PLS1Result)
    assert m2.rotation_spec is not None
    assert m2.rotation_spec.method == "varimax"
    assert m.rotation_spec is None


def test_rotate_model_predictive_invariance():
    X, y = _data()
    m = plskit.pls1_fit(X, y, k=3)
    m2 = plskit.rotate(m, method="varimax")
    # T_new @ Q_new == T_old @ Q_old (R orthogonal cancels)
    np.testing.assert_allclose(m2.T @ m2.Q, m.T @ m.Q, atol=1e-10)


def test_rotate_model_coef_unchanged():
    X, y = _data()
    m = plskit.pls1_fit(X, y, k=3)
    m2 = plskit.rotate(m, method="varimax")
    np.testing.assert_allclose(m2.coef, m.coef, atol=1e-15)
    np.testing.assert_allclose(m2.beta, m.beta, atol=1e-15)
    assert m2.intercept == m.intercept


def test_rotate_model_W_rotated_correctly():
    X, y = _data()
    m = plskit.pls1_fit(X, y, k=3)
    m2 = plskit.rotate(m, method="varimax")
    R = m2.rotation_spec.R
    np.testing.assert_allclose(m2.W, m.W @ R, atol=1e-12)
    np.testing.assert_allclose(m2.T, m.T @ R, atol=1e-12)
    np.testing.assert_allclose(m2.P, m.P @ R, atol=1e-12)


def test_rotate_model_already_rotated_raises():
    X, y = _data()
    m = plskit.pls1_fit(X, y, k=3)
    m2 = plskit.rotate(m, method="varimax")
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.rotate(m2, method="varimax")
    assert ei.value.code == "already_rotated"


# ── Bad input ─────────────────────────────────────────────────


def test_rotate_wrong_first_arg_type_raises():
    with pytest.raises(TypeError):
        plskit.rotate("not a model or array", method="varimax")
