"""Integration tests for pls1_rotation_stability."""
import numpy as np
import pytest

from plskit import (
    CIScalar,
    PlsKitError,
    RotationStabilityResult,
    pls1_rotation_stability,
)


def _synth(n=100, d=6, snr=4.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    beta = np.array([1.0] * 2 + [0.0] * (d - 2))
    y = x @ beta * snr + rng.standard_normal(n)
    return x, y


def test_none_defaults_are_recorded_as_resolved_values():
    """None (the public default for n_boot/m_rate/level) must resolve to the
    engine's own default and be recorded on the result, never as None."""
    x, y = _synth()
    out = pls1_rotation_stability(x, y, k=2, seed=7)
    assert isinstance(out, RotationStabilityResult)
    assert isinstance(out.variance_ratio, CIScalar)
    assert isinstance(out.degenerate_baseline, bool)
    assert out.n_boot == 1000
    assert out.m_rate == pytest.approx(0.7)
    assert out.level == pytest.approx(0.95)


def test_L_and_rotation_args_reach_the_engine():
    # L sets the loading basis and kaiser_normalize changes the rotation, so
    # each moves the rotated variance; a dropped argument would not.
    x, y = _synth()
    base = pls1_rotation_stability(x, y, k=2, n_boot=200, seed=7)
    with_L = pls1_rotation_stability(x, y, k=2, L=np.eye(6, 2), n_boot=200, seed=7)
    with_args = pls1_rotation_stability(
        x, y, k=2,
        rotation_args={"max_iter": 30, "tol": 1e-6, "kaiser_normalize": False},
        n_boot=200, seed=7,
    )
    assert with_L.variance_rot != base.variance_rot
    assert with_args.variance_rot != base.variance_rot
    assert with_args.method == "varimax"


@pytest.mark.parametrize(
    "kwargs, code, message",
    [
        ({"rotation_args": {"max_iter": "x"}}, "invalid_args",
         r"args\['max_iter'\] for method='varimax'"),
        ({"rotation_args": {"tol": "x"}}, "invalid_args", "must be a number"),
        ({"rotation_args": [1]}, "invalid_argument",
         "rotation_args must be a record of named values"),
        ({"n_boot": "x"}, "invalid_argument", "n_boot must be a non-negative whole number"),
        ({"k": -1}, "invalid_argument", "k must be a non-negative whole number"),
        ({"m_rate": "x"}, "invalid_argument", "m_rate must be a number"),
    ],
)
def test_unusable_values_raise_coded_errors(kwargs, code, message):
    """Unusable `rotation_args` and top-level values raise coded errors
    before anything runs."""
    x, y = _synth()
    kwargs = {"k": 2, **kwargs}
    with pytest.raises(PlsKitError, match=message) as excinfo:
        pls1_rotation_stability(x, y, **kwargs)
    assert excinfo.value.code == code


def test_unknown_rotation_method_rejected():
    x, y = _synth()
    with pytest.raises(PlsKitError) as excinfo:
        pls1_rotation_stability(x, y, k=2, rotation_method="promax",
                                n_boot=200, seed=7)
    assert excinfo.value.code == "rotation_method_not_implemented"


def test_rejects_m_less_than_k_plus_2():
    # n=20, k=4, m_rate=0.51 → m = 5 < k+2 = 6; rotation_stability rejects.
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 6))
    y = rng.standard_normal(20)
    with pytest.raises(PlsKitError) as excinfo:
        pls1_rotation_stability(x, y, k=4, n_boot=200, m_rate=0.51, seed=7)
    assert excinfo.value.code == "invalid_argument"
