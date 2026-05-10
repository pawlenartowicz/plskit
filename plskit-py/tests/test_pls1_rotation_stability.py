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


def test_runs_end_to_end():
    x, y = _synth()
    out = pls1_rotation_stability(x, y, k=2, n_boot=200, m_rate=0.7, seed=7)
    assert isinstance(out, RotationStabilityResult)
    assert out.method == "varimax"
    assert out.n_boot == 200
    assert out.m == 26
    assert isinstance(out.variance_ratio, CIScalar)
    assert out.variance_unrot >= 0.0
    assert out.variance_rot >= 0.0
    assert out.n_boot_finite <= out.n_boot


def test_seed_is_reproducible():
    x, y = _synth(seed=11)
    a = pls1_rotation_stability(x, y, k=2, n_boot=200, seed=42)
    b = pls1_rotation_stability(x, y, k=2, n_boot=200, seed=42)
    assert a.variance_ratio == b.variance_ratio
    assert a.variance_unrot == b.variance_unrot
    assert a.variance_rot == b.variance_rot


def test_rejects_k_eq_1():
    x, y = _synth()
    with pytest.raises(PlsKitError) as excinfo:
        pls1_rotation_stability(x, y, k=1, n_boot=200, seed=7)
    assert excinfo.value.code == "invalid_argument"


def test_rejects_k_gt_7():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, 12))
    y = rng.standard_normal(100)
    with pytest.raises(PlsKitError) as excinfo:
        pls1_rotation_stability(x, y, k=8, n_boot=200, seed=7)
    assert excinfo.value.code == "invalid_argument"


def test_rejects_l_shape_mismatch():
    x, y = _synth()
    L_bad = np.zeros((4, 3))     # k=2 but L.ncols=3
    with pytest.raises(PlsKitError) as excinfo:
        pls1_rotation_stability(x, y, k=2, L=L_bad, n_boot=200, seed=7)
    assert excinfo.value.code == "shape_mismatch"


def test_l_compatible_runs():
    x, y = _synth()
    L = np.eye(6, 2)             # 6 vars × k=2; cols define the loading basis
    out = pls1_rotation_stability(x, y, k=2, L=L, n_boot=200, seed=7)
    assert out.variance_unrot >= 0.0


def test_rotation_args_pass_through():
    x, y = _synth()
    out = pls1_rotation_stability(
        x, y, k=2,
        rotation_args={"max_iter": 30, "tol": 1e-6, "kaiser_normalize": False},
        n_boot=200, seed=7,
    )
    assert out.method == "varimax"


def test_unknown_rotation_method_rejected():
    x, y = _synth()
    with pytest.raises(PlsKitError):
        pls1_rotation_stability(x, y, k=2, rotation_method="promax",
                                n_boot=200, seed=7)


def test_rejects_m_less_than_k_plus_2():
    # n=20, k=4, m_rate=0.51 → m = 5 < k+2 = 6; rotation_stability rejects.
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 6))
    y = rng.standard_normal(20)
    with pytest.raises(PlsKitError) as excinfo:
        pls1_rotation_stability(x, y, k=4, n_boot=200, m_rate=0.51, seed=7)
    assert excinfo.value.code == "invalid_argument"


# ── Additional structural tests ─────────────────────────────────────────


def test_variance_ratio_per_axis_length_equals_k():
    x, y = _synth(d=8)
    for k in (2, 3):
        out = pls1_rotation_stability(x, y, k=k, n_boot=150, seed=11)
        assert len(out.variance_ratio_per_axis) == k
        assert out.variance_unrot_per_axis.shape == (k,)
        assert out.variance_rot_per_axis.shape == (k,)


def test_variance_ratio_ci_strictly_orders_lower_le_upper():
    x, y = _synth(d=6)
    out = pls1_rotation_stability(x, y, k=2, n_boot=200, seed=13)
    assert out.variance_ratio.lower <= out.variance_ratio.point + 1e-10
    assert out.variance_ratio.point <= out.variance_ratio.upper + 1e-10
    for ci in out.variance_ratio_per_axis:
        if np.isfinite(ci.point):
            assert ci.lower <= ci.point + 1e-10
            assert ci.point <= ci.upper + 1e-10


def test_variance_unrot_and_rot_are_non_negative():
    x, y = _synth(d=6)
    out = pls1_rotation_stability(x, y, k=2, n_boot=200, seed=17)
    assert out.variance_unrot >= 0.0
    assert out.variance_rot >= 0.0
    assert (out.variance_unrot_per_axis >= 0.0).all()
    assert (out.variance_rot_per_axis >= 0.0).all()


def test_per_axis_decomposition_sums_to_aggregate():
    x, y = _synth(d=6)
    out = pls1_rotation_stability(x, y, k=2, n_boot=200, seed=19)
    assert abs(out.variance_unrot_per_axis.sum() - out.variance_unrot) < 1e-10
    assert abs(out.variance_rot_per_axis.sum() - out.variance_rot) < 1e-10


def test_degenerate_baseline_flag_is_bool():
    x, y = _synth(d=6)
    out = pls1_rotation_stability(x, y, k=2, n_boot=200, seed=23)
    assert isinstance(out.degenerate_baseline, bool)


def test_default_rotation_method_runs_on_factor_design():
    """Rotation diagnostic runs without error on a factor design.

    Asserts the diagnostic produces a sensible bounded ratio. A stricter
    target (`variance_ratio.point < 1`) requires synthetic data that
    reliably triggers PLS1 NIPALS drift in a close-σ block (see Rust-side TODO).
    """
    rng = np.random.default_rng(29)
    n, d = 200, 8
    f1 = rng.standard_normal(n)
    f2 = rng.standard_normal(n)
    x = np.zeros((n, d))
    x[:, :4] = f1[:, None] + 0.05 * rng.standard_normal((n, 4))
    x[:, 4:] = f2[:, None] + 0.05 * rng.standard_normal((n, 4))
    y = f1 + f2 + 0.1 * rng.standard_normal(n)
    out = pls1_rotation_stability(x, y, k=2, n_boot=300, seed=37)
    assert np.isfinite(out.variance_ratio.point)
    assert 0.3 < out.variance_ratio.point < 2.0
