import numpy as np
import pytest
import plskit


def test_rotation_stability_with_weights():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(60, 5))
    y = X[:, 0] + 0.2 * rng.normal(size=60)
    w = rng.uniform(0.5, 2.0, size=60)
    r = plskit.pls1_rotation_stability(
        X, y, k=2, weights=w, n_boot=200, seed=0,
    )
    assert r is not None
    assert 0 < r.n_eff <= 60


def test_rotation_stability_max_skip_rate_default():
    # 5 observations have substantial weight; the rest are near-zero.
    # Full-data n_eff ≈ 5 >= k+1=4 — passes the entry-level check.
    # Each subsample of m≈18 (m=ceil(60^0.7)) will typically contain
    # 0-2 of those 5 observations, giving n_eff_sub << 4, so almost
    # all subsamples fail — skip_rate >> max_skip_rate=0.01.
    rng = np.random.default_rng(6)
    X = rng.normal(size=(60, 5))
    y = X[:, 0] + 0.2 * rng.normal(size=60)
    w = np.full(60, 1e-8)
    w[:5] = 1.0
    with pytest.raises(plskit.PlsKitResamplingDegenerate) as exc_info:
        plskit.pls1_rotation_stability(
            X, y, k=3, weights=w, n_boot=200, seed=0,
        )
    assert exc_info.value.threshold == 0.01


def test_rotation_stability_max_skip_rate_one_accepts_truncation():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(60, 5))
    y = X[:, 0] + 0.2 * rng.normal(size=60)
    w = np.full(60, 1e-8)
    w[:5] = 1.0
    r = plskit.pls1_rotation_stability(
        X, y, k=3, weights=w, n_boot=200, seed=0, max_skip_rate=1.0,
    )
    assert r is not None
