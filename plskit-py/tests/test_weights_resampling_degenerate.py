import numpy as np
import pytest
import plskit


def _pathological_setup():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(60, 5))
    y = X[:, 0] + 0.2 * rng.normal(size=60)
    # Concentrated weights: 5 rows hold most of the weight; subsamples drawn
    # without enough of those 5 rows fail the n_eff_sub >= k+1 check.
    w = np.full(60, 1e-8)
    w[:5] = 1.0
    return X, y, w


def test_rotation_stability_default_threshold_fires():
    X, y, w = _pathological_setup()
    # rotation_method="oblimin" is not implemented; "varimax" is the only
    # supported method — the test exercises the weight-degeneracy path, not
    # the rotation method itself.
    with pytest.raises(plskit.PlsKitResamplingDegenerate) as exc_info:
        plskit.pls1_rotation_stability(
            X, y, k=3, rotation_method="varimax", weights=w, n_boot=500, seed=0,
        )
    err = exc_info.value
    assert err.threshold == 0.01
    assert err.skipped > 0
    assert err.total == 500
    assert err.skip_rate > 0.01


def test_rotation_stability_max_skip_rate_one_returns_truncated():
    X, y, w = _pathological_setup()
    r = plskit.pls1_rotation_stability(
        X, y, k=3, rotation_method="varimax", weights=w,
        n_boot=500, seed=0, max_skip_rate=1.0,
    )
    assert r is not None


def test_confirmatory_ci_default_threshold_fires():
    X, y, w = _pathological_setup()
    with pytest.raises(plskit.PlsKitResamplingDegenerate):
        plskit.pls1_confirmatory_test(
            X, y, k=3, method="raw_perm",
            args={"n_perm": 200},
            ci=True, weights=w, n_boot=500, seed=0,
        )
