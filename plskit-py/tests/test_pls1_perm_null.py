"""Integration tests for pls1_perm_null."""
import numpy as np
import pytest

from plskit import PermNullResult, PlsKitError, pls1_perm_null


def _synth(n=100, d=6, snr=4.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    beta = np.array([1.0] * 2 + [0.0] * (d - 2))
    y = x @ beta * snr + rng.standard_normal(n)
    return x, y


def test_returns_perm_null_result():
    x, y = _synth()
    r = pls1_perm_null(x, y, k=2, n_perm=200, seed=7)
    assert isinstance(r, PermNullResult)
    assert r.n_perm == 200
    assert r.k == 2
    assert r.beta_ref.shape == (6,)
    assert r.beta_perm_mean.shape == (6,)
    assert r.beta_perm_sd.shape == (6,)
    assert r.beta_perm_z.shape == (6,)
    assert r.beta_perm_matrix is None


def test_return_perm_matrix_shape():
    x, y = _synth()
    r = pls1_perm_null(x, y, k=2, n_perm=200, return_perm_matrix=True, seed=7)
    assert r.beta_perm_matrix is not None
    assert r.beta_perm_matrix.shape == (200, 6)


def test_signal_voxels_have_higher_abs_z():
    x, y = _synth(n=200, d=8, snr=6.0, seed=11)
    r = pls1_perm_null(x, y, k=2, n_perm=400, seed=7)
    sig = np.abs(r.beta_perm_z[:2])
    noi = np.abs(r.beta_perm_z[2:])
    assert sig.mean() > noi.mean()


def test_signed_z_matches_beta_ref_sign():
    x, y = _synth(n=200, d=8, snr=6.0, seed=11)
    r = pls1_perm_null(x, y, k=2, n_perm=400, seed=7)
    finite = np.isfinite(r.beta_perm_z) & (np.abs(r.beta_ref) > 1e-12)
    np.testing.assert_array_equal(
        np.sign(r.beta_perm_z[finite]),
        np.sign(r.beta_ref[finite]),
    )


def test_seed_is_reproducible():
    x, y = _synth(n=120, seed=42)
    r1 = pls1_perm_null(x, y, k=2, n_perm=200, seed=99)
    r2 = pls1_perm_null(x, y, k=2, n_perm=200, seed=99)
    # beta_ref is the deterministic full-data fit — byte-exact across runs.
    np.testing.assert_array_equal(r1.beta_ref, r2.beta_ref)
    # Streaming Welford under Rayon work-stealing yields different merge trees
    # across runs → roundoff-equivalent only (matches Rust streaming-vs-retained
    # tolerance). Disable parallelism for byte-exact reproducibility.
    np.testing.assert_allclose(r1.beta_perm_z, r2.beta_perm_z, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(r1.beta_perm_sd, r2.beta_perm_sd, rtol=1e-12, atol=1e-12)


def test_streaming_matches_retained():
    x, y = _synth(n=120, seed=42)
    r_ret = pls1_perm_null(x, y, k=2, n_perm=200, return_perm_matrix=True, seed=99)
    r_str = pls1_perm_null(x, y, k=2, n_perm=200, return_perm_matrix=False, seed=99)
    np.testing.assert_allclose(r_ret.beta_perm_sd, r_str.beta_perm_sd, rtol=1e-12)
    np.testing.assert_allclose(r_ret.beta_perm_mean, r_str.beta_perm_mean, rtol=1e-12)


def test_rejects_low_n_perm():
    x, y = _synth()
    with pytest.raises(PlsKitError) as excinfo:
        pls1_perm_null(x, y, k=2, n_perm=50, seed=7)
    assert excinfo.value.code == "invalid_argument"


def test_rejects_dim_mismatch():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 5))
    y = rng.standard_normal(19)
    with pytest.raises(PlsKitError) as excinfo:
        pls1_perm_null(x, y, k=2, n_perm=200, seed=7)
    assert excinfo.value.code == "dimension_mismatch"


def test_rejects_k_exceeds_max():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 4))
    y = rng.standard_normal(20)
    with pytest.raises(PlsKitError) as excinfo:
        pls1_perm_null(x, y, k=5, n_perm=200, seed=7)
    assert excinfo.value.code == "k_exceeds_max"


def test_perm_null_with_weights_smoke():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 4))
    y = X[:, 0] + rng.normal(size=60)
    w = rng.uniform(0.5, 2.0, size=60)
    r = pls1_perm_null(X, y, k=2, n_perm=200, weights=w, seed=0)
    assert r.beta_perm_z.shape == (4,)


def test_perm_null_uniform_weights_invariance():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(60, 4))
    y = X[:, 0] + rng.normal(size=60)
    r_w = pls1_perm_null(X, y, k=2, n_perm=200, weights=np.ones(60), seed=0)
    r_n = pls1_perm_null(X, y, k=2, n_perm=200, seed=0)
    np.testing.assert_allclose(r_w.beta_perm_z, r_n.beta_perm_z, atol=1e-10)
