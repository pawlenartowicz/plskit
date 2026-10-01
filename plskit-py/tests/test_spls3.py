import numpy as np
import pytest

import plskit


def _blocks(n=200, p=12, q=4, seed=0):
    # Two planted factors: the first half of each block's columns loads on f1,
    # the second half on f2. The split is `p // 2` / `p - p // 2`, so the blocks
    # are exactly p and q columns wide for any p, q (the defaults 12 and 4 give
    # 6 + 6 and 2 + 2).
    rng = np.random.default_rng(seed)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    X = np.column_stack(
        [f1 + 0.1 * rng.normal(size=n) for _ in range(p // 2)]
        + [f2 + 0.1 * rng.normal(size=n) for _ in range(p - p // 2)]
    )
    Y = np.column_stack(
        [f1 + 0.1 * rng.normal(size=n) for _ in range(q // 2)]
        + [f2 + 0.1 * rng.normal(size=n) for _ in range(q - q // 2)]
    )
    return X, Y


def test_convergence_metadata_shapes():
    X, Y = _blocks()
    m = plskit.spls3_fit(X, Y, k=2, keep_X=4, keep_Y=2)
    # Without this the two shape checks are self-referential and hold for any
    # k_used, including 0.
    assert m.k_used == 2
    assert m.converged.shape == (m.k_used,)
    assert m.n_iter.shape == (m.k_used,)
    assert m.converged.dtype == bool


def test_max_iter_hit_reports_false_not_error():
    X, Y = _blocks()
    m = plskit.spls3_fit(X, Y, k=1, keep_X=4, keep_Y=2, max_iter=1)
    assert m.converged.tolist() == [False]
    assert m.n_iter.tolist() == [1]


@pytest.mark.parametrize("tol", [float("nan"), float("inf"), -1e-8])
def test_bad_tol_raises_off_the_dense_endpoint(tol):
    X, Y = _blocks()
    with pytest.raises(plskit.PlsKitError) as e:
        plskit.spls3_fit(X, Y, k=1, keep_X=4, keep_Y=2, tol=tol)
    assert e.value.code == "invalid_argument"
    # At the dense endpoint no alternation runs, so `tol` is never read.
    plskit.spls3_fit(X, Y, k=1, keep_X=X.shape[1], keep_Y=Y.shape[1], tol=tol)


def test_transform_round_trips_a_sparse_model():
    X, Y = _blocks()
    m = plskit.spls3_fit(X, Y, k=2, keep_X=4, keep_Y=2)
    assert m.k_used == 2
    s = plskit.pls3_transform(m, X, Y, which="both")
    np.testing.assert_allclose(s.x_scores, m.x_scores, atol=1e-10)


def test_weights_are_rejected():
    X, Y = _blocks()
    with pytest.raises(plskit.PlsKitError) as e:
        plskit.spls3_fit(X, Y, k=1, keep_X=4, keep_Y=2, weights=np.ones(X.shape[0]))
    assert e.value.code == "invalid_argument"


def test_keeps_are_positional_like_spls1_fit():
    # `keep_X` / `keep_Y` are positional, matching `spls1_fit`'s `keep`, the
    # Rust signature `spls3_fit(x, y, k, keep_x, keep_y, ...)` and the
    # `**arguments:**` row in `_docs/python/api.md` §2c.3. Under a
    # keyword-only spelling the first call below is a TypeError.
    X, Y = _blocks()
    positional = plskit.spls3_fit(X, Y, 2, 4, 2)
    keyword = plskit.spls3_fit(X, Y, k=2, keep_X=4, keep_Y=2)
    assert positional.k_used == 2
    np.testing.assert_array_equal(positional.U, keyword.U)
    np.testing.assert_array_equal(positional.V, keyword.V)


def test_k_used_truncates_when_a_later_component_is_dust():
    # X = I_3 and Y = A with both blocks declared pre-standardized, so the
    # cross-covariance the fit sees is A itself, bit for bit. A has a unit
    # (0, 0) entry and a 1e-16 block in row 1: component 1 takes the (0, 0)
    # entry exactly (keep_X=1), deflation leaves only the 1e-16 block, and
    # component 2 is numerical dust (norm ~1.4e-16, two orders of magnitude
    # under the 1e-14 floor), so the fit stops at k_used = 1 < k = 2 instead
    # of reporting a noise component. Same input as the Rust unit test
    # `spls3_dust_component_truncates_k_used`.
    X = np.eye(3)
    Y = np.zeros((3, 3))
    Y[0, 0] = 1.0
    Y[1, 1] = Y[1, 2] = 1e-16
    m = plskit.spls3_fit(
        X, Y, 2, 1, 2, pre_standardized_X=True, pre_standardized_Y=True
    )
    assert m.k_used == 1
    assert m.U.shape == (3, 1)
    assert m.V.shape == (3, 1)
    assert m.singular_values.shape == (1,)
    assert m.x_scores.shape == (3, 1)
    assert m.y_scores.shape == (3, 1)
    assert m.converged.shape == (1,)
    assert m.n_iter.shape == (1,)
    # The retained component is the (0, 0) entry, sign-pinned positive.
    np.testing.assert_array_equal(m.U[:, 0], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(m.V[:, 0], [1.0, 0.0, 0.0])
    assert m.singular_values[0] == 1.0


def test_alternation_defaults_come_from_the_engine():
    # max_iter / tol default to None on the Python signature, and the engine
    # fills in Pls3FitOpts::default() (100 sweeps, tol 1e-8), so the wrapper
    # carries no copy of those numbers.
    import inspect

    params = inspect.signature(plskit.spls3_fit).parameters
    assert params["max_iter"].default is None
    assert params["tol"].default is None
    X, Y = _blocks()
    a = plskit.spls3_fit(X, Y, k=2, keep_X=4, keep_Y=2)
    b = plskit.spls3_fit(X, Y, k=2, keep_X=4, keep_Y=2, max_iter=100, tol=1e-8)
    np.testing.assert_array_equal(a.U, b.U)
    np.testing.assert_array_equal(a.V, b.V)
    np.testing.assert_array_equal(a.n_iter, b.n_iter)
    # tol=0 never converges, so n_iter reads out the resolved max_iter default.
    m = plskit.spls3_fit(X, Y, k=2, keep_X=4, keep_Y=2, tol=0.0)
    assert m.n_iter.tolist() == [100, 100]
    assert not m.converged.any()
