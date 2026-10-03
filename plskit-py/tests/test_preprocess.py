import numpy as np
import pytest
import plskit


def _data(n=20, p=4):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    w = rng.uniform(0.5, 2.0, size=n)
    return X, y, w


def test_preprocess_all_none_returns_empty():
    r = plskit.preprocess()
    assert r.X_std is None
    assert r.Y_std is None
    assert r.weights_normalized is None
    assert r.n_eff is None


def test_preprocess_2d_y_round_trips_2d():
    X, y, w = _data()
    one = plskit.preprocess(X=X, Y=y, weights=w)
    two = plskit.preprocess(X=X, Y=y.reshape(-1, 1), weights=w)
    # Marshalling only: a 2-D Y comes back 2-D, with one mean and scale per
    # column. The per-column numbers are the core's contract
    # (preprocess_basic.rs); here column 0 just has to be the 1-D result.
    assert two.Y_std.shape == (20, 1)
    np.testing.assert_array_equal(two.Y_std[:, 0], one.Y_std)
    np.testing.assert_array_equal(np.ravel(two.Y_mean), [one.Y_mean])
    np.testing.assert_array_equal(np.ravel(two.Y_scale), [one.Y_scale])


def test_preprocess_2d_y_weights_length_mismatch_raises():
    X, y, w = _data()
    Y = y.reshape(-1, 1)
    with pytest.raises(plskit.PlsKitError) as exc_info:
        plskit.preprocess(X=X, Y=Y, weights=w[:-1])
    assert exc_info.value.code == "invalid_weights"
    assert exc_info.value.reason == "length_mismatch"


@pytest.mark.parametrize("two_d", [False, True])
def test_preprocess_x_y_row_mismatch_raises(two_d):
    X, y, _ = _data()
    Y = y.reshape(-1, 1) if two_d else y
    with pytest.raises(plskit.PlsKitError) as exc_info:
        plskit.preprocess(X=X[:-1], Y=Y)
    assert exc_info.value.code == "dimension_mismatch"



@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_preprocess_2d_y_non_finite_raises(bad):
    """A 2-D Y is validated like a 1-D one, not standardized into a NaN
    column."""
    X, y, _ = _data()
    Y = np.column_stack([y, y[::-1]])
    Y[3, 1] = bad
    with pytest.raises(plskit.PlsKitError) as exc_info:
        plskit.preprocess(X=X, Y=Y)
    assert exc_info.value.code == "non_finite_input"
