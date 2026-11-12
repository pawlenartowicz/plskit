"""Integration tests for gram_matrix and GramMatrixResult.subset."""
import numpy as np
import pytest

from plskit import GramMatrixResult, PlsKitError, gram_matrix


def _raw_x(n=30, p=12, seed=0):
    """Uncentred X with unequal column scales, so both flags matter."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, p)) * np.arange(1, p + 1) + 3.0


@pytest.mark.parametrize("flag", [False, True])
def test_gram_matrix_fields(flag):
    X = _raw_x()
    r = gram_matrix(X, pre_standardized=flag)
    assert isinstance(r, GramMatrixResult)
    assert (r.n_features, r.pre_standardized) == (12, flag)
    assert type(r.n_features) is int and type(r.pre_standardized) is bool
    Xt = X - X.mean(axis=0)
    if not flag:
        Xt = Xt / X.std(axis=0)
    want = Xt @ Xt.T
    assert r.gram.shape == (30, 30)
    np.testing.assert_allclose(r.gram, want, rtol=0, atol=1e-12 * np.abs(want).max())


def test_gram_matrix_subset():
    r = gram_matrix(_raw_x(), pre_standardized=True)
    rows = [7, 0, 3, 3, 29]
    s = r.subset(rows)
    assert isinstance(s, GramMatrixResult)
    assert (s.n_features, s.pre_standardized) == (12, True)
    np.testing.assert_array_equal(s.gram, r.gram[np.ix_(rows, rows)])
    np.testing.assert_array_equal(r.subset(np.array(rows, dtype=np.int32)).gram, s.gram)
    for bad in ([0, 30], [-1, 2], [4], [], [0.0, 1.0], [True, False], [[0, 1]]):
        with pytest.raises(PlsKitError) as e:
            r.subset(bad)
        assert e.value.code == "invalid_argument", bad
