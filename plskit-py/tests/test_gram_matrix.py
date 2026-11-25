"""Integration tests for gram_matrix and GramMatrixResult.subset."""
import numpy as np
import pytest

from plskit import (
    GramMatrixResult,
    PlsKitError,
    gram_matrix,
    gram_matrix_blocks,
    pls1_confirmatory_test,
)


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
    for bad, msg in (
        ([0, 30], "rows must index the 30 rows of the Gram matrix"),
        ([-1, 2], "rows must index the 30 rows of the Gram matrix"),
        ([2**70, 1], "rows must index the 30 rows of the Gram matrix"),
        ([4], "rows must hold at least 2 rows, got 1"),
        ([2**70], "rows must hold at least 2 rows, got 1"),
    ):
        with pytest.raises(PlsKitError) as e:
            r.subset(bad)
        assert str(e.value) == msg, bad


def _blocks(X, cuts):
    return (X[:, a:b] for a, b in zip(cuts, cuts[1:]))


@pytest.mark.parametrize("flag", [False, True])
def test_gram_matrix_blocks_equals_gram_matrix(flag):
    X = _raw_x(30, 40)
    r = gram_matrix_blocks(_blocks(X, [0, 5, 13, 40]), pre_standardized=flag)
    want = gram_matrix(X, pre_standardized=flag).gram
    assert (r.n_features, r.pre_standardized) == (40, flag)
    np.testing.assert_allclose(r.gram, want, rtol=0, atol=1e-12 * np.abs(want).max())


def test_gram_matrix_blocks_one_block():
    X = _raw_x(30, 12)
    r = gram_matrix_blocks(iter([X]))
    assert r.n_features == 12
    assert np.array_equal(r.gram, gram_matrix(X).gram)


def test_gram_matrix_blocks_float32_and_c_order():
    X = _raw_x(30, 40)
    blocks = (np.ascontiguousarray(b, dtype=np.float32) for b in _blocks(X, [0, 20, 40]))
    X32 = np.hstack([np.asarray(b, dtype=np.float32) for b in _blocks(X, [0, 20, 40])]).astype(np.float64)
    want = gram_matrix(X32).gram
    r = gram_matrix_blocks(blocks)
    np.testing.assert_allclose(r.gram, want, rtol=0, atol=1e-12 * np.abs(want).max())


def test_gram_matrix_blocks_as_gram():
    X = _raw_x(30, 40)
    y = np.random.default_rng(1).standard_normal(30)
    r = gram_matrix_blocks(_blocks(X, [0, 17, 40]))
    a = pls1_confirmatory_test(gram=r, y=y, seed=1)
    b = pls1_confirmatory_test(X, y, seed=1)
    assert a.pvalue == b.pvalue


def test_gram_matrix_blocks_errors():
    X = _raw_x(30, 12)
    with pytest.raises(PlsKitError) as e:
        gram_matrix_blocks([])
    assert e.value.code == "invalid_input"
    assert str(e.value) == "blocks holds no block"
    with pytest.raises(PlsKitError) as e:
        gram_matrix_blocks([X[:4], X[:3]])
    assert e.value.code == "shape_mismatch"
    assert str(e.value) == "block has 3 rows, the first block has 4"
    with pytest.raises(PlsKitError) as e:
        gram_matrix_blocks([X, X[:29]])
    assert e.value.code == "shape_mismatch"
    bad = X.copy()
    bad[0, 0] = np.nan
    with pytest.raises(PlsKitError) as e:
        gram_matrix_blocks([X, bad])
    assert e.value.code == "non_finite_input"
