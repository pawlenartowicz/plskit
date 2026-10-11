"""Integration tests for pls1_perm_alpha."""
import numpy as np
import pytest

import plskit
from plskit import PermAlphaResult, PlsKitError, pls1_perm_alpha


def _design(n=40, p=500, seed=0):
    """Raw X, its standardized block, the block's Gram matrix, three outcomes."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    Xs = plskit.preprocess(X).X_std
    Y = 2.0 * Xs[:, :3] + rng.standard_normal((n, 3))
    return X, Xs, Xs @ Xs.T, Y


def test_alpha_maps_onto_pls1_fit_coef():
    X, Xs, G, Y = _design()
    r = pls1_perm_alpha(G, Y, n_features=X.shape[1])
    assert isinstance(r, PermAlphaResult)
    assert r.alpha.shape == (40, 3)
    assert r.k_used.dtype == np.int64 and r.resolved.dtype == np.bool_
    assert r.resolved.all() and (r.k_used == 1).all()
    assert (r.k, r.n_features) == (1, 500)
    for j in range(3):
        coef = plskit.pls1_fit(X, Y[:, j], k=1).coef
        np.testing.assert_allclose(
            Xs.T @ r.alpha[:, j], coef, rtol=0,
            atol=1e-10 * max(1.0, np.abs(coef).max()),
        )


def test_an_all_zero_outcome_is_the_zero_model():
    X, _, G, Y = _design()
    Y[:, 1] = 0.0
    r = pls1_perm_alpha(G, Y, n_features=X.shape[1])
    assert r.resolved.tolist() == [True, True, True]
    assert r.k_used.tolist() == [1, 0, 1]
    assert (r.alpha[:, 1] == 0.0).all()
    assert not np.signbit(r.alpha[:, 1]).any()


def test_an_outcome_orthogonal_to_the_block_is_unresolved():
    rng = np.random.default_rng(1)
    n, p = 40, 10
    X = rng.standard_normal((n, p))
    Xs = plskit.preprocess(X).X_std
    # An outcome orthogonal to the ones vector and to every column of Xs.
    Q, _ = np.linalg.qr(np.column_stack([np.ones(n), Xs]))
    e = 5.0 + rng.standard_normal(n)
    for _ in range(2):
        e -= Q @ (Q.T @ e)
    Y = np.column_stack([2.0 * Xs[:, 0] + rng.standard_normal(n), e])
    r = pls1_perm_alpha(Xs @ Xs.T, Y, n_features=p)
    assert r.k_used.dtype == np.int64 and r.resolved.dtype == np.bool_
    assert r.resolved.tolist() == [True, False]
    assert r.k_used.tolist() == [1, 0]
    assert np.isnan(r.alpha[:, 1]).all()


@pytest.mark.parametrize(
    "change, code, message",
    [
        (lambda G, Y: (G, Y, {"k": 2}), "invalid_argument", "supports k = 1 only"),
        (lambda G, Y: (G + np.triu(np.full_like(G, 1e-3), 1), Y, {}),
         "invalid_input", "not symmetric"),
        (lambda G, Y: (G, Y[:, 0], {}), "invalid_argument", "Y must be 2-D"),
    ],
)
def test_bad_input_raises_its_code(change, code, message):
    X, _, G, Y = _design()
    G2, Y2, extra = change(G, Y)
    with pytest.raises(PlsKitError, match=message) as excinfo:
        pls1_perm_alpha(G2, Y2, n_features=X.shape[1], **extra)
    assert excinfo.value.code == code


def test_a_gram_with_a_zero_diagonal_and_nonzero_entries_is_invalid_input():
    # Hollow, symmetric, zero row sums: only the Cauchy-Schwarz check
    # rejects it.
    hollow = np.array([
        [0.0, 1.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0, 1.0],
        [0.0, -1.0, 1.0, 0.0],
    ])
    Y = np.array([[1.0], [1.0], [-1.0], [-1.0]])
    with pytest.raises(
        PlsKitError,
        match=r"\|gram\[1, 0\]\| = 1e0 exceeds sqrt\(gram\[1, 1\] \* gram\[0, 0\]\)",
    ) as excinfo:
        pls1_perm_alpha(hollow, Y, n_features=1)
    assert excinfo.value.code == "invalid_input"


def test_n_features_is_keyword_only():
    X, _, G, Y = _design()
    with pytest.raises(TypeError, match="positional arguments"):
        pls1_perm_alpha(G, Y, 1, X.shape[1])
