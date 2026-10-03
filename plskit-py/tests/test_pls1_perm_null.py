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


def test_none_n_perm_is_recorded_as_resolved_value():
    """None (the public default for n_perm) must resolve to the engine's
    own default and be recorded on result.n_perm, never as None."""
    x, y = _synth()
    r = pls1_perm_null(x, y, k=2, seed=7)
    assert isinstance(r, PermNullResult)
    assert r.n_perm == 1000


def test_return_perm_matrix_shape():
    x, y = _synth()
    r = pls1_perm_null(x, y, k=2, n_perm=200, return_perm_matrix=True, seed=7)
    assert r.beta_perm_matrix is not None
    assert r.beta_perm_matrix.shape == (200, 6)
    assert pls1_perm_null(x, y, k=2, n_perm=200, seed=7).beta_perm_matrix is None


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_perm": -5}, "n_perm must be a non-negative whole number"),
        ({"return_perm_matrix": 1}, "return_perm_matrix must be a bool, got 1"),
        ({"seed": "7"}, 'seed must be a whole number in \\[0, 2\\^64\\), got the string "7"'),
    ],
)
def test_unusable_top_level_values_raise_invalid_argument(kwargs, message):
    x, y = _synth()
    with pytest.raises(PlsKitError, match=message) as excinfo:
        pls1_perm_null(x, y, 1, **kwargs)
    assert excinfo.value.code == "invalid_argument"
