"""spls1 sparse-PLS1 family: wrapper defaults, arg validation, model round trip."""

import numpy as np
import pytest

import plskit
from plskit import PlsKitError


def synth(n, d, k_signal, snr, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = X[:, :k_signal].sum(axis=1) * snr + rng.standard_normal(n)
    return X, y


def test_spls1_fit_predict_roundtrip():
    X, y = synth(60, 10, 2, 4.0, 3)
    r = plskit.spls1_fit(X, y, 2, 4)
    yhat = plskit.pls1_predict(r, X)
    np.testing.assert_allclose(yhat, X @ r.beta + r.intercept, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("args", [{"bogus": 1}, {"n_folds": -1}])
def test_spls1_find_keep_optimal_rejects_unusable_args(args):
    X, y = synth(40, 5, 1, 4.0, 8)
    with pytest.raises(PlsKitError) as ei:
        plskit.spls1_find_keep_optimal(X, y, 1, args=args)
    assert ei.value.code == "invalid_args"


@pytest.mark.parametrize(
    "call",
    [
        lambda X, y: plskit.spls1_fit(X, y, 2.5, 3),
        lambda X, y: plskit.spls1_fit(X, y, 2, 3.5),
        lambda X, y: plskit.spls1_find_keep_optimal(X, y, 1.5),
        lambda X, y: plskit.spls1_find_k_optimal(X, y, 3, 2.5),
        lambda X, y: plskit.spls1_find_k_sequence(X, y, 3, 2.5),
        lambda X, y: plskit.spls1_find_k_optimal(X, y, -1, 2),
        lambda X, y: plskit.spls1_find_keep_optimal(X, y, 1, seed=-1),
        lambda X, y: plskit.spls1_fit(X, y, 2, 3, pre_standardized=1),
    ],
)
def test_spls1_family_rejects_unusable_top_level_values(call):
    X, y = synth(60, 6, 2, 4.0, 11)
    with pytest.raises(PlsKitError) as ei:
        call(X, y)
    assert ei.value.code == "invalid_argument"
