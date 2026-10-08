import numpy as np
import pytest
import plskit


def _data(n=80, d=6, k_signal=2, snr=4.0, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    beta = np.zeros(d); beta[:k_signal] = 1.0
    y = X @ beta * snr + rng.normal(size=n)
    return X, y


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"selector": "bic", "args": {"n_folds": 5}},
         "does not accept arg 'n_folds'"),
        ({"selector": "r2_se", "args": {"n_splits": 30}},
         "requires diagnostic to be set"),
        ({"selector": "bogus"}, "unknown selector: bogus"),
        ({"args": {"n_folds": "x"}},
         r"args\['n_folds'\] for method='optimal' must be a non-negative whole number"),
    ],
)
def test_optimal_seam_rejects_args_the_selection_cannot_use(kwargs, message):
    # The checks and their messages come from plskit-bind.
    X, y = _data()
    with pytest.raises(plskit.PlsKitError, match=message) as ei:
        plskit.pls1_find_k_optimal(X, y, k_max=4, seed=7, **kwargs)
    assert ei.value.code == "invalid_args"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"k_max": -1}, "k_max must be a non-negative whole number"),
        ({"selector": 1}, "selector must be a string, got 1"),
        ({"diagnostic": 2}, "diagnostic must be a string"),
        ({"args": [1]}, "args must be a record of named values"),
        ({"verbose": 1}, "verbose must be a bool"),
    ],
)
def test_optimal_rejects_unusable_top_level_values(kwargs, message):
    X, y = _data()
    kwargs = {"k_max": 4, **kwargs}
    with pytest.raises(plskit.PlsKitError, match=message) as ei:
        plskit.pls1_find_k_optimal(X, y, seed=7, **kwargs)
    assert ei.value.code == "invalid_argument"


def test_optimal_selector_none_is_the_default():
    """None for an optional argument means its default, as in plskit-bind."""
    X, y = _data()
    assert plskit.pls1_find_k_optimal(X, y, 3, selector=None, seed=7).selector == "r2_se"


@pytest.mark.parametrize("fn", ["pls1_find_k_optimal", "spls1_find_k_optimal"])
def test_optimal_diagnostic_rejects_auto(fn):
    X, y = _data()
    extra = (3,) if fn.startswith("spls1") else ()
    with pytest.raises(plskit.PlsKitError) as ei:
        getattr(plskit, fn)(X, y, 4, *extra, diagnostic="auto", seed=7)
    assert ei.value.code == "invalid_argument"
    assert str(ei.value) == "invalid argument: diagnostic does not accept 'auto'"
