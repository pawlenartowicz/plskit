import numpy as np
import pytest

import plskit


def _data(n=80, d=6, k_signal=2, snr=4.0, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    beta = np.zeros(d); beta[:k_signal] = 1.0
    y = X @ beta * snr + rng.normal(size=n)
    return X, y


def test_fit_returns_PLS1Result_with_expected_shapes():
    X, y = _data()
    m = plskit.pls1_fit(X, y, k=3)
    assert isinstance(m, plskit.PLS1Result)
    assert m.T.shape == (80, 3)
    assert m.W.shape == (6, 3)
    assert m.coef.shape == (6,)
    assert m.beta.shape == (6,)
    assert m.k_used == 3
    assert not hasattr(m, "seed")
    assert not hasattr(m, "k_was_auto")
    assert not hasattr(m, "find_k_certificate")


def test_fit_pre_standardized_passes_through():
    X, y = _data()
    Xs = (X - X.mean(0)) / X.std(0)
    ys = (y - y.mean()) / y.std()
    m = plskit.pls1_fit(Xs, ys, k=2, pre_standardized=True)
    np.testing.assert_allclose(m.beta, m.coef, atol=1e-15)
    assert m.intercept == 0.0


def test_fit_dimension_mismatch_raises():
    X = np.zeros((10, 5)); y = np.zeros(9)
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.pls1_fit(X, y, k=2)
    assert ei.value.code == "dimension_mismatch"


def test_fit_k_optimal_dispatches_to_find_k_optimal():
    # Mirrors _api.py pls1_fit dispatch for k="optimal": pls1_find_k_optimal(X, y,
    # k_max, pre_standardized=False, seed=seed, weights=None, **fk_args) — no
    # selector/diagnostic/args — so the direct call here uses the same effective
    # defaults. Update both if the dispatch in _api.py changes.
    X, y = _data()
    r = plskit.pls1_find_k_optimal(X, y, k_max=4, seed=7)
    m = plskit.pls1_fit(X, y, k="optimal", k_max=4, seed=7)
    # k_star must be stable (> 1) with _data()'s snr=4 signal, so a wrong-k bug
    # would move k_used away from 2 and break the equality below.
    assert m.k_used == r.k_star
    m_fixed = plskit.pls1_fit(X, y, k=r.k_star)
    # Post-resolution both paths call the identical Fixed-k code, so the fit
    # is bit-identical.
    assert np.array_equal(m.coef, m_fixed.coef)
    assert np.array_equal(m.beta, m_fixed.beta)
    assert m.intercept == m_fixed.intercept
    assert m.k_used == m_fixed.k_used


def test_fit_k_sequence_dispatches_to_find_k_sequence():
    # Mirrors _api.py pls1_fit dispatch for k="sequence": pls1_find_k_sequence(X, y,
    # k_max, pre_standardized=False, seed=seed, weights=None, **fk_args) — no
    # test_method/alpha/args overrides — so the direct call here uses the same
    # effective defaults (test_method="split_nb", alpha=0.05). Update both if the
    # dispatch in _api.py changes.
    X, y = _data()
    r = plskit.pls1_find_k_sequence(X, y, k_max=4, seed=7)
    m = plskit.pls1_fit(X, y, k="sequence", k_max=4, seed=7)
    assert m.k_used == r.k_star
    m_fixed = plskit.pls1_fit(X, y, k=r.k_star)
    # Post-resolution both paths call the identical Fixed-k code, so the fit
    # is bit-identical.
    assert np.array_equal(m.coef, m_fixed.coef)
    assert np.array_equal(m.beta, m_fixed.beta)
    assert m.intercept == m_fixed.intercept
    assert m.k_used == m_fixed.k_used


def test_fit_string_mode_requires_k_max():
    X, y = _data()
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.pls1_fit(X, y, k="optimal")
    assert ei.value.code == "invalid_argument"
    assert "k_max" in str(ei.value)


def test_fit_unknown_string_mode_raises():
    X, y = _data()
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.pls1_fit(X, y, k="auto", k_max=4)
    assert ei.value.code == "invalid_argument"
    assert "unknown k mode" in str(ei.value)


# Keys that always live on pls1_fit itself, never inside find_k_args.
_FIND_K_FORWARDED_ON_FIT = (
    "seed", "pre_standardized", "weights",
    "disable_parallelism", "verbose",
)


def _disallowed_for(mode: str) -> tuple[str, ...]:
    """Disallowed keys for find_k_args under a given k mode: the forwarded-on-fit
    set, plus an unknown-key sentinel, minus anything the mode does allow."""
    from plskit._api import _FIND_K_ALLOWED
    allowed = set(_FIND_K_ALLOWED[mode])
    return tuple(k for k in _FIND_K_FORWARDED_ON_FIT + ("bogus_key",)
                 if k not in allowed)


def test_fit_find_k_args_rejects_disallowed_keys():
    X, y = _data()
    for key in _disallowed_for("optimal"):
        with pytest.raises(plskit.PlsKitError) as ei:
            plskit.pls1_fit(
                X, y, k="optimal", k_max=4,
                find_k_args={key: 0},
            )
        assert ei.value.code == "invalid_args"
        assert key in str(ei.value)
        assert "allowed" in str(ei.value)


def test_fit_find_k_args_rejects_disallowed_keys_sequence():
    X, y = _data()
    # `selector` and `diagnostic` are valid for `optimal` but not for `sequence`.
    for key in _disallowed_for("sequence") + ("selector", "diagnostic"):
        with pytest.raises(plskit.PlsKitError) as ei:
            plskit.pls1_fit(
                X, y, k="sequence", k_max=4,
                find_k_args={key: 0},
            )
        assert ei.value.code == "invalid_args"
        assert key in str(ei.value)


def test_fit_find_k_args_threaded_through():
    X, y = _data()
    m = plskit.pls1_fit(
        X, y, k="optimal", k_max=4,
        find_k_args={"selector": "r2_max", "args": {"n_folds": 3}},
        seed=7,
    )
    assert 1 <= m.k_used <= 4
