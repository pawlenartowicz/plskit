"""find_k on a y with no extractable first component.

A constant y, and a y residualized on [1, X] (orthogonal to the standardized
columns of X up to rounding), both make the full-data fit return the
k_used=0 zero model. Every find_k entry point must then report k_star=0
rather than recommend a component the fit cannot produce.
"""
import warnings

import numpy as np
import pytest
import plskit


def _x(n=60, d=8, seed=5):
    return np.random.default_rng(seed).normal(size=(n, d))


def _degenerate_ys(X, seed=5):
    n = X.shape[0]
    e = np.random.default_rng(seed + 1).normal(size=n)
    A = np.column_stack([np.ones(n), X])
    y_orth = e - A @ np.linalg.lstsq(A, e, rcond=None)[0]
    return {"constant": np.full(n, 3.0), "orthogonal": y_orth}


X = _x()
YS = _degenerate_ys(X)


@pytest.mark.parametrize("name", sorted(YS))
def test_premise_fit_is_zero_model(name):
    assert plskit.pls1_fit(X, YS[name], k=2).k_used == 0


@pytest.mark.parametrize("name", sorted(YS))
@pytest.mark.parametrize("selector", ["r2_se", "r2_max", "bic"])
@pytest.mark.parametrize("sparse", [False, True])
def test_optimal_returns_k_star_zero_with_empty_scores(name, selector, sparse):
    y = YS[name]
    if sparse:
        r = plskit.spls1_find_k_optimal(X, y, 3, 4, selector=selector, seed=3)
    else:
        r = plskit.pls1_find_k_optimal(X, y, 3, selector=selector, seed=3)
    assert r.k_star == 0
    assert r.seed == 3
    if selector == "bic":
        assert r.bic_scores == {}
        assert r.cv_scores is None
    else:
        assert r.cv_scores == {}
        assert r.bic_scores is None
        assert r.cv_scores_se == ({} if selector == "r2_se" else None)
    assert r.pvalues is None and r.diagnostic is None


@pytest.mark.parametrize("name", sorted(YS))
@pytest.mark.parametrize("k", [1, 2])
def test_keep_optimal_returns_keep_star_zero_with_empty_scores(name, k):
    r = plskit.spls1_find_keep_optimal(X, YS[name], k, seed=3)
    assert r.keep_star == 0
    assert r.k == k
    assert r.seed == 3
    assert r.cv_scores == {}
    assert r.cv_scores_se == {}
    assert r.keep_grid == []


@pytest.mark.parametrize("name", sorted(YS))
@pytest.mark.parametrize(
    "diagnostic,args,expect",
    [
        ("e", None, "e"),
        ("raw_perm", {"n_perm": 20}, "raw_perm"),
        ("split_nb", {"n_splits": 5, "force": True}, "split_nb"),
    ],
)
def test_optimal_diagnostic_at_k_star_zero_is_empty(name, diagnostic, args, expect):
    r = plskit.pls1_find_k_optimal(
        X, YS[name], 3, selector="bic", diagnostic=diagnostic, args=args, seed=7,
    )
    assert r.k_star == 0
    assert r.pvalues.shape == (0,)
    assert r.diagnostic == expect
    assert (r.stable_rank is not None) == (diagnostic == "split_nb")


@pytest.mark.parametrize("name", sorted(YS))
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize(
    "test_method,args",
    [
        ("e", None),
        ("raw_perm", {"n_perm": 50}),
        ("split_exact", {"n_perm": 50, "n_splits": 5}),
        ("split_nb", {"n_splits": 10, "force": True}),
    ],
)
def test_sequence_returns_k_star_zero(name, sparse, test_method, args):
    y = YS[name]
    kw = dict(test_method=test_method, args=args, seed=3)
    if sparse:
        r = plskit.spls1_find_k_sequence(X, y, 3, 4, **kw)
    else:
        r = plskit.pls1_find_k_sequence(X, y, 3, **kw)
    assert r.k_star == 0
    assert r.pvalues.shape == (3,)
    assert r.pvalues[0] >= r.alpha
    assert np.isnan(r.pvalues[1:]).all()


@pytest.mark.parametrize("name", sorted(YS))
@pytest.mark.parametrize("selector", ["r2_se", "bic"])
def test_fit_k_optimal_raises_no_component(name, selector):
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.pls1_fit(
            X, YS[name], k="optimal", k_max=3, seed=3,
            find_k_args={"selector": selector},
        )
    assert ei.value.code == "optimal_no_component"
    # The policy is the core's (FindKOptimalOutput::k_to_fit), raised through
    # the engine's error translation, not re-implemented in Python.
    assert isinstance(ei.value.__cause__, plskit._errors._PlsKitException)


@pytest.mark.parametrize("name", sorted(YS))
def test_fit_k_sequence_raises_no_rejection(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(plskit.PlsKitError) as ei:
            plskit.pls1_fit(
                X, YS[name], k="sequence", k_max=3, seed=3,
                find_k_args={"test_method": "e"},
            )
    assert ei.value.code == "sequence_no_rejection"
    assert isinstance(ei.value.__cause__, plskit._errors._PlsKitException)
