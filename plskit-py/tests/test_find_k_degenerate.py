"""find_k on a y with no extractable first component, seen through the seam.

A constant y makes the full-data fit the k_used=0 zero model. The engine
policy (constant and orthogonal y, dense and sparse) is tested in
plskit-rs/src/find_k.rs; these tests pin how k_star=0 results and errors
cross the Python seam.
"""
import warnings

import numpy as np
import pytest
import plskit


X = np.random.default_rng(5).normal(size=(60, 8))
Y0 = np.full(60, 3.0)


@pytest.mark.parametrize("selector", ["r2_se", "r2_max", "bic"])
def test_optimal_returns_k_star_zero_with_empty_scores(selector):
    r = plskit.pls1_find_k_optimal(X, Y0, 3, selector=selector, seed=3)
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


def test_keep_optimal_returns_keep_star_zero_with_empty_scores():
    k = 1
    r = plskit.spls1_find_keep_optimal(X, Y0, k, seed=3)
    assert r.keep_star == 0
    assert r.k == k
    assert r.seed == 3
    assert r.cv_scores == {}
    assert r.cv_scores_se == {}
    assert r.keep_grid == []


@pytest.mark.parametrize(
    "diagnostic,args,expect",
    [
        ("e", None, "e"),
        ("split_nb", {"n_splits": 5, "force": True}, "split_nb"),
    ],
)
def test_optimal_diagnostic_at_k_star_zero_is_empty(diagnostic, args, expect):
    r = plskit.pls1_find_k_optimal(
        X, Y0, 3, selector="bic", diagnostic=diagnostic, args=args, seed=7,
    )
    assert r.k_star == 0
    assert r.pvalues.shape == (0,)
    assert r.diagnostic == expect
    assert (r.stable_rank is not None) == (diagnostic == "split_nb")


def test_sequence_returns_k_star_zero():
    r = plskit.pls1_find_k_sequence(X, Y0, 3, test_method="e", seed=3)
    assert r.k_star == 0
    assert r.pvalues.shape == (3,)
    assert r.pvalues[0] >= r.alpha
    assert np.isnan(r.pvalues[1:]).all()


def test_fit_k_optimal_raises_no_component():
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.pls1_fit(X, Y0, k="optimal", k_max=3, seed=3)
    assert ei.value.code == "optimal_no_component"
    # The policy is the core's (FindKOptimalOutput::k_to_fit), raised through
    # the engine's error translation, not re-implemented in Python.
    assert isinstance(ei.value.__cause__, plskit._plskit.PlsKitException)


def test_fit_k_sequence_raises_no_rejection():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(plskit.PlsKitError) as ei:
            plskit.pls1_fit(
                X, Y0, k="sequence", k_max=3, seed=3,
                find_k_args={"test_method": "e"},
            )
    assert ei.value.code == "sequence_no_rejection"
    assert isinstance(ei.value.__cause__, plskit._plskit.PlsKitException)
