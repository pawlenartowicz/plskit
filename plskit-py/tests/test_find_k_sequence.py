import warnings

import numpy as np
import pytest
import plskit


def _data(n=80, d=5, k_signal=1, snr=5.0, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    beta = np.zeros(d); beta[:k_signal] = 1.0
    y = X @ beta * snr + rng.normal(size=n)
    return X, y


def test_sequence_records_the_alpha_it_was_given():
    # 0.1 is not the default, so a dropped alpha would record 0.05.
    X, y = _data()
    r = plskit.pls1_find_k_sequence(
        X, y, k_max=4, test_method="split_nb",
        args={"n_splits": 30}, alpha=0.1, seed=7,
    )
    assert isinstance(r, plskit.FindKSequenceResult)
    assert r.alpha == 0.1


def test_sequence_score_rejected():
    X, y = _data()
    with pytest.raises(plskit.PlsKitError, match="has no sequential variant") as ei:
        plskit.pls1_find_k_sequence(X, y, k_max=4, test_method="score", seed=7)
    assert ei.value.code == "invalid_args"


def test_sequence_args_count_must_be_a_whole_number():
    X, y = _data()
    with pytest.raises(plskit.PlsKitError, match=r"args\['n_splits'\] for test_method='split_exact'") as ei:
        plskit.pls1_find_k_sequence(X, y, 2, test_method="split_exact", args={"n_splits": -1})
    assert ei.value.code == "invalid_args"


def test_sequence_auto_is_the_default_and_resolves_per_design():
    X, y = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        small = plskit.pls1_find_k_sequence(
            X, y, k_max=2, args={"n_perm": 30, "n_splits": 10}, seed=7
        )
    assert small.test_method == "split_exact"

    X, y = _data(n=300, d=10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        large = plskit.pls1_find_k_sequence(
            X, y, k_max=2, args={"n_perm": 30, "n_splits": 10}, seed=7
        )
    assert large.test_method == "split_nb"
    assert large.stable_rank is not None
    explicit = plskit.pls1_find_k_sequence(
        X, y, k_max=2, test_method="split_nb", args={"n_splits": 10}, seed=7
    )
    assert large.k_star == explicit.k_star
    np.testing.assert_array_equal(large.pvalues, explicit.pvalues)


def test_sequence_auto_args_reject_force_in_both_sequence_functions():
    X, y = _data()
    with pytest.raises(plskit.PlsKitError, match="does not accept arg 'force'") as ei:
        plskit.pls1_find_k_sequence(
            X, y, k_max=2, test_method="auto", args={"force": True}, seed=7
        )
    assert ei.value.code == "invalid_args"
    with pytest.raises(plskit.PlsKitError, match="does not accept arg 'force'") as ei:
        plskit.spls1_find_k_sequence(
            X, y, 2, 3, test_method="auto", args={"force": True}, seed=7
        )
    assert ei.value.code == "invalid_args"


def test_pls1_fit_sequence_runs_auto_when_find_k_args_set_no_test_method():
    X, y = _data(n=300, d=10)
    m = plskit.pls1_fit(
        X, y, k="sequence", k_max=2, find_k_args={"args": {"n_splits": 10}}, seed=7
    )
    assert m.selection_result.test_method == "split_nb"
    assert m.k_used == m.selection_result.k_star
