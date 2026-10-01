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


def test_sequence_no_rejection_returns_kstar_zero():
    # Pure noise + strict alpha → no rejection
    rng = np.random.default_rng(99)
    X = rng.normal(size=(60, 5))
    y = rng.normal(size=60)
    r = plskit.pls1_find_k_sequence(
        X, y, k_max=4, test_method="e",
        alpha=1e-6, seed=99,
    )
    assert r.k_star == 0


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
