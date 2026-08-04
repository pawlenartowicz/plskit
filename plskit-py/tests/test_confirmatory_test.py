import contextlib
import warnings

import numpy as np
import pytest
import plskit


@contextlib.contextmanager
def _no_warning():
    """Turn any warning inside the block into an error.

    Used to assert the *absence* of the auto-gate reroute warning — pytest has
    no negative form of `pytest.warns`."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


def _data(n=60, d=5, snr=3.0, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = X[:, 0] * snr + rng.normal(size=n)
    return X, y


def _flagged_data():
    """A design the split_nb auto-gate flags.

    These tests assert only that the gate fires, never which of its conditions
    did it — the rule lives in Rust and is tested there.
    """
    return _data(n=20, d=5, seed=3)


_SMALL_ARGS_BY_METHOD = {
    "raw_perm": {"n_perm": 100},
    "split_nb": {"n_splits": 20},
    "split_exact": {"n_perm": 100, "n_splits": 20},
    "score": {},
    "e": {},
}


@pytest.mark.parametrize("method", ["raw_perm", "split_nb", "split_exact", "score", "e"])
def test_confirmatory_methods_run(method):
    X, y = _data()
    r = plskit.pls1_confirmatory_test(
        X, y, k=1, method=method,
        args=_SMALL_ARGS_BY_METHOD[method], seed=7,
    )
    assert isinstance(r, plskit.ConfirmatoryTestResult)
    assert r.method == method
    assert r.k == 1
    assert 0.0 <= r.pvalue <= 1.0


def test_confirmatory_at_param_no_longer_accepted():
    X, y = _data()
    with pytest.raises(TypeError):
        plskit.pls1_confirmatory_test(X, y, k=1, method="split_nb", at="fitted_k")


def test_confirmatory_score_n_perm_field_is_none():
    X, y = _data()
    r = plskit.pls1_confirmatory_test(X, y, k=1, method="score", seed=7)
    assert r.n_perm is None
    assert r.n_splits is None


@pytest.mark.parametrize("removed", ["split_perm", "split_perm_nr"])
def test_removed_method_names_raise(removed):
    """The pre-split_exact names are gone from every surface that parses a
    method string — plain unknown-method errors, no deprecation shim."""
    X, y = _data()
    unknown = f"unknown method: {removed}"
    with pytest.raises(plskit.PlsKitError, match=unknown):
        plskit.pls1_confirmatory_test(X, y, k=1, method=removed)  # type: ignore[arg-type]
    with pytest.raises(plskit.PlsKitError, match=unknown):
        plskit.pls1_find_k_sequence(X, y, k_max=3, test_method=removed, seed=7)  # type: ignore[arg-type]
    with pytest.raises(plskit.PlsKitError, match=unknown):
        plskit.pls1_find_k_optimal(X, y, k_max=3, diagnostic=removed, seed=7)  # type: ignore[arg-type]


def test_split_exact_accepts_weighted_k_one_dense():
    # Weighted k=1 dense input takes the no-refit route; the weights must reach
    # the statistic, so the result differs from the unweighted run.
    X, y = _data()
    w = np.where(np.arange(X.shape[0]) % 2 == 0, 1.5, 0.5)
    args = {"n_perm": 100, "n_splits": 20}
    r_w = plskit.pls1_confirmatory_test(
        X, y, k=1, method="split_exact", args=args, weights=w, seed=7,
    )
    r_u = plskit.pls1_confirmatory_test(
        X, y, k=1, method="split_exact", args=args, seed=7,
    )
    assert isinstance(r_w, plskit.ConfirmatoryTestResult)
    assert r_w.statistic != r_u.statistic


def test_split_exact_runs_at_k_two():
    # k >= 2 sends split_exact down the per-permutation refit route; the engine
    # picks the route, so there is nothing to pass — it just has to work.
    X, y = _data()
    r = plskit.pls1_confirmatory_test(
        X, y, k=2, method="split_exact", args={"n_perm": 100, "n_splits": 20}, seed=7,
    )
    assert r.method == "split_exact"
    assert r.k == 2
    assert 0.0 < r.pvalue <= 1.0


def test_rho_hat_populated_for_split_nb_unweighted_only():
    X, y = _data()

    r_nb = plskit.pls1_confirmatory_test(
        X, y, k=1, method="split_nb", args={"n_splits": 20}, seed=7,
    )
    assert isinstance(r_nb.rho_hat, float)
    assert 0.0 <= r_nb.rho_hat <= 1.0

    w = np.where(np.arange(X.shape[0]) % 2 == 0, 1.5, 0.5)
    r_nb_weighted = plskit.pls1_confirmatory_test(
        X, y, k=1, method="split_nb", args={"n_splits": 20}, weights=w, seed=7,
    )
    assert r_nb_weighted.rho_hat is None

    r_exact = plskit.pls1_confirmatory_test(
        X, y, k=1, method="split_exact", args={"n_perm": 100, "n_splits": 20}, seed=7,
    )
    assert r_exact.rho_hat is None


def test_stable_rank_populated_for_requested_split_nb_only():
    X, y = _data()
    r_nb = plskit.pls1_confirmatory_test(
        X, y, k=1, method="split_nb", args={"n_splits": 20}, seed=7,
    )
    assert isinstance(r_nb.stable_rank, float)
    assert r_nb.stable_rank > 0.0

    for method in ("raw_perm", "split_exact", "score", "e"):
        r = plskit.pls1_confirmatory_test(
            X, y, k=1, method=method,
            args=_SMALL_ARGS_BY_METHOD[method], seed=7,
        )
        assert r.stable_rank is None, method


def test_score_has_no_sequential_variant():
    X, y = _data()
    with pytest.raises(plskit.PlsKitError):
        plskit.pls1_find_k_sequence(X, y, k_max=3, test_method="score", seed=7)  # type: ignore[arg-type]


def test_split_exact_is_a_sequential_test_method():
    X, y = _data()
    r = plskit.pls1_find_k_sequence(
        X, y, k_max=3, test_method="split_exact",
        args={"n_perm": 100, "n_splits": 20}, seed=7,
    )
    assert r.test_method == "split_exact"
    assert r.pvalues.shape == (3,)


def test_split_exact_smoke():
    X, y = _data()
    n_perm = 100
    r = plskit.pls1_confirmatory_test(
        X, y, k=1, method="split_exact", args={"n_perm": n_perm, "n_splits": 20}, seed=7,
    )
    assert 0.0 < r.pvalue <= 1.0
    assert isinstance(r.statistic, float)
    assert r.pvalue >= 1.0 / (n_perm + 1)


# ── split_nb auto-gate, seen from Python ────────────────────────────────────


def test_gate_reroutes_split_nb_and_warns():
    X, y = _flagged_data()
    with pytest.warns(UserWarning, match="rerouted") as rec:
        r = plskit.pls1_confirmatory_test(
            X, y, k=1, method="split_nb", args={"n_splits": 20}, seed=7,
        )
    assert r.method == "split_exact"
    assert isinstance(r.stable_rank, float)
    msg = str(rec[0].message)
    assert "n_perm=1000" in msg
    assert "'force': True" in msg
    assert f"{r.stable_rank:.4g}" in msg


def test_force_suppresses_the_reroute_and_the_warning():
    X, y = _flagged_data()
    with _no_warning():
        r = plskit.pls1_confirmatory_test(
            X, y, k=1, method="split_nb", args={"n_splits": 20, "force": True}, seed=7,
        )
    assert r.method == "split_nb"
    # The gate still ran and still reports what it saw under force.
    assert isinstance(r.stable_rank, float)


def test_gate_does_not_fire_on_a_healthy_design():
    X, y = _data()
    with _no_warning():
        r = plskit.pls1_confirmatory_test(
            X, y, k=1, method="split_nb", args={"n_splits": 20}, seed=7,
        )
    assert r.method == "split_nb"


def test_force_is_a_split_nb_only_arg():
    X, y = _data()
    with pytest.raises(plskit.PlsKitError, match="does not accept arg 'force'") as ei:
        plskit.pls1_confirmatory_test(
            X, y, k=1, method="split_exact", args={"n_perm": 100, "force": True}, seed=7,
        )
    assert ei.value.code == "invalid_args"


def test_force_must_be_a_bool():
    X, y = _data()
    with pytest.raises(plskit.PlsKitError, match="must be a bool") as ei:
        plskit.pls1_confirmatory_test(
            X, y, k=1, method="split_nb", args={"n_splits": 20, "force": 1.5}, seed=7,
        )
    assert ei.value.code == "invalid_args"


def test_gate_reroutes_the_optimal_diagnostic_and_warns():
    # find_k_optimal's diagnostic runs through the same hoisted gate, and it
    # takes the same `force` override, so the warning advises one.
    X, y = _flagged_data()
    with pytest.warns(UserWarning, match="rerouted") as rec:
        r = plskit.pls1_find_k_optimal(
            X, y, k_max=2, diagnostic="split_nb", args={"n_splits": 20}, seed=7,
        )
    assert r.diagnostic == "split_exact"
    assert "'force': True" in str(rec[0].message)


def test_force_suppresses_the_optimal_diagnostic_reroute():
    X, y = _flagged_data()
    with _no_warning():
        r = plskit.pls1_find_k_optimal(
            X, y, k_max=2, diagnostic="split_nb",
            args={"n_splits": 20, "force": True}, seed=7,
        )
    assert r.diagnostic == "split_nb"


def test_optimal_force_requires_a_split_nb_diagnostic():
    X, y = _data()
    with pytest.raises(plskit.PlsKitError, match="requires diagnostic to be set") as ei:
        plskit.pls1_find_k_optimal(X, y, k_max=2, args={"force": True}, seed=7)
    assert ei.value.code == "invalid_args"
    with pytest.raises(plskit.PlsKitError, match="only valid for diagnostic='split_nb'"):
        plskit.pls1_find_k_optimal(
            X, y, k_max=2, diagnostic="split_exact",
            args={"n_perm": 100, "force": True}, seed=7,
        )


def test_optimal_force_must_be_a_bool():
    X, y = _data()
    with pytest.raises(plskit.PlsKitError, match="must be a bool") as ei:
        plskit.pls1_find_k_optimal(
            X, y, k_max=2, diagnostic="split_nb",
            args={"n_splits": 20, "force": 1.5}, seed=7,
        )
    assert ei.value.code == "invalid_args"


def test_no_diagnostic_requested_does_not_warn():
    X, y = _flagged_data()
    with _no_warning():
        r = plskit.pls1_find_k_optimal(X, y, k_max=2, seed=7)
    assert r.diagnostic is None


def test_gate_reroutes_the_whole_sequence_and_warns():
    X, y = _flagged_data()
    with pytest.warns(UserWarning, match="rerouted"):
        r = plskit.pls1_find_k_sequence(
            X, y, k_max=3, test_method="split_nb", args={"n_splits": 20}, seed=7,
        )
    assert r.test_method == "split_exact"

    with _no_warning():
        r_forced = plskit.pls1_find_k_sequence(
            X, y, k_max=3, test_method="split_nb",
            args={"n_splits": 20, "force": True}, seed=7,
        )
    assert r_forced.test_method == "split_nb"


def test_sequence_results_carry_the_gate_rank():
    X, y = _flagged_data()
    with pytest.warns(UserWarning, match="rerouted") as rec:
        seq = plskit.pls1_find_k_sequence(
            X, y, k_max=3, test_method="split_nb", args={"n_splits": 20}, seed=7,
        )
    assert isinstance(seq.stable_rank, float)
    # The whole point of plumbing it through: the warning can now name both
    # numbers the rule read, with no hedge about which result type has what.
    msg = str(rec[0].message)
    assert f"{seq.stable_rank:.4g}" in msg
    assert f"{seq.n_eff:.4g}" in msg

    with pytest.warns(UserWarning, match="rerouted"):
        opt = plskit.pls1_find_k_optimal(
            X, y, k_max=2, diagnostic="split_nb", args={"n_splits": 20}, seed=7,
        )
    assert isinstance(opt.stable_rank, float)


def test_sequence_gate_rank_survives_force_and_is_absent_otherwise():
    X, y = _flagged_data()
    with _no_warning():
        forced = plskit.pls1_find_k_sequence(
            X, y, k_max=3, test_method="split_nb",
            args={"n_splits": 20, "force": True}, seed=7,
        )
    assert forced.test_method == "split_nb"
    assert isinstance(forced.stable_rank, float)

    other = plskit.pls1_find_k_sequence(
        X, y, k_max=3, test_method="raw_perm", args={"n_perm": 100}, seed=7,
    )
    assert other.stable_rank is None
    assert plskit.pls1_find_k_optimal(X, y, k_max=2, seed=7).stable_rank is None


# ── split_nb_gate, the standalone query ─────────────────────────────────────


def test_split_nb_gate_answers_what_the_test_functions_decide():
    for data, expect_fires in [(_flagged_data(), True), (_data(), False)]:
        X, y = data
        q = plskit.split_nb_gate(X)
        assert isinstance(q, plskit.SplitNbGateResult)
        assert q.fires is expect_fires
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = plskit.pls1_confirmatory_test(
                X, y, k=1, method="split_nb", args={"n_splits": 20}, seed=7,
            )
        assert q.fires == (r.method == "split_exact")
        # Same rule on the same standardized X — the numbers must match too.
        assert q.stable_rank == r.stable_rank
        assert q.n_eff == r.n_eff


def test_split_nb_gate_reads_weights():
    # n = 40 clears the size floor; these weights pull Kish n_eff under it.
    X, _ = _data(n=40, d=5, seed=8)
    w = np.where(np.arange(40) % 2 == 0, 1.0, 0.1)
    assert plskit.split_nb_gate(X).fires is False
    weighted = plskit.split_nb_gate(X, weights=w)
    assert weighted.fires is True
    assert weighted.n_eff < 40.0


def test_split_nb_gate_validates_its_input():
    X, _ = _data()
    bad = X.copy()
    bad[3, 2] = np.nan
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.split_nb_gate(bad)
    assert ei.value.code == "non_finite_input"
    with pytest.raises(plskit.PlsKitInvalidWeights):
        plskit.split_nb_gate(X, weights=np.full(X.shape[0], -1.0))
