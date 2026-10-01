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


def test_raw_perm_rejects_leave_one_out_n_folds():
    """n_folds == n through the `args` dict: every validation fold is a
    single row, so the pooled CV R² is undefined."""
    X, y = _data(n=12)
    with pytest.raises(plskit.PlsKitError) as exc_info:
        plskit.pls1_confirmatory_test(
            X, y, k=1, method="raw_perm", args={"n_folds": 12}, seed=7,
        )
    assert exc_info.value.code == "invalid_argument"


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
    for call in (
        lambda: plskit.pls1_confirmatory_test(X, y, k=1, method=removed),  # type: ignore[arg-type]
        lambda: plskit.pls1_find_k_sequence(X, y, k_max=3, test_method=removed, seed=7),  # type: ignore[arg-type]
        lambda: plskit.pls1_find_k_optimal(X, y, k_max=3, diagnostic=removed, seed=7),  # type: ignore[arg-type]
    ):
        with pytest.raises(plskit.PlsKitError, match=unknown) as ei:
            call()
        assert ei.value.code == "invalid_args"


def test_optional_fields_are_filled_only_by_unweighted_split_nb():
    # rho_hat and stable_rank are Option fields: an unweighted split_nb run
    # fills both, a weighted one drops rho_hat, and every other method
    # marshals both as None (never a recomputed stable rank).
    X, y = _data()
    nb = plskit.pls1_confirmatory_test(
        X, y, k=1, method="split_nb", args=_SMALL_ARGS_BY_METHOD["split_nb"], seed=7,
    )
    assert isinstance(nb, plskit.ConfirmatoryTestResult)
    assert isinstance(nb.rho_hat, float)
    assert 0.0 <= nb.rho_hat <= 1.0
    assert isinstance(nb.stable_rank, float)
    assert nb.stable_rank > 0.0

    w = np.where(np.arange(X.shape[0]) % 2 == 0, 1.5, 0.5)
    weighted = plskit.pls1_confirmatory_test(
        X, y, k=1, method="split_nb", args=_SMALL_ARGS_BY_METHOD["split_nb"],
        weights=w, seed=7,
    )
    assert weighted.rho_hat is None

    for method in ("raw_perm", "split_exact", "score", "e"):
        r = plskit.pls1_confirmatory_test(
            X, y, k=1, method=method, args=_SMALL_ARGS_BY_METHOD[method], seed=7,
        )
        assert r.rho_hat is None, method
        assert r.stable_rank is None, method


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
    assert isinstance(r.stable_rank, float)
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


def test_sequence_results_carry_the_gate_rank():
    X, y = _flagged_data()
    with pytest.warns(UserWarning, match="rerouted") as rec:
        seq = plskit.pls1_find_k_sequence(
            X, y, k_max=3, test_method="split_nb", args={"n_splits": 20}, seed=7,
        )
    assert seq.test_method == "split_exact"
    assert isinstance(seq.stable_rank, float)
    # The whole point of plumbing it through: the warning can now name both
    # numbers the rule read, with no hedge about which result type has what.
    msg = str(rec[0].message)
    assert f"{seq.stable_rank:.4g}" in msg
    assert f"{seq.n_eff:.4g}" in msg


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


# ── Unusable argument values ────────────────────────────────────────
#
# A value the seam cannot use raises a coded PlsKitError, never a raw
# TypeError / OverflowError: `invalid_args` inside the `args` dict,
# `invalid_argument` for a top-level argument (plskit-bind's split, so R and
# Julia raise the same codes). Each case fails before any engine work runs.
# Sibling rows for other entry points sit beside their own keepers.

_X, _Y = _data()


@pytest.mark.parametrize(
    "kwargs, code, message",
    [
        ({"method": "raw_perm", "args": {"n_perm": -5}}, "invalid_args",
         r"args\['n_perm'\] for method='raw_perm' must be a non-negative whole number, got -5"),
        ({"method": "split_exact", "args": {"n_splits": 2.5}}, "invalid_args",
         "must be a non-negative whole number, got 2.5"),
        ({"method": "raw_perm", "args": {"n_folds": True}}, "invalid_args",
         "must be a non-negative whole number, got True"),
        ({"method": "raw_perm", "args": {"n_perm": "7"}}, "invalid_args",
         'got the string "7"'),
        ({"method": "split_exact", "ci": True, "n_boot": -5}, "invalid_argument",
         "n_boot must be a non-negative whole number"),
        ({"method": "split_exact", "k": -1}, "invalid_argument",
         "k must be a non-negative whole number"),
        ({"method": "split_exact", "ci": True, "level": "x"}, "invalid_argument",
         'level must be a number, got the string "x"'),
        ({"method": "split_exact", "ci": True, "level": 10**400}, "invalid_argument",
         "level must be a number"),
        ({"method": "score", "seed": -1}, "invalid_argument",
         "seed must be a whole number"),
        ({"method": "score", "seed": 2**64}, "invalid_argument",
         "seed must be a whole number"),
        ({"method": "score", "ci": "yes"}, "invalid_argument",
         'ci must be a bool, got the string "yes"'),
        ({"method": "score", "pre_standardized": 1}, "invalid_argument",
         "pre_standardized must be a bool, got 1"),
        ({"method": "score", "disable_parallelism": 1}, "invalid_argument",
         "disable_parallelism must be a bool"),
        ({"method": "score", "verbose": "no"}, "invalid_argument",
         "verbose must be a bool"),
        ({"method": 1}, "invalid_argument", "method must be a string, got 1"),
        ({"method": "raw_perm", "args": [1]}, "invalid_argument",
         r"args must be a dict of named values, got \[1\]"),
    ],
    ids=[
        "args_negative", "args_fractional", "args_bool", "args_str",
        "n_boot", "k", "level_str", "level_overflow", "seed_negative",
        "seed_too_big", "ci_str", "pre_standardized_int",
        "disable_parallelism_int", "verbose_str", "method_int", "args_list",
    ],
)
def test_unusable_values_raise_coded_errors(kwargs, code, message):
    kwargs = {"k": 1, **kwargs}
    with pytest.raises(plskit.PlsKitError, match=message) as ei:
        plskit.pls1_confirmatory_test(_X, _Y, **kwargs)
    assert ei.value.code == code


def test_args_none_and_whole_float_counts_are_accepted():
    """An args key set to None takes the engine default, and a whole float
    is a count (plskit-bind's rules); both reach the engine as resolved
    values."""
    r = plskit.pls1_confirmatory_test(
        _X, _Y, 1, method="split_nb", args={"n_splits": 20.0, "force": None}, seed=7,
    )
    assert r.n_splits == 20
    d = plskit.pls1_confirmatory_test(
        _X, _Y, 1, method="split_nb", args={"n_splits": None}, seed=7,
    )
    assert d.n_splits == 50


@pytest.mark.parametrize("seed", [2**64 - 1, np.uint64(2**64 - 1), 0, 7.0])
def test_the_full_seed_range_is_accepted_and_echoed(seed):
    r = plskit.pls1_confirmatory_test(_X, _Y, 1, method="score", seed=seed)
    assert r.seed == int(seed) and type(r.seed) is int
