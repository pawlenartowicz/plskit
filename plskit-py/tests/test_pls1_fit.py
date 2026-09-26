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


# The seam reads a contiguous X in place (no copy) when the engine
# standardizes it, and copies any other layout; every layout must give the
# same bits, and the same error where the fit fails.
_FIT_FIELDS = ("T", "P", "W", "Q", "coef", "beta", "intercept", "n_eff")


def _layouts(X):
    n, p = X.shape
    padded = np.full((n * 2, p * 3), np.nan)
    padded[::2, ::3] = X
    offset = np.full((n + 3, p), np.nan)
    offset[2:2 + n] = X  # a C-ordered block that does not start the buffer
    read_only = np.array(X)
    read_only.flags.writeable = False
    out = {
        "C": np.ascontiguousarray(X),
        "F": np.asfortranarray(X),
        "strided": padded[::2, ::3],
        "reversed_rows": np.ascontiguousarray(X[::-1])[::-1],
        "reversed_cols": np.ascontiguousarray(X[:, ::-1])[:, ::-1],
        "offset_rows": offset[2:2 + n],
        "read_only": read_only,
    }
    if p == 1 or n == 1:
        # A unit axis with a junk stride: numpy flags it both C- and
        # F-contiguous, and the stride is never followed.
        base = np.ascontiguousarray(X).ravel()
        junk = 8 * 12345
        strides = (8, junk) if p == 1 else (junk, 8)
        out["unit_axis_junk_stride"] = np.lib.stride_tricks.as_strided(
            base, shape=(n, p), strides=strides, writeable=False)
    return out


def _outcome(call):
    try:
        d = call()
    except Exception as e:  # compared, not swallowed
        return ("error", type(e).__name__, str(e))
    return {f: np.ascontiguousarray(np.asarray(d[f], dtype=np.float64)).view(np.uint64)
            for f in _FIT_FIELDS} | {"shape": (np.shape(d["T"]), np.shape(d["P"]))}


def _same(got, ref):
    if isinstance(ref, tuple) or isinstance(got, tuple):
        return got == ref
    return got["shape"] == ref["shape"] and all(
        np.array_equal(got[f], ref[f]) for f in _FIT_FIELDS)


def _fitters(y, sparse, kw, k=3, keep=9):
    def raw(Xl):
        if sparse:
            return plskit._plskit.spls1_fit(Xl, y, k, keep, **kw)
        return plskit._plskit.pls1_fit(Xl, y, k, **kw)

    def public(Xl):
        if sparse:
            return vars(plskit.spls1_fit(Xl, y, k=k, keep=keep, **kw))
        return vars(plskit.pls1_fit(Xl, y, k=k, **kw))

    return raw, public


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("pre_standardized", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
def test_fit_bits_do_not_depend_on_x_layout(weighted, pre_standardized, sparse):
    X, y = _data(n=40, d=21, seed=5)
    X[:, 3] = 2.5  # a constant column
    if pre_standardized:
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        X = (X - X.mean(axis=0)) / sd
        y = (y - y.mean()) / y.std()
    w = np.random.default_rng(6).uniform(0.2, 3.0, size=40) if weighted else None
    raw, public = _fitters(y, sparse, dict(pre_standardized=pre_standardized, weights=w))

    ref = _outcome(lambda: raw(np.ascontiguousarray(X)))
    assert not isinstance(ref, tuple)
    for name, Xl in _layouts(X).items():
        for call in (raw, public):
            assert _same(_outcome(lambda: call(Xl)), ref), (name, call.__name__)
    # float32 input: the public API's float64 cast (then read in place)
    # against the raw extension's copy path on a strided float64 cast.
    X64 = X.astype(np.float32).astype(np.float64)
    ref32 = _outcome(lambda: raw(_layouts(X64)["strided"]))
    assert _same(_outcome(lambda: public(X.astype(np.float32))), ref32)


@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (3, 0), (1, 1), (1, 4), (4, 1), (5, 2)])
@pytest.mark.parametrize("pre_standardized", [False, True])
def test_fit_edge_shapes_do_not_depend_on_x_layout(shape, pre_standardized):
    n, p = shape
    rng = np.random.default_rng(7)
    X = rng.normal(size=shape)
    y = rng.normal(size=n)
    raw, public = _fitters(y, False, dict(pre_standardized=pre_standardized), k=1)
    for call in (raw, public):  # the two raise different exception types
        ref = _outcome(lambda: call(np.ascontiguousarray(X)))
        for name, Xl in _layouts(X).items():
            assert _same(_outcome(lambda: call(Xl)), ref), (name, call.__name__)
