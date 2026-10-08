import dataclasses

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
    assert m.weights is None
    assert m.n_eff == 80.0


def test_fit_pre_standardized_passes_through():
    X, y = _data()
    Xs = (X - X.mean(0)) / X.std(0)
    ys = (y - y.mean()) / y.std()
    m = plskit.pls1_fit(Xs, ys, k=2, pre_standardized=True)
    np.testing.assert_allclose(m.beta, m.coef, atol=1e-15)
    assert m.intercept == 0.0


@pytest.mark.parametrize(
    "find_k_args", [None, {"selector": "r2_max", "args": {"n_folds": 3}}]
)
def test_fit_k_optimal_dispatches_to_find_k_optimal(find_k_args):
    # k="optimal" must select K exactly as pls1_find_k_optimal does with the
    # same seed and find_k_args, then fit that K. The rows select different
    # K (2 and 4), so ignoring find_k_args, or its nested args, fails.
    X, y = _data()
    r = plskit.pls1_find_k_optimal(X, y, 4, seed=7, **(find_k_args or {}))
    m = plskit.pls1_fit(X, y, k="optimal", k_max=4, find_k_args=find_k_args, seed=7)
    assert r.k_star > 1
    assert m.k_used == r.k_star
    assert isinstance(m.selection_result, plskit.FindKOptimalResult)
    assert m.selection_result.selector == r.selector
    assert m.selection_result.cv_scores == r.cv_scores
    m_fixed = plskit.pls1_fit(X, y, k=r.k_star)
    assert np.array_equal(m.coef, m_fixed.coef)
    assert np.array_equal(m.beta, m_fixed.beta)
    assert m.intercept == m_fixed.intercept


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


# Keys that live on pls1_fit itself (never inside find_k_args), plus an
# unknown key; `sequence` also rejects the optimal-only keys.
_NEVER_IN_FIND_K_ARGS = (
    "seed", "pre_standardized", "weights", "verbose",
    "bogus_key",
)


@pytest.mark.parametrize(
    "mode, key",
    [("optimal", k) for k in _NEVER_IN_FIND_K_ARGS]
    + [("sequence", k) for k in _NEVER_IN_FIND_K_ARGS + ("selector", "diagnostic")],
)
def test_fit_find_k_args_rejects_disallowed_keys(mode, key):
    X, y = _data()
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.pls1_fit(X, y, k=mode, k_max=4, find_k_args={key: 0})
    assert ei.value.code == "invalid_args"
    assert key in str(ei.value)
    assert "allowed" in str(ei.value)


@pytest.mark.parametrize(
    "mode, find_k_args",
    [
        ("optimal", {"selector": 5}),
        ("sequence", {"alpha": "x"}),
        ("optimal", {"args": [1]}),
    ],
)
def test_fit_find_k_args_wrong_typed_value_is_invalid_args(mode, find_k_args):
    X, y = _data()
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.pls1_fit(X, y, k=mode, k_max=4, find_k_args=find_k_args)
    assert ei.value.code == "invalid_args"


# The seam reads an aligned C- or F-contiguous X in place and copies any
# other layout column-major. X read row-major gives the bits of the C-ordered
# reference, and other layouts agree to the corpus tolerance. The public API
# hands X on in its own layout, so a public fit has the bits of the raw fit
# of the same array, and repeating it repeats them. Every layout gives the
# same error where the fit fails.
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


def _close(got, ref, atol=1e-10, rtol=1e-14):
    """Agreement at the corpus array tolerance, `atol + rtol * |ref|`."""
    if isinstance(ref, tuple) or isinstance(got, tuple):
        return got == ref
    return got["shape"] == ref["shape"] and all(
        np.allclose(got[f].view(np.float64), ref[f].view(np.float64),
                    rtol=rtol, atol=atol, equal_nan=True)
        for f in _FIT_FIELDS)


def _read_row_major(Xl):
    """The extension reads a C-contiguous array row-major, which gives the
    bits of the C-ordered reference. Any other layout is read column-major
    and agrees within tolerance."""
    return Xl.flags.c_contiguous


def _fields(type_name, fields):
    """`make_result` for a direct extension call: the record's own fields."""
    return fields


def _fitters(y, sparse, kw, k=3, keep=9):
    def raw(Xl):
        arguments = {"X": Xl, "y": y, "k": k, **kw}
        if sparse:
            return plskit._plskit.call("spls1_fit", arguments | {"keep": keep}, _fields)[0]
        return plskit._plskit.call("pls1_fit", arguments, _fields)[0]

    def public(Xl):
        if sparse:
            return vars(plskit.spls1_fit(Xl, y, k=k, keep=keep, **kw))
        return vars(plskit.pls1_fit(Xl, y, k=k, **kw))

    return raw, public


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("pre_standardized", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
def test_fit_x_layouts_agree(weighted, pre_standardized, sparse):
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
        same = _same if _read_row_major(Xl) else _close
        got = _outcome(lambda: raw(Xl))
        assert same(got, ref), (name, "raw")
        assert _same(_outcome(lambda: public(Xl)), got), (name, "public")
        assert _same(_outcome(lambda: public(Xl)), got), (name, "public, again")
    # float32 input: the public API's float64 cast keeps the C order and is
    # read in place.
    X64 = X.astype(np.float32).astype(np.float64)
    ref32 = _outcome(lambda: raw(np.ascontiguousarray(X64)))
    assert _same(_outcome(lambda: public(X.astype(np.float32))), ref32)


@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (3, 0), (1, 1), (1, 4), (4, 1), (5, 2)])
@pytest.mark.parametrize("pre_standardized", [False, True])
def test_fit_edge_shapes_x_layouts_agree(shape, pre_standardized):
    n, p = shape
    rng = np.random.default_rng(7)
    X = rng.normal(size=shape)
    y = rng.normal(size=n)
    raw, public = _fitters(y, False, dict(pre_standardized=pre_standardized), k=1)
    for call in (raw, public):  # the two raise different exception types
        ref = _outcome(lambda: call(np.ascontiguousarray(X)))
        for name, Xl in _layouts(X).items():
            same = _same if _read_row_major(Xl) else _close
            assert same(_outcome(lambda: call(Xl)), ref), (name, call.__name__)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("pre_standardized", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_fit_non_finite_x_raises_in_both_orders(order, pre_standardized, weighted):
    X, y = _data(n=30, d=5, seed=3)
    X[7, 2] = np.inf
    w = None
    if weighted:
        w = np.ones(30)
        w[7] = 0.0  # a zero weight does not hide the non-finite entry
    # The raw extension, in both orders it reads in place.
    arguments = {"X": np.asarray(X, order=order), "y": y, "k": 2,
                 "pre_standardized": pre_standardized, "weights": w}
    with pytest.raises(plskit._plskit.PlsKitException) as ei:
        plskit._plskit.call("pls1_fit", arguments, _fields)
    assert ei.value.code == "non_finite_input"


@pytest.mark.parametrize(
    "bad_k", [2.7, np.float64(1.5), True, -1, 1e20, 2**63, np.array(2.5)]
)
def test_fit_rejects_a_non_count_k(bad_k):
    X, y = _data()
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.pls1_fit(X, y, k=bad_k)
    assert ei.value.code == "invalid_argument"
    assert "whole number" in str(ei.value)


@pytest.mark.parametrize(
    "extra, message",
    [
        ({"seed": -1}, "seed must be a whole number"),
        ({"pre_standardized": "no"}, "pre_standardized must be a bool"),
        ({"k": "optimal", "k_max": 3, "find_k_args": [1]},
         "find_k_args must be a record of named values"),
    ],
)
def test_fit_rejects_unusable_top_level_values(extra, message):
    """Checked on every call, not only on the branch that reads them: the
    seed of an int-k fit is unused but still validated."""
    X, y = _data()
    kwargs = {"k": 2, **extra}
    with pytest.raises(plskit.PlsKitError, match=message) as ei:
        plskit.pls1_fit(X, y, **kwargs)
    assert ei.value.code == "invalid_argument"


def _array_cases():
    X, y = _data()
    Y = np.column_stack([y, X[:, 0]])
    m = plskit.pls1_fit(X, y, k=2)
    ragged = [[1.0, 2.0]] * 79 + [[1.0]]
    return {
        "X string": (lambda: plskit.pls1_fit("abc", y, k=1),
                     'X must be a numeric matrix, got the string "abc"'),
        "y strings": (lambda: plskit.pls1_fit(X, ["a"] * 80, k=1),
                      "y must be a numeric vector, got an array of dtype <U1"),
        "X ragged": (lambda: plskit.pls1_fit(ragged, y, k=1),
                     "X must be a numeric matrix, got a value numpy cannot read as one"),
        "X complex": (lambda: plskit.pls1_fit(X + 0j, y, k=1),
                      "X must be a numeric matrix, got an array of dtype complex128"),
        "X objects": (lambda: plskit.pls1_fit(np.full((80, 6), object()), y, k=1),
                      "X must be a numeric matrix, got a value numpy cannot read as one"),
        "X numeric strings as objects": (
            lambda: plskit.pls1_fit(np.full((80, 6), "1.5", dtype=object), y, k=1),
            'X must be a numeric matrix, got an object array holding the string "1.5"'),
        "weights string": (lambda: plskit.pls1_fit(X, y, k=1, weights="w"),
                           'weights must be a numeric vector, got the string "w"'),
        "pls3 Y string": (lambda: plskit.pls3_fit(X, "abc", k=1),
                          'Y must be a numeric matrix, got the string "abc"'),
        "preprocess Y complex": (lambda: plskit.preprocess(Y=Y * 1j),
                                 "Y must be a numeric array, got an array of dtype complex128"),
        "model array": (lambda: plskit.pls1_predict(
                            dataclasses.replace(m, T=np.full(m.T.shape, "a")), X),
                        "model.T must be a numeric array, got an array of dtype <U1"),
    }


@pytest.mark.parametrize("case", list(_array_cases()))
def test_unreadable_array_is_invalid_argument(case):
    """An array argument numpy cannot read as real numbers raises
    `invalid_argument` naming the argument, as plskit-bind does, not numpy's
    `ValueError` / `TypeError`, and a complex one is not cast to its real
    part."""
    call, message = _array_cases()[case]
    with pytest.raises(plskit.PlsKitError) as ei:
        call()
    assert ei.value.code == "invalid_argument"
    assert str(ei.value).startswith(message), str(ei.value)


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda X, y: plskit.pls1_fit(np.zeros((4, 3, 2)), y, k=1),
         "X must be 2-D, got 3-D"),
        (lambda X, y: plskit.pls1_fit(X, np.array(3.0), k=1),
         "y must be 1-D, got 0-D"),
        (lambda X, y: plskit.preprocess(Y=np.array(3.0)),
         "Y must be 1-D or 2-D, got 0-D"),
        (lambda X, y: plskit.preprocess(Y=np.zeros((4, 3, 2))),
         "Y must be 1-D or 2-D, got 3-D"),
    ],
)
def test_array_of_the_wrong_dimension_is_invalid_argument(call, message):
    """An array of three or more dimensions, or a 0-D value where a vector
    is expected, raises `invalid_argument` naming the argument."""
    X, y = _data()
    with pytest.raises(plskit.PlsKitError) as ei:
        call(X, y)
    assert ei.value.code == "invalid_argument"
    assert str(ei.value).startswith(message), str(ei.value)


@pytest.mark.parametrize(
    "call, code, message",
    [
        (lambda X, y: plskit.pls1_predict(np.arange(4), X),
         "invalid_argument", "model must be a PLS1Result, got a vector"),
        (lambda X, y: plskit.pls1_fit(X, y, k=np.array([2])),
         "invalid_argument", "k must be a non-negative whole number, got a vector"),
        (lambda X, y: plskit.pls1_fit(X, y, k=2, pre_standardized=np.ones((2, 2), dtype=bool)),
         "invalid_argument", "pre_standardized must be a bool, got a matrix"),
        (lambda X, y: plskit.pls1_confirmatory_test(
            X, y, k=1, test_method="raw_perm", args={"n_perm": np.array([20])}, seed=1),
         "invalid_args",
         "args['n_perm'] for test_method='raw_perm' must be a non-negative whole number, "
         "got a vector"),
        (lambda X, y: plskit.pls1_fit(X, y, k=np.array(["a"])),
         "invalid_argument", "k has unsupported array dtype <U1"),
    ],
)
def test_array_where_no_array_fits_is_refused_by_what_the_parameter_takes(call, code, message):
    """An integer or bool array passed where the parameter takes no array is
    refused as a float64 one is: by what the parameter takes, not by its
    dtype. An array of no real dtype is refused by its dtype alone."""
    X, y = _data()
    with pytest.raises(plskit.PlsKitError) as ei:
        call(X, y)
    assert ei.value.code == code
    assert str(ei.value) == message


def test_bool_arrays_are_read_as_zero_one():
    """A bool data array is accepted and read as 1/0 (numpy's cast); a bool
    flag stays a flag."""
    X, y = _data()
    Xb, yb = X > 0, y > np.median(y)
    ref = plskit.pls1_fit(
        Xb.astype(np.float64), yb.astype(np.float64), k=2, weights=np.ones(len(y))
    )
    got = plskit.pls1_fit(Xb, yb, k=2, weights=np.ones(len(y), dtype=bool))
    np.testing.assert_array_equal(got.coef, ref.coef)
    np.testing.assert_array_equal(
        plskit.pls1_predict(ref, Xb), plskit.pls1_predict(ref, Xb.astype(np.float64))
    )
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.pls1_fit(X, y, k=2, pre_standardized=np.array([True, False]))
    assert ei.value.code == "invalid_argument"


def test_predict_reads_a_0d_array_model_field_as_its_item():
    """A scalar model field held as a 0-D array, as `np.load` returns the
    fields of a saved model, is read as its item."""
    X, y = _data()
    m = plskit.pls1_fit(X, y, k=2)
    flagged = dataclasses.replace(m, pre_standardized=np.array(False))
    np.testing.assert_array_equal(
        plskit.pls1_predict(flagged, X), plskit.pls1_predict(m, X)
    )


@pytest.mark.parametrize("missing", ["None", "pd.NA"])
def test_missing_value_in_an_object_array_is_non_finite(missing):
    """`None` and pandas' `pd.NA` (a nullable-dtype frame with a gap) both
    read as NaN, so the engine reports `non_finite_input` for either."""
    X, y = _data()
    X = X.astype(object)
    X[3, 1] = None if missing == "None" else pytest.importorskip("pandas").NA
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.pls1_fit(X, y, k=2)
    assert ei.value.code == "non_finite_input"


@pytest.mark.parametrize("whole_k", [2.0, np.array(2), np.int64(2)])
def test_fit_accepts_a_whole_number_k(whole_k):
    X, y = _data()
    a = plskit.pls1_fit(X, y, k=whole_k)
    b = plskit.pls1_fit(X, y, k=2)
    np.testing.assert_array_equal(a.coef, b.coef)


@pytest.mark.parametrize(
    "extra", [{"k_max": 4}, {"find_k_args": {"bogus": 1}}, {"find_k_args": {}}]
)
def test_fit_rejects_selection_options_with_an_integer_k(extra):
    X, y = _data()
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.pls1_fit(X, y, k=2, **extra)
    assert ei.value.code == "invalid_argument"
    assert str(ei.value) == (
        "k_max and find_k_args apply only when k is 'optimal' or 'sequence'"
    )
