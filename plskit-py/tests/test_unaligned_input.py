"""Array marshalling: misaligned float64 input, and X's memory layout.

A byte-offset view of a buffer (`buf[1:].view(np.float64)`) can be
C-contiguous and still misaligned (`flags.aligned` is False). Rust may not
read `f64` data through a reference to misaligned memory, so the extension
refuses such an array and the Python wrapper copies it into an aligned one
first. Every result must be bit-identical to the same values passed aligned.

A C- or F-ordered float64 X is handed to the extension as it is, not copied.
The block inputs (`Y`, `Y_new`, `W`, `L`) are read in place when F-ordered
and copied column-major otherwise; every layout gives the same bits.
"""
import dataclasses

import numpy as np
import pytest

import plskit
from plskit import _api, _plskit


def _misaligned(a):
    a = np.asarray(a, dtype=np.float64)
    buf = np.zeros(a.nbytes + 16, dtype=np.uint8)
    start = (1 - buf.ctypes.data) % 8  # data starts at an address that is 1 mod 8
    v = buf[start:start + a.nbytes].view(np.float64).reshape(a.shape)
    v[...] = a
    if a.size:
        assert not v.flags.aligned
    assert v.flags.c_contiguous
    return v


def _bits(obj):
    """Nested structure of a result with every float replaced by its bits."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _bits(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {k: _bits(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_bits(v) for v in obj)
    if isinstance(obj, np.ndarray) and obj.dtype.kind == "f":
        return (obj.shape, np.ascontiguousarray(obj, dtype=np.float64).view(np.uint64).tobytes())
    if isinstance(obj, float):
        return ("f", np.float64(obj).view(np.uint64).item())
    if isinstance(obj, np.ndarray):
        return (obj.shape, obj.dtype.str, obj.tobytes())
    return obj


def _data():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 6))
    y = X[:, 0] - X[:, 1] + rng.normal(size=30)
    Y = np.column_stack([y, X[:, 2] + rng.normal(size=30)])
    w = rng.uniform(0.5, 2.0, size=30)
    return X, y, Y, w


def _with_arrays(model, mk):
    """`model` with every array field passed through `mk`."""
    return dataclasses.replace(model, **{
        f.name: mk(getattr(model, f.name))
        for f in dataclasses.fields(model)
        if isinstance(getattr(model, f.name), np.ndarray)
    })


def _fields(type_name, fields):
    """`make_result` for a direct extension call: the record's own fields."""
    return fields


# Direct extension calls, one per place an array can sit in the arguments.
def _direct_cases():
    X, y, Y, _ = _data()
    m1 = plskit.pls1_fit(X, y, k=2)
    m3 = plskit.pls3_fit(X, Y, k=2)
    return {
        "X": ("pls1_fit", lambda mk: {"X": mk(X), "y": y, "k": 2}),
        "vector": ("pls1_fit", lambda mk: {"X": X, "y": mk(y), "k": 2}),
        "block": ("pls3_fit", lambda mk: {"X": X, "Y": mk(Y), "k": 2}),
        "model_field": ("pls1_predict", lambda mk: {
            "model": dataclasses.replace(m1, T=mk(m1.T)), "X_new": X}),
        "model_dict_entry": ("pls3_transform", lambda mk: {
            "model": {**vars(m3), "U": mk(m3.U)}, "X_new": X, "which": "x_scores"}),
    }


@pytest.mark.parametrize("name", list(_direct_cases()))
def test_extension_refuses_a_misaligned_array(name):
    function, arguments = _direct_cases()[name]
    # The same call with the array aligned runs, so alignment alone is refused.
    _plskit.call(function, arguments(np.ascontiguousarray), _fields)
    with pytest.raises(_plskit.PlsKitException) as ei:
        _plskit.call(function, arguments(_misaligned), _fields)
    assert ei.value.code == "invalid_argument"


# In every case the first array `mk` makes is X (`W`, or the model's `T`,
# for `rotate`): `test_public_api_hands_the_extension_x_uncopied` looks for
# that one.
def _public_cases():
    X, y, Y, w = _data()
    m1 = plskit.pls1_fit(X, y, k=2)
    m3 = plskit.pls3_fit(X, Y, k=2)
    return {
        "pls1_fit": lambda mk: plskit.pls1_fit(mk(X), mk(y), k=2, weights=mk(w)),
        "spls1_fit": lambda mk: plskit.spls1_fit(mk(X), mk(y), k=2, keep=3, weights=mk(w)),
        "pls1_predict": lambda mk: plskit.pls1_predict(
            X_new=mk(X), model=_with_arrays(m1, mk)),
        "pls3_fit": lambda mk: plskit.pls3_fit(mk(X), mk(Y), k=2),
        "pls3_transform": lambda mk: plskit.pls3_transform(
            X_new=mk(X), Y_new=mk(Y), model=_with_arrays(m3, mk)),
        "preprocess": lambda mk: plskit.preprocess(mk(X), mk(Y), weights=mk(w)),
        "rotate": lambda mk: plskit.rotate(
            mk(np.asarray(m1.W)), method="varimax", L=mk(m1.P)),
        "rotate_model": lambda mk: plskit.rotate(_with_arrays(m1, mk), method="varimax"),
        "split_nb_gate": lambda mk: plskit.split_nb_gate(mk(X), weights=mk(w)),
        "pls1_perm_null": lambda mk: plskit.pls1_perm_null(
            mk(X), mk(y), k=1, n_perm=100, seed=4, weights=mk(w)),
        "pls1_find_k_optimal": lambda mk: plskit.pls1_find_k_optimal(
            mk(X), mk(y), 3, seed=5, weights=mk(w)),
        "pls1_confirmatory_test": lambda mk: plskit.pls1_confirmatory_test(
            mk(X), mk(y), k=1, test_method="split_exact", seed=6, weights=mk(w)),
    }


def _record_extension_arrays(monkeypatch):
    """The list that collects every array the wrapper hands the extension
    from here on, arrays inside a model included."""
    seen = []

    def collect(value):
        if isinstance(value, np.ndarray):
            seen.append(value)
        elif isinstance(value, dict):
            for v in value.values():
                collect(v)
        elif dataclasses.is_dataclass(value):
            for f in dataclasses.fields(value):
                collect(getattr(value, f.name))

    real = _plskit.call

    def call(name, arguments, make_result):
        collect(arguments)
        return real(name, arguments, make_result)

    monkeypatch.setattr(_plskit, "call", call)
    return seen


@pytest.mark.parametrize("name", list(_public_cases()))
def test_public_api_aligns_misaligned_input(name, monkeypatch):
    """A misaligned input gives the bits of the same values aligned, and
    every array that reaches the extension is aligned."""
    call = _public_cases()[name]
    ref = _bits(call(np.ascontiguousarray))
    seen = _record_extension_arrays(monkeypatch)
    assert _bits(call(_misaligned)) == ref
    assert seen and all(v.flags.aligned for v in seen)


def test_aligned_f64_returns_aligned():
    X = _misaligned(np.arange(6.0).reshape(2, 3))
    out = _api._aligned_f64(X, "X")
    assert out.flags.aligned and out.flags.c_contiguous
    assert np.array_equal(out, X)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("name", list(_public_cases()))
def test_public_api_hands_the_extension_x_uncopied(name, order, monkeypatch):
    """The wrapper passes a contiguous float64 X (`W`, or the model's `T`,
    for `rotate`) on in its own order: the array in the extension's
    arguments is the caller's."""
    call = _public_cases()[name]
    seen = _record_extension_arrays(monkeypatch)
    held = []

    def mk(a):
        held.append(np.array(a, dtype=np.float64, order=order))
        return held[-1]

    call(mk)
    assert any(v is held[0] for v in seen), [v.flags for v in seen]


def _block_layouts(a):
    """The same values in every layout the seam tells apart: C and F order
    (F also at an address 8 bytes past a buffer's start), and the strided
    and negative-stride views the extension copies."""
    a = np.asarray(a, dtype=np.float64)
    big = np.zeros((2 * a.shape[0], 2 * a.shape[1]))
    big[1::2, ::2] = a
    f_offset = np.zeros(a.size + 1)[1:].reshape(a.shape[::-1]).T
    f_offset[...] = a
    out = {
        "C": np.ascontiguousarray(a),
        "F": np.asfortranarray(a),
        "F_offset": f_offset,
        "strided": big[1::2, ::2],
        "reversed_rows": np.ascontiguousarray(a[::-1])[::-1],
        "reversed_cols": np.asfortranarray(a[:, ::-1])[:, ::-1],
    }
    assert out["F_offset"].flags.f_contiguous and out["F_offset"].flags.aligned
    for v in out.values():
        assert np.array_equal(v, a)
    return out


def _block_cases():
    # A shape and seed on which a row-major read of a C-ordered Y moves the
    # last bits of a pre-standardized k=1 PLS3 fit (seen on arm64 macOS),
    # so there the "C" layout below discriminates between reading C order
    # in place and copying it column-major.
    rng = np.random.default_rng(5)
    n, p, q = 143, 46, 32
    X = rng.standard_normal((n, p))
    Y = rng.standard_normal((n, q)) + X[:, :1]
    Xs = (X - X.mean(0)) / X.std(0, ddof=1)
    Ys = (Y - Y.mean(0)) / Y.std(0, ddof=1)
    y = Y[:, 0]
    pre = {"pre_standardized_X": True, "pre_standardized_Y": True}
    m3 = plskit.pls3_fit(Xs, Ys, k=2, **pre)
    m1 = plskit.pls1_fit(X, y, k=3)
    W = np.asarray(m1.W)
    L = W @ rng.standard_normal((3, 3))
    return {
        "pls3_fit": (Y, lambda Yv: plskit.pls3_fit(X, Yv, k=2)),
        "pls3_fit_pre": (Ys, lambda Yv: plskit.pls3_fit(Xs, Yv, k=1, **pre)),
        "spls3_fit_pre": (Ys, lambda Yv: plskit.spls3_fit(Xs, Yv, 1, 20, 10, **pre)),
        "pls3_confirmatory_test": (Y, lambda Yv: plskit.pls3_confirmatory_test(
            X, Yv, k=1, test_method="split_exact", args={"n_perm": 99}, seed=7)),
        "pls3_transform": (Ys, lambda Yv: plskit.pls3_transform(m3, Y_new=Yv, which="y_scores")),
        "preprocess": (Y, lambda Yv: plskit.preprocess(X, Yv)),
        "rotate_W": (W, lambda Wv: plskit.rotate(Wv, method="varimax")),
        "rotate_L": (L, lambda Lv: plskit.rotate(W, method="varimax", L=Lv)),
        "pls1_rotation_stability_L": (L, lambda Lv: plskit.pls1_rotation_stability(
            X, y, 3, L=Lv, n_boot=100, seed=3)),
    }


@pytest.mark.parametrize("name", list(_block_cases()))
def test_block_inputs_give_the_same_bits_in_every_layout(name):
    """`Y`, `Y_new`, `W` and `L` in any layout give the bits of the
    column-major copy the extension makes of a strided one."""
    value, call = _block_cases()[name]
    layouts = _block_layouts(value)
    ref = _bits(call(layouts.pop("strided")))
    for label, v in layouts.items():
        assert _bits(call(v)) == ref, label
