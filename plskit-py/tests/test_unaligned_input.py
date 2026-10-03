"""Array marshalling: misaligned float64 input, and X's memory layout.

A byte-offset view of a buffer (`buf[1:].view(np.float64)`) can be
C-contiguous and still misaligned (`flags.aligned` is False). Rust may not
read `f64` data through a reference to misaligned memory, so both the Python
wrapper and the extension seam must copy such an array into an aligned one
before reading it. Every result must be bit-identical to the same values
passed aligned.

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


# Raw extension calls: the seam alone must cope, including arrays that
# reach it inside a model dict and a 2-D `Y` given to `preprocess`.
def _raw_cases():
    X, y, Y, w = _data()
    m1 = _plskit.pls1_fit(X, y, 2, weights=w)
    m3 = _plskit.pls3_fit(X, Y, 2)
    W = np.asarray(m1["W"])

    def mis_dict(d, mk):
        return {k: (mk(v) if isinstance(v, np.ndarray) else v) for k, v in d.items()}

    return {
        "pls1_fit": lambda mk: _plskit.pls1_fit(mk(X), mk(y), 2, weights=mk(w)),
        "pls1_fit_pre": lambda mk: _plskit.pls1_fit(
            mk((X - X.mean(0)) / X.std(0)), mk((y - y.mean()) / y.std()), 2,
            pre_standardized=True, weights=mk(w)),
        "spls1_fit": lambda mk: _plskit.spls1_fit(mk(X), mk(y), 2, 3, weights=mk(w)),
        "pls1_predict": lambda mk: _plskit.pls1_predict(mis_dict(m1, mk), mk(X)),
        "pls3_fit": lambda mk: _plskit.pls3_fit(mk(X), mk(Y), 2),
        "pls3_transform": lambda mk: _plskit.pls3_transform(mis_dict(m3, mk), mk(X), mk(Y)),
        "rotate": lambda mk: _plskit.rotate(mk(W), method="varimax", l=mk(np.asarray(m1["P"]))),
        "split_nb_gate": lambda mk: _plskit.split_nb_gate(mk(X), weights=mk(w)),
        "preprocess_1d": lambda mk: _plskit.preprocess(x=mk(X), y=mk(y), weights=mk(w)),
        "preprocess_2d": lambda mk: _plskit.preprocess(x=mk(X), y=mk(Y), weights=mk(w)),
    }


@pytest.mark.parametrize("name", list(_raw_cases()))
def test_raw_extension_reads_misaligned_arrays(name):
    call = _raw_cases()[name]
    ref = _bits(call(np.ascontiguousarray))
    got = _bits(call(_misaligned))
    assert got == ref


def _public_cases():
    X, y, Y, w = _data()
    m1 = plskit.pls1_fit(X, y, k=2)
    return {
        "pls1_fit": lambda mk: plskit.pls1_fit(mk(X), mk(y), k=2, weights=mk(w)),
        "spls1_fit": lambda mk: plskit.spls1_fit(mk(X), mk(y), k=2, keep=3, weights=mk(w)),
        "pls1_predict": lambda mk: plskit.pls1_predict(m1, mk(X)),
        "pls3_fit": lambda mk: plskit.pls3_fit(mk(X), mk(Y), k=2),
        "preprocess": lambda mk: plskit.preprocess(mk(X), mk(Y), weights=mk(w)),
        "rotate": lambda mk: plskit.rotate(mk(np.asarray(m1.W)), method="varimax"),
        "pls1_perm_null": lambda mk: plskit.pls1_perm_null(
            mk(X), mk(y), k=1, n_perm=100, seed=4, weights=mk(w)),
        "pls1_find_k_optimal": lambda mk: plskit.pls1_find_k_optimal(
            mk(X), mk(y), 3, seed=5, weights=mk(w)),
        "pls1_confirmatory_test": lambda mk: plskit.pls1_confirmatory_test(
            mk(X), mk(y), k=1, test_method="split_exact", seed=6, weights=mk(w)),
    }


@pytest.mark.parametrize("name", list(_public_cases()))
def test_public_api_hands_the_seam_misaligned_arrays(name, monkeypatch):
    """Every array the public API validates reaches the extension misaligned,
    so each entry's seam path is exercised, not just the wrapper's copy."""
    call = _public_cases()[name]
    ref = _bits(call(np.ascontiguousarray))
    real = _api._ensure_array

    def ensure_then_misalign(x, name, ndim):
        return _misaligned(real(x, name, ndim))

    monkeypatch.setattr(_api, "_ensure_array", ensure_then_misalign)
    assert _bits(call(np.ascontiguousarray)) == ref


def test_ensure_array_returns_aligned():
    X = _misaligned(np.arange(6.0).reshape(2, 3))
    out = _api._ensure_array(X, "X", 2)
    assert out.flags.aligned and out.flags.c_contiguous
    assert np.array_equal(out, X)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("name", list(_public_cases()))
def test_public_api_hands_the_extension_x_uncopied(name, order, monkeypatch):
    """The wrapper passes a contiguous float64 X (or `W`, for `rotate`) on
    in its own order: the array the extension receives is the caller's."""
    call = _public_cases()[name]
    seen = []

    def record(fn):
        def wrapped(*args, **kwargs):
            seen.extend(v for v in (*args, *kwargs.values()) if isinstance(v, np.ndarray))
            return fn(*args, **kwargs)
        return wrapped

    for attr in dir(_plskit):
        fn = getattr(_plskit, attr)
        if callable(fn) and not isinstance(fn, type):
            monkeypatch.setattr(_plskit, attr, record(fn))
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
