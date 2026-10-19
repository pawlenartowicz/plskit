"""Run every fixture in testdata/ against the live Python wrapper."""

from __future__ import annotations

import dataclasses
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

import plskit

# $PLSKIT_TESTDATA, else the monorepo's testdata/. A missing corpus skips the
# module unless PLSKIT_REQUIRE_CORPUS is "1"/"true" (CI), which makes it an
# error. Same variable names and semantics as helper-npz.R::corpus_root and
# corpus.jl: change together.
ROOT = Path(os.environ.get("PLSKIT_TESTDATA")
            or Path(__file__).resolve().parents[2] / "testdata")
MANIFEST = ROOT / "manifest.json"

# Corpus tolerances (testdata/README.md "Tolerance"), numpy's rule
# |actual - expected| <= atol + rtol * |expected|. The rule is owned by
# plskit-testdata-gen/src/settle.rs::close; assert_close and the direct
# assert_allclose calls below mirror it: change together.
ATOL_SCALAR = 1e-12
ATOL_ARRAY = 1e-10
RTOL = 1e-14

_MISSING = object()
# Fields of a CIScalar that a fixture key can end in.
CI_PARTS = ("point", "lower", "upper", "sd")


if not MANIFEST.exists():
    _msg = (f"{MANIFEST} missing: from the workspace root, run "
            f"`cargo run -p plskit-testdata-gen -- --testdata-root testdata`, "
            f"or set PLSKIT_TESTDATA")
    if os.environ.get("PLSKIT_REQUIRE_CORPUS", "").lower() in ("1", "true"):
        pytest.fail(f"PLSKIT_REQUIRE_CORPUS is set but {_msg}", pytrace=False)
    pytest.skip(_msg, allow_module_level=True)

CASES = json.loads(MANIFEST.read_text())["cases"]
assert CASES, f"{MANIFEST} lists no cases"


def load_npz(rel_path: str) -> dict:
    with np.load(ROOT / rel_path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def assert_close(actual, expected, name: str):
    if expected is None:
        assert actual is None, f"{name}: expected None, got {actual!r}"
        return
    # 0-d string ndarray, 1-D uint8 byte buffer (npz writer convention — see
    # plskit-testdata-gen/src/npz.rs::add_string), or plain str — compare as
    # strings.
    if (hasattr(expected, "dtype") and expected.dtype == np.uint8
            and getattr(expected, "ndim", 0) == 1):
        expected_val = expected.tobytes().decode("utf-8")
    elif (hasattr(expected, "item") and hasattr(expected, "dtype") and
          expected.dtype.kind in ("U", "S")):
        expected_val = expected.item()
    else:
        expected_val = expected
    if isinstance(expected_val, (str, bytes)):
        actual_str = actual.decode() if isinstance(actual, bytes) else str(actual)
        exp_str = expected_val.decode() if isinstance(expected_val, bytes) else expected_val
        assert actual_str == exp_str, f"{name}: expected {exp_str!r}, got {actual_str!r}"
        return
    # Mirrors plskit-testdata-gen/src/settle.rs::close: change together.
    if np.isscalar(expected) or (hasattr(expected, "shape") and expected.shape == ()):
        np.testing.assert_allclose(actual, float(expected), rtol=RTOL, atol=ATOL_SCALAR,
                                   err_msg=name)
    else:
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL_ARRAY,
                                   err_msg=name)


def _resolve_corpus_weights(case, kw, inputs):
    """`weights` in manifest kwargs is the descriptor string "nonuniform" (the
    only descriptor in use); the actual array lives in the NPZ. Both directions
    of mismatch are bugs: assert they agree, then substitute when present."""
    has_descriptor = kw.get("weights") == "nonuniform"
    has_array = "weights" in inputs
    assert has_descriptor == has_array, (
        f"{case['name']}: manifest weights descriptor and NPZ weights array disagree"
    )
    if has_array:
        kw["weights"] = inputs["weights"]
    return kw


def _fields(rec):
    if isinstance(rec, dict):
        return rec
    return {f.name: getattr(rec, f.name) for f in dataclasses.fields(rec)}


def _is_record(v):
    return dataclasses.is_dataclass(v) and not isinstance(v, type)


def _lookup(rec, key):
    """The field a fixture key names: exact, then ASCII case-insensitive, then
    into nested result dataclasses, recursively. `_MISSING` if absent."""
    fields = _fields(rec)
    if key in fields:
        return fields[key]
    for name, v in fields.items():
        if name.lower() == key.lower():
            return v
    for v in fields.values():
        if _is_record(v):
            found = _lookup(v, key)
            if found is not _MISSING:
                return found
    return _MISSING


def _resolve(rec, key, expected):
    """The result value a fixture key names, or `_MISSING`. `{f}__keys` /
    `{f}__values` encode an int-keyed dict field, `{f}_{point|lower|upper|sd}`
    (`{f}_k{i}_{part}` for items of a list field) a CIScalar. Same encoding
    and lookup order as resolve() in test-corpus.R and corpus.jl: change
    together."""
    if key.endswith("__keys"):
        m = _lookup(rec, key[:-6])
        return sorted(m) if isinstance(m, dict) else _MISSING
    if key.endswith("__values"):
        base = key[:-8]
        m = _lookup(rec, base)
        if not isinstance(m, dict):
            return _MISSING
        keys = [int(k) for k in expected[base + "__keys"].tolist()]
        return [m[k] for k in keys] if all(k in m for k in keys) else _MISSING
    found = _lookup(rec, key)
    if found is not _MISSING:
        return found
    base, _, part = key.rpartition("_")
    if part not in CI_PARTS:
        return _MISSING
    ci = _lookup(rec, base)
    if ci is _MISSING:
        m = re.fullmatch(r"(.*)_k([0-9]+)", base)
        if not m:
            return _MISSING
        items = _lookup(rec, m[1])
        i = int(m[2])
        if not isinstance(items, (list, tuple)) or i >= len(items):
            return _MISSING
        # The fixture must hold every item, so an extra result item fails.
        held = {int(j) for j in re.findall(rf"^{re.escape(m[1])}_k([0-9]+)_", "\n".join(expected), re.M)}
        assert len(items) == len(held), f"{m[1]}: {len(items)} items, fixture holds {len(held)}"
        ci = items[i]
    return getattr(ci, part, _MISSING) if _is_record(ci) else _MISSING


def _comparable(v):
    """Bool results compare against the fixture's integer encoding; lists
    (K grids, p-value lists) against its arrays."""
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, np.ndarray) and v.dtype == bool:
        return v.astype(np.int64)
    if isinstance(v, (list, tuple)):
        return np.asarray(v)
    return v


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_corpus_case(case):
    fn = case["function"]
    inputs = load_npz(case["inputs"])
    expected = load_npz(case["outputs"])
    assert expected, f"{case['name']}: fixture has no outputs"
    X = inputs.get("X")
    y = inputs.get("y")
    kwargs = _resolve_corpus_weights(case, dict(case["kwargs"]), inputs)

    if fn == "pls1_fit":
        # Every manifest kwarg goes through, `seed` included: with a string
        # `k` ("sequence" / "optimal") pls1_fit forwards it to the K
        # selection, which is how the generator selected K for those
        # fixtures (the Rust corpus arm passes it the same way). With an
        # int `k` the fit is deterministic and pls1_fit ignores `seed`.
        # The generator refuses to write a "sequence" fixture whose
        # selection rejects no component, so a `sequence_no_rejection`
        # error here is a regression, not a stale fixture.
        r = plskit.pls1_fit(X, y, **kwargs)
    elif fn == "pls1_confirmatory_test":
        k = kwargs.pop("k")
        r = plskit.pls1_confirmatory_test(X, y, k, **kwargs)
    elif fn == "pls1_find_k_optimal":
        k_max = kwargs.pop("k_max")
        r = plskit.pls1_find_k_optimal(X, y, k_max, **kwargs)
    elif fn == "pls1_find_k_sequence":
        k_max = kwargs.pop("k_max")
        r = plskit.pls1_find_k_sequence(X, y, k_max, **kwargs)
    elif fn == "pls1_predict":
        model = plskit.pls1_fit(inputs["X_train"], inputs["y_train"], k=int(kwargs["k"]))
        r = {"y_pred": plskit.pls1_predict(model, inputs["X_new"]), "model": model}
    elif fn == "rotate":
        model = plskit.pls1_fit(X, y, k=int(kwargs["k"]))
        r = plskit.rotate(model.W, method=kwargs.get("method", "varimax"))
    elif fn == "preprocess":
        r = plskit.preprocess(X=X, Y=y, weights=kwargs.get("weights"))
    elif fn == "pls1_perm_null":
        # `d` / `n` describe the generated data, not the call. Everything
        # else goes through, so a kwarg this arm does not know about fails
        # the call instead of being dropped.
        for key in ("d", "n"):
            kwargs.pop(key, None)
        k = int(kwargs.pop("k"))
        kwargs["n_perm"] = int(kwargs["n_perm"])
        r = plskit.pls1_perm_null(X, y, k, **kwargs)
    elif fn == "pls1_perm_alpha":
        r = plskit.pls1_perm_alpha(
            inputs["gram"], inputs["Y"], k=kwargs["k"], n_features=kwargs["n_features"]
        )
    elif fn == "pls1_rotation_stability":
        # As for pls1_perm_null: only the data-shape keys are dropped.
        for key in ("d", "n"):
            kwargs.pop(key, None)
        k = int(kwargs.pop("k"))
        for key, conv in (("n_boot", int), ("m_rate", float), ("level", float)):
            if key in kwargs:
                kwargs[key] = conv(kwargs[key])
        r = plskit.pls1_rotation_stability(X, y, k, **kwargs)
    elif fn == "spls1_fit":
        # `seed` here is the data-generation seed: spls1_fit is deterministic
        # and takes none (the Rust corpus arm ignores it too).
        kwargs.pop("seed", None)
        k = int(kwargs.pop("k"))
        keep = int(kwargs.pop("keep"))
        r = plskit.spls1_fit(X, y, k, keep, **kwargs)
    elif fn == "spls1_find_keep_optimal":
        k = int(kwargs.pop("k"))
        r = plskit.spls1_find_keep_optimal(X, y, k, **kwargs)
    elif fn == "spls1_find_k_optimal":
        k_max = kwargs.pop("k_max")
        keep = int(kwargs.pop("keep"))
        r = plskit.spls1_find_k_optimal(X, y, k_max, keep, **kwargs)
    elif fn == "spls1_find_k_sequence":
        k_max = kwargs.pop("k_max")
        keep = int(kwargs.pop("keep"))
        r = plskit.spls1_find_k_sequence(X, y, k_max, keep, **kwargs)
    elif fn == "pls3_fit":
        k = kwargs.pop("k")
        kwargs.pop("seed", None)          # data-generation seed; pls3_fit takes none
        r = plskit.pls3_fit(X, inputs["Y"], k, **kwargs)
    elif fn == "pls3_transform":
        m = plskit.pls3_fit(X, inputs["Y"], kwargs["k"])
        r = plskit.pls3_transform(m, inputs["X_new"], inputs["Y_new"], which=kwargs["which"])
    elif fn == "spls3_fit":
        k = kwargs.pop("k")
        keep_X = kwargs.pop("keep_X")
        keep_Y = kwargs.pop("keep_Y")
        r = plskit.spls3_fit(X, inputs["Y"], k, keep_X=keep_X, keep_Y=keep_Y, **kwargs)
    elif fn == "pls3_confirmatory_test":
        k = kwargs.pop("k")
        r = plskit.pls3_confirmatory_test(X, inputs["Y"], k, **kwargs)
    else:
        pytest.fail(f"{case['name']}: no Python arm for manifest function {fn!r}")

    for key in sorted(expected):
        found = _resolve(r, key, expected)
        if found is _MISSING:
            pytest.fail(f"{case['name']}.{key}: no matching result field", pytrace=False)
        assert_close(_comparable(found), expected[key], f"{case['name']}.{key}")
