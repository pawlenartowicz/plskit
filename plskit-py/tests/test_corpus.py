"""Run every fixture in testdata/ against the live Python wrapper."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import plskit

ROOT = Path(__file__).resolve().parents[2] / "testdata"
MANIFEST = ROOT / "manifest.json"

# Corpus tolerances (testdata/README.md "Tolerance"), numpy's rule
# |actual - expected| <= atol + rtol * |expected|, as in plskit-rs/tests/corpus.rs.
ATOL_SCALAR = 1e-12
ATOL_ARRAY = 1e-10
RTOL = 1e-14


def load_manifest():
    if not MANIFEST.exists():
        pytest.skip(
            f"{MANIFEST} missing: from the workspace root, run "
            f"`cargo run -p plskit-testdata-gen -- --testdata-root testdata`"
        )
    return json.loads(MANIFEST.read_text())["cases"]


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


@pytest.mark.parametrize("case", load_manifest(), ids=lambda c: c["name"])
def test_corpus_case(case):
    fn = case["function"]
    inputs = load_npz(case["inputs"])
    expected = load_npz(case["outputs"])
    X = inputs.get("X")
    y = inputs.get("y")
    kwargs = case["kwargs"]

    if fn == "pls1_fit":
        # Every manifest kwarg goes through, `seed` included: with a string
        # `k` ("sequence" / "optimal") pls1_fit forwards it to the K
        # selection, which is how the generator selected K for those
        # fixtures (the Rust corpus arm passes it the same way). With an
        # int `k` the fit is deterministic and pls1_fit ignores `seed`.
        # The generator refuses to write a "sequence" fixture whose
        # selection rejects no component, so a `sequence_no_rejection`
        # error here is a regression, not a stale fixture.
        fit_kwargs = _resolve_corpus_weights(case, dict(kwargs), inputs)
        r = plskit.pls1_fit(X, y, **fit_kwargs)
        for field in ["coef", "beta", "intercept", "k_used"]:
            if field in expected:
                assert_close(getattr(r, field), expected[field], f"{case['name']}.{field}")
    elif fn == "pls1_confirmatory_test":
        kw = dict(kwargs); k = kw.pop("k")
        kw = _resolve_corpus_weights(case, kw, inputs)
        r = plskit.pls1_confirmatory_test(X, y, k, **kw)
        for field in ["pvalue", "statistic", "method", "k", "n_perm", "n_splits", "seed", "stable_rank"]:
            if field in expected:
                assert_close(getattr(r, field), expected[field], f"{case['name']}.{field}")
        # CI fixture: when ci=True kwarg is set, also pin the CI bundle.
        if kw.get("ci"):
            assert r.ci is not None, f"{case['name']}: ci=True but result.ci is None"
            for ci_scalar_field in ["n_boot", "m", "m_rate", "level",
                                    "n_boot_finite", "n_boot_finite_holdout_corr"]:
                if ci_scalar_field in expected:
                    assert_close(
                        getattr(r.ci, ci_scalar_field),
                        expected[ci_scalar_field],
                        f"{case['name']}.ci.{ci_scalar_field}",
                    )
            for ci_arr_field in ["beta_sign_z", "beta_sign_z_signed",
                                 "leverage_ci_lower", "leverage_ci_upper",
                                 "leverage_se",
                                 "beta_ci_lower", "beta_ci_upper", "beta_se"]:
                if ci_arr_field in expected:
                    assert_close(
                        getattr(r.ci, ci_arr_field),
                        expected[ci_arr_field],
                        f"{case['name']}.ci.{ci_arr_field}",
                    )
            for composite in ["holdout_corr"]:
                ci_obj = getattr(r.ci, composite)
                for sub in ["point", "lower", "upper", "sd"]:
                    key = f"{composite}_{sub}"
                    if key in expected:
                        assert_close(
                            getattr(ci_obj, sub),
                            expected[key],
                            f"{case['name']}.ci.{composite}.{sub}",
                        )
    elif fn == "pls1_find_k_optimal":
        kw = dict(kwargs); k_max = kw.pop("k_max")
        r = plskit.pls1_find_k_optimal(X, y, k_max, **kw)
        for field in ["k_star", "selector", "pvalues", "diagnostic", "seed"]:
            if field in expected:
                assert_close(getattr(r, field), expected[field], f"{case['name']}.{field}")
        # cv_scores / cv_scores_se / bic_scores arrive as flattened keys/values
        for d_field in ("cv_scores", "cv_scores_se", "bic_scores"):
            keys_k = f"{d_field}__keys"
            if keys_k in expected:
                ks = expected[keys_k]
                vs = expected[f"{d_field}__values"]
                actual_dict = getattr(r, d_field)
                assert actual_dict is not None, f"{case['name']}.{d_field}"
                for k_int, v_exp in zip(ks.tolist(), vs.tolist()):
                    np.testing.assert_allclose(actual_dict[int(k_int)], v_exp,
                                               atol=ATOL_ARRAY, rtol=RTOL,
                                               err_msg=f"{case['name']}.{d_field}[{k_int}]")
    elif fn == "pls1_find_k_sequence":
        kw = dict(kwargs); k_max = kw.pop("k_max")
        r = plskit.pls1_find_k_sequence(X, y, k_max, **kw)
        for field in ["k_star", "pvalues", "test_method", "alpha", "seed"]:
            if field in expected:
                assert_close(getattr(r, field), expected[field], f"{case['name']}.{field}")
    elif fn == "pls1_predict":
        X_train = inputs["X_train"]
        y_train = inputs["y_train"]
        X_new = inputs["X_new"]
        k = int(kwargs["k"])
        model = plskit.pls1_fit(X_train, y_train, k=k)
        y_pred = plskit.pls1_predict(model, X_new)
        if "y_pred" in expected:
            assert_close(y_pred, expected["y_pred"], f"{case['name']}.y_pred")
        for field in ["coef", "beta", "intercept", "k_used"]:
            if field in expected:
                assert_close(getattr(model, field), expected[field], f"{case['name']}.{field}")
    elif fn == "rotate":
        k = int(kwargs["k"])
        method = kwargs.get("method", "varimax")
        model = plskit.pls1_fit(inputs["X"], inputs["y"], k=k)
        r = plskit.rotate(model.W, method=method)
        if "w_rot" in expected:
            assert_close(r.W_rot, expected["w_rot"], f"{case['name']}.w_rot")
        if "r" in expected:
            assert_close(r.spec.R, expected["r"], f"{case['name']}.r")
        if "sweeps" in expected:
            assert_close(r.spec.sweeps, expected["sweeps"], f"{case['name']}.sweeps")
        if "v_converged" in expected:
            assert_close(r.spec.V_converged, expected["v_converged"], f"{case['name']}.v_converged")
    elif fn == "preprocess":
        r = plskit.preprocess(
            X=inputs.get("X"),
            Y=inputs.get("y"),
            weights=inputs.get("weights"),
        )
        # npz uses lowercase y_std/y_mean/y_scale; PreprocessResult uses uppercase Y_*
        field_map = [
            ("X_std", "X_std"), ("X_mean", "X_mean"), ("X_scale", "X_scale"),
            ("Y_std", "y_std"), ("Y_mean", "y_mean"), ("Y_scale", "y_scale"),
            ("weights_normalized", "weights_normalized"), ("n_eff", "n_eff"),
        ]
        for attr, key in field_map:
            if key in expected:
                assert_close(getattr(r, attr), expected[key], f"{case['name']}.{key}")
    elif fn == "pls1_perm_null":
        # `d` / `n` describe the generated data, not the call. Everything
        # else goes through, so a kwarg this arm does not know about fails
        # the call instead of being dropped.
        kw = {k: v for k, v in kwargs.items() if k not in ("d", "n")}
        kw = _resolve_corpus_weights(case, kw, inputs)
        k = int(kw.pop("k"))
        kw["n_perm"] = int(kw["n_perm"])
        r = plskit.pls1_perm_null(inputs["X"], inputs["y"], k, **kw)
        for field in ["beta_ref", "beta_perm_mean", "beta_perm_sd", "beta_perm_z",
                      "n_perm", "k", "seed", "n_eff", "beta_perm_matrix"]:
            if field in expected:
                assert_close(getattr(r, field), expected[field], f"{case['name']}.{field}")
    elif fn == "pls1_rotation_stability":
        # As for pls1_perm_null: only the data-shape keys are dropped.
        kw = {k: v for k, v in kwargs.items() if k not in ("d", "n")}
        kw = _resolve_corpus_weights(case, kw, inputs)
        k = int(kw.pop("k"))
        for key, conv in (("n_boot", int), ("m_rate", float), ("level", float)):
            if key in kw:
                kw[key] = conv(kw[key])
        r = plskit.pls1_rotation_stability(inputs["X"], inputs["y"], k, **kw)
        # CIScalar bundle for variance_ratio (overall)
        for sub in ["point", "lower", "upper", "sd"]:
            key = f"variance_ratio_{sub}"
            if key in expected:
                assert_close(getattr(r.variance_ratio, sub), expected[key],
                             f"{case['name']}.{key}")
        # CIScalar per-axis bundles encoded as variance_ratio_per_axis_k{i}_{sub}
        assert len(r.variance_ratio_per_axis) == k, case["name"]
        for i, ci_scalar in enumerate(r.variance_ratio_per_axis):
            for sub in ["point", "lower", "upper", "sd"]:
                key = f"variance_ratio_per_axis_k{i}_{sub}"
                if key in expected:
                    assert_close(getattr(ci_scalar, sub), expected[key],
                                 f"{case['name']}.{key}")
        for field in ["variance_unrot", "variance_rot",
                      "variance_unrot_per_axis", "variance_rot_per_axis",
                      "n_boot", "m", "seed", "m_rate", "level", "n_boot_finite"]:
            if field in expected:
                assert_close(getattr(r, field), expected[field], f"{case['name']}.{field}")
        if "degenerate_baseline" in expected:
            assert bool(r.degenerate_baseline) == bool(int(expected["degenerate_baseline"])), \
                f"{case['name']}.degenerate_baseline mismatch"
    elif fn == "spls1_fit":
        # `seed` here is the data-generation seed: spls1_fit is deterministic
        # and takes none (the Rust corpus arm ignores it too).
        kw = {k: v for k, v in kwargs.items() if k not in ("seed",)}
        kw = _resolve_corpus_weights(case, kw, inputs)
        k_int = int(kw.pop("k"))
        keep = int(kw.pop("keep"))
        r = plskit.spls1_fit(X, y, k_int, keep, **kw)
        for field in ["coef", "beta", "intercept", "k_used", "keep"]:
            if field in expected:
                assert_close(getattr(r, field), expected[field], f"{case['name']}.{field}")
    elif fn == "spls1_find_keep_optimal":
        kw = _resolve_corpus_weights(case, dict(kwargs), inputs)
        k_int = int(kw.pop("k"))
        r = plskit.spls1_find_keep_optimal(X, y, k_int, **kw)
        for field in ["keep_star", "k", "seed"]:
            if field in expected:
                assert_close(getattr(r, field), expected[field], f"{case['name']}.{field}")
        if "keep_grid" in expected:
            assert r.keep_grid == [int(v) for v in expected["keep_grid"].tolist()], \
                f"{case['name']}.keep_grid"
        for d_field in ("cv_scores", "cv_scores_se"):
            keys_k = f"{d_field}__keys"
            if keys_k in expected:
                ks = expected[keys_k]
                vs = expected[f"{d_field}__values"]
                actual_dict = getattr(r, d_field)
                for k_int2, v_exp in zip(ks.tolist(), vs.tolist()):
                    np.testing.assert_allclose(actual_dict[int(k_int2)], v_exp,
                                               atol=ATOL_ARRAY, rtol=RTOL,
                                               err_msg=f"{case['name']}.{d_field}[{k_int2}]")
    elif fn == "spls1_find_k_optimal":
        kw = _resolve_corpus_weights(case, dict(kwargs), inputs)
        k_max = kw.pop("k_max")
        keep = int(kw.pop("keep"))
        r = plskit.spls1_find_k_optimal(X, y, k_max, keep, **kw)
        for field in ["k_star", "selector", "pvalues", "diagnostic", "seed"]:
            if field in expected:
                assert_close(getattr(r, field), expected[field], f"{case['name']}.{field}")
        for d_field in ("cv_scores", "cv_scores_se", "bic_scores"):
            keys_k = f"{d_field}__keys"
            if keys_k in expected:
                ks = expected[keys_k]
                vs = expected[f"{d_field}__values"]
                actual_dict = getattr(r, d_field)
                assert actual_dict is not None, f"{case['name']}.{d_field}"
                for k_int2, v_exp in zip(ks.tolist(), vs.tolist()):
                    np.testing.assert_allclose(actual_dict[int(k_int2)], v_exp,
                                               atol=ATOL_ARRAY, rtol=RTOL,
                                               err_msg=f"{case['name']}.{d_field}[{k_int2}]")
    elif fn == "spls1_find_k_sequence":
        kw = _resolve_corpus_weights(case, dict(kwargs), inputs)
        k_max = kw.pop("k_max")
        keep = int(kw.pop("keep"))
        r = plskit.spls1_find_k_sequence(X, y, k_max, keep, **kw)
        for field in ["k_star", "pvalues", "test_method", "alpha", "seed"]:
            if field in expected:
                assert_close(getattr(r, field), expected[field], f"{case['name']}.{field}")
    elif fn == "pls3_fit":
        kw = dict(kwargs)
        k = kw.pop("k")
        kw.pop("seed", None)          # data-generation seed; pls3_fit takes none
        r = plskit.pls3_fit(X, inputs["Y"], k, **kw)
        for field in ["U", "V", "singular_values", "x_scores", "y_scores", "k_used"]:
            if field in expected:
                assert_close(getattr(r, field), expected[field], f"{case['name']}.{field}")
    elif fn == "pls3_transform":
        k = kwargs["k"]
        which = kwargs["which"]
        m = plskit.pls3_fit(X, inputs["Y"], k)
        sc = plskit.pls3_transform(m, inputs["X_new"], inputs["Y_new"], which=which)
        for field in ["x_scores", "y_scores"]:
            if field in expected:
                assert_close(getattr(sc, field), expected[field], f"{case['name']}.{field}")
    elif fn == "spls3_fit":
        kw = dict(kwargs)
        k = kw.pop("k")
        keep_X = kw.pop("keep_X")
        keep_Y = kw.pop("keep_Y")
        r = plskit.spls3_fit(X, inputs["Y"], k, keep_X=keep_X, keep_Y=keep_Y, **kw)
        for field in [
            "U", "V", "singular_values", "x_scores", "y_scores", "k_used",
            "keep_X", "keep_Y",
        ]:
            if field in expected:
                assert_close(getattr(r, field), expected[field], f"{case['name']}.{field}")
        if "converged" in expected:
            assert_close(
                r.converged.astype(np.int64), expected["converged"],
                f"{case['name']}.converged",
            )
        if "n_iter" in expected:
            assert_close(
                r.n_iter.astype(np.int64), expected["n_iter"],
                f"{case['name']}.n_iter",
            )
    elif fn == "pls3_confirmatory_test":
        kw = dict(kwargs)
        k = kw.pop("k")
        r = plskit.pls3_confirmatory_test(X, inputs["Y"], k, **kw)
        for field in [
            "pvalue", "statistic", "method", "k", "n_perm", "n_splits",
            "n_eff", "seed", "stable_rank",
        ]:
            if field in expected:
                assert_close(getattr(r, field), expected[field], f"{case['name']}.{field}")
    else:
        pytest.fail(f"{case['name']}: no Python arm for manifest function {fn!r}")
