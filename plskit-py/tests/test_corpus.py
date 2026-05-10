"""Run every fixture in testdata/ against the live Python wrapper."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import plskit

ROOT = Path(__file__).resolve().parents[2] / "testdata"
MANIFEST = ROOT / "manifest.json"


def load_manifest():
    if not MANIFEST.exists():
        pytest.skip(
            f"{MANIFEST} missing — run "
            f"`cargo run -p plskit-testdata-gen -- --testdata-root plskit/testdata`"
        )
    return json.loads(MANIFEST.read_text())["cases"]


def test_manifest_is_v2():
    if not MANIFEST.exists():
        pytest.skip(f"{MANIFEST} missing")
    m = json.loads(MANIFEST.read_text())
    assert m["schema_version"] == 2
    assert "producing_version" in m
    assert all("outputs" in c and "hashes" in c for c in m["cases"])


def load_npz(rel_path: str) -> dict:
    with np.load(ROOT / rel_path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def assert_close(actual, expected, name: str, atol_scalar=1e-12, atol_array=1e-10):
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
        np.testing.assert_allclose(actual, float(expected), rtol=0, atol=atol_scalar,
                                   err_msg=name)
    else:
        np.testing.assert_allclose(actual, expected, rtol=0, atol=atol_array,
                                   err_msg=name)


@pytest.mark.parametrize("case", load_manifest(), ids=lambda c: c["name"])
def test_corpus_case(case):
    fn = case["function"]
    inputs = load_npz(case["inputs"])
    expected = load_npz(case["outputs"])
    X = inputs.get("X")
    y = inputs.get("y")
    kwargs = case["kwargs"]

    if fn == "pls1_fit":
        # `seed` is no longer accepted on pls1_fit (deterministic kernel — see
        # 2026-04-30 remediation E3); strip it from corpus kwargs and skip the
        # `seed` field check on the fixture.
        fit_kwargs = {k: v for k, v in kwargs.items() if k != "seed"}
        try:
            r = plskit.pls1_fit(X, y, **fit_kwargs)
        except plskit.PlsKitError as exc:
            # E1 in 2026-04-30 remediation: pls1_fit(k="sequence") now raises
            # when find_k_sequence rejects no component, where it previously
            # fell back to K=1. Pre-E1 fixtures captured the K=1 fallback
            # outputs and are stale for this case; the raise itself is the
            # correct new behavior.
            if exc.code == "sequence_no_rejection":
                pytest.skip(
                    f"fixture {case['name']} pre-dates E1 sequence_no_rejection "
                    f"raise; regenerate testdata to refresh"
                )
            raise
        for field in ["coef", "beta", "intercept", "k_used"]:
            if field in expected:
                assert_close(getattr(r, field), expected[field], f"{case['name']}.{field}")
    elif fn == "pls1_confirmatory_test":
        kw = dict(kwargs); k = kw.pop("k")
        r = plskit.pls1_confirmatory_test(X, y, k, **kw)
        for field in ["pvalue", "statistic", "method", "k", "n_perm", "n_splits", "seed"]:
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
                                               atol=1e-10,
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
        kw = {k: v for k, v in kwargs.items() if k not in ("d", "n")}
        k = int(kw.pop("k"))
        n_perm = int(kw.pop("n_perm"))
        seed = kw.pop("seed", None)
        disable_parallelism = kw.pop("disable_parallelism", False)
        r = plskit.pls1_perm_null(
            inputs["X"], inputs["y"], k,
            n_perm=n_perm,
            seed=seed,
            disable_parallelism=disable_parallelism,
        )
        for field in ["beta_ref", "beta_perm_mean", "beta_perm_sd", "beta_perm_z",
                      "n_perm", "k", "seed", "n_eff"]:
            if field in expected:
                assert_close(getattr(r, field), expected[field], f"{case['name']}.{field}")
    elif fn == "pls1_rotation_stability":
        kw = {k: v for k, v in kwargs.items() if k not in ("d", "n")}
        k = int(kw.pop("k"))
        n_boot = int(kw.pop("n_boot"))
        m_rate = float(kw.pop("m_rate"))
        level = float(kw.pop("level"))
        seed = kw.pop("seed", None)
        disable_parallelism = kw.pop("disable_parallelism", False)
        r = plskit.pls1_rotation_stability(
            inputs["X"], inputs["y"], k,
            n_boot=n_boot,
            m_rate=m_rate,
            level=level,
            seed=seed,
            disable_parallelism=disable_parallelism,
        )
        # CIScalar bundle for variance_ratio (overall)
        for sub in ["point", "lower", "upper", "sd"]:
            key = f"variance_ratio_{sub}"
            if key in expected:
                assert_close(getattr(r.variance_ratio, sub), expected[key],
                             f"{case['name']}.{key}")
        # CIScalar per-axis bundles encoded as variance_ratio_per_axis_k{i}_{sub}
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
    else:
        pytest.skip(f"unknown function {fn}")
