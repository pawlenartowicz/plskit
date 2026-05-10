//! Smoke tests for `pls1_confirmatory_test` cases (Family D of Task 5).

use ndarray_npy::NpzReader;
use plskit_testdata_gen::cases::pls1_confirmatory_test::{
    e, raw_perm, score, split_nb, split_nb_ci, split_perm,
};
use plskit_testdata_gen::manifest::Case;
use tempfile::tempdir;

/// Assert that a case runner emits both fixture files, returns the expected
/// `name`/`function` strings, stores correct relative paths, and produces
/// 64-character hex hashes.
fn check_basic(case_fn: fn(&std::path::Path) -> anyhow::Result<Case>, expected_name: &str) {
    let dir = tempdir().unwrap();
    let case = case_fn(dir.path()).unwrap();
    assert_eq!(case.name, expected_name);
    assert_eq!(case.function, "pls1_confirmatory_test");
    assert_eq!(case.inputs, "inputs/pls1_confirmatory_inputs.npz");
    assert_eq!(
        case.outputs,
        format!("outputs/pls1_confirmatory_test/{expected_name}.npz")
    );
    assert!(dir.path().join(&case.inputs).exists());
    assert!(dir.path().join(&case.outputs).exists());
    assert_eq!(case.hashes.inputs_sha256.len(), 64);
    assert_eq!(case.hashes.outputs_sha256.len(), 64);
}

#[test]
fn raw_perm_emits_files() {
    check_basic(raw_perm, "pls1_confirmatory_raw_perm");
}

#[test]
fn split_nb_emits_files() {
    check_basic(split_nb, "pls1_confirmatory_split_nb");
}

#[test]
fn split_perm_emits_files() {
    check_basic(split_perm, "pls1_confirmatory_split_perm");
}

#[test]
fn score_emits_files() {
    check_basic(score, "pls1_confirmatory_score");
}

#[test]
fn e_emits_files() {
    check_basic(e, "pls1_confirmatory_e");
}

#[test]
fn split_nb_ci_emits_files() {
    check_basic(split_nb_ci, "pls1_confirmatory_split_nb_ci");
}

#[test]
fn split_nb_ci_includes_ci_bundle_fields() {
    let dir = tempdir().unwrap();
    let case = split_nb_ci(dir.path()).unwrap();
    let f = std::fs::File::open(dir.path().join(&case.outputs)).unwrap();
    let mut npz = NpzReader::new(f).unwrap();
    let names: Vec<String> = npz.names().unwrap();
    for required in [
        "n_boot",
        "m",
        "m_rate",
        "level",
        "beta_sign_z",
        "beta_sign_z_signed",
        "leverage_ci_lower",
        "leverage_ci_upper",
        "leverage_se",
        "beta_ci_lower",
        "beta_ci_upper",
        "beta_se",
        "holdout_corr_point",
        "holdout_corr_lower",
        "holdout_corr_upper",
        "holdout_corr_sd",
        "n_boot_finite",
        "n_boot_finite_holdout_corr",
    ] {
        assert!(
            names.contains(&required.to_string()),
            "CI variant missing field: {required}"
        );
    }
}
