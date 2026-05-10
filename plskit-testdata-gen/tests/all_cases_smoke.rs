//! End-to-end smoke test for `cases::all_cases` — confirms every fixture
//! family registers correctly and the full corpus contains the expected
//! number of entries.

use plskit_testdata_gen::cases::all_cases;
use std::collections::HashSet;
use tempfile::tempdir;

#[test]
fn all_cases_produces_full_corpus() {
    let dir = tempdir().unwrap();
    let cases = all_cases(dir.path()).unwrap();
    assert_eq!(
        cases.len(),
        25,
        "expected exactly 25 cases, got {}",
        cases.len()
    );
    let functions: HashSet<_> = cases.iter().map(|c| c.function.clone()).collect();
    for f in [
        "pls1_fit",
        "pls1_find_k_optimal",
        "pls1_find_k_sequence",
        "pls1_confirmatory_test",
        "pls1_predict",
        "rotate",
        "preprocess",
        "pls1_perm_null",
        "pls1_rotation_stability",
    ] {
        assert!(functions.contains(f), "missing function family: {f}");
    }
    let names: HashSet<_> = cases.iter().map(|c| c.name.clone()).collect();
    assert!(names.contains("pls1_fit_small_n50_d10_k1"));
    assert!(names.contains("pls1_confirmatory_split_nb_ci"));
    assert!(names.contains("pls1_fit_skinny_n200_d5_k1"));
    assert!(names.contains("pls1_predict_basic_n80_d6_k2"));
    assert!(names.contains("rotate_varimax_d6_k2"));
    assert!(names.contains("preprocess_n50_d10_with_weights"));
    assert!(names.contains("pls1_perm_null_basic_n80_d6_k2"));
    assert!(names.contains("pls1_rotation_stability_n80_d6_k2"));
    for c in &cases {
        assert_eq!(
            c.hashes.inputs_sha256.len(),
            64,
            "case {} has bad inputs hash",
            c.name
        );
        assert_eq!(
            c.hashes.outputs_sha256.len(),
            64,
            "case {} has bad outputs hash",
            c.name
        );
        assert!(
            dir.path().join(&c.inputs).exists(),
            "inputs missing for {}",
            c.name
        );
        assert!(
            dir.path().join(&c.outputs).exists(),
            "outputs missing for {}",
            c.name
        );
        assert!(
            c.tolerance.is_some(),
            "tolerance should be populated for {}",
            c.name
        );
    }
}
