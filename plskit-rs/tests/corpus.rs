//! Integration test: load deterministic fixtures from `testdata/`
//! and assert the Rust core reproduces the expected output bit-near.
//! Resampling-based fixtures are exercised from plskit-py/tests/test_corpus.py
//! to avoid duplicating the wrapper-side kwarg dispatch in Rust.

use std::fs;
use std::path::PathBuf;

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .join("testdata")
}

#[test]
fn deterministic_pls1_fit_cases_match_corpus() {
    let manifest_path = corpus_dir().join("manifest.json");
    assert!(
        manifest_path.exists(),
        "corpus manifest missing at {} — RULE 3 requires the testdata corpus; run scripts/generate.py. \
         testdata/ is never excluded from the repo, so a `cargo test` checkout always has it.",
        manifest_path.display()
    );
    let manifest: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&manifest_path).unwrap()).unwrap();
    let cases = manifest["cases"].as_array().unwrap();
    let mut tested = 0;
    for case in cases {
        if case["function"].as_str() != Some("pls1_fit") {
            continue;
        }
        let kwargs = &case["kwargs"];
        // Skip string-mode k (e.g. "sequence", "optimal") — those paths use
        // RNG and are covered exhaustively from plskit-py/tests/test_corpus.py.
        if kwargs["k"].is_string() {
            continue;
        }
        let k = usize::try_from(kwargs["k"].as_u64().expect("fixed k")).expect("k fits usize");

        let inputs_path = corpus_dir().join(case["inputs"].as_str().unwrap());
        let expected_path = corpus_dir().join(case["outputs"].as_str().unwrap());
        let (x, y, weights) = load_inputs(&inputs_path);
        let expected = load_expected(&expected_path);

        // Manifest "weights": "nonuniform" descriptor and NPZ weights array must agree:
        // either both present or both absent.
        let case_name = case["name"].as_str().unwrap();
        let manifest_has_descriptor =
            kwargs.get("weights").and_then(|v| v.as_str()) == Some("nonuniform");
        let npz_has_array = weights.is_some();
        assert_eq!(
            manifest_has_descriptor, npz_has_array,
            "{case_name}: manifest weights descriptor and NPZ weights array disagree"
        );

        let m = plskit::pls1_fit(
            x.as_ref(),
            y.as_ref(),
            plskit::KSpec::Fixed(k),
            weights.as_ref().map(faer::Col::as_ref),
            plskit::FitOpts::default(),
        )
        .expect("fit");

        let coef_expected = expected.get("coef").expect("coef in expected");
        assert_col_close(
            &m.coef,
            coef_expected,
            1e-10,
            case["name"].as_str().unwrap(),
        );
        let beta_expected = expected.get("beta").expect("beta in expected");
        assert_col_close(
            &m.beta,
            beta_expected,
            1e-10,
            case["name"].as_str().unwrap(),
        );
        tested += 1;
    }
    assert!(tested > 0, "no deterministic pls1_fit cases found");
}

// Bridge ndarray-npy → faer at the test-only seam.
// `ndarray` is a dev-dependency only; nothing in the production build uses it.
fn load_inputs(path: &PathBuf) -> (faer::Mat<f64>, faer::Col<f64>, Option<faer::Col<f64>>) {
    let bytes = fs::read(path).unwrap();
    let mut npz = ndarray_npy::NpzReader::new(std::io::Cursor::new(bytes)).unwrap();
    let x_nd: ndarray::Array2<f64> = npz.by_name("X.npy").unwrap();
    let y_nd: ndarray::Array1<f64> = npz.by_name("y.npy").unwrap();
    let (n, d) = x_nd.dim();
    let x = faer::Mat::<f64>::from_fn(n, d, |i, j| x_nd[(i, j)]);
    let y = faer::Col::<f64>::from_fn(y_nd.len(), |i| y_nd[i]);
    // Load weights when present — absent in unweighted cases.
    // Not-found → None; any other error (wrong dtype, shape, corrupt) → panic.
    let weights = match npz.by_name::<ndarray::OwnedRepr<f64>, ndarray::Ix1>("weights.npy") {
        Ok(w_nd) => Some(faer::Col::<f64>::from_fn(w_nd.len(), |i| w_nd[i])),
        Err(ndarray_npy::ReadNpzError::Zip(zip::result::ZipError::FileNotFound)) => None,
        Err(err) => panic!(
            "load_inputs({}): unexpected error reading weights.npy: {err}",
            path.display()
        ),
    };
    (x, y, weights)
}

fn load_expected(path: &PathBuf) -> std::collections::HashMap<String, ndarray::ArrayD<f64>> {
    let bytes = fs::read(path).unwrap();
    let mut npz = ndarray_npy::NpzReader::new(std::io::Cursor::new(bytes)).unwrap();
    let names: Vec<String> = npz.names().unwrap();
    let mut out = std::collections::HashMap::new();
    for n in names {
        if let Ok(a) = npz.by_name::<ndarray::OwnedRepr<f64>, ndarray::IxDyn>(&n) {
            out.insert(n.trim_end_matches(".npy").to_string(), a);
        }
    }
    out
}

fn assert_col_close(
    actual: &faer::Col<f64>,
    expected: &ndarray::ArrayD<f64>,
    atol: f64,
    name: &str,
) {
    assert_eq!(actual.nrows(), expected.len(), "{name}: length mismatch");
    for (i, e) in expected.iter().enumerate() {
        let a = actual[i];
        let diff = (a - e).abs();
        assert!(diff < atol, "{name}: |{a} - {e}| = {diff} > {atol}");
    }
}
