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
    if !manifest_path.exists() {
        eprintln!("manifest.json missing — run scripts/generate.py first; skipping");
        return;
    }
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
        let (x, y) = load_inputs(&inputs_path);
        let expected = load_expected(&expected_path);

        let m = plskit::pls1_fit(
            x.as_ref(),
            y.as_ref(),
            plskit::KSpec::Fixed(k),
            None,
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
fn load_inputs(path: &PathBuf) -> (faer::Mat<f64>, faer::Col<f64>) {
    let bytes = fs::read(path).unwrap();
    let mut npz = ndarray_npy::NpzReader::new(std::io::Cursor::new(bytes)).unwrap();
    let x_nd: ndarray::Array2<f64> = npz.by_name("X.npy").unwrap();
    let y_nd: ndarray::Array1<f64> = npz.by_name("y.npy").unwrap();
    let (n, d) = x_nd.dim();
    let x = faer::Mat::<f64>::from_fn(n, d, |i, j| x_nd[(i, j)]);
    let y = faer::Col::<f64>::from_fn(y_nd.len(), |i| y_nd[i]);
    (x, y)
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
