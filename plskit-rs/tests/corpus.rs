//! Integration test: load every fixture in `testdata/` and assert the Rust
//! core reproduces the frozen output, at the corpus tolerances (scalars
//! `atol=1e-12`, arrays `atol=1e-10`; integer and string fields exactly).
//!
//! One test per function family, each walking every manifest case of that
//! family. The call each arm makes is the one `plskit-testdata-gen` made to
//! produce the fixture, with its options read from the manifest `kwargs`
//! (the same kwargs `plskit-py/tests/test_corpus.py` forwards to the Python
//! wrapper) and everything the kwargs leave out at the core's defaults.
//! Resampling fixtures are reproducible here because the seed is in the
//! kwargs and the output is byte-identical across thread counts for a fixed
//! seed.
//!
//! Every arm checks every field its fixture holds, which is at least the
//! field set the Python arm compares: [`Fixture::finish`] fails a case that
//! leaves a stored field unchecked, so a field added to the generator
//! cannot go unverified on the Rust side. [`all_manifest_functions_are_covered`]
//! fails when the manifest gains a function family with no arm here.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use plskit::{
    CIOpts, ConfirmatoryArgs, ConfirmatoryMethod, ConfirmatoryTestInput, ConfirmatoryTestOpts,
    FindKOptimalOpts, FindKSequenceOpts, FindKeepOptimalOpts, FitOpts, KSpec, PermNullOpts,
    Pls3ConfirmatoryTestOpts, Pls3FitOpts, PreprocessInput, RotationMethod,
    RotationStabilityMethod, RotationStabilityOpts, Selector, TransformWhich, VarimaxArgs,
};
use serde_json::Value;

/// Corpus tolerance for 0-D fields.
const ATOL_SCALAR: f64 = 1e-12;
/// Corpus tolerance for array fields (and the flattened `{k: score}` maps).
const ATOL_ARRAY: f64 = 1e-10;

/// Function families with an arm in this file. Kept in step with the
/// `#[test]` functions below by [`all_manifest_functions_are_covered`].
const COVERED_FUNCTIONS: &[&str] = &[
    "pls1_fit",
    "pls1_find_k_optimal",
    "pls1_find_k_sequence",
    "pls1_confirmatory_test",
    "pls1_predict",
    "rotate",
    "preprocess",
    "pls1_perm_null",
    "pls1_rotation_stability",
    "spls1_fit",
    "spls1_find_keep_optimal",
    "spls1_find_k_optimal",
    "spls1_find_k_sequence",
    "pls3_fit",
    "pls3_transform",
    "pls3_confirmatory_test",
    "spls3_fit",
];

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .join("testdata")
}

fn manifest_cases() -> Vec<Value> {
    let manifest_path = corpus_dir().join("manifest.json");
    assert!(
        manifest_path.exists(),
        "corpus manifest missing at {}: the reference corpus is required \
         (`cargo run -p plskit-testdata-gen -- --testdata-root testdata`). \
         testdata/ is never excluded from the repo, so a `cargo test` checkout always has it.",
        manifest_path.display()
    );
    let manifest: Value =
        serde_json::from_str(&fs::read_to_string(&manifest_path).unwrap()).unwrap();
    manifest["cases"].as_array().unwrap().clone()
}

/// Every manifest case whose `function` field is `function`.
fn cases_for(function: &str) -> Vec<Value> {
    manifest_cases()
        .into_iter()
        .filter(|case| case["function"].as_str() == Some(function))
        .collect()
}

/// Run `check` on every case of `function`; fail if there are none.
///
/// Every case runs with faer's global parallelism disabled (process-wide;
/// this file is its own test binary). faer's operator `*` and high-level
/// decompositions read that global, whose default degree is the Rayon pool
/// size, so a core path reading it would round differently per
/// `RAYON_NUM_THREADS`; disabled, such a read panics and names the site.
fn for_each_case(function: &str, mut check: impl FnMut(&Value, &Inputs, &Fixture)) {
    faer::disable_global_parallelism();
    let cases = cases_for(function);
    assert!(!cases.is_empty(), "no {function} cases in the manifest");
    for case in &cases {
        let inputs = Inputs::load(&corpus_dir().join(case["inputs"].as_str().unwrap()));
        let fx = Fixture::load(case);
        check(case, &inputs, &fx);
        fx.finish();
    }
}

#[test]
fn all_manifest_functions_are_covered() {
    let in_manifest: BTreeSet<String> = manifest_cases()
        .iter()
        .map(|c| c["function"].as_str().unwrap().to_string())
        .collect();
    let covered: BTreeSet<String> = COVERED_FUNCTIONS.iter().map(ToString::to_string).collect();
    let missing: Vec<&String> = in_manifest.difference(&covered).collect();
    assert!(
        missing.is_empty(),
        "manifest functions with no Rust corpus arm: {missing:?}"
    );
}

// ── Inputs ────────────────────────────────────────────────────────────────

/// A fixture's input arrays, by npz entry name (`X`, `y`, `Y`, `weights`,
/// `X_train`, …).
struct Inputs(HashMap<String, ndarray::ArrayD<f64>>);

impl Inputs {
    fn load(path: &Path) -> Self {
        Self(read_npz::<f64>(path))
    }

    fn get(&self, name: &str) -> &ndarray::ArrayD<f64> {
        self.0
            .get(name)
            .unwrap_or_else(|| panic!("input array {name} missing"))
    }

    fn mat(&self, name: &str) -> faer::Mat<f64> {
        let a = self.get(name);
        assert_eq!(a.ndim(), 2, "input {name} is not 2-D");
        faer::Mat::<f64>::from_fn(a.shape()[0], a.shape()[1], |i, j| a[[i, j]])
    }

    fn col(&self, name: &str) -> faer::Col<f64> {
        let a = self.get(name);
        assert_eq!(a.ndim(), 1, "input {name} is not 1-D");
        faer::Col::<f64>::from_fn(a.len(), |i| a[[i]])
    }

    /// The `weights` input, cross-checked against the manifest's
    /// `"weights": "nonuniform"` descriptor: either both are present or
    /// neither is (the same check `test_corpus.py` makes).
    fn weights(&self, case: &Value) -> Option<faer::Col<f64>> {
        let has_descriptor =
            case["kwargs"].get("weights").and_then(Value::as_str) == Some("nonuniform");
        let has_array = self.0.contains_key("weights");
        assert_eq!(
            has_descriptor, has_array,
            "{}: manifest weights descriptor and NPZ weights array disagree",
            case["name"]
        );
        has_array.then(|| self.col("weights"))
    }
}

/// Every entry of `path` that deserializes as element type `T`, keyed by
/// name without the `.npy` suffix. Entries of any other dtype are skipped.
fn read_npz<T: ndarray_npy::ReadableElement>(path: &Path) -> HashMap<String, ndarray::ArrayD<T>> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut npz = ndarray_npy::NpzReader::new(std::io::Cursor::new(bytes)).unwrap();
    let names: Vec<String> = npz.names().unwrap();
    let mut out = HashMap::new();
    for n in names {
        if let Ok(a) = npz.by_name::<ndarray::OwnedRepr<T>, ndarray::IxDyn>(&n) {
            out.insert(n.trim_end_matches(".npy").to_string(), a);
        }
    }
    out
}

// ── Expected outputs ──────────────────────────────────────────────────────

/// A case's frozen outputs, split by dtype as the generator writes them:
/// `f64` values, `i64` counts / flags / map keys, and strings (a 1-D `u8`
/// array of UTF-8 bytes, `plskit-testdata-gen/src/npz.rs::add_string`).
/// Records which fields have been compared; [`Fixture::finish`] requires
/// all of them.
struct Fixture {
    name: String,
    f64s: HashMap<String, ndarray::ArrayD<f64>>,
    i64s: HashMap<String, ndarray::ArrayD<i64>>,
    strs: HashMap<String, String>,
    checked: RefCell<BTreeSet<String>>,
}

impl Fixture {
    fn load(case: &Value) -> Self {
        let path = corpus_dir().join(case["outputs"].as_str().unwrap());
        let strs = read_npz::<u8>(&path)
            .into_iter()
            .map(|(k, v)| {
                let s = String::from_utf8(v.iter().copied().collect())
                    .unwrap_or_else(|e| panic!("{k}: not UTF-8: {e}"));
                (k, s)
            })
            .collect();
        Self {
            name: case["name"].as_str().unwrap().to_string(),
            f64s: read_npz::<f64>(&path),
            i64s: read_npz::<i64>(&path),
            strs,
            checked: RefCell::new(BTreeSet::new()),
        }
    }

    fn has(&self, field: &str) -> bool {
        self.f64s.contains_key(field)
            || self.i64s.contains_key(field)
            || self.strs.contains_key(field)
    }

    fn mark(&self, field: &str) {
        self.checked.borrow_mut().insert(field.to_string());
    }

    fn f64_field(&self, field: &str) -> &ndarray::ArrayD<f64> {
        self.mark(field);
        self.f64s
            .get(field)
            .unwrap_or_else(|| panic!("{}: f64 field {field} missing from fixture", self.name))
    }

    fn i64_field(&self, field: &str) -> &ndarray::ArrayD<i64> {
        self.mark(field);
        self.i64s
            .get(field)
            .unwrap_or_else(|| panic!("{}: integer field {field} missing from fixture", self.name))
    }

    /// 0-D `f64` field, `atol = 1e-12`.
    fn scalar(&self, field: &str, actual: f64) {
        let e = self.f64_field(field);
        assert_eq!(
            e.ndim(),
            0,
            "{}.{field}: fixture value is not 0-D",
            self.name
        );
        let e = *e.iter().next().unwrap();
        assert!(
            close(actual, e, ATOL_SCALAR),
            "{}.{field}: |{actual} - {e}| = {:e} > {ATOL_SCALAR:e}",
            self.name,
            (actual - e).abs()
        );
    }

    /// 1-D `f64` field, `atol = 1e-10`.
    fn array(&self, field: &str, actual: &[f64]) {
        let e = self.f64_field(field);
        assert_eq!(
            e.ndim(),
            1,
            "{}.{field}: fixture value is not 1-D",
            self.name
        );
        assert_eq!(
            actual.len(),
            e.len(),
            "{}.{field}: length mismatch",
            self.name
        );
        for (i, (&a, &ev)) in actual.iter().zip(e.iter()).enumerate() {
            assert!(
                close(a, ev, ATOL_ARRAY),
                "{}.{field}[{i}]: |{a} - {ev}| = {:e} > {ATOL_ARRAY:e}",
                self.name,
                (a - ev).abs()
            );
        }
    }

    fn col(&self, field: &str, actual: &faer::Col<f64>) {
        let v: Vec<f64> = (0..actual.nrows()).map(|i| actual[i]).collect();
        self.array(field, &v);
    }

    /// 2-D `f64` field, `atol = 1e-10`.
    fn mat(&self, field: &str, actual: &faer::Mat<f64>) {
        let e = self.f64_field(field);
        assert_eq!(
            e.ndim(),
            2,
            "{}.{field}: fixture value is not 2-D",
            self.name
        );
        assert_eq!(
            (actual.nrows(), actual.ncols()),
            (e.shape()[0], e.shape()[1]),
            "{}.{field}: shape mismatch",
            self.name
        );
        for i in 0..actual.nrows() {
            for j in 0..actual.ncols() {
                let (a, ev) = (actual[(i, j)], e[[i, j]]);
                assert!(
                    close(a, ev, ATOL_ARRAY),
                    "{}.{field}[{i},{j}]: |{a} - {ev}| = {:e} > {ATOL_ARRAY:e}",
                    self.name,
                    (a - ev).abs()
                );
            }
        }
    }

    /// 0-D integer field, exact.
    fn int(&self, field: &str, actual: impl TryInto<i64, Error: std::fmt::Debug>) {
        let e = self.i64_field(field);
        assert_eq!(
            e.ndim(),
            0,
            "{}.{field}: fixture value is not 0-D",
            self.name
        );
        let actual: i64 = actual.try_into().unwrap();
        assert_eq!(actual, *e.iter().next().unwrap(), "{}.{field}", self.name);
    }

    /// 1-D integer field, exact.
    fn ints(&self, field: &str, actual: &[i64]) {
        let e = self.i64_field(field);
        assert_eq!(
            e.ndim(),
            1,
            "{}.{field}: fixture value is not 1-D",
            self.name
        );
        let e: Vec<i64> = e.iter().copied().collect();
        assert_eq!(actual, e.as_slice(), "{}.{field}", self.name);
    }

    /// String field, exact.
    fn string(&self, field: &str, actual: &str) {
        self.mark(field);
        let e = self
            .strs
            .get(field)
            .unwrap_or_else(|| panic!("{}: string field {field} missing from fixture", self.name));
        assert_eq!(actual, e, "{}.{field}", self.name);
    }

    /// An `Option` output the generator writes only when it is `Some`: the
    /// fixture holds the field exactly when `actual` is `Some`, and then
    /// `check` compares it.
    fn optional<T>(&self, field: &str, actual: Option<T>, check: impl FnOnce(&Self, T)) {
        match (self.has(field), actual) {
            (true, Some(v)) => check(self, v),
            (false, None) => {}
            (true, None) => panic!("{}.{field}: in the fixture but None", self.name),
            (false, Some(_)) => panic!("{}.{field}: Some but not in the fixture", self.name),
        }
    }

    /// A `{k: score}` map, stored flattened as `{field}__keys` (i64) and
    /// `{field}__values` (f64), at the array tolerance.
    fn score_map(&self, field: &str, actual: &BTreeMap<usize, f64>) {
        let keys: Vec<i64> = actual.keys().map(|&k| i64::try_from(k).unwrap()).collect();
        let values: Vec<f64> = actual.values().copied().collect();
        self.ints(&format!("{field}__keys"), &keys);
        self.array(&format!("{field}__values"), &values);
    }

    fn opt_score_map(&self, field: &str, actual: Option<&BTreeMap<usize, f64>>) {
        let keys = format!("{field}__keys");
        match (self.has(&keys), actual) {
            (true, Some(m)) => self.score_map(field, m),
            (false, None) => {}
            (has, _) => panic!(
                "{}.{field}: fixture has it = {has}, output has it = {}",
                self.name, !has
            ),
        }
    }

    /// Fail unless every field in the fixture was compared.
    fn finish(self) {
        let checked = self.checked.into_inner();
        let unchecked: Vec<&String> = self
            .f64s
            .keys()
            .chain(self.i64s.keys())
            .chain(self.strs.keys())
            .filter(|k| !checked.contains(*k))
            .collect();
        assert!(
            unchecked.is_empty(),
            "{}: fixture fields never compared: {unchecked:?}",
            self.name
        );
    }
}

/// `|a − e| ≤ atol`, with NaN equal to NaN and an infinity equal to itself
/// (numpy's `assert_allclose` defaults, which the Python side uses). The
/// exact `==` is what makes `inf` match `inf`, where the difference is NaN.
#[allow(clippy::float_cmp)]
fn close(a: f64, e: f64, atol: f64) -> bool {
    a == e || (a.is_nan() && e.is_nan()) || (a - e).abs() <= atol
}

// ── kwargs ────────────────────────────────────────────────────────────────

fn kw_usize(kw: &Value, key: &str) -> usize {
    usize::try_from(
        kw[key]
            .as_u64()
            .unwrap_or_else(|| panic!("kwarg {key} missing or not an integer: {kw}")),
    )
    .unwrap()
}

fn kw_opt_usize(kw: &Value, key: &str) -> Option<usize> {
    kw.get(key)
        .map(|v| usize::try_from(v.as_u64().expect("integer kwarg")).unwrap())
}

fn kw_opt_f64(kw: &Value, key: &str) -> Option<f64> {
    kw.get(key).map(|v| v.as_f64().expect("float kwarg"))
}

fn kw_bool(kw: &Value, key: &str) -> bool {
    kw.get(key)
        .is_some_and(|v| v.as_bool().expect("bool kwarg"))
}

fn kw_seed(kw: &Value) -> Option<u64> {
    kw.get("seed").map(|v| v.as_u64().expect("seed"))
}

fn method_from_str(s: &str) -> ConfirmatoryMethod {
    match s {
        "raw_perm" => ConfirmatoryMethod::RawPerm,
        "split_nb" => ConfirmatoryMethod::SplitNb,
        "split_exact" => ConfirmatoryMethod::SplitExact,
        "score" => ConfirmatoryMethod::Score,
        "e" => ConfirmatoryMethod::E,
        other => panic!("unknown method {other}"),
    }
}

/// `ConfirmatoryArgs` for `method` from the manifest's `args` object, any
/// count it omits at the core's per-method default.
fn confirmatory_args(method: ConfirmatoryMethod, args: &Value) -> ConfirmatoryArgs {
    match ConfirmatoryArgs::defaults_for(method) {
        ConfirmatoryArgs::RawPerm { n_perm, n_folds } => ConfirmatoryArgs::RawPerm {
            n_perm: kw_opt_usize(args, "n_perm").unwrap_or(n_perm),
            n_folds: kw_opt_usize(args, "n_folds").unwrap_or(n_folds),
        },
        ConfirmatoryArgs::SplitNb { n_splits, force } => ConfirmatoryArgs::SplitNb {
            n_splits: kw_opt_usize(args, "n_splits").unwrap_or(n_splits),
            force: args.get("force").map_or(force, |v| v.as_bool().unwrap()),
        },
        ConfirmatoryArgs::SplitExact { n_perm, n_splits } => ConfirmatoryArgs::SplitExact {
            n_perm: kw_opt_usize(args, "n_perm").unwrap_or(n_perm),
            n_splits: kw_opt_usize(args, "n_splits").unwrap_or(n_splits),
        },
        a @ (ConfirmatoryArgs::Score | ConfirmatoryArgs::E) => a,
    }
}

/// `FindKOptimalOpts` from `selector` / `diagnostic` / `args` / `seed`.
fn find_k_optimal_opts(kw: &Value) -> FindKOptimalOpts {
    let d = FindKOptimalOpts::default();
    let args = &kw["args"];
    FindKOptimalOpts {
        selector: match kw["selector"].as_str().expect("selector") {
            "r2_se" => Selector::R2Se,
            "r2_max" => Selector::R2Max,
            "bic" => Selector::Bic,
            other => panic!("unknown selector {other}"),
        },
        diagnostic: kw
            .get("diagnostic")
            .map(|v| method_from_str(v.as_str().unwrap())),
        n_folds: kw_opt_usize(args, "n_folds").unwrap_or(d.n_folds),
        n_perm: kw_opt_usize(args, "n_perm").unwrap_or(d.n_perm),
        n_splits: kw_opt_usize(args, "n_splits").unwrap_or(d.n_splits),
        seed: kw_seed(kw),
        ..d
    }
}

/// `FindKSequenceOpts` from `test_method` / `alpha` / `args` / `seed`.
fn find_k_sequence_opts(kw: &Value) -> FindKSequenceOpts {
    let d = FindKSequenceOpts::default();
    let args = &kw["args"];
    FindKSequenceOpts {
        test_method: kw
            .get("test_method")
            .map_or(d.test_method, |v| method_from_str(v.as_str().unwrap())),
        alpha: kw_opt_f64(kw, "alpha").unwrap_or(d.alpha),
        n_perm: kw_opt_usize(args, "n_perm").unwrap_or(d.n_perm),
        n_splits: kw_opt_usize(args, "n_splits").unwrap_or(d.n_splits),
        seed: kw_seed(kw),
        ..d
    }
}

// ── Shared field sets ─────────────────────────────────────────────────────

/// The `Pls1Model` fields every PLS1-fit fixture records.
fn check_pls1_model(fx: &Fixture, m: &plskit::Pls1Model) {
    fx.col("coef", &m.coef);
    fx.col("beta", &m.beta);
    fx.scalar("intercept", m.intercept);
    fx.int("k_used", m.k_used);
}

/// The fields every `pls1_find_k_optimal` / `spls1_find_k_optimal` fixture records.
fn check_find_k_optimal(fx: &Fixture, r: &plskit::FindKOptimalOutput) {
    fx.int("k_star", r.k_star);
    fx.string("selector", &r.selector);
    fx.opt_score_map("cv_scores", r.cv_scores.as_ref());
    fx.opt_score_map("cv_scores_se", r.cv_scores_se.as_ref());
    fx.opt_score_map("bic_scores", r.bic_scores.as_ref());
    fx.optional("pvalues", r.pvalues.as_ref(), |fx, p| fx.col("pvalues", p));
    fx.optional("diagnostic", r.diagnostic.as_deref(), |fx, s| {
        fx.string("diagnostic", s);
    });
    fx.int("seed", r.seed);
}

/// The fields every `pls1_find_k_sequence` / `spls1_find_k_sequence` fixture records.
fn check_find_k_sequence(fx: &Fixture, r: &plskit::FindKSequenceOutput) {
    fx.int("k_star", r.k_star);
    fx.col("pvalues", &r.pvalues);
    fx.string("test_method", &r.test_method);
    fx.scalar("alpha", r.alpha);
    fx.int("seed", r.seed);
}

/// A `CIScalar` stored as four 0-D fields `{prefix}_{point,lower,upper,sd}`.
fn check_ci_scalar(fx: &Fixture, prefix: &str, ci: &plskit::CIScalar) {
    fx.scalar(&format!("{prefix}_point"), ci.point);
    fx.scalar(&format!("{prefix}_lower"), ci.lower);
    fx.scalar(&format!("{prefix}_upper"), ci.upper);
    fx.scalar(&format!("{prefix}_sd"), ci.sd);
}

/// The three saliences fields every PLS3-family fixture records.
fn check_svd_outputs(fx: &Fixture, m: &plskit::Pls3Model) {
    fx.mat("U", &m.u_saliences);
    fx.mat("V", &m.v_saliences);
    fx.col("singular_values", &m.singular_values);
}

// ── PLS1 ──────────────────────────────────────────────────────────────────

/// Fixed-`k` cases fit directly. `k = "sequence"` mirrors the Python
/// wrapper and the generator: `pls1_find_k_sequence` at the core defaults
/// with the manifest seed, then a fixed fit at its `k_star`.
#[test]
fn pls1_fit_cases_match_corpus() {
    for_each_case("pls1_fit", |case, inputs, fx| {
        let kw = &case["kwargs"];
        let (x, y) = (inputs.mat("X"), inputs.col("y"));
        let weights = inputs.weights(case);
        let wref = weights.as_ref().map(faer::Col::as_ref);
        let k = match &kw["k"] {
            Value::String(mode) if mode == "sequence" => {
                let seq = plskit::pls1_find_k_sequence(
                    x.as_ref(),
                    y.as_ref(),
                    kw_usize(kw, "k_max"),
                    wref,
                    find_k_sequence_opts(kw),
                )
                .expect("find_k_sequence");
                assert!(
                    seq.k_star > 0,
                    "{}: sequence rejected no component",
                    fx.name
                );
                seq.k_star
            }
            Value::String(mode) => panic!("{}: unknown k mode {mode}", fx.name),
            _ => kw_usize(kw, "k"),
        };
        let m = plskit::pls1_fit(
            x.as_ref(),
            y.as_ref(),
            KSpec::Fixed(k),
            wref,
            FitOpts::default(),
        )
        .expect("fit");
        check_pls1_model(fx, &m);
    });
}

#[test]
fn pls1_find_k_optimal_cases_match_corpus() {
    for_each_case("pls1_find_k_optimal", |case, inputs, fx| {
        let kw = &case["kwargs"];
        let weights = inputs.weights(case);
        let r = plskit::pls1_find_k_optimal(
            inputs.mat("X").as_ref(),
            inputs.col("y").as_ref(),
            kw_usize(kw, "k_max"),
            weights.as_ref().map(faer::Col::as_ref),
            find_k_optimal_opts(kw),
        )
        .expect("find_k_optimal");
        check_find_k_optimal(fx, &r);
    });
}

#[test]
fn pls1_find_k_sequence_cases_match_corpus() {
    for_each_case("pls1_find_k_sequence", |case, inputs, fx| {
        let kw = &case["kwargs"];
        let weights = inputs.weights(case);
        let r = plskit::pls1_find_k_sequence(
            inputs.mat("X").as_ref(),
            inputs.col("y").as_ref(),
            kw_usize(kw, "k_max"),
            weights.as_ref().map(faer::Col::as_ref),
            find_k_sequence_opts(kw),
        )
        .expect("find_k_sequence");
        check_find_k_sequence(fx, &r);
    });
}

/// Every `pls1_confirmatory_test` fixture: the five methods, weighted and
/// unweighted, the wide `raw_perm` case that takes the Gram route, and the
/// `split_nb` case carrying the subsampling CI bundle.
#[test]
fn pls1_confirmatory_test_cases_match_corpus() {
    for_each_case("pls1_confirmatory_test", |case, inputs, fx| {
        let kw = &case["kwargs"];
        let (x, y) = (inputs.mat("X"), inputs.col("y"));
        let weights = inputs.weights(case);
        let method = method_from_str(kw["method"].as_str().expect("method"));
        let ci = kw_bool(kw, "ci").then(|| {
            let d = CIOpts::default();
            CIOpts {
                n_boot: kw_opt_usize(kw, "n_boot").unwrap_or(d.n_boot),
                m_rate: kw_opt_f64(kw, "m_rate").unwrap_or(d.m_rate),
                level: kw_opt_f64(kw, "level").unwrap_or(d.level),
                max_failure_rate: kw_opt_f64(kw, "max_failure_rate").unwrap_or(d.max_failure_rate),
            }
        });
        let r = plskit::pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: kw_usize(kw, "k"),
                weights: weights.as_ref().map(faer::Col::as_ref),
            },
            ConfirmatoryTestOpts {
                args: confirmatory_args(method, &kw["args"]),
                seed: kw_seed(kw),
                disable_parallelism: kw_bool(kw, "disable_parallelism"),
                ci,
                ..ConfirmatoryTestOpts::default()
            },
        )
        .expect("pls1_confirmatory_test");

        fx.scalar("pvalue", r.pvalue);
        fx.scalar("statistic", r.statistic);
        fx.string("method", &r.method);
        fx.int("k", r.k);
        fx.optional("n_perm", r.n_perm, |fx, v| fx.int("n_perm", v));
        fx.optional("n_splits", r.n_splits, |fx, v| fx.int("n_splits", v));
        fx.optional("stable_rank", r.stable_rank, |fx, v| {
            fx.scalar("stable_rank", v);
        });
        fx.int("seed", r.seed);

        assert_eq!(r.ci.is_some(), ci.is_some(), "{}: ci presence", fx.name);
        if let Some(c) = &r.ci {
            fx.int("n_boot", c.n_boot);
            fx.int("m", c.m);
            fx.scalar("m_rate", c.m_rate);
            fx.scalar("level", c.level);
            fx.array("beta_sign_z", &c.beta_sign_z);
            fx.array("beta_sign_z_signed", &c.beta_sign_z_signed);
            fx.array("leverage_ci_lower", &c.leverage_ci_lower);
            fx.array("leverage_ci_upper", &c.leverage_ci_upper);
            fx.array("leverage_se", &c.leverage_se);
            fx.array("beta_ci_lower", &c.beta_ci_lower);
            fx.array("beta_ci_upper", &c.beta_ci_upper);
            fx.array("beta_se", &c.beta_se);
            check_ci_scalar(fx, "holdout_corr", &c.holdout_corr);
            fx.int("n_boot_finite", c.n_boot_finite);
            fx.int("n_boot_finite_holdout_corr", c.n_boot_finite_holdout_corr);
        }
    });
}

#[test]
fn pls1_predict_cases_match_corpus() {
    for_each_case("pls1_predict", |case, inputs, fx| {
        let m = plskit::pls1_fit(
            inputs.mat("X_train").as_ref(),
            inputs.col("y_train").as_ref(),
            KSpec::Fixed(kw_usize(&case["kwargs"], "k")),
            None,
            FitOpts::default(),
        )
        .expect("fit");
        let y_pred = plskit::pls1_predict(&m, inputs.mat("X_new").as_ref()).expect("predict");
        fx.col("y_pred", &y_pred);
        check_pls1_model(fx, &m);
    });
}

/// Varimax on the `W*` of a fixed-`k` fit (`Pls1Model::w_star`, the `W`
/// the Python wrapper passes to `rotate`).
#[test]
fn rotate_cases_match_corpus() {
    for_each_case("rotate", |case, inputs, fx| {
        let kw = &case["kwargs"];
        assert_eq!(kw["method"].as_str(), Some("varimax"), "{}", fx.name);
        let m = plskit::pls1_fit(
            inputs.mat("X").as_ref(),
            inputs.col("y").as_ref(),
            KSpec::Fixed(kw_usize(kw, "k")),
            None,
            FitOpts::default(),
        )
        .expect("fit");
        let r = plskit::rotate(
            m.w_star.as_ref(),
            RotationMethod::Varimax(VarimaxArgs::default()),
            None,
        )
        .expect("rotate");
        fx.mat("w_rot", &r.w_rot);
        fx.mat("r", &r.r);
        fx.int("sweeps", r.sweeps);
        fx.scalar("v_converged", r.v_converged);
    });
}

#[test]
fn preprocess_cases_match_corpus() {
    for_each_case("preprocess", |case, inputs, fx| {
        let x = inputs.0.contains_key("X").then(|| inputs.mat("X"));
        let y = inputs.0.contains_key("y").then(|| inputs.col("y"));
        let weights = inputs.weights(case);
        let r = plskit::preprocess(PreprocessInput {
            x: x.as_ref().map(faer::Mat::as_ref),
            y: y.as_ref().map(faer::Col::as_ref),
            weights: weights.as_ref().map(faer::Col::as_ref),
        })
        .expect("preprocess");
        fx.optional("X_std", r.x_std.as_ref(), |fx, (xs, mean, scale)| {
            fx.mat("X_std", xs);
            fx.col("X_mean", mean);
            fx.col("X_scale", scale);
        });
        fx.optional("y_std", r.y_std.as_ref(), |fx, (ys, mean, scale)| {
            fx.col("y_std", ys);
            fx.scalar("y_mean", *mean);
            fx.scalar("y_scale", *scale);
        });
        fx.optional(
            "weights_normalized",
            r.weights_normalized.as_ref(),
            |fx, w| fx.col("weights_normalized", w),
        );
        fx.optional("n_eff", r.n_eff, |fx, v| fx.scalar("n_eff", v));
    });
}

#[test]
fn pls1_perm_null_cases_match_corpus() {
    for_each_case("pls1_perm_null", |case, inputs, fx| {
        let kw = &case["kwargs"];
        let weights = inputs.weights(case);
        let r = plskit::pls1_perm_null(
            inputs.mat("X").as_ref(),
            inputs.col("y").as_ref(),
            kw_usize(kw, "k"),
            weights.as_ref().map(faer::Col::as_ref),
            PermNullOpts {
                n_perm: kw_usize(kw, "n_perm"),
                return_perm_matrix: false,
                pre_standardized: false,
                disable_parallelism: kw_bool(kw, "disable_parallelism"),
                verbose: false,
            },
            kw_seed(kw),
        )
        .expect("perm_null");
        fx.array("beta_ref", &r.beta_ref);
        fx.array("beta_perm_mean", &r.beta_perm_mean);
        fx.array("beta_perm_sd", &r.beta_perm_sd);
        fx.array("beta_perm_z", &r.beta_perm_z);
        fx.int("n_perm", r.n_perm);
        fx.int("k", r.k);
        fx.int("seed", r.seed);
        fx.scalar("n_eff", r.n_eff);
    });
}

#[test]
fn pls1_rotation_stability_cases_match_corpus() {
    for_each_case("pls1_rotation_stability", |case, inputs, fx| {
        let kw = &case["kwargs"];
        let d = RotationStabilityOpts::default();
        let r = plskit::pls1_rotation_stability(
            inputs.mat("X").as_ref(),
            inputs.col("y").as_ref(),
            kw_usize(kw, "k"),
            RotationStabilityMethod::Varimax(VarimaxArgs::default()),
            None,
            None,
            RotationStabilityOpts {
                n_boot: kw_opt_usize(kw, "n_boot").unwrap_or(d.n_boot),
                m_rate: kw_opt_f64(kw, "m_rate").unwrap_or(d.m_rate),
                level: kw_opt_f64(kw, "level").unwrap_or(d.level),
                seed: kw_seed(kw),
                disable_parallelism: kw_bool(kw, "disable_parallelism"),
                ..d
            },
        )
        .expect("rotation_stability");
        check_ci_scalar(fx, "variance_ratio", &r.variance_ratio);
        for (a, ci) in r.variance_ratio_per_axis.iter().enumerate() {
            check_ci_scalar(fx, &format!("variance_ratio_per_axis_k{a}"), ci);
        }
        fx.scalar("variance_unrot", r.variance_unrot);
        fx.scalar("variance_rot", r.variance_rot);
        fx.array("variance_unrot_per_axis", &r.variance_unrot_per_axis);
        fx.array("variance_rot_per_axis", &r.variance_rot_per_axis);
        fx.int("n_boot", r.n_boot);
        fx.int("m", r.m);
        fx.int("seed", r.seed);
        fx.scalar("m_rate", r.m_rate);
        fx.scalar("level", r.level);
        fx.int("degenerate_baseline", i64::from(r.degenerate_baseline));
        fx.int("n_boot_finite", r.n_boot_finite);
    });
}

// ── sPLS1 ─────────────────────────────────────────────────────────────────

#[test]
fn spls1_fit_cases_match_corpus() {
    for_each_case("spls1_fit", |case, inputs, fx| {
        let kw = &case["kwargs"];
        let weights = inputs.weights(case);
        let m = plskit::spls1_fit(
            inputs.mat("X").as_ref(),
            inputs.col("y").as_ref(),
            KSpec::Fixed(kw_usize(kw, "k")),
            kw_usize(kw, "keep"),
            weights.as_ref().map(faer::Col::as_ref),
            FitOpts::default(),
        )
        .expect("spls1_fit");
        check_pls1_model(fx, &m);
        fx.int("keep", m.keep.expect("spls1_fit always sets keep"));
    });
}

#[test]
fn spls1_find_keep_optimal_cases_match_corpus() {
    for_each_case("spls1_find_keep_optimal", |case, inputs, fx| {
        let kw = &case["kwargs"];
        let weights = inputs.weights(case);
        let r = plskit::spls1_find_keep_optimal(
            inputs.mat("X").as_ref(),
            inputs.col("y").as_ref(),
            kw_usize(kw, "k"),
            weights.as_ref().map(faer::Col::as_ref),
            FindKeepOptimalOpts {
                seed: kw_seed(kw),
                ..FindKeepOptimalOpts::default()
            },
        )
        .expect("find_keep_optimal");
        fx.int("keep_star", r.keep_star);
        fx.int("k", r.k);
        fx.score_map("cv_scores", &r.cv_scores);
        fx.score_map("cv_scores_se", &r.cv_scores_se);
        let grid: Vec<i64> = r
            .keep_grid
            .iter()
            .map(|&v| i64::try_from(v).unwrap())
            .collect();
        fx.ints("keep_grid", &grid);
        fx.int("seed", r.seed);
    });
}

#[test]
fn spls1_find_k_optimal_cases_match_corpus() {
    for_each_case("spls1_find_k_optimal", |case, inputs, fx| {
        let kw = &case["kwargs"];
        let weights = inputs.weights(case);
        let r = plskit::spls1_find_k_optimal(
            inputs.mat("X").as_ref(),
            inputs.col("y").as_ref(),
            kw_usize(kw, "k_max"),
            kw_usize(kw, "keep"),
            weights.as_ref().map(faer::Col::as_ref),
            find_k_optimal_opts(kw),
        )
        .expect("spls1_find_k_optimal");
        check_find_k_optimal(fx, &r);
    });
}

#[test]
fn spls1_find_k_sequence_cases_match_corpus() {
    for_each_case("spls1_find_k_sequence", |case, inputs, fx| {
        let kw = &case["kwargs"];
        let weights = inputs.weights(case);
        let r = plskit::spls1_find_k_sequence(
            inputs.mat("X").as_ref(),
            inputs.col("y").as_ref(),
            kw_usize(kw, "k_max"),
            kw_usize(kw, "keep"),
            weights.as_ref().map(faer::Col::as_ref),
            find_k_sequence_opts(kw),
        )
        .expect("spls1_find_k_sequence");
        check_find_k_sequence(fx, &r);
    });
}

// ── PLS3 ──────────────────────────────────────────────────────────────────

#[test]
fn pls3_fit_cases_match_corpus() {
    for_each_case("pls3_fit", |case, inputs, fx| {
        let m = plskit::pls3_fit(
            inputs.mat("X").as_ref(),
            inputs.mat("Y").as_ref(),
            kw_usize(&case["kwargs"], "k"),
            None,
            Pls3FitOpts::default(),
        )
        .expect("pls3_fit");
        check_svd_outputs(fx, &m);
        fx.mat("x_scores", &m.x_scores);
        fx.mat("y_scores", &m.y_scores);
        fx.int("k_used", m.k_used);
    });
}

#[test]
fn pls3_transform_cases_match_corpus() {
    for_each_case("pls3_transform", |case, inputs, fx| {
        let kw = &case["kwargs"];
        let m = plskit::pls3_fit(
            inputs.mat("X").as_ref(),
            inputs.mat("Y").as_ref(),
            kw_usize(kw, "k"),
            None,
            Pls3FitOpts::default(),
        )
        .expect("pls3_fit");
        let which = match kw["which"].as_str().expect("which") {
            "x_scores" => TransformWhich::XScores,
            "y_scores" => TransformWhich::YScores,
            "both" => TransformWhich::Both,
            other => panic!("unknown which {other}"),
        };
        let (x_new, y_new) = (inputs.mat("X_new"), inputs.mat("Y_new"));
        let s = plskit::pls3_transform(&m, Some(x_new.as_ref()), Some(y_new.as_ref()), which)
            .expect("pls3_transform");
        fx.optional("x_scores", s.x_scores.as_ref(), |fx, v| {
            fx.mat("x_scores", v);
        });
        fx.optional("y_scores", s.y_scores.as_ref(), |fx, v| {
            fx.mat("y_scores", v);
        });
    });
}

#[test]
fn pls3_confirmatory_test_cases_match_corpus() {
    for_each_case("pls3_confirmatory_test", |case, inputs, fx| {
        let kw = &case["kwargs"];
        let method = method_from_str(kw["method"].as_str().expect("method"));
        let r = plskit::pls3_confirmatory_test(
            inputs.mat("X").as_ref(),
            inputs.mat("Y").as_ref(),
            kw_usize(kw, "k"),
            Pls3ConfirmatoryTestOpts {
                args: confirmatory_args(method, &kw["args"]),
                seed: kw_seed(kw),
                disable_parallelism: kw_bool(kw, "disable_parallelism"),
                ..Pls3ConfirmatoryTestOpts::default()
            },
        )
        .expect("pls3_confirmatory_test");
        fx.scalar("pvalue", r.pvalue);
        fx.scalar("statistic", r.statistic);
        fx.string("method", &r.method);
        fx.int("k", r.k);
        fx.optional("n_perm", r.n_perm, |fx, v| fx.int("n_perm", v));
        fx.optional("n_splits", r.n_splits, |fx, v| fx.int("n_splits", v));
        // The generator records `stable_rank` on the `split_nb` case only,
        // so it is compared where the fixture stores it.
        if fx.has("stable_rank") {
            fx.scalar("stable_rank", r.stable_rank.expect("stable_rank"));
        }
        fx.scalar("n_eff", r.n_eff);
        fx.int("seed", r.seed);
    });
}

/// `spls3_fit` is deterministic for the same reason `pls3_fit` is: the
/// alternation is seeded from the leading singular pair, never from an RNG.
/// `converged` and `n_iter` are the fields a `max_iter` / `tol` mismatch
/// between the generator and the manifest would move first.
#[test]
fn spls3_fit_cases_match_corpus() {
    for_each_case("spls3_fit", |case, inputs, fx| {
        let kw = &case["kwargs"];
        let d = Pls3FitOpts::default();
        let opts = Pls3FitOpts {
            max_iter: kw_opt_usize(kw, "max_iter").unwrap_or(d.max_iter),
            tol: kw_opt_f64(kw, "tol").unwrap_or(d.tol),
            ..d
        };
        let m = plskit::spls3_fit(
            inputs.mat("X").as_ref(),
            inputs.mat("Y").as_ref(),
            kw_usize(kw, "k"),
            kw_usize(kw, "keep_X"),
            kw_usize(kw, "keep_Y"),
            None,
            opts,
        )
        .expect("spls3_fit");
        check_svd_outputs(fx, &m);
        fx.mat("x_scores", &m.x_scores);
        fx.mat("y_scores", &m.y_scores);
        fx.int("k_used", m.k_used);
        fx.int("keep_X", m.keep_x.expect("spls3_fit always sets keep_x"));
        fx.int("keep_Y", m.keep_y.expect("spls3_fit always sets keep_y"));
        // `converged` is written as 0/1: the npz writer has no bool dtype.
        let converged: Vec<i64> = m
            .converged
            .as_ref()
            .expect("spls3_fit always sets converged")
            .iter()
            .map(|&b| i64::from(b))
            .collect();
        fx.ints("converged", &converged);
        let n_iter: Vec<i64> = m
            .n_iter
            .as_ref()
            .expect("spls3_fit always sets n_iter")
            .iter()
            .map(|&v| i64::try_from(v).unwrap())
            .collect();
        fx.ints("n_iter", &n_iter);
    });
}
