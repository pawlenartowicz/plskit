//! Route pins for the corpus fixtures that exist to exercise one route, and
//! for the shapes of `tests/byte_parity.rs` / `tests/thread_count_parity.rs`
//! that exist to exercise a route.
//!
//! A route fixture is only useful while its shape, `k` and replicate count
//! select the route it was generated for. Each test here reads the
//! fixture's manifest entry and the shape of its `X` input and asserts the
//! route rules in force, so a later rule change that silently moves a
//! fixture onto another route fails here instead of turning the fixture
//! into a test of something else. Test-only.
//!
//! This is the crate's one module of route pins: a new route rule adds its
//! pins here, next to the ones it may flip, never in a module of its own.

use std::path::PathBuf;

use serde_json::Value;

use crate::dual_route::use_dual_route;
use crate::perm_null::perm_null_route;
use crate::signal_test::{
    raw_perm_k1_gram_route, raw_perm_route, split_exact_no_refit_route, split_exact_refit_route,
    ReplicateRoute,
};

fn testdata() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .join("testdata")
}

/// The manifest entry of fixture `name`.
fn case(name: &str) -> Value {
    let text = std::fs::read_to_string(testdata().join("manifest.json")).expect("manifest");
    let manifest: Value = serde_json::from_str(&text).expect("manifest json");
    manifest["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .find(|c| c["name"] == name)
        .unwrap_or_else(|| panic!("fixture {name} is not in testdata/manifest.json"))
        .clone()
}

/// `(n, p)` of the fixture's `X` input.
fn x_shape(case: &Value) -> (usize, usize) {
    let path = testdata().join(case["inputs"].as_str().expect("inputs path"));
    let bytes = std::fs::read(&path).expect("inputs npz");
    let mut npz = ndarray_npy::NpzReader::new(std::io::Cursor::new(bytes)).expect("npz");
    let entry = npz
        .names()
        .expect("npz names")
        .into_iter()
        .find(|n| n.trim_end_matches(".npy") == "X")
        .expect("X input");
    let x = npz
        .by_name::<ndarray::OwnedRepr<f64>, ndarray::IxDyn>(&entry)
        .expect("X array");
    (x.shape()[0], x.shape()[1])
}

fn kw(v: &Value, key: &str) -> usize {
    usize::try_from(v[key].as_u64().unwrap_or_else(|| panic!("kwarg {key}"))).expect("usize")
}

#[test]
fn perm_null_wide_fixtures_are_dense_and_n_space_eligible() {
    // A dense, unweighted input with `k + 1 <= n` whose shape passes
    // `use_dual_route(n, p, n_perm, k)` is what the n-space route serves.
    // These fixtures were generated on the primal route and are the n-space
    // route's reference, so they must stay inside that set.
    for (name, k_expected) in [
        ("pls1_perm_null_wide_n60_d3000_k1", 1),
        ("pls1_perm_null_wide_n60_d3000_k2", 2),
    ] {
        let c = case(name);
        let (n, p) = x_shape(&c);
        let kwargs = &c["kwargs"];
        let (k, n_perm) = (kw(kwargs, "k"), kw(kwargs, "n_perm"));
        assert_eq!(k, k_expected, "{name}: k");
        assert!(
            kwargs.get("weights").is_none(),
            "{name}: must be unweighted"
        );
        assert!(k < n, "{name}: k + 1 <= n");
        assert!(
            use_dual_route(n, p, n_perm, k),
            "{name}: shape passes use_dual_route"
        );
        assert!(
            crate::dual_route::nspace_eligible_perm_null(n, p, n_perm, k, true),
            "{name}: n-space eligible"
        );
        assert_eq!(
            perm_null_route(n, p, n_perm, k, false),
            ReplicateRoute::Nspace,
            "{name} must take the n-space route"
        );
    }
}

#[test]
fn perm_null_weighted_fixture_stays_on_the_primal_hoist() {
    let name = "pls1_perm_null_weighted_n80_d6_k2";
    let c = case(name);
    let (n, p) = x_shape(&c);
    let kwargs = &c["kwargs"];
    assert_eq!(kwargs["weights"], "nonuniform", "{name}: weighted");
    // Weighted input never takes the n-space route, and this narrow shape
    // would not pass its flop rule either. The p-space Gram route does serve
    // weighted input, but this shape's work is under its work floor.
    let (n_perm, k) = (kw(kwargs, "n_perm"), kw(kwargs, "k"));
    assert!(!use_dual_route(n, p, n_perm, k));
    assert!(!crate::gram_p::use_gram_p_route(n, p, n_perm, k), "{name}");
    assert_eq!(
        perm_null_route(n, p, n_perm, k, true),
        ReplicateRoute::Primal,
        "{name}: selector"
    );
}

#[test]
fn raw_perm_wide_k2_takes_the_n_space_route() {
    let name = "pls1_confirmatory_raw_perm_wide_k2";
    let c = case(name);
    let (n, p) = x_shape(&c);
    let kwargs = &c["kwargs"];
    assert_eq!(kwargs["test_method"], "raw_perm");
    assert!(kwargs.get("weights").is_none());
    let k = kw(kwargs, "k");
    let (n_perm, n_folds) = (
        kw(&kwargs["args"], "n_perm"),
        kw(&kwargs["args"], "n_folds"),
    );
    // The K = 1 Gram closed form does not claim it.
    assert!(!raw_perm_k1_gram_route(
        n, n_folds, p, n_perm, k, true, false
    ));
    assert!(
        crate::dual_route::nspace_eligible_raw_perm(n, n_folds, p, n_perm, k, true),
        "{name}: n-space eligible"
    );
    assert_eq!(
        raw_perm_route(n, n_folds, p, n_perm, k, None, false),
        ReplicateRoute::Nspace,
        "{name} must take the n-space route"
    );
    // The n-space rule's parts: k <= smallest training fold - 1 and the
    // flop rule at the largest training fold.
    let n_tr_min = n - n.div_ceil(n_folds);
    let n_tr_max = n - n / n_folds;
    assert!(k >= 2 && k < n_tr_min);
    assert!(use_dual_route(n_tr_max, p, n_perm + 1, k));
}

#[test]
fn split_exact_wide_k2_takes_the_refit_route_and_is_n_space_eligible() {
    let name = "pls1_confirmatory_split_exact_wide_k2";
    let c = case(name);
    let (n, p) = x_shape(&c);
    let kwargs = &c["kwargs"];
    assert_eq!(kwargs["test_method"], "split_exact");
    assert!(kwargs.get("weights").is_none());
    let k = kw(kwargs, "k");
    let n_perm = kw(&kwargs["args"], "n_perm");
    assert!(
        !split_exact_no_refit_route(k, None),
        "dense k = {k}: refit route"
    );
    let (n_train, _) = crate::resample::split_sizes(n, k);
    assert!(
        crate::dual_route::nspace_eligible_split_exact(n_train, p, n_perm, k, true),
        "{name}: n-space eligible"
    );
    assert_eq!(
        split_exact_refit_route(n, p, n_perm, k, None, false),
        ReplicateRoute::Nspace,
        "{name} must take the n-space route"
    );
    assert!(use_dual_route(n_train, p, n_perm + 1, k));
}

#[test]
fn sequence_keep3_fixture_reaches_the_sparse_refit_route() {
    let name = "spls1_find_k_sequence_split_exact_keep3";
    let c = case(name);
    let (n, p) = x_shape(&c);
    let kwargs = &c["kwargs"];
    assert_eq!(kwargs["test_method"], "split_exact");
    let keep = kw(kwargs, "keep");
    let n_perm = kw(&kwargs["args"], "n_perm");
    assert!(keep < p, "keep must select fewer than all columns");
    // Every step tests at k = 1 with `keep` set.
    assert!(!split_exact_no_refit_route(1, Some(keep)));
    assert_eq!(
        split_exact_refit_route(n, p, n_perm, 1, Some(keep), false),
        ReplicateRoute::Primal
    );
}

#[test]
fn raw_perm_wide_k1_fixture_takes_the_special_route() {
    // `pls1_confirmatory_raw_perm_wide` is the corpus reference of the K = 1
    // Gram closed form: its selector must keep choosing `Special`.
    let name = "pls1_confirmatory_raw_perm_wide";
    let c = case(name);
    let (n, p) = x_shape(&c);
    let kwargs = &c["kwargs"];
    assert_eq!(kwargs["test_method"], "raw_perm");
    assert!(kwargs.get("weights").is_none());
    let k = kw(kwargs, "k");
    let (n_perm, n_folds) = (
        kw(&kwargs["args"], "n_perm"),
        kw(&kwargs["args"], "n_folds"),
    );
    assert!(
        raw_perm_k1_gram_route(n, n_folds, p, n_perm, k, true, false),
        "{name}: predicate"
    );
    assert_eq!(
        raw_perm_route(n, n_folds, p, n_perm, k, None, false),
        ReplicateRoute::Special,
        "{name}"
    );
}

#[test]
fn split_exact_k1_fixtures_take_the_special_route() {
    // The corpus references of the K = 1 no-refit route, weighted and wide
    // (n 60, d 3000) included.
    for (name, weighted) in [
        ("pls1_confirmatory_split_exact_k1", false),
        ("pls1_confirmatory_weighted_split_exact", true),
        ("pls1_confirmatory_split_exact_wide_k1", false),
    ] {
        let c = case(name);
        let (n, p) = x_shape(&c);
        let kwargs = &c["kwargs"];
        assert_eq!(kwargs["test_method"], "split_exact");
        assert_eq!(kwargs.get("weights").is_some(), weighted, "{name}: weights");
        assert!(kwargs.get("keep").is_none(), "{name}: dense");
        let (k, n_perm) = (kw(kwargs, "k"), kw(&kwargs["args"], "n_perm"));
        assert!(split_exact_no_refit_route(k, None), "{name}: predicate");
        assert_eq!(
            split_exact_refit_route(n, p, n_perm, k, None, weighted),
            ReplicateRoute::Special,
            "{name}"
        );
    }
}

#[test]
fn narrow_fixtures_stay_on_the_primal_route() {
    // The narrow (n 80, d 6) fixtures are the primal route's references: no
    // Gram route may claim them. `weighted_split_exact_k2` is the corpus
    // reference of the weighted refit.
    let pin = |name: &str| {
        let c = case(name);
        let (n, p) = x_shape(&c);
        let kwargs = &c["kwargs"];
        let weighted = kwargs.get("weights").is_some();
        assert!(kwargs.get("keep").is_none(), "{name}: dense");
        let (k, args) = (kw(kwargs, "k"), &kwargs["args"]);
        let route = match kwargs["test_method"].as_str() {
            Some("raw_perm") => raw_perm_route(
                n,
                kw(args, "n_folds"),
                p,
                kw(args, "n_perm"),
                k,
                None,
                weighted,
            ),
            Some("split_exact") => {
                split_exact_refit_route(n, p, kw(args, "n_perm"), k, None, weighted)
            }
            _ => perm_null_route(n, p, kw(kwargs, "n_perm"), k, weighted),
        };
        assert_eq!(route, ReplicateRoute::Primal, "{name}");
    };
    pin("pls1_perm_null_basic_n80_d6_k2");
    pin("pls1_confirmatory_raw_perm");
    pin("pls1_confirmatory_split_exact");
    pin("pls1_confirmatory_weighted_split_exact_k2");
}

// ── Gram-backend tall fixtures ─────────────────────────────────────────
//
// Generated by the explicit-deflation kernel of plskit 0.5.0 and earlier (see
// CHANGELOG), so each is an independent reference for the Gram route, but only
// while it stays Gram-sized. Each pin asserts the site selector's route
// (`GramP`; the n-space route must never claim these tall shapes), the
// route rule `gram_p::use_gram_p_route` on the block, and the shape numbers: the X
// backend's work per block `B·(2k + 1)·n_tr·p` against the starting work
// floor, and the cost rule `gram_p::gram_p_cost_rule`. If a later change
// raises the floor above a fixture's work, or moves the cost rule's
// boundary past a fixture, the pin fails and the rule or the fixture needs
// revisiting.

/// Starting value of `GRAM_P_MIN_WORK`, in multiply-adds.
const GRAM_P_MIN_WORK_START: f64 = 1e8;

/// The numbers of one fixed block: `n_tr` training rows, `p` features, `b`
/// replicates (`n_perm` for `perm_null`, `n_perm + 1` elsewhere), `k`
/// components per replicate fit. The X backend's work `b·(2k + 1)·n_tr·p`
/// equals `expected_work` and clears the starting floor, and the cost rule
/// holds.
fn assert_gram_sized(name: &str, n_tr: usize, p: usize, b: usize, k: usize, expected_work: f64) {
    let work = (b * (2 * k + 1) * n_tr * p) as f64;
    assert!(
        (work - expected_work).abs() < 0.5,
        "{name}: work B·(2k+1)·n_tr·p = {work:e}, expected {expected_work:e}"
    );
    assert!(
        work >= GRAM_P_MIN_WORK_START,
        "{name}: work {work:e} is under the starting floor {GRAM_P_MIN_WORK_START:e}"
    );
    assert!(
        crate::gram_p::gram_p_cost_rule(n_tr as f64, p as f64, b as f64, k as f64),
        "{name}: gram_p_cost_rule fails"
    );
}

/// `pls1_perm_null_tall_n2000_d50_k3` pins the Gram-p route at `perm_null`.
#[test]
fn perm_null_tall_n2000_d50_k3_route_and_gram_shape() {
    let name = "pls1_perm_null_tall_n2000_d50_k3";
    let c = case(name);
    let (n, p) = x_shape(&c);
    let kwargs = &c["kwargs"];
    let (k, n_perm) = (kw(kwargs, "k"), kw(kwargs, "n_perm"));
    assert_eq!(
        (n, p, k, n_perm),
        (2000, 50, 3, 1000),
        "{name}: n, p, k, n_perm"
    );
    assert!(kwargs.get("weights").is_none(), "{name}: unweighted");
    assert_eq!(
        perm_null_route(n, p, n_perm, k, false),
        crate::signal_test::ReplicateRoute::GramP,
        "{name}: route"
    );
    assert!(crate::gram_p::use_gram_p_route(n, p, n_perm, k), "{name}");
    assert_gram_sized(name, n, p, n_perm, k, 7.0e8);
}

/// `pls1_perm_null_tall_weighted_n2000_d50_k2` pins the Gram-p route at
/// `perm_null` on weighted input.
#[test]
fn perm_null_tall_weighted_n2000_d50_k2_route_and_gram_shape() {
    let name = "pls1_perm_null_tall_weighted_n2000_d50_k2";
    let c = case(name);
    let (n, p) = x_shape(&c);
    let kwargs = &c["kwargs"];
    let (k, n_perm) = (kw(kwargs, "k"), kw(kwargs, "n_perm"));
    assert_eq!(
        (n, p, k, n_perm),
        (2000, 50, 2, 1000),
        "{name}: n, p, k, n_perm"
    );
    assert_eq!(kwargs["weights"], "nonuniform", "{name}: weighted");
    assert_eq!(
        perm_null_route(n, p, n_perm, k, true),
        crate::signal_test::ReplicateRoute::GramP,
        "{name}: route"
    );
    assert!(crate::gram_p::use_gram_p_route(n, p, n_perm, k), "{name}");
    assert_gram_sized(name, n, p, n_perm, k, 5.0e8);
}

/// `pls1_confirmatory_raw_perm_tall_k2` pins the Gram-p route at `raw_perm`.
#[test]
fn raw_perm_tall_k2_route_and_gram_shape() {
    let name = "pls1_confirmatory_raw_perm_tall_k2";
    let c = case(name);
    let (n, p) = x_shape(&c);
    let kwargs = &c["kwargs"];
    assert_eq!(kwargs["test_method"], "raw_perm", "{name}: test_method");
    assert!(kwargs.get("weights").is_none(), "{name}: unweighted");
    let k = kw(kwargs, "k");
    let (n_perm, n_folds) = (
        kw(&kwargs["args"], "n_perm"),
        kw(&kwargs["args"], "n_folds"),
    );
    assert_eq!(
        (n, p, k, n_perm, n_folds),
        (2000, 50, 2, 1000, 5),
        "{name}: n, p, k, n_perm, n_folds"
    );
    assert_eq!(
        crate::signal_test::raw_perm_route(n, n_folds, p, n_perm, k, None, false),
        crate::signal_test::ReplicateRoute::GramP,
        "{name}: route"
    );
    // The largest training fold, as `run_raw_perm` computes it.
    let n_tr = n - n / n_folds;
    assert_eq!(n_tr, 1600, "{name}: n_tr");
    assert!(
        crate::gram_p::use_gram_p_route(n_tr, p, n_perm + 1, k),
        "{name}"
    );
    assert_gram_sized(name, n_tr, p, n_perm + 1, k, 4.004e8);
}

/// `spls1_find_k_sequence_split_exact_tall_keep10` pins the Gram-p route at
/// the sparse `split_exact` refit.
#[test]
fn sequence_split_exact_tall_keep10_route_and_gram_shape() {
    let name = "spls1_find_k_sequence_split_exact_tall_keep10";
    let c = case(name);
    let (n, p) = x_shape(&c);
    let kwargs = &c["kwargs"];
    assert_eq!(kwargs["test_method"], "split_exact", "{name}: test_method");
    let (keep, n_perm) = (kw(kwargs, "keep"), kw(&kwargs["args"], "n_perm"));
    assert_eq!(
        (n, p, keep, n_perm),
        (2000, 200, 10, 1000),
        "{name}: n, p, keep, n_perm"
    );
    // Every step tests at k = 1 on the deflated residual, with `keep` set,
    // so the sparse refit route runs.
    assert_eq!(
        crate::signal_test::split_exact_refit_route(n, p, n_perm, 1, Some(keep), false),
        crate::signal_test::ReplicateRoute::GramP,
        "{name}: route"
    );
    let (n_tr, _) = crate::resample::split_sizes(n, 1);
    assert_eq!(n_tr, 1000, "{name}: n_tr");
    assert!(
        crate::gram_p::use_gram_p_route(n_tr, p, n_perm + 1, 1),
        "{name}"
    );
    assert_gram_sized(name, n_tr, p, n_perm + 1, 1, 6.006e8);
}

// ── Route pins for tests/byte_parity.rs and tests/thread_count_parity.rs ──
//
// Integration tests cannot call the crate's route selectors, so the shapes
// those files use to exercise the n-space and p-space Gram routes are
// pinned here, copied from them.

#[test]
fn byte_parity_shapes_take_their_routes() {
    use ReplicateRoute::{GramP, Nspace};
    // perm_null_wide_nspace_byte_parity (and thread_count_parity's n-space
    // perm_null row, k = 2): 40 x 2000, n_perm = 100.
    for k in [1, 2] {
        assert_eq!(
            perm_null_route(40, 2000, 100, k, false),
            Nspace,
            "perm_null k={k}"
        );
    }
    // confirmatory_raw_perm_wide_k2 / _many_folds_k2: n_perm = 50, n_folds 5
    // and 20.
    for n_folds in [5, 20] {
        assert_eq!(
            raw_perm_route(40, n_folds, 2000, 50, 2, None, false),
            Nspace,
            "raw_perm n_folds={n_folds}"
        );
    }
    // confirmatory_split_exact_wide_k2: n_perm = 50; thread_count_parity's
    // split_exact_nspace_is_pool_size_invariant: n_perm = 57.
    for n_perm in [50, 57] {
        assert_eq!(
            split_exact_refit_route(40, 2000, n_perm, 2, None, false),
            Nspace,
            "split_exact n_perm={n_perm}"
        );
    }
    // perm_null_gram_p_byte_parity (and thread_count_parity's p-space
    // perm_null row, dense): 2000 x 50, n_perm = 300, k = 2.
    for weighted in [false, true] {
        assert_eq!(
            perm_null_route(2000, 50, 300, 2, weighted),
            GramP,
            "perm_null weighted={weighted}"
        );
    }
    // confirmatory_raw_perm_gram_p_byte_parity: n_folds 5, n_perm 300, k 2.
    assert_eq!(
        raw_perm_route(2000, 5, 50, 300, 2, None, false),
        GramP,
        "raw_perm"
    );
    // confirmatory_split_exact_gram_p_keep_byte_parity: 2000 x 200,
    // n_perm 400, k 1, keep 10.
    assert_eq!(
        split_exact_refit_route(2000, 200, 400, 1, Some(10), false),
        GramP,
        "split_exact keep"
    );
}
