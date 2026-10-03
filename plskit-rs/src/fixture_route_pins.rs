//! Route pins for the corpus fixtures that exist to exercise one route, and
//! for the shapes of `tests/thread_count_parity.rs` that exist to exercise a
//! route.
//!
//! A route fixture is only useful while its shape, `k` and replicate count
//! select the route it was generated for. The table here reads each
//! fixture's manifest entry and the shape of its `X` input and asserts the
//! route its selector returns, so a later rule change that silently moves a
//! fixture onto another route fails here instead of turning the fixture
//! into a test of something else. Test-only.
//!
//! This is the crate's one module of route pins: a new route rule adds its
//! pins here, next to the ones it may flip, never in a module of its own.

use std::path::PathBuf;

use serde_json::Value;

use crate::perm_null::perm_null_route;
use crate::signal_test::{raw_perm_route, split_exact_refit_route, ReplicateRoute};

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

/// The route the site selector returns for fixture `name`, from its
/// manifest entry (function, method, `k`, replicate count, `keep`,
/// weights) and the shape of its `X` input.
fn route_of(name: &str) -> ReplicateRoute {
    let c = case(name);
    let (n, p) = x_shape(&c);
    let kwargs = &c["kwargs"];
    let weighted = kwargs.get("weights").is_some();
    let keep = kwargs.get("keep").map(|_| kw(kwargs, "keep"));
    let function = c["function"].as_str().expect("function");
    if function == "pls1_perm_null" {
        return perm_null_route(n, p, kw(kwargs, "n_perm"), kw(kwargs, "k"), weighted);
    }
    let args = &kwargs["args"];
    let n_perm = kw(args, "n_perm");
    match (function, kwargs["test_method"].as_str()) {
        ("pls1_confirmatory_test", Some("raw_perm")) => raw_perm_route(
            n,
            kw(args, "n_folds"),
            p,
            n_perm,
            kw(kwargs, "k"),
            keep,
            weighted,
        ),
        ("pls1_confirmatory_test", Some("split_exact")) => {
            split_exact_refit_route(n, p, n_perm, kw(kwargs, "k"), keep, weighted)
        }
        // Every step of the sequence tests at k = 1 on the deflated residual.
        ("spls1_find_k_sequence", Some("split_exact")) => {
            split_exact_refit_route(n, p, n_perm, 1, keep, weighted)
        }
        other => panic!("{name}: no route selector for {other:?}"),
    }
}

#[test]
fn route_fixtures_take_their_routes() {
    use ReplicateRoute::{GramP, Nspace, Primal, Special};
    for (name, route) in [
        // Wide (n 60, d 3000), dense and unweighted: generated on the
        // primal route, the n-space route's references.
        ("pls1_perm_null_wide_n60_d3000_k1", Nspace),
        ("pls1_perm_null_wide_n60_d3000_k2", Nspace),
        ("pls1_confirmatory_raw_perm_wide_k2", Nspace),
        ("pls1_confirmatory_split_exact_wide_k2", Nspace),
        // The references of the K = 1 Gram closed form (`raw_perm`) and of
        // the K = 1 no-refit route (`split_exact`), weighted and wide
        // included.
        ("pls1_confirmatory_raw_perm_wide", Special),
        ("pls1_confirmatory_split_exact_k1", Special),
        ("pls1_confirmatory_weighted_split_exact", Special),
        ("pls1_confirmatory_split_exact_wide_k1", Special),
        // Narrow (n 80, d 6): the primal route's references, under the
        // p-space Gram route's work floor. `weighted_split_exact_k2` is
        // the reference of the weighted refit, `keep3` of the sparse one.
        ("pls1_perm_null_basic_n80_d6_k2", Primal),
        ("pls1_perm_null_weighted_n80_d6_k2", Primal),
        ("pls1_confirmatory_raw_perm", Primal),
        ("pls1_confirmatory_split_exact", Primal),
        ("pls1_confirmatory_weighted_split_exact_k2", Primal),
        ("spls1_find_k_sequence_split_exact_keep3", Primal),
        // Tall: generated by the explicit-deflation kernel of plskit 0.5.0
        // and earlier (see CHANGELOG), so each is an independent reference
        // for the p-space Gram route while it stays Gram-sized. A change
        // that raises the work floor above a fixture's work, or moves the
        // cost rule's boundary past it, fails here, and the rule or the
        // fixture needs revisiting.
        ("pls1_perm_null_tall_n2000_d50_k3", GramP),
        ("pls1_perm_null_tall_weighted_n2000_d50_k2", GramP),
        ("pls1_confirmatory_raw_perm_tall_k2", GramP),
        ("spls1_find_k_sequence_split_exact_tall_keep10", GramP),
    ] {
        assert_eq!(route_of(name), route, "{name}");
    }
}

// ── Route pins for tests/thread_count_parity.rs ──
//
// Integration tests cannot call the crate's route selectors, so the shapes
// that file uses to exercise the n-space and p-space Gram routes are
// pinned here, copied from it.

#[test]
fn thread_count_parity_shapes_take_their_routes() {
    use ReplicateRoute::{GramP, Nspace};
    // thread_count_parity's n-space perm_null row: 40 x 2000, n_perm = 100,
    // k = 2.
    assert_eq!(
        perm_null_route(40, 2000, 100, 2, false),
        Nspace,
        "perm_null"
    );
    // thread_count_parity's split_exact_nspace_is_pool_size_invariant:
    // 40 x 2000, n_perm = 57, k = 2.
    assert_eq!(
        split_exact_refit_route(40, 2000, 57, 2, None, false),
        Nspace,
        "split_exact"
    );
    // thread_count_parity's p-space perm_null row, dense: 2000 x 50,
    // n_perm = 300, k = 2.
    assert_eq!(perm_null_route(2000, 50, 300, 2, false), GramP, "perm_null");
    // thread_count_parity's n-space raw_perm rows: 40 x 2000, n_perm = 50,
    // k = 2, n_folds 5 and 20.
    for n_folds in [5, 20] {
        assert_eq!(
            raw_perm_route(40, n_folds, 2000, 50, 2, None, false),
            Nspace,
            "raw_perm n_folds={n_folds}"
        );
    }
    // thread_count_parity's p-space raw_perm row: 2000 x 50, n_folds 5,
    // n_perm 300, k 2.
    assert_eq!(
        raw_perm_route(2000, 5, 50, 300, 2, None, false),
        GramP,
        "raw_perm"
    );
    // thread_count_parity's split_exact keep row: 2000 x 200, n_perm 400,
    // k 1, keep 10.
    assert_eq!(
        split_exact_refit_route(2000, 200, 400, 1, Some(10), false),
        GramP,
        "split_exact keep"
    );
}
