//! Route pins for the corpus fixtures that exist to exercise one route, and
//! for the shapes of `tests/thread_count_parity.rs` that exist to exercise a
//! route.
//!
//! A route fixture is only useful while its shape, `k` and replicate count
//! select the route it was generated for. The table here reads each
//! fixture's manifest entry and the shape of its `X` input and asserts the
//! route its selector returns, so a later rule change that silently moves a
//! fixture onto another route fails here instead of turning the fixture
//! into a test of something else. The `auto` fixtures are pinned by the
//! method stored in their outputs, so a regeneration cannot quietly give a
//! fixture named after one method another. Test-only.
//!
//! This is the crate's one module of route pins: a new route rule adds its
//! pins here, next to the ones it may flip, never in a module of its own.

use std::path::PathBuf;

use serde_json::Value;

use crate::perm_null::perm_null_route;
use crate::resample::split_sizes;
use crate::signal_test::{
    K1Route, ReplicateRoute, k1_route, raw_perm_route, split_exact_refit_route, test_moments_route,
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

/// The fixture's `X` input.
fn x_input(case: &Value) -> faer::Mat<f64> {
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
    faer::Mat::from_fn(x.shape()[0], x.shape()[1], |i, j| x[[i, j]])
}

/// `(n, p)` of the fixture's `X` input.
fn x_shape(case: &Value) -> (usize, usize) {
    let x = x_input(case);
    (x.nrows(), x.ncols())
}

/// The `test_method` stored in fixture `name`'s outputs (UTF-8 bytes).
fn stored_test_method(name: &str) -> String {
    let path = testdata().join(case(name)["outputs"].as_str().expect("outputs path"));
    let bytes = std::fs::read(&path).expect("outputs npz");
    let mut npz = ndarray_npy::NpzReader::new(std::io::Cursor::new(bytes)).expect("npz");
    let entry = npz
        .names()
        .expect("npz names")
        .into_iter()
        .find(|n| n.trim_end_matches(".npy") == "test_method")
        .expect("test_method output");
    let m = npz
        .by_name::<ndarray::OwnedRepr<u8>, ndarray::Ix1>(&entry)
        .expect("test_method array");
    String::from_utf8(m.to_vec()).expect("UTF-8")
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
            let n_splits = kw(args, "n_splits");
            split_exact_refit_route(
                n,
                p,
                n_perm,
                n_splits,
                kw(kwargs, "k"),
                keep,
                weighted,
                false,
            )
        }
        // Every step of the sequence tests at k = 1 on the deflated residual.
        ("spls1_find_k_sequence", Some("split_exact")) => {
            let n_splits = kw(args, "n_splits");
            split_exact_refit_route(n, p, n_perm, n_splits, 1, keep, weighted, false)
        }
        other => panic!("{name}: no route selector for {other:?}"),
    }
}

/// How the `split_exact` fixture `name` scores its test halves, from the
/// same manifest entry and `X` shape as [`route_of`]: on the K = 1 no-refit
/// route (`Special`) the `k1_route` arm and whether route A scores them from
/// moments (`test_moments_route`); on the `GramP` refit route `None` and
/// whether that loop does; on any other route `None` and false. The
/// sequence steps test at k = 1; `n_perm + 1` outcome columns over
/// `n_splits` splits.
fn split_exact_arm_of(name: &str) -> (Option<K1Route>, bool) {
    let c = case(name);
    let (n, p) = x_shape(&c);
    let kwargs = &c["kwargs"];
    let args = &kwargs["args"];
    let weighted = kwargs.get("weights").is_some();
    let k = if c["function"] == "spls1_find_k_sequence" {
        1
    } else {
        kw(kwargs, "k")
    };
    let (n_cols, n_splits) = (kw(args, "n_perm") + 1, kw(args, "n_splits"));
    let (n_train, n_test) = split_sizes(n, k);
    // At the default budget, so `PLSKIT_MEMORY_BUDGET` in the test environment cannot move the pin.
    let moments = || {
        crate::fit::with_budget(crate::fit::DEFAULT_MEMORY_BUDGET, || {
            test_moments_route(n, n_test, p, n_cols, n_splits, weighted)
        })
    };
    match route_of(name) {
        ReplicateRoute::Special => {
            let arm = k1_route(n, n_train, n_test, p, n_cols, n_splits, false);
            (Some(arm), arm == K1Route::A && moments())
        }
        ReplicateRoute::GramP => (None, moments()),
        _ => (None, false),
    }
}

#[test]
fn route_fixtures_take_their_routes() {
    use ReplicateRoute::{GramP, Nspace, Primal, Special};
    for (name, route) in [
        // Wide (n 60, d 3000), dense and unweighted: generated on the
        // primal route, the n-space route's references.
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
    // The arms inside those routes that score the test halves: route G
    // reads the all-row Gram matrix (wide), route A scores them from
    // moments where it is cheaper and unweighted, and so does the tall
    // Gram-p refit loop. No corpus shape takes route B, or the Gram-p
    // moment arm at k >= 2.
    for (name, arm) in [
        (
            "pls1_confirmatory_split_exact_wide_k1",
            (Some(K1Route::G), false),
        ),
        ("pls1_confirmatory_split_exact_k1", (Some(K1Route::A), true)),
        (
            "pls1_confirmatory_weighted_split_exact",
            (Some(K1Route::A), false),
        ),
        (
            "spls1_find_k_sequence_split_exact_tall_keep10",
            (None, true),
        ),
    ] {
        assert_eq!(split_exact_arm_of(name), arm, "{name}");
    }
}

#[test]
fn auto_fixtures_resolve_to_their_named_method() {
    for (name, method) in [
        ("pls1_confirmatory_auto_split_exact", "split_exact"),
        ("pls1_confirmatory_auto_split_nb", "split_nb"),
        ("pls3_confirmatory_auto_split_exact", "split_exact"),
        ("pls1_find_k_sequence_auto", "split_exact"),
    ] {
        assert_eq!(stored_test_method(name), method, "{name}");
    }
}

// ── Route pins for tests/thread_count_parity.rs ──
//
// Integration tests cannot call the crate's route selectors, so the shapes
// that file uses to exercise the n-space and p-space Gram routes and the
// K = 1 Gram closed form are pinned here, copied from it.

#[test]
fn thread_count_parity_shapes_take_their_routes() {
    use ReplicateRoute::{GramP, Nspace, Special};
    // thread_count_parity's n-space perm_null row: 40 x 2000, n_perm = 100,
    // k = 2.
    assert_eq!(
        perm_null_route(40, 2000, 100, 2, false),
        Nspace,
        "perm_null"
    );
    // thread_count_parity's split_exact_nspace_is_pool_size_invariant:
    // 40 x 2000, n_perm = 57, n_splits = 10, k = 2.
    assert_eq!(
        split_exact_refit_route(40, 2000, 57, 10, 2, None, false, false),
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
    // n_splits 5, k 1, keep 10.
    assert_eq!(
        split_exact_refit_route(2000, 200, 400, 5, 1, Some(10), false, false),
        GramP,
        "split_exact keep"
    );
    // thread_count_parity's K = 1 Gram raw_perm row: 40 x 20000,
    // n_folds 5, n_perm 50.
    assert_eq!(
        raw_perm_route(40, 5, 20_000, 50, 1, None, false),
        Special,
        "raw_perm k=1"
    );

    // thread_count_parity's three 20,000-column rows are there to sum over
    // several column blocks: each spans more than two blocks at the row
    // count its route passes to `linalg::stream_block_width`. Shapes copied
    // from that file, row counts from the routes (change together).
    //
    // The K = 1 Gram raw_perm row above: 5 folds of 8 rows, so
    // `dual_route::pls1_cv_r2_columns` builds its all-row Gram matrix
    // (`HeldOutGram::from_x`) in blocks sized for all 40 rows.
    assert!(
        20_000 > 2 * crate::linalg::stream_block_width(40),
        "raw_perm k=1 blocks"
    );
    // thread_count_parity's split_nb_gate 40 x 20000 row:
    // `signal_test::resolve_split_nb` sizes its blocks for all 40 rows.
    assert!(
        20_000 > 2 * crate::linalg::stream_block_width(40),
        "split_nb_gate blocks"
    );
    // thread_count_parity's pls3 split_exact 60 x 20000 x 3 row, k = 1:
    // the Gram route reads X through `HeldOutGram::from_x`, which streams
    // it in blocks sized for all 60 rows.
    assert!(
        20_000 > 2 * crate::linalg::stream_block_width(60),
        "pls3 split_exact blocks"
    );
}
