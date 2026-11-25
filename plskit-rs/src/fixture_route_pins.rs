//! Route pins for the corpus fixtures that exist to exercise one route, and
//! for the shapes of `tests/thread_count_parity.rs` that exist to exercise a
//! route. The file also holds the `pool_cases` section, a test that the
//! thread-count "pool" cases reach a pool-sensitive split.
//!
//! A route fixture is only useful while its shape, `k` and replicate count
//! select the route it was generated for. The table here reads each
//! fixture's manifest entry and the shapes of its inputs and asserts the
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
use crate::pls3_signal_test::pls3_split_exact_route;
use crate::signal_test::{
    K1Route, PrimalSource, ReplicateRoute, SplitRoute, TestArm, raw_perm_route,
    split_exact_refit_route, split_nb_route_choice,
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

/// The fixture's input `name`, a 2-D array.
fn input(case: &Value, name: &str) -> faer::Mat<f64> {
    let path = testdata().join(case["inputs"].as_str().expect("inputs path"));
    let bytes = std::fs::read(&path).expect("inputs npz");
    let mut npz = ndarray_npy::NpzReader::new(std::io::Cursor::new(bytes)).expect("npz");
    let entry = npz
        .names()
        .expect("npz names")
        .into_iter()
        .find(|n| n.trim_end_matches(".npy") == name)
        .unwrap_or_else(|| panic!("{name} input"));
    let x = npz
        .by_name::<ndarray::OwnedRepr<f64>, ndarray::IxDyn>(&entry)
        .unwrap_or_else(|e| panic!("{name} array: {e}"));
    faer::Mat::from_fn(x.shape()[0], x.shape()[1], |i, j| x[[i, j]])
}

/// `(n, p)` of the fixture's `X` input.
fn x_shape(case: &Value) -> (usize, usize) {
    let x = input(case, "X");
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
/// weights) and the shapes of its inputs.
fn route_of(name: &str) -> ReplicateRoute {
    // At the default budget, so `PLSKIT_MEMORY_BUDGET` in the test environment cannot move the pin.
    crate::fit::with_budget(crate::fit::DEFAULT_MEMORY_BUDGET, || {
        let c = case(name);
        let (n, p) = x_shape(&c);
        let kwargs = &c["kwargs"];
        let weighted = kwargs.get("weights").is_some();
        let keep = kwargs.get("keep").map(|_| kw(kwargs, "keep"));
        let function = c["function"].as_str().expect("function");
        if function == "pls1_perm_null" {
            return perm_null_route(n, p, kw(kwargs, "n_perm"), kw(kwargs, "k"), weighted, false)
                .unwrap()
                .route;
        }
        let args = &kwargs["args"];
        match (function, kwargs["test_method"].as_str()) {
            ("pls1_confirmatory_test", Some("raw_perm")) => {
                raw_perm_route(
                    n,
                    kw(args, "n_folds"),
                    p,
                    kw(args, "n_perm"),
                    &[kw(kwargs, "k")],
                    keep,
                    weighted,
                )
                .unwrap()
                .route
            }
            ("pls1_confirmatory_test", Some("split_exact")) => {
                let n_splits = kw(args, "n_splits");
                split_exact_refit_route(
                    n,
                    p,
                    kw(args, "n_perm"),
                    n_splits,
                    &[kw(kwargs, "k")],
                    keep,
                    weighted,
                    false,
                )
                .unwrap()
                .route
                .route
            }
            // Every step of the sequence tests at k = 1 on the deflated residual.
            ("spls1_find_k_sequence", Some("split_exact")) => {
                let n_splits = kw(args, "n_splits");
                split_exact_refit_route(
                    n,
                    p,
                    kw(args, "n_perm"),
                    n_splits,
                    &[1],
                    keep,
                    weighted,
                    false,
                )
                .unwrap()
                .route
                .route
            }
            // When it takes X (not a Gram matrix), the real call owns the
            // gate's Gram matrix exactly when n <= p and frees it off the
            // routes that do not read it (`signal_test::resolve_split_nb`,
            // change together).
            ("pls1_confirmatory_test", Some("split_nb")) => {
                split_nb_route_choice(
                    n,
                    p,
                    kw(args, "n_splits"),
                    kw(kwargs, "k"),
                    keep,
                    weighted,
                    n <= p,
                    false,
                )
                .unwrap_or_else(|e| panic!("{name}: {e}"))
                .route
                .route
            }
            ("pls3_confirmatory_test", Some("split_exact")) => pls3_split_route_of(name).route,
            other => panic!("{name}: no route selector for {other:?}"),
        }
    })
}

/// The route [`pls3_split_exact_route`] returns for the PLS3 `split_exact`
/// fixture `name`, at the default budget, from its manifest entry and the
/// shapes of its `X` and `Y` inputs.
fn pls3_split_route_of(name: &str) -> SplitRoute {
    let c = case(name);
    let (n, p) = x_shape(&c);
    let q = input(&c, "Y").ncols();
    let kwargs = &c["kwargs"];
    let args = &kwargs["args"];
    let keep_x = kwargs.get("keep_X").map(|_| kw(kwargs, "keep_X"));
    // At the default budget, so `PLSKIT_MEMORY_BUDGET` in the test environment cannot move the pin.
    crate::fit::with_budget(crate::fit::DEFAULT_MEMORY_BUDGET, || {
        pls3_split_exact_route(
            keep_x,
            n,
            p,
            q,
            kw(args, "n_perm"),
            kw(args, "n_splits"),
            false,
        )
    })
    .unwrap_or_else(|e| panic!("{name}: {e}"))
    .route
}

/// How the `split_exact` fixture `name` scores its test halves, from the
/// same manifest entry and `X` shape as [`route_of`], as its selector
/// (`split_exact_refit_route`) picks them: on the K = 1 no-refit route
/// (`Special`) its `K1Route` and whether it scores them from moments (with
/// or without the cells); on the `GramP` refit route `None` and whether
/// that loop does; on any other route `None` and false. The sequence steps
/// test at k = 1; `n_perm + 1` outcome columns over `n_splits` splits.
fn split_exact_arm_of(name: &str) -> (Option<K1Route>, bool) {
    let c = case(name);
    let (n, p) = x_shape(&c);
    let kwargs = &c["kwargs"];
    let args = &kwargs["args"];
    let weighted = kwargs.get("weights").is_some();
    let keep = kwargs.get("keep").map(|_| kw(kwargs, "keep"));
    let k = if c["function"] == "spls1_find_k_sequence" {
        1
    } else {
        kw(kwargs, "k")
    };
    let (n_perm, n_splits) = (kw(args, "n_perm"), kw(args, "n_splits"));
    // At the default budget, so `PLSKIT_MEMORY_BUDGET` in the test environment cannot move the pin.
    let route = crate::fit::with_budget(crate::fit::DEFAULT_MEMORY_BUDGET, || {
        split_exact_refit_route(n, p, n_perm, n_splits, &[k], keep, weighted, false)
    })
    .unwrap()
    .route;
    let moments = route.arm != TestArm::Rows;
    match route.route {
        ReplicateRoute::Special => (Some(route.k1), moments),
        ReplicateRoute::GramP => (None, moments),
        _ => (None, false),
    }
}

#[test]
fn route_fixtures_take_their_routes() {
    use ReplicateRoute::{GramP, Nspace, Special};
    let primal = ReplicateRoute::Primal(PrimalSource::Copy);
    for (name, route) in [
        // Wide (n 60, d 3000), dense and unweighted: generated on the
        // primal route, the n-space route's references.
        ("pls1_perm_null_wide_n60_d3000_k2", Nspace),
        ("pls1_confirmatory_raw_perm_wide_k2", Nspace),
        ("pls1_confirmatory_split_exact_wide_k2", Nspace),
        // The fixture that runs the n-space route at k = 1 (wide, dense,
        // unweighted).
        ("pls1_confirmatory_raw_perm_wide", Nspace),
        // The references of the K = 1 no-refit route (`split_exact`),
        // weighted and wide included.
        ("pls1_confirmatory_split_exact_k1", Special),
        ("pls1_confirmatory_weighted_split_exact", Special),
        ("pls1_confirmatory_split_exact_wide_k1", Special),
        // Narrow (n 80, d 6): the primal route's references, under the
        // p-space Gram route's work floor. `weighted_split_exact_k2` is
        // the reference of the weighted refit, `keep3` of the sparse one.
        ("pls1_perm_null_basic_n80_d6_k2", primal),
        ("pls1_perm_null_weighted_n80_d6_k2", primal),
        ("pls1_confirmatory_raw_perm", primal),
        ("pls1_confirmatory_split_exact", primal),
        ("pls1_confirmatory_weighted_split_exact_k2", primal),
        ("spls1_find_k_sequence_split_exact_keep3", primal),
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
        // PLS3 `split_exact`: wide (n 30, p 100, q 3) on the Gram route,
        // narrow (n 80, p 6, q 3) on the primal route.
        ("pls3_confirmatory_split_exact_wide", Nspace),
        ("pls3_confirmatory_split_exact", primal),
        // `split_nb` at k = 2 on the wide (n 60, d 3000) inputs: the
        // n-space route's only `split_nb` fixture.
        ("pls1_confirmatory_split_nb_wide_k2", Nspace),
    ] {
        assert_eq!(route_of(name), route, "{name}");
    }
    // The narrow PLS3 fixture scores its test halves from the cells.
    assert_eq!(
        pls3_split_route_of("pls3_confirmatory_split_exact").arm,
        TestArm::Cells,
        "pls3_confirmatory_split_exact"
    );
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
// that file uses to exercise the n-space and p-space Gram routes are pinned
// here, copied from it.

#[test]
fn thread_count_parity_shapes_take_their_routes() {
    // At the default budget, as `route_of`.
    crate::fit::with_budget(crate::fit::DEFAULT_MEMORY_BUDGET, || {
        use ReplicateRoute::{GramP, Nspace};
        // thread_count_parity's n-space perm_null row: 40 x 2000, n_perm = 100,
        // k = 2.
        assert_eq!(
            perm_null_route(40, 2000, 100, 2, false, true)
                .unwrap()
                .route,
            Nspace,
            "perm_null"
        );
        // thread_count_parity's split_exact_nspace_is_pool_size_invariant:
        // 40 x 2000, n_perm = 57, n_splits = 10, k = 2.
        assert_eq!(
            split_exact_refit_route(40, 2000, 57, 10, &[2], None, false, false)
                .unwrap()
                .route
                .route,
            Nspace,
            "split_exact"
        );
        // thread_count_parity's p-space perm_null row, dense: 2000 x 50,
        // n_perm = 300, k = 2.
        assert_eq!(
            perm_null_route(2000, 50, 300, 2, false, true)
                .unwrap()
                .route,
            GramP,
            "perm_null"
        );
        // thread_count_parity's n-space raw_perm rows: 40 x 2000, n_perm = 50,
        // k = 2, n_folds 5 and 20.
        for n_folds in [5, 20] {
            assert_eq!(
                raw_perm_route(40, n_folds, 2000, 50, &[2], None, false)
                    .unwrap()
                    .route,
                Nspace,
                "raw_perm n_folds={n_folds}"
            );
        }
        // thread_count_parity's p-space raw_perm row: 2000 x 50, n_folds 5,
        // n_perm 300, k 2.
        assert_eq!(
            raw_perm_route(2000, 5, 50, 300, &[2], None, false)
                .unwrap()
                .route,
            GramP,
            "raw_perm"
        );
        // thread_count_parity's split_exact keep row: 2000 x 200, n_perm 400,
        // n_splits 5, k 1, keep 10.
        assert_eq!(
            split_exact_refit_route(2000, 200, 400, 5, &[1], Some(10), false, false)
                .unwrap()
                .route
                .route,
            GramP,
            "split_exact keep"
        );
        // thread_count_parity's n-space raw_perm row at k = 1: 40 x 20000,
        // n_folds 5, n_perm 50.
        assert_eq!(
            raw_perm_route(40, 5, 20_000, 50, &[1], None, false)
                .unwrap()
                .route,
            Nspace,
            "raw_perm k=1"
        );

        // thread_count_parity's three 20,000-column rows are there to sum over
        // several column blocks: each spans more than two blocks at the row
        // count its route passes to `linalg::stream_block_width`. Shapes copied
        // from that file, row counts from the routes (change together).
        //
        // The k = 1 raw_perm row above: 5 folds of 8 rows, so the n-space
        // route builds its all-row Gram matrix (`signal_test::block_gram`,
        // `HeldOutGram::from_x`) in blocks sized for all 40 rows.
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
    });
}

// ── Pool sensitivity of tests/thread_count_parity.rs's "pool" cases ──
//
// That file compares each call across pool sizes, which only has teeth for
// a call that reaches a split whose rounding follows the degree of a
// pool-sized `Par`. Each check it labels `// Pool:` is run here under
// `fit::install_pool_sized` in a 1-thread and a 5-thread pool, and the two
// renderings must differ, so a case that falls below every split fails by
// name. Shapes, seeds and calls are copied from that file (change together).
mod pool_cases {
    use std::fmt::Debug;

    use faer::{Col, Mat, MatRef};

    use crate::fit::install_pool_sized;
    use crate::{
        CIOpts, ConfirmatoryArgs, ConfirmatoryMethod, ConfirmatoryTestInput, ConfirmatoryTestOpts,
        FindKOptimalOpts, FindKSequenceOpts, FitOpts, PermNullOpts, Pls3FitOpts, Selector,
        pls1_confirmatory_test, pls1_find_k_optimal, pls1_find_k_sequence, pls1_fit,
        pls1_perm_null, pls3_fit, split_nb_gate, spls3_fit,
    };

    include!("../tests/common/synth.rs");

    #[allow(clippy::many_single_char_names)]
    fn two_block(n: usize, p: usize, q: usize, seed: u64) -> (Mat<f64>, Mat<f64>) {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let x = Mat::<f64>::from_fn(n, p, |_, _| rng.random_range(-1.0..1.0));
        let y = Mat::<f64>::from_fn(n, q, |i, j| x[(i, j)] + 0.5 * rng.random_range(-1.0..1.0));
        (x, y)
    }

    fn row_weights(n: usize) -> Col<f64> {
        Col::<f64>::from_fn(n, |i| 0.5 + f64::from(u32::try_from(i % 4).unwrap()) * 0.25)
    }

    fn row_major(x: &Mat<f64>) -> Vec<f64> {
        (0..x.nrows())
            .flat_map(|i| (0..x.ncols()).map(move |j| x[(i, j)]))
            .collect()
    }

    fn perm_opts() -> PermNullOpts {
        PermNullOpts {
            n_perm: 100,
            return_perm_matrix: true,
            pre_standardized: false,
        }
    }

    fn confirmatory(
        x: &Mat<f64>,
        y: &Col<f64>,
        k: usize,
        weights: Option<&Col<f64>>,
        args: ConfirmatoryArgs,
        ci: Option<CIOpts>,
        seed: u64,
    ) -> crate::ConfirmatoryTestOutput {
        pls1_confirmatory_test(
            ConfirmatoryTestInput::Raw {
                x: x.as_ref(),
                y: y.as_ref(),
                k: Some(&[k]),
                weights: weights.map(Col::as_ref),
                covariates: None,
                null_y: None,
            },
            ConfirmatoryTestOpts {
                args,
                seed: Some(seed),
                ci,
                ..Default::default()
            },
        )
        .unwrap()
    }

    /// Collects every case whose renderings in the two pools are equal.
    struct Cases {
        pools: [rayon::ThreadPool; 2],
        ran: usize,
        same: Vec<String>,
    }

    impl Cases {
        fn new() -> Self {
            let pool = |t| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(t)
                    .build()
                    .expect("rayon pool")
            };
            Self {
                pools: [pool(1), pool(5)],
                ran: 0,
                same: Vec::new(),
            }
        }

        fn check<T: Debug>(&mut self, name: &str, f: impl Fn() -> T + Sync) {
            let [one, five] = self
                .pools
                .each_ref()
                .map(|p| install_pool_sized(p, || format!("{:?}", f())));
            self.ran += 1;
            if one == five {
                self.same.push(name.to_owned());
            }
        }
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn thread_count_parity_pool_cases_are_pool_sensitive() {
        // `with_thread_limit` runs a call with a smaller cap on its own
        // cached pool, whose workers the pool-sized switch does not reach.
        assert!(
            std::env::var_os(crate::fit::NUM_THREADS_ENV).is_none(),
            "{} must be unset",
            crate::fit::NUM_THREADS_ENV
        );
        let mut c = Cases::new();

        // pls1_fit_auto_is_pool_size_invariant
        for (n, d, k) in [
            (2000usize, 600usize, 1usize),
            (2000, 600, 5),
            (20_000, 60, 5),
        ] {
            let (x, y) = synth(n, d, 1.0, 3);
            c.check(&format!("pls1_fit {n}x{d} k={k}"), || {
                pls1_fit(x.as_ref(), y.as_ref(), k, None, None, FitOpts::default()).unwrap()
            });
        }
        let (x_wide, y_wide) = synth(203, 5000, 1.0, 5);
        let x_wide_t = x_wide.transpose().to_owned();
        c.check("pls1_fit row-major view 203x5000 k=3", || {
            pls1_fit(
                x_wide_t.transpose(),
                y_wide.as_ref(),
                3,
                None,
                None,
                FitOpts::default(),
            )
            .unwrap()
        });
        let pp = crate::preprocess(crate::PreprocessInput {
            x: Some(x_wide.as_ref()),
            y: Some(y_wide.as_ref()),
            weights: None,
        })
        .unwrap();
        let (x_std, _, _) = pp.x_std.unwrap();
        let (y_std, _, _) = pp.y_std.unwrap();
        let x_std_t = x_std.transpose().to_owned();
        c.check("pls1_fit pre_standardized row-major 203x5000 k=3", || {
            pls1_fit(
                x_std_t.transpose(),
                y_std.as_ref(),
                3,
                None,
                None,
                FitOpts {
                    pre_standardized: true,
                    ..FitOpts::default()
                },
            )
            .unwrap()
        });

        // pls1_fit_weighted_row_major_is_pool_size_invariant
        let w = row_weights(203);
        c.check("pls1_fit weighted row-major view 203x5000 k=3", || {
            pls1_fit(
                x_wide_t.transpose(),
                y_wide.as_ref(),
                3,
                Some(w.as_ref()),
                None,
                FitOpts::default(),
            )
            .unwrap()
        });

        // pls3_fits_auto_are_pool_size_invariant
        let (x, y) = two_block(200, 2000, 6, 31);
        c.check("pls3_fit 200x2000x6 k=3", || {
            pls3_fit(x.as_ref(), y.as_ref(), 3, None, Pls3FitOpts::default()).unwrap()
        });
        let (xs, ys) = two_block(100, 20_000, 4, 32);
        c.check("spls3_fit 100x20000x4 k=2 keep=500,2", || {
            spls3_fit(
                xs.as_ref(),
                ys.as_ref(),
                2,
                500,
                2,
                None,
                Pls3FitOpts::default(),
            )
            .unwrap()
        });

        // other_pls1_confirmatory_methods_are_pool_size_invariant
        let (xt, yt) = synth(2000, 600, 1.0, 7);
        c.check("e 2000x600 k=2", || {
            confirmatory(&xt, &yt, 2, None, ConfirmatoryArgs::E, None, 13)
        });
        c.check("split_nb_gate 300x400", || {
            split_nb_gate(xt.as_ref().subrows(0, 300).subcols(0, 400), None, false).unwrap()
        });

        // perm_null_is_pool_size_invariant
        let (x, y) = synth(2000, 600, 1.0, 9);
        c.check("perm_null 2000x600 k=2", || {
            pls1_perm_null(x.as_ref(), y.as_ref(), 2, None, None, perm_opts(), Some(15)).unwrap()
        });

        // find_k_is_pool_size_invariant
        let (xt, yt) = synth(2000, 600, 1.0, 21);
        c.check("find_k_optimal bic 2000x600", || {
            pls1_find_k_optimal(
                xt.as_ref(),
                yt.as_ref(),
                3,
                None,
                None,
                FindKOptimalOpts {
                    selector: Selector::Bic,
                    seed: Some(1),
                    ..Default::default()
                },
            )
            .unwrap()
        });
        let r2_se = FindKOptimalOpts {
            selector: Selector::R2Se,
            seed: Some(3),
            ..Default::default()
        };
        c.check("find_k_optimal r2_se 2000x600", || {
            pls1_find_k_optimal(xt.as_ref(), yt.as_ref(), 3, None, None, r2_se).unwrap()
        });
        let xt_row_major = row_major(&xt);
        let xt_row_view = MatRef::from_row_major_slice(&xt_row_major, 2000, 600);
        c.check("find_k_optimal r2_se 2000x600 row-major", || {
            pls1_find_k_optimal(xt_row_view, yt.as_ref(), 3, None, None, r2_se).unwrap()
        });
        c.check("find_k_sequence split_nb 2000x600", || {
            pls1_find_k_sequence(
                xt.as_ref(),
                yt.as_ref(),
                2,
                None,
                None,
                FindKSequenceOpts {
                    test_method: ConfirmatoryMethod::SplitNb,
                    n_splits: 6,
                    force: true,
                    alpha: 0.5,
                    seed: Some(2),
                    ..Default::default()
                },
            )
            .unwrap()
        });

        assert_eq!(c.ran, 15, "pool cases run");
        assert!(
            c.same.is_empty(),
            "{} pool case(s) render the same in 1- and 5-thread pools under a pool-sized Par:\n{}",
            c.same.len(),
            c.same.join("\n")
        );
    }
}
