//! The parity cases of `tests/memory_budget_parity.rs`, run in the crate so
//! that the routes and batch sizes of their loops can be read: each case,
//! on pools of 1 and `MAX_THREADS` threads, takes one route at its low
//! budget (its final `needed`), its middle one and the default, and at the
//! middle one runs every loop of more than one unit in batches of at least
//! two and fewer than all its units; on one thread it reaches its low
//! budget in one `MemoryBudget` round per loop that raises. The budgets go
//! through `fit::with_budget`; the integration test passes them through the
//! environment and compares the results.

use faer::linalg::svd::{ComputeSvdVectors, svd_scratch};
use faer::{Col, Mat};

use crate::fit::{DEFAULT_MEMORY_BUDGET, with_budget};
use crate::resample::with_batch_log;
use crate::route_select::with_route_log;
use crate::{
    CIOpts, ConfirmatoryArgs, ConfirmatoryTestInput, ConfirmatoryTestOpts, ConfirmatoryTestOutput,
    FindKOptimalOpts, FindKOptimalOutput, FindKeepOptimalOpts, Pls3ConfirmatoryTestOpts,
    PlsKitError, PlsKitResult, ProfilesFindNOpts, Selector, pls1_confirmatory_test,
    pls1_find_k_optimal, pls3_confirmatory_test, profiles_find_n, spls1_find_keep_optimal,
};

include!("../tests/common/synth.rs");
include!("../tests/common/budget_cases.rs");

/// The loops of a case that raise `MemoryBudget` one after another as its
/// budget rises from 1: `split_nb`'s gate (on wide X: the cases run under
/// `force`, where the gate builds only the wide Gram matrix) and its splits,
/// `profiles_find_n`'s parallel analysis and its folds, the test and the CI;
/// one otherwise.
fn raising_loops(name: &str) -> usize {
    let wide_split_nb = name.contains("split_nb") && name != "pls3 split_nb dense 40x15";
    if wide_split_nb || name.starts_with("profiles_find_n") || name.starts_with("ci ") {
        2
    } else {
        1
    }
}

/// What one run of a case did on the calling thread: the routes its
/// selectors chose as `(route, n_units, lanes)` and its other loops'
/// `(n_units, batch)`.
type Trace = (Vec<(String, usize, usize)>, Vec<(usize, usize)>);

/// Runs `c` at `budget` and returns its result and trace.
fn traced(c: &Case<'_>, budget: u64) -> (PlsKitResult<String>, Trace) {
    let ((out, routes), batches) =
        with_batch_log(|| with_route_log(|| with_budget(budget, || (c.run)())));
    (out, (routes, batches))
}

#[test]
fn budget_keeps_route_and_mid_runs_lanes() {
    let data = parity_data();
    let mut failures = Vec::new();
    for c in parity_cases(&data) {
        for threads in [1, MAX_THREADS] {
            pool(threads).install(|| {
                let (low, (low_trace, _), rounds) = final_needed(c.name, |b| {
                    let (out, trace) = traced(&c, b);
                    out.map(|_| trace)
                });
                let mid = c.mid(low);
                let (mid_out, (mid_routes, mid_batches)) = traced(&c, mid);
                let (default_out, (default_routes, _)) = traced(&c, DEFAULT_MEMORY_BUDGET);
                mid_out.unwrap_or_else(|e| panic!("{} at mid: {e:?}", c.name));
                default_out.unwrap_or_else(|e| panic!("{} at the default: {e:?}", c.name));
                // The primal route's two sources give the same bits where
                // both are candidates (`source_flip_keeps_bits`), so the
                // source is not compared.
                let routes = |log: &[(String, usize, usize)]| {
                    log.iter()
                        .map(|(r, _, _)| {
                            r.replace("Primal(Copy)", "Primal")
                                .replace("Primal(Gather)", "Primal")
                        })
                        .collect::<Vec<_>>()
                };
                let at = format!("{} on {threads} thread(s)", c.name);
                let want = raising_loops(c.name);
                // On one thread a selector's check and its re-check read the
                // same bytes, so each loop raises once: a loop that needs more
                // than its selector counted adds a round.
                if threads == 1 && rounds != want {
                    failures.push(format!("{at}: {rounds} MemoryBudget round(s), want {want}"));
                }
                println!(
                    "{at}: low {low} after {rounds} round(s), mid {mid}, routes {mid_routes:?}, batches {mid_batches:?}"
                );
                if routes(&low_trace) != routes(&mid_routes)
                    || routes(&mid_routes) != routes(&default_routes)
                {
                    failures.push(format!(
                        "{at}: routes low {low} {:?}, mid {mid} {:?}, default {:?}",
                        routes(&low_trace),
                        routes(&mid_routes),
                        routes(&default_routes)
                    ));
                }
                // Every loop of more than one unit runs a partial batch:
                // more than one unit at once, not all of them (a loop of two
                // units runs both).
                let mid_lanes = mid_routes
                    .iter()
                    .map(|&(_, n, lanes)| (n, lanes))
                    .chain(mid_batches.iter().copied());
                let partial = |n: usize, k: usize| (2..n.max(3)).contains(&k);
                let off: Vec<_> = mid_lanes.filter(|&(n, k)| n > 1 && !partial(n, k)).collect();
                if !off.is_empty()
                    || (mid_routes.is_empty() && mid_batches.is_empty())
                {
                    failures.push(format!(
                        "{at}: mid {mid} runs a loop one unit at a time or all at once (or no loop): routes {mid_routes:?}, batches {mid_batches:?}"
                    ));
                }
            });
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
