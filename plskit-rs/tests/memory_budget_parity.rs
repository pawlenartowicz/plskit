//! `PLSKIT_MEMORY_BUDGET` end to end: the budget must not change the bits of
//! a resampling call whose route it keeps, and every call fails at budget 1
//! and runs at its final `needed`.
//!
//! The tests re-run themselves in child processes (`common::run_child`),
//! one per budget, because the library reads the variable at each call's
//! entry and the test thread cannot change the environment safely.
//! `memory_budget_does_not_change_results` runs every resampling loop the
//! budget batches through its public entry point (the cases of
//! `common/budget_cases.rs`) at three budgets: the call's final `needed`
//! (`low`, found per pool by `final_needed` from budget 1), `Case::mid` over
//! it (more than one unit at once, fewer than all) and unset (the default),
//! each inside Rayon pools of 1 and 4 threads (`ThreadPoolBuilder::install`).
//! The `Debug` renderings of the six results of a case must be identical
//! (see `thread_count_parity.rs` for why that compares bits). That the
//! three budgets take one route and that `mid` runs partial batches is
//! checked in the crate (`src/memory_budget_cases.rs`), where the route is
//! visible. `final_needed_runs` covers the functions the budget bounds at
//! these shapes.

mod common;

use std::ffi::OsStr;

use common::{child_arg, emit_result, run_child, synth};
use faer::linalg::svd::{ComputeSvdVectors, svd_scratch};
use faer::{Col, Mat};
use plskit::{
    CIOpts, ConfirmatoryArgs, ConfirmatoryTestInput, ConfirmatoryTestOpts, ConfirmatoryTestOutput,
    FindKOptimalOpts, FindKOptimalOutput, FindKeepOptimalOpts, FitOpts, PermNullOpts,
    Pls3ConfirmatoryTestOpts, PlsKitError, PlsKitResult, ProfilesFindNOpts, Selector, factors,
    gram_matrix, pca_basis, pls1_confirmatory_test, pls1_find_k_optimal, pls1_fit, pls1_perm_alpha,
    pls1_perm_alpha_gram, pls1_perm_null, pls3_confirmatory_test, profiles_find_n,
    spls1_find_keep_optimal,
};

include!("common/budget_cases.rs");

const BUDGET_VAR: &str = "PLSKIT_MEMORY_BUDGET";
/// A `PLSKIT_NUM_THREADS` cap would run the calls on plskit's own pool, not on
/// the pools installed by the test.
const THREADS_VAR: &str = "PLSKIT_NUM_THREADS";
/// Must equal the name of the test function.
const TEST: &str = "memory_budget_does_not_change_results";
const FINAL_TEST: &str = "final_needed_runs";

/// The message of the `InvalidArgument` that `pls1_fit` returns while the
/// variable holds a bad value.
fn bad_budget_message(x: &Mat<f64>, y: &Col<f64>) -> String {
    match pls1_fit(x.as_ref(), y.as_ref(), 2, None, None, FitOpts::default()) {
        Err(PlsKitError::InvalidArgument(msg)) => msg,
        other => panic!("{BUDGET_VAR} bad: {:?}", other.map(|_| ())),
    }
}

/// Emits a call's result for the parent: `ok` and its rendering, or
/// `memory_budget` and the bytes it needs. Any other error fails the child.
fn emit_call(name: &str, out: PlsKitResult<String>) {
    match out {
        Ok(s) => emit_result(&format!("ok {s}")),
        Err(PlsKitError::MemoryBudget { needed, .. }) => {
            emit_result(&format!("memory_budget {needed}"));
        }
        Err(e) => panic!("{name}: {e:?}"),
    }
}

/// Reads a line of `emit_call` back into the call's result at `budget`.
fn parse_call(line: &str, budget: u64) -> PlsKitResult<String> {
    if let Some(s) = line.strip_prefix("ok ") {
        return Ok(s.to_owned());
    }
    let needed = line
        .strip_prefix("memory_budget ")
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| panic!("child result {line:?}"));
    Err(PlsKitError::MemoryBudget { budget, needed })
}

/// Runs a child of `test` with the budget `budget` (`None`: unset) and no
/// thread cap.
fn run_budget_child(test: &str, arg: &str, budget: Option<u64>) -> Vec<String> {
    let budget = budget.map(|b| b.to_string());
    run_child(
        test,
        arg,
        &[
            (THREADS_VAR, None),
            (BUDGET_VAR, budget.as_deref().map(OsStr::new)),
        ],
    )
}

/// The result of case `i` on a pool of `threads` threads at `budget`, run in
/// a child.
fn case_at(i: usize, threads: usize, budget: u64) -> PlsKitResult<String> {
    let lines = run_budget_child(TEST, &format!("{i} {threads}"), Some(budget));
    assert_eq!(lines.len(), 1, "case {i}: child results {lines:?}");
    parse_call(&lines[0], budget)
}

#[test]
fn memory_budget_does_not_change_results() {
    let data = parity_data();
    let cases = parity_cases(&data);
    // Child argument: `bad` checks the bad-value message, `all` runs every
    // case on both pools, `<case> <threads>` one case on one pool.
    if let Some(arg) = child_arg() {
        if arg == "bad" {
            emit_result(&bad_budget_message(&data.x, &data.y));
            return;
        }
        let (only, pools) = match arg.split_once(' ') {
            Some((i, t)) => (i.parse::<usize>().ok(), vec![pool(t.parse().unwrap())]),
            None => (None, vec![pool(1), pool(MAX_THREADS)]),
        };
        for (i, c) in cases.iter().enumerate() {
            if only.is_none_or(|o| o == i) {
                for pl in &pools {
                    emit_call(c.name, pl.install(|| (c.run)()));
                }
            }
        }
        return;
    }

    let unset = run_budget_child(TEST, "all", None);
    assert_eq!(unset.len(), 2 * cases.len(), "unset budget results");
    let mut failures = Vec::new();
    for (i, c) in cases.iter().enumerate() {
        let mut runs: Vec<(String, String)> = Vec::new();
        for (t, unset_line) in [1, MAX_THREADS].into_iter().zip(&unset[2 * i..2 * i + 2]) {
            let (low, low_out, _) = final_needed(c.name, |b| case_at(i, t, b));
            let mid = c.mid(low);
            let mid_out =
                case_at(i, t, mid).unwrap_or_else(|e| panic!("{} at mid {mid}: {e:?}", c.name));
            let unset_out =
                parse_call(unset_line, 0).unwrap_or_else(|e| panic!("{} unset: {e:?}", c.name));
            runs.push((format!("budget {low} (low) on {t} thread(s)"), low_out));
            runs.push((format!("budget {mid} (mid) on {t} thread(s)"), mid_out));
            runs.push((format!("budget unset on {t} thread(s)"), unset_out));
        }
        let (label0, out0) = &runs[0];
        if let Some((label, out)) = runs[1..].iter().find(|(_, o)| o != out0) {
            let at = out0
                .bytes()
                .zip(out.bytes())
                .position(|(a, b)| a != b)
                .unwrap_or_else(|| out0.len().min(out.len()));
            let lo = at.saturating_sub(100);
            let ctx = |s: &str| s.get(lo..(at + 40).min(s.len())).unwrap_or("").to_owned();
            failures.push(format!(
                "{}: {label0} and {label} differ at byte {at}\n  ...{}\n  ...{}",
                c.name,
                ctx(out0),
                ctx(out)
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} case(s) depend on the budget or the pool:\n{}",
        failures.len(),
        failures.join("\n")
    );

    // A bad value is rejected before any work, quoted as given.
    let bad = |v: &OsStr| {
        run_child(TEST, "bad", &[(THREADS_VAR, None), (BUDGET_VAR, Some(v))]).swap_remove(0)
    };
    let msg = bad(OsStr::new("2X"));
    assert!(msg.contains(BUDGET_VAR) && msg.contains("\"2X\""), "{msg}");
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let msg = bad(OsStr::from_bytes(b"2\xffG"));
        assert!(
            msg.contains(BUDGET_VAR) && msg.contains(r#""2\xFFG""#),
            "{msg}"
        );
    }
}

/// A `pls1_confirmatory_test` call that picks its own `k` (`k = None`).
fn pls1_test_k_none(
    x: &Mat<f64>,
    y: &Col<f64>,
    weights: Option<&Col<f64>>,
) -> PlsKitResult<ConfirmatoryTestOutput> {
    pls1_confirmatory_test(
        ConfirmatoryTestInput::Raw {
            x: x.as_ref(),
            y: y.as_ref(),
            k: None,
            weights: weights.map(Col::as_ref),
            covariates: None,
            null_y: None,
        },
        ConfirmatoryTestOpts {
            args: SPLIT_EXACT,
            seed: Some(40),
            ..Default::default()
        },
    )
}

type Call<'a> = (String, Box<dyn Fn() -> PlsKitResult<String> + 'a>);

/// Every function whose memory the budget bounds, at the parity shapes:
/// wide (`PN × PP`, PLS3 with `PQ` columns of Y) and tall (`GRAM_P_N ×
/// GRAM_P_P`).
#[allow(clippy::too_many_lines)]
fn final_calls<'a>(
    d: &'a ParityData,
    x3t: &'a Mat<f64>,
    y3t: &'a Mat<f64>,
    wt: &'a Col<f64>,
) -> Vec<Call<'a>> {
    let mut calls: Vec<Call<'a>> = Vec::new();
    let mut add =
        |name: String, f: Box<dyn Fn() -> PlsKitResult<String> + 'a>| calls.push((name, f));
    let ci = CIOpts {
        n_boot: P_BOOT,
        m_rate: P_M_RATE,
        ..CIOpts::default()
    };
    add(
        "split_exact k=None weighted wide".into(),
        Box::new(move || render(pls1_test_k_none(&d.x, &d.y, Some(&d.w)))),
    );
    add(
        "score weighted tall".into(),
        Box::new(move || {
            let args = ConfirmatoryArgs::Score { n_perm: Some(19) };
            render(pls1_test(&d.xg, &d.yg, 2, Some(wt), args, None, 41))
        }),
    );
    add(
        "ci (score) wide".into(),
        Box::new(move || {
            let args = ConfirmatoryArgs::Score { n_perm: None };
            render(pls1_test(&d.x, &d.y, 2, None, args, Some(ci), 42))
        }),
    );
    add(
        "pls1_find_k_optimal r2_se wide".into(),
        Box::new(move || render(find_k(&d.x, &d.y, None, 43))),
    );
    add(
        "spls1_find_keep_optimal wide".into(),
        Box::new(move || {
            let opts = FindKeepOptimalOpts {
                n_folds: P_UNITS,
                seed: Some(44),
            };
            render(spls1_find_keep_optimal(
                d.x.as_ref(),
                d.y.as_ref(),
                2,
                None,
                None,
                opts,
            ))
        }),
    );
    add(
        "profiles_find_n wide".into(),
        Box::new(move || {
            let opts = ProfilesFindNOpts {
                seed: Some(45),
                ..Default::default()
            };
            render(profiles_find_n(d.x.as_ref(), d.y.as_ref(), 2, 9, opts))
        }),
    );
    for (shape, x, y, x3, y3) in [
        ("wide", &d.x, &d.y, &d.x3, &d.y3),
        ("tall", &d.xg, &d.yg, x3t, y3t),
    ] {
        add(
            format!("split_exact k=None {shape}"),
            Box::new(move || render(pls1_test_k_none(x, y, None))),
        );
        add(
            format!("split_nb k=2 {shape}"),
            Box::new(move || render(pls1_test(x, y, 2, None, SPLIT_NB, None, 46))),
        );
        add(
            format!("raw_perm k=2 {shape}"),
            Box::new(move || {
                let args = ConfirmatoryArgs::RawPerm {
                    n_perm: 19,
                    n_folds: P_UNITS,
                };
                render(pls1_test(x, y, 2, None, args, None, 47))
            }),
        );
        for (method, args) in [
            ("split_exact", SPLIT_EXACT),
            ("split_nb", SPLIT_NB),
            (
                "sv_perm",
                ConfirmatoryArgs::SvPerm {
                    n_perm: 19,
                    rotate: false,
                },
            ),
        ] {
            add(
                format!("pls3 {method} {shape}"),
                Box::new(move || render(pls3_test(x3, y3, args, None, None, 48))),
            );
        }
        add(
            format!("pls1_perm_null {shape}"),
            Box::new(move || {
                let opts = PermNullOpts {
                    n_perm: 100,
                    ..Default::default()
                };
                render(pls1_perm_null(
                    x.as_ref(),
                    y.as_ref(),
                    2,
                    None,
                    None,
                    opts,
                    Some(49),
                ))
            }),
        );
        // Three outcome columns: `y` and two rotations of its rows.
        let ys = move || Mat::<f64>::from_fn(y.nrows(), 3, |i, j| y[(i + j) % y.nrows()]);
        add(
            format!("pls1_perm_alpha {shape}"),
            Box::new(move || {
                let g = gram_matrix(x.as_ref(), false)?;
                render(pls1_perm_alpha(
                    g.gram.as_ref(),
                    ys().as_ref(),
                    2,
                    g.n_features,
                ))
            }),
        );
        add(
            format!("pls1_perm_alpha_gram {shape}"),
            Box::new(move || {
                render(pls1_perm_alpha_gram(
                    &gram_matrix(x.as_ref(), false)?,
                    ys().as_ref(),
                    2,
                ))
            }),
        );
        add(
            format!("factors {shape}"),
            Box::new(move || {
                let fit = pls1_fit(x.as_ref(), y.as_ref(), 2, None, None, FitOpts::default())?;
                render(factors(fit.coef.as_ref(), x.as_ref(), 1, false, None))
            }),
        );
        add(
            format!("pca_basis {shape}"),
            Box::new(move || render(pca_basis(x.as_ref(), false))),
        );
    }
    calls
}

#[test]
fn final_needed_runs() {
    let data = parity_data();
    let (x3t, y3t) = two_block(GRAM_P_N, GRAM_P_P, PQ, 5);
    let wt = row_weights(GRAM_P_N);
    let calls = final_calls(&data, &x3t, &y3t, &wt);
    if let Some(arg) = child_arg() {
        let (name, run) = &calls[arg.parse::<usize>().unwrap()];
        // Only the outcome matters here; some outputs render on many lines.
        emit_call(name, run().map(|_| String::new()));
        return;
    }
    for (i, (name, _)) in calls.iter().enumerate() {
        let (needed, _, rounds) = final_needed(name, |b| {
            let lines = run_budget_child(FINAL_TEST, &i.to_string(), Some(b));
            parse_call(&lines[0], b)
        });
        println!("{name}: final needed {needed} after {rounds} round(s)");
    }
}
