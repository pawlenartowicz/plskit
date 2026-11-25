//! The peak memory of a call under a real cap: each call runs in a child
//! process started under `systemd-run --user --scope -p MemoryMax=<cap>
//! -p MemorySwapMax=0`, with `cap = base + budget + margin`, where `base` is
//! the `VmHWM` of a child that builds the case's inputs and calls nothing and
//! the margin is `max(32 MiB, budget / 20)`. The call must return `Ok`; a
//! child the kernel kills and a child whose call returns an error fail the
//! case, reported apart.
//!
//! Per case and thread count (1 and 4, `PLSKIT_NUM_THREADS`), uncapped
//! children find the call's final `needed` from budget 1
//! (`final_needed`); capped children then run at `needed` and `4·needed`,
//! and once at budget `0` on 1 thread, capped as at the 1-thread `needed`.
//! Every capped child prints the `memory.max` of its own cgroup, which must
//! equal the cap, and the routes its loops took (`route_select::take_routes`,
//! which the `route-log` feature of the tests turns on); at `4·needed` the
//! rows the spec names a Gram route for must take it.
//!
//! Linux only and ignored: it needs `systemd-run --user`, which CI does not
//! have. Run by hand:
//! `cargo test --release --test memory_bound -- --ignored --nocapture`.
// The published crate drops the self dev-dependency that turns on `route-log`.
#![cfg(all(target_os = "linux", feature = "route-log"))]
// `common/budget_cases.rs` is included for its call helpers, not its table.
#![allow(dead_code)]

mod common;

use std::os::unix::process::ExitStatusExt;
use std::process::Command;
use std::sync::Mutex;

use common::{CHILD_VAR, RESULT_PREFIX, child_arg, emit_result, synth};
use faer::linalg::svd::{ComputeSvdVectors, svd_scratch};
use faer::{Col, Mat};
use plskit::{
    CIOpts, ConfirmatoryArgs, ConfirmatoryMethod, ConfirmatoryTestInput, ConfirmatoryTestOpts,
    ConfirmatoryTestOutput, FindKOptimalOpts, FindKOptimalOutput, FindKSequenceOpts,
    FindKeepOptimalOpts, FitOpts, PermNullOpts, Pls3ConfirmatoryTestOpts, PlsKitError,
    PlsKitResult, ProfilesFindNOpts, Selector, factors, gram_matrix, pca_basis,
    pls1_confirmatory_test, pls1_find_k_optimal, pls1_find_k_sequence, pls1_fit, pls1_perm_alpha,
    pls1_perm_null, pls3_confirmatory_test, pls3_confirmatory_test_gram, profiles_find_n,
    split_nb_gate, spls1_find_keep_optimal,
};

include!("common/budget_cases.rs");

const BUDGET_VAR: &str = "PLSKIT_MEMORY_BUDGET";
const THREADS_VAR: &str = "PLSKIT_NUM_THREADS";
/// Must equal the names of the test functions.
const KILL_TEST: &str = "cap_kills_an_over_budget_child";
const TEST: &str = "peak_within_cap";
const MIB: u64 = 1 << 20;
/// About 100 MB of X each.
const WIDE: (usize, usize) = (250, 50_000);
const TALL: (usize, usize) = (50_000, 250);
/// Rows of the tall `pls1_perm_alpha` case: its input is the `n × n` Gram
/// matrix, about 100 MB here (20 GB at `TALL`'s rows).
const ALPHA_TALL_N: usize = 3_500;
/// Shape of the no-refit case: 16 MB of X, 320 MB of outcome columns.
const NR_N: usize = 20_000;
const NR_P: usize = 100;
/// Rows of the Gram input cases: a caller's Gram matrix of about 100 MB.
const GRAM_N: usize = 3_500;
/// Columns of the covariates in the covariate case.
const COV_Q: usize = 3;
/// Shape of the moments case, with 1000 outcome columns over 64 splits: on
/// 1 thread at `needed` the selector took the no-refit moments arm on the
/// test machine. The row only prints its route (the choice depends on
/// faer's buffer size, so on the CPU); `route_select`'s own tests assert
/// the arm.
const MOM_N: usize = 1_000;
const MOM_P: usize = 200;
/// Shape of the n-space kernel case: 312 MB of X, halves of 3000 rows.
const NS_N: usize = 6_000;
const NS_P: usize = 6_500;

/// One capped child at a time, whichever test starts it.
static SERIAL: Mutex<()> = Mutex::new(());

fn margin(budget: u64) -> u64 {
    (32 * MIB).max(budget / 20)
}

/// `VmHWM` of this process, in bytes.
fn vm_hwm() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").expect("/proc/self/status");
    let kb = status
        .lines()
        .find_map(|l| l.strip_prefix("VmHWM:"))
        .and_then(|v| v.trim().strip_suffix("kB"))
        .and_then(|v| v.trim().parse::<u64>().ok())
        .expect("VmHWM");
    kb * 1024
}

/// `memory.max` of this process's cgroup (`max` when uncapped).
fn memory_max() -> String {
    let cgroup = std::fs::read_to_string("/proc/self/cgroup").expect("/proc/self/cgroup");
    let path = cgroup
        .lines()
        .find_map(|l| l.strip_prefix("0::"))
        .expect("cgroup v2 path");
    std::fs::read_to_string(format!("/sys/fs/cgroup{path}/memory.max"))
        .expect("memory.max")
        .trim()
        .to_owned()
}

/// Prints this child's line for the parent: `VmHWM`, `memory.max`, the
/// routes its loops took (each once, in order, without spaces; `-` for none)
/// and the outcome.
fn emit_child(outcome: &str) {
    let mut routes: Vec<String> = Vec::new();
    for r in plskit::route_select::take_routes() {
        let r = r.replace(' ', "");
        if !routes.contains(&r) {
            routes.push(r);
        }
    }
    let routes = if routes.is_empty() {
        "-".to_owned()
    } else {
        routes.join(";")
    };
    emit_result(&format!("{} {} {routes} {outcome}", vm_hwm(), memory_max()));
}

enum Outcome {
    Ok,
    Budget(u64),
    Error(String),
    Killed,
}

struct Run {
    outcome: Outcome,
    hwm: u64,
    max: String,
    routes: String,
}

/// Runs `test` again in a child whose `child_arg()` is `arg`, at `budget`
/// (`None`: unset) on `threads` threads, under a `MemoryMax` of `cap` bytes
/// when given.
fn child(test: &str, arg: &str, budget: Option<u64>, threads: usize, cap: Option<u64>) -> Run {
    let exe = std::env::current_exe().expect("current_exe");
    let mut cmd = match cap {
        Some(cap) => {
            let mut c = Command::new("systemd-run");
            c.args(["--user", "--scope", "--quiet", "-p"])
                .arg(format!("MemoryMax={cap}"))
                .args(["-p", "MemorySwapMax=0"])
                .arg(&exe);
            c
        }
        None => Command::new(&exe),
    };
    cmd.args([test, "--exact", "--ignored", "--nocapture"])
        .env(CHILD_VAR, arg)
        .env(THREADS_VAR, threads.to_string());
    match budget {
        Some(b) => cmd.env(BUDGET_VAR, b.to_string()),
        None => cmd.env_remove(BUDGET_VAR),
    };
    let out = cmd.output().expect("spawn child");
    let failed = |outcome| Run {
        outcome,
        hwm: 0,
        max: "-".into(),
        routes: "-".into(),
    };
    if out.status.signal() == Some(9) {
        return failed(Outcome::Killed);
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout.lines().find_map(|l| l.strip_prefix(RESULT_PREFIX));
    let Some(line) = line.filter(|_| out.status.success()) else {
        let stderr = String::from_utf8_lossy(&out.stderr);
        let tail: Vec<&str> = stderr.lines().rev().take(5).collect();
        return failed(Outcome::Error(format!("{} {tail:?}", out.status)));
    };
    let mut parts = line.splitn(4, ' ');
    let mut next = || parts.next().expect("child line");
    let hwm = next().parse().expect("VmHWM");
    let max = next().to_owned();
    let routes = next().to_owned();
    let outcome = match next() {
        "ok" => Outcome::Ok,
        s => match s.strip_prefix("memory_budget ") {
            Some(n) => Outcome::Budget(n.parse().expect("needed")),
            None => Outcome::Error(s.to_owned()),
        },
    };
    Run {
        outcome,
        hwm,
        max,
        routes,
    }
}

#[test]
#[ignore = "needs systemd-run --user; run by hand"]
fn cap_kills_an_over_budget_child() {
    if let Some(arg) = child_arg() {
        let mib: usize = arg.parse().unwrap();
        std::hint::black_box(vec![1u8; mib << 20]);
        emit_child("ok");
        return;
    }
    let _serial = SERIAL.lock().unwrap();
    let cap = 64 * MIB;
    let big = child(KILL_TEST, "192", None, 1, Some(cap));
    assert!(
        matches!(big.outcome, Outcome::Killed),
        "192 MiB under 64 MiB"
    );
    let small = child(KILL_TEST, "32", None, 1, Some(cap));
    assert!(matches!(small.outcome, Outcome::Ok), "32 MiB under 64 MiB");
    assert_eq!(small.max, "67108864");
}

/// Builds a case's inputs; with `call`, runs the call on them.
type Build = Box<dyn Fn(bool) -> PlsKitResult<()>>;

struct Bound {
    name: String,
    shape: &'static str,
    run: Build,
}

fn bound(
    name: &str,
    shape: &'static str,
    run: impl Fn(bool) -> PlsKitResult<()> + 'static,
) -> Bound {
    Bound {
        name: name.to_owned(),
        shape,
        run: Box::new(run),
    }
}

/// Runs `f` when `call`.
fn then<T>(call: bool, f: impl FnOnce() -> PlsKitResult<T>) -> PlsKitResult<()> {
    if call { f().map(drop) } else { Ok(()) }
}

fn pls1_k_none(
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
            seed: Some(50),
            ..Default::default()
        },
    )
}

/// The rows of the spec's table: every function the budget bounds.
#[allow(clippy::too_many_lines)]
fn bound_cases() -> Vec<Bound> {
    let (wn, wp) = WIDE;
    let (tn, tp) = TALL;
    let raw_perm = ConfirmatoryArgs::RawPerm {
        n_perm: 19,
        n_folds: P_UNITS,
    };
    let mut cases = vec![
        bound("pls1 split_exact k=None weighted", "wide", move |call| {
            let (x, y) = synth(wn, wp, 1.0, 51);
            let w = row_weights(wn);
            then(call, || pls1_k_none(&x, &y, Some(&w)))
        }),
        bound("pls1 score weighted", "tall", move |call| {
            let (x, y) = synth(tn, tp, 1.0, 52);
            let w = row_weights(tn);
            let args = ConfirmatoryArgs::Score { n_perm: Some(19) };
            then(call, || pls1_test(&x, &y, 2, Some(&w), args, None, 53))
        }),
        bound("pls1 score with ci", "wide", move |call| {
            let (x, y) = synth(wn, wp, 1.0, 54);
            let args = ConfirmatoryArgs::Score { n_perm: None };
            let ci = CIOpts {
                n_boot: P_BOOT,
                m_rate: P_M_RATE,
                ..CIOpts::default()
            };
            then(call, || pls1_test(&x, &y, 2, None, args, Some(ci), 55))
        }),
        bound("pls1_find_k_optimal r2_se", "wide", move |call| {
            let (x, y) = synth(wn, wp, 1.0, 56);
            then(call, || find_k(&x, &y, None, 57))
        }),
        bound("spls1_find_keep_optimal", "wide", move |call| {
            let (x, y) = synth(wn, wp, 1.0, 58);
            let opts = FindKeepOptimalOpts {
                n_folds: P_UNITS,
                seed: Some(59),
            };
            then(call, || {
                spls1_find_keep_optimal(x.as_ref(), y.as_ref(), 2, None, None, opts)
            })
        }),
        bound("profiles_find_n", "wide", move |call| {
            let (x, y) = synth(wn, wp, 1.0, 60);
            let opts = ProfilesFindNOpts {
                seed: Some(61),
                ..Default::default()
            };
            then(call, || profiles_find_n(x.as_ref(), y.as_ref(), 2, 9, opts))
        }),
        // The no-refit body at a realistic `n_perm`, where each split holds
        // its rows of the outcome columns and their scores.
        bound(
            "pls1 split_exact k=1 n_perm 1999",
            "20000x100",
            move |call| {
                let (x, y) = synth(NR_N, NR_P, 1.0, 74);
                let args = ConfirmatoryArgs::SplitExact {
                    n_perm: 1999,
                    n_splits: P_UNITS,
                };
                then(call, || pls1_test(&x, &y, 1, None, args, None, 75))
            },
        ),
        // The Gram input form: the call writes a centred copy of the
        // caller's `n × n` matrix and holds it through the split loop.
        bound("pls1 split_exact k=2 gram input", "3500x250", move |call| {
            let (x, y) = synth(GRAM_N, TALL.1, 1.0, 78);
            let g = gram_matrix(x.as_ref(), false)?;
            drop(x);
            then(call, || {
                pls1_confirmatory_test(
                    ConfirmatoryTestInput::Gram {
                        gram: &g,
                        y: y.as_ref(),
                        k: Some(&[2]),
                        covariates: None,
                        null_y: None,
                    },
                    ConfirmatoryTestOpts {
                        args: SPLIT_EXACT,
                        seed: Some(79),
                        ..Default::default()
                    },
                )
            })
        }),
        bound("pls3 split_exact gram input", "3500x250", move |call| {
            let (x, y) = two_block(GRAM_N, TALL.1, PQ, 80);
            let g = gram_matrix(x.as_ref(), false)?;
            drop(x);
            then(call, || {
                pls3_confirmatory_test_gram(
                    &g,
                    y.as_ref(),
                    1,
                    Pls3ConfirmatoryTestOpts {
                        args: SPLIT_EXACT,
                        seed: Some(81),
                        ..Default::default()
                    },
                )
            })
        }),
        // With covariates the call writes the adjusted copy of X and holds
        // it through the test and the CI.
        bound("pls1 split_exact k=2 covariates ci", "wide", move |call| {
            let (wn, wp) = WIDE;
            let (x, y) = synth(wn, wp, 1.0, 82);
            let (z, _) = synth(wn, COV_Q, 1.0, 83);
            let ci = CIOpts {
                n_boot: P_BOOT,
                m_rate: P_M_RATE,
                ..CIOpts::default()
            };
            then(call, || {
                pls1_confirmatory_test(
                    ConfirmatoryTestInput::Raw {
                        x: x.as_ref(),
                        y: y.as_ref(),
                        k: Some(&[2]),
                        weights: None,
                        covariates: Some(z.as_ref()),
                        null_y: None,
                    },
                    ConfirmatoryTestOpts {
                        args: SPLIT_EXACT,
                        seed: Some(84),
                        ci: Some(ci),
                        ..Default::default()
                    },
                )
            })
        }),
        // The no-refit body scoring its test halves from moments.
        bound("pls1 split_exact k=1 moments", "1000x200", move |call| {
            let (x, y) = synth(MOM_N, MOM_P, 1.0, 89);
            let args = ConfirmatoryArgs::SplitExact {
                n_perm: 999,
                n_splits: 64,
            };
            then(call, || pls1_test(&x, &y, 1, None, args, None, 90))
        }),
        // A k sequence of two steps: the steps hold the standardized copy of
        // X and, from step 2, the deflated residual. `alpha` just below 1
        // lets step 1 reject at 19 permutations, so step 2 runs.
        bound("pls1_find_k_sequence k_max=2", "wide", move |call| {
            let (wn, wp) = WIDE;
            let (x, y) = synth(wn, wp, 1.0, 85);
            let opts = FindKSequenceOpts {
                test_method: ConfirmatoryMethod::SplitExact,
                // every p below 1 rejects
                alpha: 1.0 - f64::EPSILON,
                n_perm: 19,
                n_splits: P_UNITS,
                seed: Some(86),
                ..Default::default()
            };
            then(call, || {
                let r = pls1_find_k_sequence(x.as_ref(), y.as_ref(), 2, None, None, opts)?;
                if r.pvalues[1].is_nan() {
                    return Err(PlsKitError::InvalidArgument("step 2 did not run".into()));
                }
                Ok(r)
            })
        }),
        // The unforced `split_nb` gate on a caller's Gram matrix: the
        // solver of its stable rank beside the centred copy. The gate clears
        // this design, so `split_nb` runs.
        bound("pls1 split_nb k=1 gram input", "3500x250", move |call| {
            let (x, y) = synth(GRAM_N, TALL.1, 1.0, 87);
            let g = gram_matrix(x.as_ref(), false)?;
            drop(x);
            then(call, || {
                pls1_confirmatory_test(
                    ConfirmatoryTestInput::Gram {
                        gram: &g,
                        y: y.as_ref(),
                        k: Some(&[1]),
                        covariates: None,
                        null_y: None,
                    },
                    ConfirmatoryTestOpts {
                        args: SPLIT_NB_UNFORCED,
                        seed: Some(88),
                        ..Default::default()
                    },
                )
                .and_then(ran_split_nb)
            })
        }),
        // The unforced gate's own buffers on wide X under weights: the Gram
        // matrix, its row-weighted copy and the solver.
        bound("split_nb_gate weighted", "wide", move |call| {
            let (x, _) = synth(wn, wp, 1.0, 91);
            let w = row_weights(wn);
            then(call, || split_nb_gate(x.as_ref(), Some(w.as_ref()), false))
        }),
        // The n-space refit route where its kernel runs are large: 4 runs of
        // 16 columns at once on 4 threads hold about 60 MB, over the margin.
        bound(
            "pls1 split_exact k=8 n_perm 199",
            "6000x6500",
            move |call| {
                let (x, y) = synth(NS_N, NS_P, 1.0, 76);
                let args = ConfirmatoryArgs::SplitExact {
                    n_perm: 199,
                    n_splits: P_UNITS,
                };
                then(call, || pls1_test(&x, &y, 8, None, args, None, 77))
            },
        ),
    ];
    for (shape, (n, p)) in [("wide", WIDE), ("tall", TALL)] {
        cases.push(bound("pls1 split_exact k=None", shape, move |call| {
            let (x, y) = synth(n, p, 1.0, 62);
            then(call, || pls1_k_none(&x, &y, None))
        }));
        cases.push(bound("pls1 split_nb k=2", shape, move |call| {
            let (x, y) = synth(n, p, 1.0, 63);
            then(call, || pls1_test(&x, &y, 2, None, SPLIT_NB, None, 64))
        }));
        // The unforced gate: on wide X the Gram matrix and the solver, on
        // tall X the standardized copy, the `p × p` Gram matrix and the
        // solver. It runs no split.
        cases.push(bound("split_nb_gate", shape, move |call| {
            let (x, _) = synth(n, p, 1.0, 92);
            then(call, || split_nb_gate(x.as_ref(), None, false))
        }));
        cases.push(bound("pls1 raw_perm k=2", shape, move |call| {
            let (x, y) = synth(n, p, 1.0, 65);
            then(call, || pls1_test(&x, &y, 2, None, raw_perm, None, 66))
        }));
        for (name, args) in [
            ("pls3 split_exact", SPLIT_EXACT),
            ("pls3 split_nb", SPLIT_NB),
            (
                "pls3 sv_perm",
                ConfirmatoryArgs::SvPerm {
                    n_perm: 19,
                    rotate: false,
                },
            ),
        ] {
            cases.push(bound(name, shape, move |call| {
                let (x, y) = two_block(n, p, PQ, 67);
                then(call, || pls3_test(&x, &y, args, None, None, 68))
            }));
        }
        cases.push(bound("pls1_perm_null", shape, move |call| {
            let (x, y) = synth(n, p, 1.0, 69);
            // The floor of `n_perm`.
            let opts = PermNullOpts {
                n_perm: 100,
                ..Default::default()
            };
            then(call, || {
                pls1_perm_null(x.as_ref(), y.as_ref(), 2, None, None, opts, Some(70))
            })
        }));
        let alpha_n = if shape == "tall" { ALPHA_TALL_N } else { n };
        cases.push(bound("pls1_perm_alpha", shape, move |call| {
            let (x, y) = synth(alpha_n, p, 1.0, 71);
            let g = gram_matrix(x.as_ref(), false)?;
            drop(x);
            // Three outcome columns: `y` and two rotations of its rows.
            let ys = Mat::<f64>::from_fn(alpha_n, 3, |i, j| y[(i + j) % alpha_n]);
            then(call, || {
                pls1_perm_alpha(g.gram.as_ref(), ys.as_ref(), 2, g.n_features)
            })
        }));
        cases.push(bound("factors", shape, move |call| {
            let (x, y) = synth(n, p, 1.0, 72);
            let fit = pls1_fit(x.as_ref(), y.as_ref(), 2, None, None, FitOpts::default())?;
            then(call, || {
                factors(fit.coef.as_ref(), x.as_ref(), 1, false, None)
            })
        }));
        cases.push(bound("pca_basis", shape, move |call| {
            let (x, _) = synth(n, p, 1.0, 73);
            then(call, || pca_basis(x.as_ref(), false))
        }));
    }
    cases
}

/// `split_nb` without `force`, so the auto-gate runs first.
const SPLIT_NB_UNFORCED: ConfirmatoryArgs = ConfirmatoryArgs::SplitNb {
    n_splits: P_UNITS,
    force: false,
};

/// `Err` when the gate rerouted the call to `split_exact`.
fn ran_split_nb(o: ConfirmatoryTestOutput) -> PlsKitResult<ConfirmatoryTestOutput> {
    if o.test_method == "split_nb" {
        Ok(o)
    } else {
        Err(PlsKitError::InvalidArgument(format!(
            "the gate rerouted split_nb to {}",
            o.test_method
        )))
    }
}

/// The route the spec's table names for a row at `4·needed` (its `Debug`
/// name, `Primal` for either source). Gram routes are asserted on both thread
/// counts; at 4 threads the primal rows' larger `needed` can move them, so
/// those rows assert on one thread only.
fn spec_route(name: &str, shape: &str) -> Option<&'static str> {
    match (name, shape) {
        ("pls1 split_exact k=None" | "pls1 raw_perm k=2", "wide") => Some("Nspace"),
        ("pls1 split_exact k=None", "tall") => Some("GramP"),
        ("pls1 split_exact k=None weighted", "wide") | ("pls1 raw_perm k=2", "tall") => {
            Some("Primal")
        }
        _ => None,
    }
}

#[test]
#[ignore = "needs systemd-run --user; run by hand"]
#[allow(clippy::cast_precision_loss)] // the printed VmHWM / cap
fn peak_within_cap() {
    let cases = bound_cases();
    // Child argument: `build <case>` builds the inputs only, `call <case>`
    // also runs the call.
    if let Some(arg) = child_arg() {
        let (mode, i) = arg.split_once(' ').expect("child argument");
        let outcome = match (cases[i.parse::<usize>().unwrap()].run)(mode == "call") {
            Ok(()) => "ok".to_owned(),
            Err(PlsKitError::MemoryBudget { needed, .. }) => format!("memory_budget {needed}"),
            Err(e) => format!("error {e}").replace(['\n', '\r'], " "),
        };
        emit_child(&outcome);
        return;
    }
    let _serial = SERIAL.lock().unwrap();
    println!(
        "{:<34} {:<5} {:>7} {:>12} {:>12} {:>11}  outcome  routes",
        "function", "shape", "threads", "budget", "cap", "VmHWM / cap"
    );
    let mut failures = Vec::new();
    for (i, c) in cases.iter().enumerate() {
        let base = child(TEST, &format!("build {i}"), None, 1, None);
        assert!(
            matches!(base.outcome, Outcome::Ok),
            "{}: base child",
            c.name
        );
        let base = base.hwm;
        let call = format!("call {i}");
        let mut runs = Vec::new();
        let mut needed_1 = 0;
        for threads in [1, 4] {
            let (needed, (), _) = final_needed(&c.name, |b| {
                match child(TEST, &call, Some(b), threads, None).outcome {
                    Outcome::Ok => Ok(()),
                    Outcome::Budget(needed) => Err(PlsKitError::MemoryBudget { budget: b, needed }),
                    Outcome::Error(e) => panic!("{} {} at {b} uncapped: {e}", c.name, c.shape),
                    Outcome::Killed => panic!("{} {} at {b} uncapped: killed", c.name, c.shape),
                }
            });
            if threads == 1 {
                needed_1 = needed;
            }
            runs.extend([(threads, needed, false), (threads, 4 * needed, true)]);
        }
        runs.push((1, 0, false));
        for (threads, budget, at_4x) in runs {
            let capped_as = if budget == 0 { needed_1 } else { budget };
            // `memory.max` holds whole pages.
            let cap = (base + capped_as + margin(capped_as)) / 4096 * 4096;
            let r = child(TEST, &call, Some(budget), threads, Some(cap));
            let outcome = match &r.outcome {
                Outcome::Ok => "ok".to_owned(),
                Outcome::Killed => "killed".to_owned(),
                Outcome::Budget(n) => format!("error: memory_budget {n}"),
                Outcome::Error(e) => format!("error: {e}"),
            };
            let row = format!(
                "{:<34} {:<5} {threads:>7} {budget:>12} {cap:>12} {:>11.3}  {outcome}  {}",
                c.name,
                c.shape,
                r.hwm as f64 / cap as f64,
                r.routes
            );
            println!("{row}");
            let want =
                spec_route(&c.name, c.shape).filter(|w| at_4x && (threads == 1 || *w != "Primal"));
            if !matches!(r.outcome, Outcome::Ok) {
                failures.push(row);
            } else if r.max != cap.to_string() {
                failures.push(format!("{row}: memory.max {} is not the cap", r.max));
            } else if let Some(want) = want.filter(|w| !r.routes.contains(w)) {
                failures.push(format!("{row}: not on the {want} route"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} capped run(s) failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
