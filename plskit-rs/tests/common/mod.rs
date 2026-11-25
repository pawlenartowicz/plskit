//! Helpers shared by the integration tests.
#![allow(dead_code)]

use std::ffi::OsStr;
use std::process::Command;

use faer::{Col, ColRef, Mat, MatRef};
use plskit::PermNullOpts;

pub const CHILD_VAR: &str = "PLSKIT_TEST_CHILD";
pub const RESULT_PREFIX: &str = "PLSKIT_RESULT ";

/// The argument `run_child` passed to this process: `Some` in a child,
/// `None` in the parent.
pub fn child_arg() -> Option<String> {
    std::env::var(CHILD_VAR).ok()
}

/// Prints one result line for the parent. A result is a single line, so a
/// value with a line break is a bug of the test.
pub fn emit_result(value: &str) {
    assert!(
        !value.contains(['\n', '\r']),
        "result spans lines: {value:?}"
    );
    // The leading line break keeps the prefix at the start of its line after
    // the `test <name> ... ` text libtest prints without a newline.
    println!("\n{RESULT_PREFIX}{value}");
}

/// Runs the test `test` (its exact name) again in a child process whose
/// `child_arg()` is `arg`, and returns the values of its `emit_result` calls in
/// order. `env` lists the variables to set (`Some`) or remove (`None`) in the
/// child; the child inherits every other variable. Panics with the child's
/// output if it fails or emits nothing.
pub fn run_child(test: &str, arg: &str, env: &[(&str, Option<&OsStr>)]) -> Vec<String> {
    let mut cmd = Command::new(std::env::current_exe().expect("current_exe"));
    cmd.args([test, "--exact", "--nocapture", "--test-threads=1"])
        .env(CHILD_VAR, arg);
    for (name, value) in env {
        match value {
            Some(v) => cmd.env(name, v),
            None => cmd.env_remove(name),
        };
    }
    let out = cmd.output().expect("spawn child");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let results: Vec<String> = stdout
        .lines()
        .filter_map(|l| l.strip_prefix(RESULT_PREFIX))
        .map(str::to_owned)
        .collect();
    assert!(
        out.status.success() && !results.is_empty(),
        "child `{test}` (arg `{arg}`, env {env:?}) {}\n--- stdout ---\n{stdout}\n--- stderr ---\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    results
}

include!("synth.rs");

pub fn perm_opts() -> PermNullOpts {
    PermNullOpts {
        n_perm: 100,
        return_perm_matrix: true,
        pre_standardized: false,
    }
}

/// Expand `(x, y)` by integer weights into row-duplicated `(x_dup, y_dup)`.
pub fn duplicate_rows(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    w_int: &[u32],
) -> (Mat<f64>, Col<f64>) {
    let n = x.nrows();
    let p = x.ncols();
    let total: usize = w_int.iter().map(|&w| w as usize).sum();
    let mut x_dup = Mat::<f64>::zeros(total, p);
    let mut y_dup = Col::<f64>::zeros(total);
    let mut row = 0;
    for i in 0..n {
        for _ in 0..w_int[i] {
            for j in 0..p {
                x_dup[(row, j)] = x[(i, j)];
            }
            y_dup[row] = y[i];
            row += 1;
        }
    }
    (x_dup, y_dup)
}
