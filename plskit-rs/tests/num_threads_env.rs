//! `PLSKIT_NUM_THREADS` end to end. The test re-runs itself in one child
//! process per variable value (`common::run_child`): the variable is read from
//! the process environment, which the test thread cannot change safely.

mod common;

use std::ffi::OsStr;

use common::{child_arg, emit_result, perm_opts, run_child, synth};
use plskit::{PlsKitError, pls1_perm_null};

const VAR: &str = "PLSKIT_NUM_THREADS";
/// Must equal the name of the test function.
const TEST: &str = "plskit_num_threads_caps_the_pool_without_changing_results";

#[test]
fn plskit_num_threads_caps_the_pool_without_changing_results() {
    if let Some(arg) = child_arg() {
        // 2000x600 k=2: the reference fit is past ParChoice::Auto's 1e6
        // threshold, so the call reaches a parallel faer split as well as the
        // replicate loop.
        let (x, y) = synth(2000, 600, 1.0, 9);
        let out = pls1_perm_null(x.as_ref(), y.as_ref(), 2, None, perm_opts(), Some(15))
            .map(|o| format!("{o:?}"));
        match (arg.as_str(), out) {
            ("ok", Ok(s)) => emit_result(&s),
            ("err", Err(PlsKitError::InvalidArgument(msg))) => emit_result(&msg),
            (arg, out) => panic!("{VAR} run `{arg}`: {out:?}"),
        }
        return;
    }
    let child = |value: Option<&str>, arg| {
        run_child(TEST, arg, &[(VAR, value.map(OsStr::new))]).swap_remove(0)
    };
    let unset = child(None, "ok");
    // "0" means no cap; a cap far above the core count runs on the current
    // pool without starting threads.
    for v in ["1", "3", "0", "100000"] {
        assert_eq!(child(Some(v), "ok"), unset, "{VAR}={v}");
    }
    let msg = child(Some("abc"), "err");
    assert!(msg.contains(VAR) && msg.contains("\"abc\""), "{msg}");
}
