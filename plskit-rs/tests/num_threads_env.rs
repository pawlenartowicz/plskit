//! `PLSKIT_NUM_THREADS` end to end. A test binary of its own with a single
//! `#[test]`: it sets a process-wide environment variable, and the tests of
//! one binary run on parallel threads.

mod common;

use common::{perm_opts, synth};
use plskit::{pls1_perm_null, PlsKitError};

#[test]
fn plskit_num_threads_caps_the_pool_without_changing_results() {
    const VAR: &str = "PLSKIT_NUM_THREADS";
    // 2000x600 k=2: the reference fit is past ParChoice::Auto's 1e6
    // threshold, so the call reaches a parallel faer split as well as the
    // replicate loop.
    let (x, y) = synth(2000, 600, 1.0, 9);
    let run = || {
        pls1_perm_null(x.as_ref(), y.as_ref(), 2, None, perm_opts(), Some(15))
            .map(|o| format!("{o:?}"))
    };
    std::env::remove_var(VAR);
    let unset = run().unwrap();
    // "0" means no cap; a cap far above the core count runs on the current
    // pool without starting threads.
    for v in ["1", "3", "0", "100000"] {
        std::env::set_var(VAR, v);
        assert_eq!(run().unwrap(), unset, "{VAR}={v}");
    }
    std::env::set_var(VAR, "abc");
    match run() {
        Err(PlsKitError::InvalidArgument(msg)) => {
            assert!(msg.contains(VAR) && msg.contains("\"abc\""), "{msg}");
        }
        other => panic!("{VAR}=abc: {other:?}"),
    }
    std::env::remove_var(VAR);
}
