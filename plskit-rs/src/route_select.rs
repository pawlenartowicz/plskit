//! The route choice of a resampling loop: among the routes the loop's
//! eligibility rules allow, the one with the least work per square root of
//! the units that fit the memory budget at once; between routes that give
//! the same bits, the units the pool can run at once decide.

use crate::error::{PlsKitError, PlsKitResult};
use crate::resample::{units_at_once, units_fitting};

/// One route a loop can take. `work` is its side of the loop's cost rules,
/// in primal multiply-adds; `bytes(threads)` returns `(call_bytes,
/// unit_bytes)` when `threads` threads run the loop: the bytes held across
/// all units and those of one unit. Candidates with the same `Some` group
/// give the same bits (the primal route's copy and gather sources).
pub(crate) struct Candidate<'a, R> {
    pub route: R,
    pub work: f64,
    pub bytes: &'a dyn Fn(usize) -> (usize, usize),
    pub group: Option<u8>,
}

/// The chosen route, its bytes at the pool's thread count and how many of
/// its units run at once (`resample::units_at_once`).
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Choice<R> {
    pub route: R,
    pub call_bytes: usize,
    pub unit_bytes: usize,
    pub lanes: usize,
}

/// Pick one of `candidates`, listed in tie order, for a loop of `n_units`
/// units, at the budget the call fixed (`fit::memory_budget`), each
/// candidate's call bytes with `fit::held_bytes` added.
///
/// The choice evaluates every `bytes` at one thread, so the route never
/// depends on the pool, except between candidates of one group. At a
/// nonzero budget it takes the candidate with the least `work / √lanes`
/// among those whose call and one unit fit, the first on ties; at budget
/// `0` the one with the least `call + unit`, ties to the smaller `work`,
/// then the first. When the candidate taken at a nonzero budget has a
/// group, the members of that group then compete again at the pool's
/// `T = rayon::current_num_threads()`: the least `work / √min(lanes, T)`,
/// `lanes` from their bytes at `T` (`resample::units_at_once`), among those
/// that fit there, the first on ties. So on one thread the copy source wins
/// wherever it fits, and gather's extra lanes count only up to the threads
/// that run them; the bits do not move. The chosen candidate's bytes are
/// then taken at `T`, and its lanes from them and from faer's multiply
/// buffers on those threads (`resample::units_at_once`).
///
/// # Errors
/// `MemoryBudget` when no candidate fits at one thread (`needed` the least
/// `call + unit`, plus the call's multiply buffers, `fit::gemm_bytes`, so
/// that the call runs there), or when the chosen one does not fit at the
/// pool's thread count (`needed` its bytes there).
pub(crate) fn select<R: Copy + std::fmt::Debug>(
    n_units: usize,
    candidates: &[Candidate<'_, R>],
) -> PlsKitResult<Choice<R>> {
    debug_assert!(!candidates.is_empty(), "a loop has at least one route");
    let budget = crate::fit::memory_budget();
    let held = crate::fit::held_bytes();
    let need = |c: &Candidate<'_, R>| {
        let (call, unit) = (c.bytes)(1);
        call.checked_add(unit)
            .and_then(|b| b.checked_add(held))
            .and_then(|b| u64::try_from(b).ok())
            .unwrap_or(u64::MAX)
    };
    let best = if budget == 0 {
        candidates
            .iter()
            .min_by(|a, b| need(a).cmp(&need(b)).then(a.work.total_cmp(&b.work)))
    } else {
        candidates
            .iter()
            .filter_map(|c| {
                let (call, unit) = (c.bytes)(1);
                let lanes = units_fitting(n_units, call, unit).ok()?;
                Some((c, c.work / (lanes.max(1) as f64).sqrt()))
            })
            .min_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(c, _)| c)
    };
    let Some(best) = best else {
        return Err(PlsKitError::MemoryBudget {
            budget,
            needed: candidates
                .iter()
                .map(need)
                .min()
                .unwrap_or(0)
                .saturating_add(crate::fit::gemm_bytes() as u64),
        });
    };
    let threads = rayon::current_num_threads();
    let best = match best.group {
        Some(_) if budget > 0 => candidates
            .iter()
            .filter(|c| c.group == best.group)
            .filter_map(|c| {
                let (call, unit) = (c.bytes)(threads);
                let lanes = units_at_once(n_units, call, unit).ok()?.min(threads);
                Some((c, c.work / (lanes.max(1) as f64).sqrt()))
            })
            .min_by(|(_, a), (_, b)| a.total_cmp(b))
            .map_or(best, |(c, _)| c),
        _ => best,
    };
    let (call_bytes, unit_bytes) = (best.bytes)(threads);
    let lanes = units_at_once(n_units, call_bytes, unit_bytes)?;
    #[cfg(test)]
    ROUTE_LOG.with(|log| {
        if let Some(log) = log.borrow_mut().as_mut() {
            log.push((format!("{:?}", best.route), n_units, lanes));
        }
    });
    #[cfg(feature = "route-log")]
    ROUTES_TAKEN
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(format!("{:?}", best.route));
    Ok(Choice {
        route: best.route,
        call_bytes,
        unit_bytes,
        lanes,
    })
}

/// Every route [`select`] has chosen in this process, on any thread
/// (feature `route-log`).
#[cfg(feature = "route-log")]
static ROUTES_TAKEN: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// The routes the loops of this process have taken since the last call, in
/// order, in their `Debug` form, from every thread (feature `route-log`,
/// off by default): for a test that runs one call per process.
#[cfg(feature = "route-log")]
#[must_use]
pub fn take_routes() -> Vec<String> {
    std::mem::take(
        &mut *ROUTES_TAKEN
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    )
}

#[cfg(test)]
thread_local! {
    /// Set by `with_route_log`; [`select`] appends to it while it is `Some`.
    static ROUTE_LOG: std::cell::RefCell<Option<Vec<(String, usize, usize)>>> =
        const { std::cell::RefCell::new(None) };
}

/// Run `f` and return its result with `(route, n_units, lanes)` for every
/// [`select`] that succeeds on the calling thread while `f` runs, in call
/// order, `route` in its `Debug` form (test builds). Choices made on other
/// threads, inside units that run on Rayon workers, are not recorded. The
/// previous log is restored afterwards, also when `f` panics.
#[cfg(test)]
pub(crate) fn with_route_log<T>(f: impl FnOnce() -> T) -> (T, Vec<(String, usize, usize)>) {
    struct Restore(Option<Vec<(String, usize, usize)>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            ROUTE_LOG.with(|log| *log.borrow_mut() = self.0.take());
        }
    }
    let _restore = Restore(ROUTE_LOG.with(|log| log.replace(Some(Vec::new()))));
    let out = f();
    let log = ROUTE_LOG
        .with(|log| log.borrow_mut().take())
        .unwrap_or_default();
    (out, log)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::with_budget;

    fn fixed(call: usize, unit: usize) -> impl Fn(usize) -> (usize, usize) {
        move |_| (call, unit)
    }

    fn pool(threads: usize) -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
    }

    fn needed(r: PlsKitResult<Choice<char>>) -> u64 {
        match r {
            Err(PlsKitError::MemoryBudget { needed, .. }) => needed,
            other => panic!("expected MemoryBudget, got {other:?}"),
        }
    }

    /// Budget 100: A (work 10, unit 50) runs 2 lanes, 10/√2 ≈ 7.07; B (work
    /// 8, unit 100) runs 1, 8. A wins on less work per √lanes.
    #[test]
    fn select_takes_least_work_per_sqrt_lanes() {
        let (a, b) = (fixed(0, 50), fixed(0, 100));
        let cs = [
            Candidate {
                route: 'A',
                work: 10.0,
                bytes: &a,
                group: None,
            },
            Candidate {
                route: 'B',
                work: 8.0,
                bytes: &b,
                group: None,
            },
        ];
        let c = with_budget(100, || select(4, &cs)).unwrap();
        let want = Choice {
            route: 'A',
            call_bytes: 0,
            unit_bytes: 50,
            lanes: 2,
        };
        assert_eq!(c, want);
    }

    #[test]
    fn select_ties_to_list_order() {
        let a = fixed(10, 20);
        let cs = [
            Candidate {
                route: 'A',
                work: 5.0,
                bytes: &a,
                group: None,
            },
            Candidate {
                route: 'B',
                work: 5.0,
                bytes: &a,
                group: None,
            },
        ];
        assert_eq!(with_budget(100, || select(3, &cs)).unwrap().route, 'A');
        assert_eq!(with_budget(0, || select(3, &cs)).unwrap().route, 'A');
    }

    #[test]
    fn select_errors_with_least_need() {
        let (a, b) = (fixed(5, 10), fixed(2, 10));
        let cs = [
            Candidate {
                route: 'A',
                work: 1.0,
                bytes: &a,
                group: None,
            },
            Candidate {
                route: 'B',
                work: 1.0,
                bytes: &b,
                group: None,
            },
        ];
        assert_eq!(needed(with_budget(10, || select(3, &cs))), 12);
    }

    /// Budget 0 takes the least memory, whatever the work, one unit at a
    /// time; equal memory goes to the smaller work.
    #[test]
    fn select_at_zero_takes_least_memory() {
        let (a, b) = (fixed(0, 50), fixed(0, 30));
        let cs = [
            Candidate {
                route: 'A',
                work: 10.0,
                bytes: &a,
                group: None,
            },
            Candidate {
                route: 'B',
                work: 99.0,
                bytes: &b,
                group: None,
            },
            Candidate {
                route: 'C',
                work: 98.0,
                bytes: &b,
                group: None,
            },
        ];
        let c = with_budget(0, || select(4, &cs)).unwrap();
        assert_eq!((c.route, c.lanes), ('C', 1));
    }

    /// A need past `usize` is `u64::MAX`, at a budget near it, and never
    /// wraps into a fit.
    #[test]
    fn select_saturates() {
        let (a, b) = (
            fixed(usize::MAX, 1),
            fixed(usize::MAX / 2 + 1, usize::MAX / 2 + 1),
        );
        let cs = [
            Candidate {
                route: 'A',
                work: 1.0,
                bytes: &a,
                group: None,
            },
            Candidate {
                route: 'B',
                work: 1.0,
                bytes: &b,
                group: None,
            },
        ];
        assert_eq!(
            needed(with_budget(u64::MAX - 1, || select(3, &cs))),
            u64::MAX
        );
        assert_eq!(needed(with_budget(u64::MAX, || select(3, &cs))), u64::MAX);
    }

    /// Bytes that grow with the threads do not move the choice. At budget
    /// 1000, A (work 10, call 150 per thread, unit 100) runs 8 lanes at one
    /// thread (10/√8 ≈ 3.5) and beats B (work 6, unit 400, 2 lanes, 6/√2 ≈
    /// 4.2); at 5 threads A would run 2 lanes (7.1) and lose, but stays the
    /// choice, with the 5-thread bytes and lanes.
    #[test]
    fn route_ignores_threads() {
        let run = |threads| {
            pool(threads).install(|| {
                let (a, b) = (|t: usize| (150 * t, 100), fixed(0, 400));
                let cs = [
                    Candidate {
                        route: 'A',
                        work: 10.0,
                        bytes: &a,
                        group: None,
                    },
                    Candidate {
                        route: 'B',
                        work: 6.0,
                        bytes: &b,
                        group: None,
                    },
                ];
                with_budget(1000, || select(16, &cs)).unwrap()
            })
        };
        let (one, five) = (run(1), run(5));
        assert_eq!((one.route, one.call_bytes, one.lanes), ('A', 150, 8));
        assert_eq!((five.route, five.call_bytes, five.lanes), ('A', 750, 2));
    }

    /// faer's multiply buffers enter only the real check: counted in the
    /// choice, a 5000-byte term would leave B (4 units at once, 2 with the
    /// term) behind A (4 at once either way), so B, the choice without it,
    /// is taken on pools of 1 and 5, with 2 lanes.
    #[test]
    fn gemm_term_stays_out_of_the_route_choice() {
        let run = |threads| {
            pool(threads).install(|| {
                let (a, b) = (fixed(0, 1000), fixed(0, 2000));
                let cs = [
                    Candidate {
                        route: 'A',
                        work: 10.0,
                        bytes: &a,
                        group: None,
                    },
                    Candidate {
                        route: 'B',
                        work: 9.0,
                        bytes: &b,
                        group: None,
                    },
                ];
                with_budget(10_000, || {
                    let _gemm = crate::fit::hold_gemm_bytes(5000);
                    select(4, &cs).unwrap()
                })
            })
        };
        for threads in [1, 5] {
            let c = run(threads);
            assert_eq!((c.route, c.lanes), ('B', 2), "{threads} thread(s)");
        }
    }

    /// A primal-like pair at budget 1000 over 16 units: copy (work 10, unit
    /// 300, 3 lanes) and gather (work 15, unit 100, 10 lanes), one group,
    /// beside a grouped-off route Z (work 30, unit 1000, 1 lane). The
    /// one-thread score picks gather (15/√10 ≈ 4.7 against 10/√3 ≈ 5.8);
    /// capped at the pool, copy wins on 1 thread (10 against 15) and on 2
    /// (7.1 against 10.6), gather on 8 (3 lanes 5.8 against 8 lanes 5.3).
    #[test]
    fn group_competes_at_the_pool() {
        let run = |threads| {
            pool(threads).install(|| {
                let (copy, gather, z) = (fixed(0, 300), fixed(0, 100), fixed(0, 1000));
                let cs = [
                    Candidate {
                        route: 'Z',
                        work: 30.0,
                        bytes: &z,
                        group: None,
                    },
                    Candidate {
                        route: 'C',
                        work: 10.0,
                        bytes: &copy,
                        group: Some(0),
                    },
                    Candidate {
                        route: 'G',
                        work: 15.0,
                        bytes: &gather,
                        group: Some(0),
                    },
                ];
                with_budget(1000, || select(16, &cs).unwrap())
            })
        };
        let (one, two, eight) = (run(1), run(2), run(8));
        assert_eq!((one.route, one.lanes), ('C', 3));
        assert_eq!((two.route, two.lanes), ('C', 3));
        assert_eq!((eight.route, eight.lanes), ('G', 10));
    }

    /// A group member that fits only at one thread is passed over for one
    /// that fits at the pool: copy (work 1, call 300 per thread, unit 100)
    /// fits 7 lanes at one thread and none at four, gather (work 2, unit
    /// 100) fits 10 everywhere.
    #[test]
    fn group_takes_the_member_that_fits_the_pool() {
        let c = pool(4).install(|| {
            let (copy, gather) = (|t: usize| (300 * t, 100), fixed(0, 100));
            let cs = [
                Candidate {
                    route: 'C',
                    work: 1.0,
                    bytes: &copy,
                    group: Some(0),
                },
                Candidate {
                    route: 'G',
                    work: 2.0,
                    bytes: &gather,
                    group: Some(0),
                },
            ];
            with_budget(1000, || select(16, &cs).unwrap())
        });
        assert_eq!((c.route, c.lanes), ('G', 10));
    }

    /// Fits at one thread (40 bytes), not at four (160): the error names
    /// the four-thread need.
    #[test]
    fn select_rechecks_at_real_threads() {
        let r = pool(4).install(|| {
            let a = |t: usize| (40 * t, 0);
            let cs = [Candidate {
                route: 'A',
                work: 1.0,
                bytes: &a,
                group: None,
            }];
            with_budget(100, || select(3, &cs))
        });
        assert_eq!(needed(r), 160);
    }
}
