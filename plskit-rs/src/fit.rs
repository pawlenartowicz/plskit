//! PLS1 fit: the NIPALS PLS1 model, computed by Improved Kernel PLS (one
//! shared component loop, `pls1_component_loop`, with the X backend; see
//! `pls1_kernel`). Public entry points: `pls1_fit`, `spls1_fit`.

use faer::linalg::matmul::matmul;
use faer::{Accum, Col, ColRef, Mat, MatRef, Par};

use crate::error::{PlsKitError, PlsKitResult};

pub(crate) mod lanes;

/// Parallelism strategy for the PLS1 kernel (`pls1_kernel`) called by `pls1_fit`.
///
/// `Auto` (the default) selects per-fit based on problem size:
/// runs sequentially when `n * d * k < 1_000_000` and otherwise on the
/// current Rayon pool (an installed pool, if the call runs inside one),
/// split into a fixed number of pieces (see below). The `1_000_000` threshold reflects the
/// crossover measured on Arrow Lake-H: at smaller sizes faer's matmul
/// dispatch over-eagerly parallelizes and the thread-overhead dominates
/// (~1.8× slowdown observed at `(200, 800, 5)`).
///
/// The parallel arm splits each product into a fixed number of pieces
/// (8), however many threads the pool has, so a fit's bits
/// depend on its inputs and on whether `Auto` chose the parallel arm (a
/// function of the shape), never on the pool size: `PLSKIT_NUM_THREADS=1`
/// and `PLSKIT_NUM_THREADS=16` give the same result. `Seq` and the parallel
/// arm of `Auto` can differ from each other in the last bits.
///
/// The chosen arm also runs the fit's pass over `X` before the kernel: the
/// standardization moments of a raw `X`, over strips of columns (also in
/// the PLS3 fits), and the sum of squares of a pre-standardized `X`, in
/// pieces. Both give the same bits on either arm.
///
/// Inside the resampling functions, the PLS1 fit of one resample takes
/// `Auto`, so a large fit still uses the cores when the memory budget
/// (`PLSKIT_MEMORY_BUDGET`) lets only a few resamples run at once. This
/// holds for the per-split fits of `pls1_confirmatory_test` with
/// `ConfirmatoryMethod::SplitNb` and `ConfirmatoryMethod::SplitExact`, the
/// cross-validation fold fits of `pls1_find_k_optimal`,
/// `spls1_find_k_optimal` and `spls1_find_keep_optimal`, and the subsample
/// and bootstrap fits of the confirmatory CI (`ConfirmatoryTestOpts::ci`). The fold fits of
/// `ConfirmatoryMethod::RawPerm`, the permutation refits of
/// `pls1_perm_null` and the per-split fits of `pls3_confirmatory_test` are
/// each sequential (the resamples themselves can still run in parallel).
//
// Route detail. `Auto` (where the loop runs its units through
// `resample::map_units`): `signal_test::split_column_r` through
// `signal_test::SPLIT_FIT_PAR`, which also covers the first product
// `ŝ₁ = Xs'ys` of `split_exact`'s Gram-p arm (formed under the `Par` that
// fit resolves); the two `find_k` fold fits; `subsample::fit_rows`. `Seq`:
// the products of the `raw_perm` fold fits (`signal_test::fold_columns_lanes`)
// and the Gram-p arm's first product on its folds; the permutation refits of
// `pls1_perm_null` and the Gram-p arm's first product there; the
// per-replicate products of the Gram-p and n-space backends beyond that
// first product; and the per-split PLS3 fit of `pls3_signal_test`, which
// runs every product with `Par::Seq`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ParChoice {
    /// Decide per fit based on `n * d * k`.
    Auto,
    /// Force sequential execution.
    Seq,
}

/// Knobs for `pls1_fit`.
#[derive(Debug, Clone, Copy)]
pub struct FitOpts {
    /// Skip the centering/scaling step; caller asserts X and y are already standardized.
    ///
    /// **Scale contract.** When `pre_standardized=true`, the PLS1 kernel
    /// stops a component on `‖X_a'y_a‖ < 1e-14` and on `t't < 1e-14`
    /// (`t = X_a w`), and on `‖X_a'y_a‖` at the rounding floor relative to
    /// `‖X‖_F·‖y‖`, which is scale-free. An X whose largest entry (with
    /// weights, of the √w-scaled rows) is outside `[2^-20, 2^20]` is fitted
    /// divided by a power of two near that entry and the fit is mapped back
    /// to X's scale, as if fitted in exact arithmetic, so X's scale does not
    /// cut components. y is used as given and not divided, so for a y far
    /// below unit scale the `‖X_a'y_a‖` stop still depends on its scale.
    /// Callers passing `pre_standardized=true` must ensure the inputs are
    /// genuinely zero-mean / unit-variance (or at least scale-comparable
    /// to that). The default `pre_standardized=false` path absorbs raw
    /// scale automatically and is unaffected by this contract.
    ///
    /// As a guard, when `pre_standardized=true`, the public `pls1_fit`
    /// returns `InvalidInput` if the PLS1 kernel truncates below the requested `k`.
    /// The crate's own per-fold and per-split fits (CV folds, per-half split fits,
    /// permutation refits, the BIC full-k fit, the sequential deflation
    /// fit) tolerate truncation by design.
    pub pre_standardized: bool,
    /// Parallelism strategy for the fit's pass over `X` (the standardization
    /// moments or the sum of squares) and the PLS1 kernel. See `ParChoice`.
    pub par: ParChoice,
}

impl Default for FitOpts {
    fn default() -> Self {
        Self {
            pre_standardized: false,
            par: ParChoice::Auto,
        }
    }
}

/// Crate-internal knobs of [`pls1_fit_impl`], beside the public [`FitOpts`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct InternalFitOpts {
    /// When true (default), the fit errors with `InvalidWeights{reason:"insufficient_effective_n"}`
    /// (non-uniform weights) or `InvalidArgument` (uniform or absent weights)
    /// if `n_eff < k + 1`. Set to false for per-iteration internal calls (CV folds,
    /// bootstrap subsamples) where the upstream accumulator handles degeneracy.
    /// See `docs/internals/n-eff-check.md`.
    pub(crate) check_n_eff: bool,
    /// Sparse keep-count (spls1 family plumbing): retain the `keep`
    /// largest-|w| coordinates per component, zero the rest — hard selection
    /// at the keep-th order statistic of |w|; exact ties break by lowest
    /// column index (reproducibility contract). `None` (default) = dense
    /// PLS1. Wrapper surfaces never expose this on dense functions;
    /// call `spls1_fit` instead of setting it directly.
    pub(crate) keep: Option<usize>,
}

impl Default for InternalFitOpts {
    fn default() -> Self {
        Self {
            check_n_eff: true,
            keep: None,
        }
    }
}

/// Number of pieces every parallel faer call in this crate is split into.
///
/// faer's `Par::rayon(0)` (and its global default, which operator `*` and
/// the high-level decompositions read) takes the degree from
/// `rayon::current_num_threads()`, and the degree is part of the
/// arithmetic, not only of the scheduling: a column-major GEMV `X v` sums
/// `degree` partial products over column blocks, in order, so the rounding
/// of `t = X r` changes with the pool size. A fixed degree fixes the split;
/// Rayon then only schedules the pieces, on however many threads it has.
///
/// 8 spreads the kernel GEMVs over the cores of a typical workstation. The
/// column-major GEMV adds its partial vectors in one serial pass (`n` adds
/// per piece), so a much larger degree costs more than it gains. A pool
/// with fewer threads runs the 8 pieces on the threads it has; a larger
/// pool leaves the rest free for that call.
pub(crate) const PAR_DEGREE: usize = 8;

/// The one parallel `faer::Par` of this crate: `Par::Rayon(PAR_DEGREE)`.
/// Never `Par::rayon(0)`, whose degree is the pool size (see
/// [`PAR_DEGREE`]).
pub(crate) fn par_fixed() -> Par {
    Par::rayon(PAR_DEGREE)
}

/// The number of pieces the crate's own row-split kernels cut their work
/// into: [`PAR_DEGREE`]. It sets the pieces of
/// `linalg::row_major_t_mul`'s parallel arm and of both arms of
/// `linalg::sum_of_squares`.
pub(crate) fn par_degree() -> usize {
    PAR_DEGREE
}

/// Environment variable that caps plskit's thread count; read once per
/// top-level call by [`with_thread_limit`].
pub(crate) const NUM_THREADS_ENV: &str = "PLSKIT_NUM_THREADS";

/// `PLSKIT_NUM_THREADS` as read from the environment: unset or `0` → `None`
/// (no cap: run on the current Rayon pool); a positive integer → `Some(n)`,
/// the cap; anything else is an error that quotes the value (no trimming,
/// so stray whitespace shows in the message).
pub(crate) fn parse_num_threads(raw: Option<&str>) -> PlsKitResult<Option<usize>> {
    match raw {
        None | Some("0") => Ok(None),
        Some(s) => match s.parse::<usize>() {
            Ok(n) if n >= 1 && s.bytes().all(|c| c.is_ascii_digit()) => Ok(Some(n)),
            _ => Err(PlsKitError::InvalidArgument(format!(
                "{NUM_THREADS_ENV} must be a positive integer or 0, got {s:?}"
            ))),
        },
    }
}

/// Environment variable that sets the memory budget of the resampling
/// loops; see [`memory_budget`]. Read once per top-level call by
/// [`with_thread_limit`].
pub(crate) const MEMORY_BUDGET_ENV: &str = "PLSKIT_MEMORY_BUDGET";

/// Budget in bytes when `PLSKIT_MEMORY_BUDGET` is unset: 4 GiB.
pub(crate) const DEFAULT_MEMORY_BUDGET: u64 = 4 << 30;

/// `PLSKIT_MEMORY_BUDGET` as read from the environment: unset → the default;
/// otherwise a whole number of bytes, optionally followed by a unit letter
/// `K`, `M`, `G` or `T` (powers of 1024), optionally followed by `B`
/// (after digits or a unit letter) or `iB` (after a unit letter only), in
/// any case. `0` is a valid budget (one unit at a time). Anything else, or
/// a value past `u64`, is an error that quotes the value (no trimming, so
/// stray whitespace shows in the message).
pub(crate) fn parse_memory_budget(raw: Option<&str>) -> PlsKitResult<u64> {
    let Some(s) = raw else {
        return Ok(DEFAULT_MEMORY_BUDGET);
    };
    let parsed = (|| {
        let mut t = s.as_bytes();
        let unit_at = |t: &[u8]| -> Option<u32> {
            match t.last()?.to_ascii_uppercase() {
                b'K' => Some(10),
                b'M' => Some(20),
                b'G' => Some(30),
                b'T' => Some(40),
                _ => None,
            }
        };
        if t.len() >= 2 && t[t.len() - 2..].eq_ignore_ascii_case(b"ib") {
            t = &t[..t.len() - 2];
            unit_at(t)?;
        } else if t.last().is_some_and(|c| c.eq_ignore_ascii_case(&b'b')) {
            t = &t[..t.len() - 1];
        }
        let shift = unit_at(t);
        if shift.is_some() {
            t = &t[..t.len() - 1];
        }
        if t.is_empty() || !t.iter().all(u8::is_ascii_digit) {
            return None;
        }
        let n: u64 = std::str::from_utf8(t).ok()?.parse().ok()?;
        n.checked_mul(1_u64 << shift.unwrap_or(0))
    })();
    parsed.ok_or_else(|| bad_memory_budget(&format!("{s:?}")))
}

fn bad_memory_budget(quoted: &str) -> PlsKitError {
    PlsKitError::InvalidArgument(format!(
        "{MEMORY_BUDGET_ENV} must be a whole number of bytes with an optional K, M, G or T \
         and an optional B or iB (for example 2G), got {quoted}"
    ))
}

thread_local! {
    static MEMORY_BUDGET_OVERRIDE: std::cell::Cell<Option<u64>> =
        const { std::cell::Cell::new(None) };
    static HELD_BYTES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static GEMM_BYTES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Test builds read this in place of the `PLSKIT_MEMORY_BUDGET`
/// environment variable when it is set: safe code cannot change the
/// environment of a running process. It is process-wide, so only a test
/// that runs alone in a child process sets it.
#[cfg(test)]
static MEMORY_BUDGET_VAR: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

fn memory_budget_var() -> Result<String, std::env::VarError> {
    #[cfg(test)]
    if let Some(s) = MEMORY_BUDGET_VAR
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
    {
        return Ok(s);
    }
    std::env::var(MEMORY_BUDGET_ENV)
}

/// Bytes of working memory the resampling loops may spend on the buffers
/// that grow with the data: the call's value when one is set on this
/// thread (by [`with_budget`], which [`with_thread_limit`] and the batching
/// helpers of `resample` call), else `PLSKIT_MEMORY_BUDGET`, the default
/// when it is unset or does not parse (the entry points reject a bad value
/// up front).
pub(crate) fn memory_budget() -> u64 {
    if let Some(b) = MEMORY_BUDGET_OVERRIDE.with(std::cell::Cell::get) {
        return b;
    }
    match memory_budget_var() {
        Ok(s) => parse_memory_budget(Some(&s)).unwrap_or(DEFAULT_MEMORY_BUDGET),
        Err(_) => DEFAULT_MEMORY_BUDGET,
    }
}

/// Run `f` with [`memory_budget`] fixed to `bytes` on the calling thread
/// (loop sites that `f` reaches on this thread see it; other threads see
/// it only where a batching helper of `resample` passes it on). The
/// previous value is restored afterwards, also when `f` panics.
pub(crate) fn with_budget<T>(bytes: u64, f: impl FnOnce() -> T) -> T {
    struct Restore(Option<u64>);
    impl Drop for Restore {
        fn drop(&mut self) {
            MEMORY_BUDGET_OVERRIDE.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(MEMORY_BUDGET_OVERRIDE.with(|c| c.replace(Some(bytes))));
    let _held = hold_bytes(0);
    let _gemm = hold_gemm_bytes(0);
    f()
}

/// Bytes the call holds on this thread beside the loops it runs (the
/// covariate-adjusted copy of X in `find_k`): `resample::units_at_once`
/// and `route_select::select` count them as call bytes of every loop.
/// [`with_budget`] sets them to 0 for its scope, so the units of a loop,
/// which the batching helpers run under it, and a public call's body do not
/// count them.
pub(crate) fn held_bytes() -> usize {
    HELD_BYTES.with(std::cell::Cell::get)
}

/// Set [`held_bytes`] to `bytes` on the calling thread until the returned
/// guard drops, which restores the previous value (also on a panic).
#[must_use = "held bytes count only while the guard lives"]
pub(crate) fn hold_bytes(bytes: usize) -> HeldBytes {
    HeldBytes::set(&HELD_BYTES, bytes)
}

/// faer's multiply buffers (`resample::gemm_buffer_bytes`) that
/// `resample::units_at_once` adds to every loop's call bytes at the real
/// thread count: set by [`with_thread_limit`] on a call that takes its
/// budget from the environment, and 0 under every other [`with_budget`]
/// (a loop's units, a call made inside another, which the outer call
/// counted, and tests that fix a budget).
pub(crate) fn gemm_bytes() -> usize {
    GEMM_BYTES.with(std::cell::Cell::get)
}

/// Set [`gemm_bytes`] to `bytes` on the calling thread until the returned
/// guard drops, which restores the previous value (also on a panic).
#[must_use = "the multiply buffers count only while the guard lives"]
pub(crate) fn hold_gemm_bytes(bytes: usize) -> HeldBytes {
    HeldBytes::set(&GEMM_BYTES, bytes)
}

/// The guard of [`hold_bytes`] and of [`hold_gemm_bytes`]: the thread-local it
/// set and the value it replaced.
pub(crate) struct HeldBytes(
    &'static std::thread::LocalKey<std::cell::Cell<usize>>,
    usize,
);

impl HeldBytes {
    fn set(key: &'static std::thread::LocalKey<std::cell::Cell<usize>>, bytes: usize) -> Self {
        Self(key, key.with(|c| c.replace(bytes)))
    }
}

impl Drop for HeldBytes {
    fn drop(&mut self) {
        self.0.with(|c| c.set(self.1));
    }
}

/// The plskit-owned pool for the last `n` asked for. One size is cached:
/// a different `n` builds a new pool and drops the old one once its last
/// running call finishes (each call holds its own `Arc`).
static NUM_THREADS_POOL: std::sync::Mutex<Option<(usize, std::sync::Arc<rayon::ThreadPool>)>> =
    std::sync::Mutex::new(None);

/// Run `f` on at most `n` threads: on the current pool when `n` is `None`
/// or not below the current pool's size (the core count when called
/// outside any pool), otherwise on a plskit pool of `n`
/// threads, so a cap at or above the pool size never starts threads. The
/// cache lock is held only to fetch or replace the pool, never while `f`
/// runs, so concurrent calls with different `n` do not serialize.
pub(crate) fn run_with_threads<T: Send>(
    n: Option<usize>,
    f: impl FnOnce() -> T + Send,
) -> PlsKitResult<T> {
    // Outside any pool, `rayon::current_num_threads()` would start Rayon's
    // global pool of all cores just to read its size; compare to the core
    // count instead, so a cap never leaves an idle global pool behind.
    let pool_size = if rayon::current_thread_index().is_some() {
        rayon::current_num_threads()
    } else {
        std::thread::available_parallelism().map_or(1, std::num::NonZero::get)
    };
    let Some(n) = n.filter(|&n| n < pool_size) else {
        return Ok(f());
    };
    let pool = {
        let mut cache = NUM_THREADS_POOL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match cache.as_ref() {
            Some((m, p)) if *m == n => std::sync::Arc::clone(p),
            _ => {
                let p = std::sync::Arc::new(
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(n)
                        .build()
                        .map_err(|e| {
                            PlsKitError::Internal(format!(
                                "could not start a {n}-thread pool for {NUM_THREADS_ENV}: {e}"
                            ))
                        })?,
                );
                *cache = Some((n, std::sync::Arc::clone(&p)));
                p
            }
        }
    };
    Ok(pool.install(f))
}

/// Every public entry point that reaches Rayon runs its body through this:
/// it reads `PLSKIT_NUM_THREADS` once and runs `f` on the matching pool. It
/// also reads `PLSKIT_MEMORY_BUDGET` once (a bad value is rejected before
/// any work) and runs `f` under [`with_budget`] with it, so every loop of
/// the call sees that value; on a thread that already has a call's budget
/// (a call made inside another), that budget is kept. A call that reads the
/// budget also sets [`gemm_bytes`] for its loops.
/// Crate-internal callers call the `_impl` functions instead, so inner fits
/// inside replicate workers never re-read the variables.
pub(crate) fn with_thread_limit<T: Send>(
    f: impl FnOnce() -> PlsKitResult<T> + Send,
) -> PlsKitResult<T> {
    let n = match std::env::var(NUM_THREADS_ENV) {
        Ok(s) => parse_num_threads(Some(&s))?,
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(raw)) => {
            return Err(PlsKitError::InvalidArgument(format!(
                "{NUM_THREADS_ENV} must be a positive integer or 0, got {raw:?}"
            )));
        }
    };
    let budget = match memory_budget_var() {
        Ok(s) => parse_memory_budget(Some(&s))?,
        Err(std::env::VarError::NotPresent) => DEFAULT_MEMORY_BUDGET,
        Err(std::env::VarError::NotUnicode(raw)) => {
            return Err(bad_memory_budget(&format!("{raw:?}")));
        }
    };
    let outer = MEMORY_BUDGET_OVERRIDE.with(std::cell::Cell::get);
    run_with_threads(n, move || {
        with_budget(outer.unwrap_or(budget), || {
            // Read on the thread that runs the call's loops.
            let gemm = if outer.is_none() {
                crate::resample::gemm_buffer_bytes()
            } else {
                0
            };
            let _gemm = hold_gemm_bytes(gemm);
            f()
        })
    })?
}

/// Translate a `ParChoice` into a concrete `faer::Par` for the given problem size.
///
/// `n` is the number of samples and `d` the number of features. `n_sweeps`
/// is the number of `O(n * d)` passes over `X` the caller performs, so that
/// `n * d * n_sweeps` estimates the flop count of the dominant kernel: the
/// requested component count `k` for PLS1 (the PLS1 kernel reads `X`
/// `2k + 1` times; `pls1_fit` passes `k`, the count the `1_000_000`
/// threshold was measured with), and the target count `q` for PLS3 (the `X'Y` and score
/// matmuls are `n * p * q`).
///
/// `ParChoice::Seq` always maps to `Par::Seq`; `ParChoice::Auto` uses
/// [`par_fixed`] when `n * d * n_sweeps >= 1_000_000`, else `Par::Seq`.
/// Saturating arithmetic guards against `usize` overflow on absurd inputs.
pub(crate) fn resolve_par(choice: ParChoice, n: usize, d: usize, n_sweeps: usize) -> Par {
    match choice {
        ParChoice::Seq => Par::Seq,
        ParChoice::Auto => {
            let work = n.saturating_mul(d).saturating_mul(n_sweeps);
            if work >= 1_000_000 {
                par_fixed()
            } else {
                Par::Seq
            }
        }
    }
}

/// Owned PLS1 fit. Fields use long `snake_case` names;
/// the wrapper translates to short Python-facing names at the FFI seam.
#[derive(Debug, Clone)]
pub struct Pls1Model {
    /// X-scores `T`; shape `(n_samples, k_used)`.
    pub t_scores: Mat<f64>,
    /// X-loadings `P`; shape `(n_features, k_used)`.
    pub p_loadings: Mat<f64>,
    /// X-weights `W` (raw NIPALS weights, unit-normed per component); shape `(n_features, k_used)`.
    /// Note: this is raw W, not the modified W* = W·(P'W)^{-1} used to back-solve coefficients — see `pls1_coef_at_k`.
    pub w_star: Mat<f64>,
    /// y-loadings `Q`; shape `(k_used,)`.
    pub q_loadings: Col<f64>,
    /// Regression coefficients in standardized space; shape `(n_features,)`.
    pub coef: Col<f64>,
    /// Regression coefficients back-projected to raw X scale; shape `(n_features,)`.
    pub beta: Col<f64>,
    /// y intercept in raw scale (0 when `pre_standardized=true` without
    /// covariates; with covariates, the intercept of the fit
    /// `ŷ = intercept + X·β + Z·covariate_coef` in either case).
    pub intercept: f64,
    /// Number of components actually retained (≤ requested `k`).
    pub k_used: usize,
    /// Echoes the caller's `pre_standardized` flag.
    pub pre_standardized: bool,
    /// Resolved (post-normalization) weight vector. `None` when input was uniform
    /// or absent (all-equal weights fit identically to none). Length = `n_samples` when present.
    pub weights: Option<Col<f64>>,
    /// Kish's effective sample size. Equals `n_samples` for uniform/absent weights.
    pub n_eff: f64,
    /// Resolved sparse keep-count. `None` for dense fits — mirrors the
    /// `weights: Option<Col<f64>>` precedent.
    pub keep: Option<usize>,
    /// Coefficients of the covariates `Z`, one per column of `Z`, 0 at a
    /// dropped (aliased) column; shape `(p_z,)`. `None` when fitted without
    /// covariates.
    pub covariate_coef: Option<Col<f64>>,
    /// `rank([1, Z]) − 1`, the covariates kept. `None` when fitted without
    /// covariates.
    pub n_covariates: Option<usize>,
}

/// Validate weights and return `(normalized weights, n_eff)`; `(None, n)`
/// when `weights` is `None` and when every weight is equal. All-equal
/// weights are no weights (uniform-weight invariance), so every downstream
/// branch on `w_norm.is_some()` takes the unweighted path and the call is
/// bit-identical to one without weights.
/// That includes `n_eff`, which is exactly `n` (Kish's ratio of equal
/// non-unit weights can round an ulp below it).
///
/// # Errors
/// - `InvalidWeights { reason: "length_mismatch" }` if `weights.len() != n`
/// - `NonFiniteInput` for any NaN / infinity
/// - `InvalidWeights { reason: "negative" }` for any `w < 0`
/// - `InvalidWeights { reason: "all_zero" }` if `Σw == 0`
///
/// All-equal means every normalized entry is 1.0 within 1e-12; such weights
/// come back as `None` with `n_eff = n`, exactly like absent weights.
/// Top-level entries that take a component count call
/// `validate_weights_for_k` instead.
pub(crate) fn validate_and_normalize_weights(
    weights: Option<ColRef<'_, f64>>,
    n: usize,
) -> PlsKitResult<(Option<Col<f64>>, f64)> {
    let Some(w) = weights else {
        #[allow(clippy::cast_precision_loss)]
        return Ok((None, n as f64));
    };

    if w.nrows() != n {
        // Weights length is a weights problem, not a dimension mismatch between X and y.
        // Mirrors the same convention in preprocess.rs — change together.
        return Err(PlsKitError::InvalidWeights {
            reason: "length_mismatch",
        });
    }
    // mirrors preprocess.rs weight-validation loop — change together.
    for i in 0..n {
        if !w[i].is_finite() {
            return Err(PlsKitError::NonFiniteInput);
        }
        if w[i] < 0.0 {
            return Err(PlsKitError::InvalidWeights { reason: "negative" });
        }
    }
    let wn = crate::linalg::normalize_weights(w)
        .ok_or(PlsKitError::InvalidWeights { reason: "all_zero" })?;
    if weights_all_equal(wn.as_ref()) {
        #[allow(clippy::cast_precision_loss)]
        return Ok((None, n as f64));
    }
    let n_eff = crate::linalg::compute_n_eff(wn.as_ref());
    Ok((Some(wn), n_eff))
}

/// The all-equal predicate on normalized weights `w'` (mean 1): every
/// entry is 1.0 within 1e-12. Such weights are no weights, and their
/// `n_eff` is exactly `n`; `preprocess` reports the same `n_eff` for them.
pub(crate) fn weights_all_equal(wn: ColRef<'_, f64>) -> bool {
    let max_dev = (0..wn.nrows())
        .map(|i| (wn[i] - 1.0).abs())
        .fold(0.0_f64, f64::max);
    max_dev < 1e-12
}

/// `validate_and_normalize_weights` followed by the `n_eff ≥ k + 1` check:
/// the one call every top-level entry with weights and a component count
/// makes. Returns `(normalized weights, n_eff)`. Whether the weights are "in
/// play" for the error kind is read off the normalized vector, never off
/// the caller's `Option`, so all-equal weights fail exactly as absent ones.
///
/// # Errors
/// Those of `validate_and_normalize_weights`, then
/// `InvalidWeights { reason: "insufficient_effective_n" }` (non-uniform
/// weights) or `InvalidArgument` (uniform or absent weights) when
/// `n_eff < k + 1`.
pub(crate) fn validate_weights_for_k(
    weights: Option<ColRef<'_, f64>>,
    n: usize,
    k: usize,
) -> PlsKitResult<(Option<Col<f64>>, f64)> {
    let (w_norm, n_eff) = validate_and_normalize_weights(weights, n)?;
    check_n_eff_for_k(n_eff, k, w_norm.is_some())?;
    Ok((w_norm, n_eff))
}

/// Check that every entry of `x` is finite. Used at top-level public
/// entry points to guarantee the boundary contract; per-iteration inner
/// callers can rely on this having run upstream.
///
/// # Errors
/// `NonFiniteInput` on any NaN or infinity.
pub(crate) fn check_finite_mat(x: MatRef<'_, f64>) -> PlsKitResult<()> {
    let n = x.nrows();
    let d = x.ncols();
    // A row-major view (a Rust caller's row-major `MatRef`) is swept row by
    // row over contiguous slices; the answer does not depend on the order.
    if x.try_as_col_major().is_none() {
        if let Some(xr) = x.try_as_row_major() {
            for i in 0..n {
                // A non-short-circuiting fold, so the row sweep vectorizes.
                let row_ok = xr
                    .row(i)
                    .as_slice()
                    .iter()
                    .fold(true, |ok, v| ok & v.is_finite());
                if !row_ok {
                    return Err(PlsKitError::NonFiniteInput);
                }
            }
            return Ok(());
        }
    }
    // j-outer/i-inner matches faer's column-major storage (cache-friendly sweep).
    for j in 0..d {
        for i in 0..n {
            if !x[(i, j)].is_finite() {
                return Err(PlsKitError::NonFiniteInput);
            }
        }
    }
    Ok(())
}

/// Check that every entry of `y` is finite.
///
/// # Errors
/// `NonFiniteInput` on any NaN or infinity.
pub(crate) fn check_finite_col(y: ColRef<'_, f64>) -> PlsKitResult<()> {
    let n = y.nrows();
    for i in 0..n {
        if !y[i].is_finite() {
            return Err(PlsKitError::NonFiniteInput);
        }
    }
    Ok(())
}

/// Check that effective sample size supports the requested number of components.
///
/// When `weighted=true` (non-uniform observation weights), returns
/// `Err(InvalidWeights { reason: "insufficient_effective_n" })` when `n_eff < k + 1`.
/// When `weighted=false` (uniform or absent weights), returns
/// `Err(InvalidArgument)` — no weights are in play, so the failure is a plain
/// data-size problem, not a weights problem (callers branch on `code()`).
/// Reached only through `validate_weights_for_k`, which derives `weighted`
/// from the normalized vector; see `docs/internals/n-eff-check.md`.
///
/// # Errors
/// `InvalidWeights { reason: "insufficient_effective_n" }` (non-uniform
/// weights) or `InvalidArgument` (uniform or absent weights) when
/// `n_eff < k + 1`.
fn check_n_eff_for_k(n_eff: f64, k: usize, weighted: bool) -> PlsKitResult<()> {
    #[allow(clippy::cast_precision_loss)]
    if n_eff < (k as f64) + 1.0 {
        return Err(if weighted {
            PlsKitError::InvalidWeights {
                reason: "insufficient_effective_n",
            }
        } else {
            PlsKitError::InvalidArgument(format!(
                "insufficient n for k={k}: need n >= k + 1 (got n={n_eff})"
            ))
        });
    }
    Ok(())
}

/// Range-check a keep-count against the dimension it selects within,
/// naming both in the message. `validate_keep` is the PLS1 spelling
/// (`keep` within `n_features`); the sPLS3 sides pass their own names.
///
/// # Errors
/// `InvalidArgument` when `keep == 0` or `keep > dim`.
pub(crate) fn validate_keep_arg(
    keep: usize,
    dim: usize,
    arg: &str,
    dim_name: &str,
) -> PlsKitResult<()> {
    if keep == 0 {
        return Err(PlsKitError::InvalidArgument(format!(
            "{arg} must be >= 1 ({arg}=0 would produce an empty component)"
        )));
    }
    if keep > dim {
        return Err(PlsKitError::InvalidArgument(format!(
            "{arg}={keep} exceeds {dim_name}={dim}"
        )));
    }
    Ok(())
}

/// PLS1 spelling of [`validate_keep_arg`].
///
/// # Errors
/// `InvalidArgument` when `keep == 0` or `keep > n_features`.
pub(crate) fn validate_keep(keep: usize, n_features: usize) -> PlsKitResult<()> {
    validate_keep_arg(keep, n_features, "keep", "n_features")
}

/// The argument checks `pls1_fit` makes after the shapes and before the
/// weights, in its order: finite y, `k ≥ 1`, `k ≤ n_features`, `keep`. A replicate loop that
/// checked its prepared X once (per fold or per split) runs only these per
/// outcome column, and fails with the error `pls1_fit` would return.
///
/// # Errors
/// `NonFiniteInput`, `InvalidArgument` (for `k = 0` or a bad `keep`) or
/// `KExceedsMax`.
pub(crate) fn check_fit_y_and_k(
    n_features: usize,
    y: ColRef<'_, f64>,
    k: usize,
    keep: Option<usize>,
) -> PlsKitResult<()> {
    check_finite_col(y)?;
    if k == 0 {
        return Err(PlsKitError::InvalidArgument("k must be >= 1".into()));
    }
    if k > n_features {
        return Err(PlsKitError::KExceedsMax {
            k,
            k_max: n_features,
        });
    }
    if let Some(kp) = keep {
        validate_keep(kp, n_features)?;
    }
    Ok(())
}

/// Fit a PLS1 regression: the NIPALS PLS1 model, computed by Improved
/// Kernel PLS (`pls1_kernel`).
///
/// # Shapes
/// - `x`: `(n_samples, n_features)`
/// - `y`: `(n_samples,)`
/// - `weights`: optional per-observation weights; `None` is equivalent to all-ones.
/// - `covariates`: optional `Z`, `(n_samples, p_z)`. With `Z`, X is
///   standardized (or used as given under `pre_standardized`), then X and y
///   are replaced by their weighted least-squares residuals on `[1, Z]`
///   (`crate::covariates`), and the PLS1 fit runs on those residuals. The
///   scores, loadings, weights and `coef` describe the adjusted data;
///   `beta`, `intercept` and `covariate_coef` give
///   `ŷ = intercept + X·β + Z·covariate_coef` on the raw scale. A column of
///   `Z` that is constant or a linear combination of the intercept and
///   earlier columns is dropped.
/// - returns `Pls1Model { t_scores: (n_samples, k_used), p_loadings: (n_features, k_used),
///   w_star: (n_features, k_used), q_loadings: (k_used,), coef: (n_features,),
///   beta: (n_features,), ... }`
///
/// # Errors
/// - `PlsKitError::DimensionMismatch` when `y.nrows() != x.nrows()`
/// - `PlsKitError::InvalidWeights { reason: "length_mismatch" }` when `weights.len() != n`
/// - `PlsKitError::InvalidArgument` when `k == 0`
/// - `PlsKitError::KExceedsMax` when `k > n_features`
/// - `PlsKitError::NonFiniteInput` when X, y, or weights contains NaN/inf
/// - `PlsKitError::InvalidWeights` for negative, all-zero, or insufficient-`n_eff` weights
/// - `PlsKitError::ShapeMismatch` when `covariates` does not have `n_samples` rows
/// - `PlsKitError::NonFiniteInput` when `covariates` contains NaN/inf
/// - `PlsKitError::InsufficientResidualDf` when `n_samples − rank([1, Z]) < k + 1`
///
/// # Panics
/// Never (all internal indexing guarded by validated shapes).
#[allow(clippy::many_single_char_names, clippy::too_many_lines)]
pub fn pls1_fit(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    weights: Option<ColRef<'_, f64>>,
    covariates: Option<MatRef<'_, f64>>,
    opts: FitOpts,
) -> PlsKitResult<Pls1Model> {
    crate::fit::with_thread_limit(|| {
        pls1_fit_impl(
            x,
            y,
            k,
            weights,
            covariates,
            opts,
            InternalFitOpts::default(),
        )
    })
}

/// Body of [`pls1_fit`], on the caller's pool.
#[allow(clippy::many_single_char_names, clippy::too_many_lines)]
pub(crate) fn pls1_fit_impl(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k_requested: usize,
    weights: Option<ColRef<'_, f64>>,
    covariates: Option<MatRef<'_, f64>>,
    opts: FitOpts,
    inner: InternalFitOpts,
) -> PlsKitResult<Pls1Model> {
    let n_samples = x.nrows();
    let n_features = x.ncols();
    if y.nrows() != n_samples {
        return Err(PlsKitError::DimensionMismatch {
            x: (n_samples, n_features),
            y: y.nrows(),
        });
    }
    // X's finiteness is read from the standardization moments or the
    // sum-of-squares screen below. When a later check fails first, the exact
    // scan runs before its error is returned, so a non-finite X is still
    // reported ahead of y, k, keep and the weights, as the replicate loops
    // (which check X first) report it.
    // Weights: finite, non-negative, Σw > 0, normalized to mean 1.
    let checked = check_fit_y_and_k(n_features, y, k_requested, inner.keep).and_then(|()| {
        if inner.check_n_eff {
            validate_weights_for_k(weights, n_samples, k_requested)
        } else {
            validate_and_normalize_weights(weights, n_samples)
        }
    });
    let (w_norm, n_eff_val) = match checked {
        Ok(v) => v,
        Err(e) => {
            check_finite_mat(x)?;
            return Err(e);
        }
    };
    let wref: Option<ColRef<'_, f64>> = w_norm.as_ref().map(Col::as_ref);

    if let Some(z) = covariates {
        return pls1_fit_covariates(x, y, z, k_requested, weights, wref, n_eff_val, opts, inner);
    }

    // √w' row factor. Row-scaling is the Cholesky factor of diag(w'),
    // *not* preprocessing, so it applies even when pre_standardized=true.
    let sqw: Option<Col<f64>> = wref.map(crate::linalg::sqrt_col);

    // The arm the kernel's products take (`pls1_fit_prepared_fro` and
    // `pls1_fit_implicit` resolve the same one). The moments map their
    // column strips, and the pre-standardized sum of squares its pieces, in
    // parallel on it, with the same bits on either arm.
    let fit_par = resolve_par(opts.par, n_samples, n_features, k_requested);

    // Standardization moments, or none when pre_standardized, weighted when
    // weights is Some. The standardized X itself is written only when its
    // columns' means are too large for the implicit products (below).
    let mut fx: Option<crate::linalg::FitX> =
        (!opts.pre_standardized).then(|| crate::linalg::fit_x_moments(x, wref, fit_par));
    let (ys_owned, y_mean, y_scale) = if opts.pre_standardized {
        (None, 0.0, 1.0)
    } else {
        let (zs, ym, ysc) = crate::linalg::standardize1_weighted(y, wref);
        (Some(zs), ym, ysc)
    };
    let ys_view: ColRef<'_, f64> = match &ys_owned {
        Some(a) => a.as_ref(),
        None => y,
    };

    // A pre-standardized X is still to be row-scaled; the implicit kernel
    // applies √w' itself.
    let x_scaled_owned: Option<Mat<f64>> = match (&fx, &sqw) {
        (None, Some(s)) => Some(scale_rows(x, s.as_ref())),
        _ => None,
    };
    let y_scaled_owned: Option<Col<f64>> = sqw.as_ref().map(|s| scale_col(ys_view, s.as_ref()));

    // The pre-standardized path's kernel input.
    let x_for_nipals: MatRef<'_, f64> = match &x_scaled_owned {
        Some(a) => a.as_ref(),
        None => x,
    };
    let y_for_nipals: ColRef<'_, f64> = match &y_scaled_owned {
        Some(a) => a.as_ref(),
        None => ys_view,
    };

    // X's finiteness and ‖X‖_F (for the kernel's truncation floor) come from
    // the pass already made: a non-finite entry makes its column's mean
    // non-finite, and the pre-standardized sum of squares is non-finite for a
    // non-finite entry or an overflowing square. A non-finite verdict is
    // confirmed by the exact scan, so a finite X is never rejected.
    // A pre-standardized X whose largest entry `m` (with weights, of the
    // √w'-scaled rows) is outside `[2^-E, 2^E]`
    // (`linalg::PRE_STANDARDIZED_BAND_EXP`) is divided by a power of two `c`
    // near it, as the held-out tests divide it
    // (`linalg::pre_standardized_scale`), and the fit is mapped back to X's
    // scale below. The sum of squares `ss` spares the common case that scan:
    // it is on the rows the kernel reads, the rows `m` is read on, so
    // `m² ≤ ss ≤ nd·m²`, and inside `[nd·2^-(2E-2), 2^(2E-2)]` the entry `m`
    // is inside the band, weighted or not. The window sits two binary orders
    // inside the band's edges, so rounding in the sum cannot skip a rescale.
    let mut x_div: Option<(Mat<f64>, f64)> = None;
    let x_fro = if let Some(f) = &fx {
        if !f.mean.iter().chain(f.scale.iter()).all(|v| v.is_finite()) {
            check_finite_mat(x)?;
        }
        f.fro
    } else {
        let mut ss = crate::linalg::sum_of_squares(x_for_nipals, fit_par);
        let nd = (n_samples * n_features) as f64;
        let e2 = 2 * crate::linalg::PRE_STANDARDIZED_BAND_EXP - 2;
        if !(nd * 2f64.powi(-e2)..=2f64.powi(e2)).contains(&ss) {
            if !ss.is_finite() {
                check_finite_mat(x)?;
            }
            x_div = crate::linalg::pre_standardized_div(x, wref).map(|(xd, c)| match &sqw {
                Some(s) => (scale_rows(xd.as_ref(), s.as_ref()), c),
                None => (xd, c),
            });
            if let Some((xd, _)) = &x_div {
                ss = crate::linalg::sum_of_squares(xd.as_ref(), fit_par);
            }
        }
        if ss.is_finite() {
            ss.sqrt()
        } else {
            check_finite_mat(x)?;
            x_for_nipals.norm_l2()
        }
    };
    let (x_mean, x_scale) = match &fx {
        Some(f) => (f.mean.clone(), f.scale.clone()),
        None => (
            Col::<f64>::zeros(n_features),
            Col::<f64>::from_fn(n_features, |_| 1.0),
        ),
    };

    // Within `IMPLICIT_MAX_MEAN_RATIO` the kernel forms the standardized
    // products from the raw X. Past it, or when the implicit fit cannot decide
    // where to stop, X is standardized into a copy (its layout, rows scaled
    // by √w'), and the fit is the copy's to the bit.
    let implicit_fit = match &fx {
        Some(f) => {
            let ratio = f.max_mean_ratio();
            if ratio > IMPLICIT_MAX_MEAN_RATIO {
                None
            } else {
                pls1_fit_implicit(
                    x,
                    y_for_nipals,
                    crate::linalg::owned_col_slice(&f.mean),
                    crate::linalg::owned_col_slice(&f.scale),
                    sqw.as_ref().map(crate::linalg::owned_col_slice),
                    k_requested,
                    inner.keep,
                    opts.par,
                    x_fro,
                    ratio,
                    n_samples,
                )
            }
        }
        None => None,
    };
    // The moments of a finite X are finite, but `x − mean` can still
    // overflow (entries near `±f64::MAX` of both signs), and the fit then is
    // not finite. An implicit fit that is not finite is refitted on the
    // written copy, as `pls3::prepare` refits an implicit `X̃'Ỹ` that
    // overflows (change together); a fit on a written matrix that is not
    // finite scans that matrix, and a non-finite entry there is non-finite
    // input, the answer `signal_test::prepare_cv_fold` gives the same X. A
    // finite fit pays for no scan.
    let implicit_fit = implicit_fit.filter(|f| f.coef.is_all_finite());
    let mut fit = match (implicit_fit, fx.as_mut()) {
        (Some(fit), _) => fit,
        (None, Some(f)) => {
            f.materialize(x, sqw.as_ref().map(Col::as_ref));
            let fit = pls1_fit_prepared_fro(
                f.xs(),
                y_for_nipals,
                k_requested,
                inner.keep,
                opts.par,
                x_fro,
            );
            if !fit.coef.is_all_finite() {
                check_finite_mat(f.xs())?;
            }
            fit
        }
        (None, None) => {
            let xk = x_div.as_ref().map_or(x_for_nipals, |(xd, _)| xd.as_ref());
            let fit =
                pls1_fit_prepared_fro(xk, y_for_nipals, k_requested, inner.keep, opts.par, x_fro);
            if !fit.coef.is_all_finite() {
                check_finite_mat(xk)?;
            }
            fit
        }
    };
    // The fit of X/c has the scores T/c, the same weights and loadings,
    // and the y-loadings and coefficients times c.
    if let Some((_, c)) = x_div {
        for j in 0..fit.t_scores.ncols() {
            for i in 0..fit.t_scores.nrows() {
                fit.t_scores[(i, j)] *= c;
            }
            fit.q_loadings[j] /= c;
        }
        for j in 0..n_features {
            fit.coef[j] /= c;
        }
    }
    let k_used = fit.k_used;
    if opts.pre_standardized && inner.check_n_eff && k_used < k_requested {
        return Err(PlsKitError::InvalidInput(format!(
            "pls1_fit(pre_standardized=true) truncated to k_used={k_used} < requested k={k_requested}: \
             NIPALS stopped at component {kth} (‖X_a'y_a‖ < 1e-14 or t't < 1e-14, or \
             ‖X_a'y_a‖ at the rounding floor relative to ‖X‖_F·‖y‖). Either X or y is \
             exhausted (fewer than k informative directions: lower k; at k_used=0, y is \
             orthogonal to X up to rounding), or y is far below unit scale (X is rescaled \
             when far from it, y is used as given; see `FitOpts::pre_standardized`): \
             re-fit with `pre_standardized=false` to let plskit standardize.",
            kth = k_used + 1
        )));
    }
    let coef = fit.coef;

    // Back-project to raw scale: beta[j] = coef[j] * y_scale / x_scale[j]
    let beta = if opts.pre_standardized {
        coef.clone()
    } else {
        Col::<f64>::from_fn(n_features, |j| coef[j] * y_scale / x_scale[j])
    };
    let intercept = if opts.pre_standardized {
        0.0
    } else {
        // y_hat_raw = mean_y + sum_j beta_j (x_j - mean_x_j)
        let dot: f64 = (0..n_features).map(|j| beta[j] * x_mean[j]).sum();
        y_mean - dot
    };

    Ok(Pls1Model {
        t_scores: fit.t_scores,
        p_loadings: fit.p_loadings,
        w_star: fit.w_star,
        q_loadings: fit.q_loadings,
        coef,
        beta,
        intercept,
        k_used,
        pre_standardized: opts.pre_standardized,
        weights: w_norm,
        n_eff: n_eff_val,
        keep: inner.keep,
        covariate_coef: None,
        n_covariates: None,
    })
}

/// The covariate branch of [`pls1_fit_impl`], after the weights are
/// validated (`wref` is their normalized form). The fit on the adjusted
/// data `(X̃, ỹ)` runs with `pre_standardized` semantics. On the raw scale,
/// `β_j = coef_j · s_y / s_j` (`β = coef` under `pre_standardized`), and the
/// covariate part is one WLS regression: `d` = coefficients of
/// `y − (X − 1·μ_X')·β` on `[1, Z]`, which is `γ_y − Γ_X·β` (`γ_y`, `Γ_X` the
/// coefficients of y and of the centred X on `[1, Z]`), so
/// `ŷ = [1, Z]·γ_y + (X − 1·μ_X' − [1, Z]·Γ_X)·β` multiplies out to
/// `intercept + X·β + Z·covariate_coef` with `intercept = d₀ − μ_X'·β` and
/// `covariate_coef = d₁..` (`μ_X = 0` under `pre_standardized`).
#[allow(clippy::many_single_char_names, clippy::too_many_arguments)]
fn pls1_fit_covariates(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    z: MatRef<'_, f64>,
    k_requested: usize,
    weights: Option<ColRef<'_, f64>>,
    wref: Option<ColRef<'_, f64>>,
    n_eff: f64,
    opts: FitOpts,
    inner: InternalFitOpts,
) -> PlsKitResult<Pls1Model> {
    check_finite_mat(x)?;
    let mut adj = crate::covariates::adjust(x, y, z, wref, opts.pre_standardized)?;
    adj.basis.check_df(k_requested)?;
    // Weighted, the adjusted rows are scaled by √w' in place and fitted
    // unweighted: the weighted fit's kernel input without its n×p copy.
    let sqw = wref
        .map(crate::linalg::sqrt_col)
        .filter(|s| rows_scale_in_band(adj.x.as_ref(), s.as_ref()));
    if let Some(s) = &sqw {
        scale_rows_in_place(&mut adj.x, s.as_ref());
        for i in 0..adj.y.nrows() {
            adj.y[i] *= s[i];
        }
    }
    // The truncation guard of a pre-standardized fit applies only when the
    // caller passed `pre_standardized`; the n_eff check has run above.
    let mut m = pls1_fit_impl(
        adj.x.as_ref(),
        adj.y.as_ref(),
        k_requested,
        if sqw.is_some() { None } else { weights },
        None,
        FitOpts {
            pre_standardized: true,
            ..opts
        },
        InternalFitOpts {
            check_n_eff: inner.check_n_eff && opts.pre_standardized,
            ..inner
        },
    )?;
    if sqw.is_some() {
        m.weights = wref.map(|w| w.to_owned());
        m.n_eff = n_eff;
    }
    if !opts.pre_standardized {
        m.beta = Col::<f64>::from_fn(m.coef.nrows(), |j| m.coef[j] * adj.y_scale / adj.x_scale[j]);
    }
    // r = y − (X − 1·μ')·β, centred term by term so that large column
    // means do not cancel inside the product.
    let mut r = y.to_owned();
    for j in 0..x.ncols() {
        let (mu, bj) = (adj.x_mean[j], m.beta[j]);
        for i in 0..x.nrows() {
            r[i] -= (x[(i, j)] - mu) * bj;
        }
    }
    let d = adj.basis.coef(r.as_ref().as_mat());
    let full = adj.basis.full_coef(d.col(0));
    let mu_beta: f64 = (0..x.ncols()).map(|j| adj.x_mean[j] * m.beta[j]).sum();
    m.intercept = full[0] - mu_beta;
    m.covariate_coef = Some(Col::<f64>::from_fn(adj.basis.p_z(), |c| full[1 + c]));
    m.n_covariates = Some(adj.basis.n_covariates());
    m.pre_standardized = opts.pre_standardized;
    Ok(m)
}

/// Sparse PLS1 fit, head of the `spls1_*` family: the PLS1 (NIPALS) model
/// with hard keep-count selection on the weight vector per component (Chun &
/// Keleş 2010 lineage, keep-count formulation), so each latent direction
/// loads on exactly `keep` X variables. Everything downstream of the
/// selection step (rotation, scores, loadings, the update of `X'y`,
/// `coef = W(P'W)⁻¹Q`, raw-scale β, intercept) is the same code as
/// `pls1_fit`; `keep = n_features` reduces bit-exactly to the dense fit.
/// `keep` is one scalar for all `k` components. Selection on `w` does not
/// guarantee a nested selection path across components; tune `keep`, do not
/// interpret the path.
///
/// `covariates` is `pls1_fit`'s: selection acts on the adjusted X.
///
/// # Errors
/// Everything `pls1_fit` returns, plus `InvalidArgument` for `keep == 0`
/// or `keep > n_features`.
pub fn spls1_fit(
    x: MatRef<'_, f64>,
    y: ColRef<'_, f64>,
    k: usize,
    keep: usize,
    weights: Option<ColRef<'_, f64>>,
    covariates: Option<MatRef<'_, f64>>,
    opts: FitOpts,
) -> PlsKitResult<Pls1Model> {
    crate::fit::with_thread_limit(|| {
        pls1_fit_impl(
            x,
            y,
            k,
            weights,
            covariates,
            opts,
            InternalFitOpts {
                keep: Some(keep),
                ..InternalFitOpts::default()
            },
        )
    })
}

/// The kernel-and-coefficient tail of `pls1_fit`, on arrays that are
/// already standardized and already √w-scaled. `pls1_fit` runs it as
/// [`pls1_fit_prepared_fro`] when it does not standardize, or when it
/// standardizes X into a copy (past [`IMPLICIT_MAX_MEAN_RATIO`]); otherwise
/// it runs [`pls1_fit_implicit`] on the raw X, which returns this struct
/// too. Callers that prepare the arrays once for many replicates run
/// [`check_finite_mat`] on X and [`check_fit_y_and_k`] first when their
/// inputs can fail them.
pub(crate) struct PreparedFit {
    /// X-scores `T` of the matrix the kernel ran on; `(n, k_used)`.
    pub(crate) t_scores: Mat<f64>,
    /// X-loadings `P`; `(d, k_used)`.
    pub(crate) p_loadings: Mat<f64>,
    /// Raw weights `W`; `(d, k_used)`.
    pub(crate) w_star: Mat<f64>,
    /// y-loadings `Q`; `(k_used,)`.
    pub(crate) q_loadings: Col<f64>,
    /// `W (P'W)⁻¹ Q` in the prepared (standardized) space; `(d,)`.
    pub(crate) coef: Col<f64>,
    /// Components kept (≤ `k`).
    pub(crate) k_used: usize,
}

/// [`PreparedFit`] of `xs`, `ys`, with `‖xs‖_F` supplied as `x_fro`
/// (see [`pls1_kernel`]): a loop over one fixed block takes `xs.norm_l2()`
/// once and passes it to every call. `par` is resolved from `xs`'s shape and
/// `k` exactly as `pls1_fit` resolves it.
pub(crate) fn pls1_fit_prepared_fro(
    xs: MatRef<'_, f64>,
    ys: ColRef<'_, f64>,
    k: usize,
    keep: Option<usize>,
    par: ParChoice,
    x_fro: f64,
) -> PreparedFit {
    let par = resolve_par(par, xs.nrows(), xs.ncols(), k);
    let (t_mat, p_mat, w_mat, q_vec) = pls1_kernel(xs, ys, k, keep, par, x_fro);
    let k_used = w_mat.ncols();
    let coef = pls1_coef_at_k(&w_mat, &p_mat, &q_vec, k_used, par);
    PreparedFit {
        t_scores: t_mat,
        p_loadings: p_mat,
        w_star: w_mat,
        q_loadings: q_vec,
        coef,
        k_used,
    }
}

/// [`pls1_fit_prepared_fro`] on `Xs = diag(sqw)·(x − 1·mean')·diag(1/scale)`
/// through [`ImplicitXBackend`], without writing `Xs`. `x_fro` is `‖Xs‖_F`;
/// `mean_ratio` is `max_j |mean_j| / scale_j`; `n_rows` is the row count
/// the truncation floor takes: `x.nrows()` for a fit of all of X, a fold's
/// training rows (zero-weight rows included) for a cross-validation fold,
/// which passes all of X with `sqw` 0 on its validation rows. A row whose
/// `sqw` is 0 is a zero row of `Xs` (while `x·u` is finite there). `None`
/// when a component's stop decision falls in the backend's rounding band;
/// the caller then fits the standardized copy.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pls1_fit_implicit(
    x: MatRef<'_, f64>,
    ys: ColRef<'_, f64>,
    mean: &[f64],
    scale: &[f64],
    sqw: Option<&[f64]>,
    k: usize,
    keep: Option<usize>,
    par: ParChoice,
    x_fro: f64,
    mean_ratio: f64,
    n_rows: usize,
) -> Option<PreparedFit> {
    let par = resolve_par(par, x.nrows(), x.ncols(), k);
    let w_floor = w_rel_floor(n_rows, x.ncols(), x_fro, ys.norm_l2());
    let mut backend = ImplicitXBackend {
        x,
        ys,
        mean,
        scale,
        sqw,
        par,
        band: 2.0 * (1.0 + mean_ratio) * NIPALS_ABS_FLOOR.max(w_floor),
    };
    let s0 = backend.xs_t_mul(ys, 1.0);
    let LoopOutcome::Done { t, p, w, q } = pls1_component_loop(&mut backend, s0, w_floor, k, keep)
    else {
        return None;
    };
    let t = t.unwrap_or_else(|| Mat::<f64>::zeros(x.nrows(), 0));
    let k_used = w.ncols();
    let coef = pls1_coef_at_k(&w, &p, &q, k_used, par);
    Some(PreparedFit {
        t_scores: t,
        p_loadings: p,
        w_star: w,
        q_loadings: q,
        coef,
        k_used,
    })
}

/// The exact √w vector `pls1_fit` scales rows by when handed weights `w`:
/// `sqrt(normalize_weights(w)[i])`. `pls1_fit` renormalizes the weights it
/// is given, so a caller that hoists the scaling out of a replicate loop
/// uses this, not `√w`: the two can differ in the last bit.
///
/// # Panics
/// When `Σw == 0`, where `pls1_fit` returns `InvalidWeights`. Callers pass
/// weights that went through `validate_and_normalize_weights` or a per-fold
/// renormalization (`None` for a zero-sum slice), whose sum is positive.
pub(crate) fn fit_row_scale(w: ColRef<'_, f64>) -> Col<f64> {
    let wn = crate::linalg::normalize_weights(w).expect("fit_row_scale: positive weight sum");
    crate::linalg::sqrt_col(wn.as_ref())
}

/// `sqw[i] * x[(i, j)]`, the operand order of `pls1_fit`'s row scaling.
pub(crate) fn scale_rows(x: MatRef<'_, f64>, sqw: ColRef<'_, f64>) -> Mat<f64> {
    Mat::<f64>::from_fn(x.nrows(), x.ncols(), |i, j| sqw[i] * x[(i, j)])
}

/// [`scale_rows`] written over `x`, the same bits without the copy.
pub(crate) fn scale_rows_in_place(x: &mut Mat<f64>, sqw: ColRef<'_, f64>) {
    for j in 0..x.ncols() {
        for i in 0..x.nrows() {
            x[(i, j)] *= sqw[i];
        }
    }
}

/// Whether the unweighted pre-standardized fit of `scale_rows(x, sqw)` and
/// `scale_col(y, sqw)` takes the kernel input of the weighted one of
/// `(x, y)` with `√w' = sqw`: when the scaled rows have their largest entry
/// inside `[2^-E, 2^E]` (`linalg::PRE_STANDARDIZED_BAND_EXP`), neither fit
/// divides by a power of two (`linalg::pre_standardized_scale`; both read
/// it off the scaled rows). `|sqw·x|` is `sqw·|x|` to the bit.
pub(crate) fn rows_scale_in_band(x: MatRef<'_, f64>, sqw: ColRef<'_, f64>) -> bool {
    let mut ms = 0.0_f64;
    for j in 0..x.ncols() {
        for i in 0..x.nrows() {
            ms = ms.max(sqw[i] * x[(i, j)].abs());
        }
    }
    let e = crate::linalg::PRE_STANDARDIZED_BAND_EXP;
    (2f64.powi(-e)..=2f64.powi(e)).contains(&ms)
}

/// `sqw[i] * y[i]`.
pub(crate) fn scale_col(y: ColRef<'_, f64>, sqw: ColRef<'_, f64>) -> Col<f64> {
    Col::<f64>::from_fn(y.nrows(), |i| sqw[i] * y[i])
}

/// Zero all but the `keep` largest-|w| coordinates (hard thresholding at
/// the keep-th order statistic of |w| — NOT soft thresholding; survivors
/// keep their magnitudes). Exact ties break deterministically: order by
/// (|w| desc, index asc), so the lowest column index wins.
///
/// The index tiebreak is what makes an *unstable* partial select safe
/// here, and is why this is `select_nth_unstable_by` rather than a full
/// sort. `total_cmp` is a total order on bit patterns (and `.abs()`
/// collapses `+0.0` / `-0.0` onto one key), while `a.cmp(&b)` returns
/// `Equal` only for `a == b`, which never happens between two distinct
/// elements of a permutation of `0..d`. So no two elements ever compare
/// `Equal`, the sorted order is unique, and the selected *set* is fixed by
/// the comparator alone, independent of which algorithm computes it.
/// Drop the tiebreak and that stops being true; the comparator and the
/// unstable select stand or fall together.
///
/// Shared with `pls3::spls3_component`, which applies it to `u` and to
/// `v` in turn. Do not copy it: the tie rule is the reproducibility
/// contract and must have exactly one implementation.
pub(crate) fn hard_select_keep(w: &mut Col<f64>, keep: usize) {
    let d = w.nrows();
    // `keep >= d` selects everything, so there is no complement to zero.
    // Guarded rather than left to `select_nth_unstable_by`, which panics
    // on `keep == d`: this function is total and stays total.
    if keep >= d {
        return;
    }
    let mut idx: Vec<usize> = (0..d).collect();
    idx.select_nth_unstable_by(keep, |&a, &b| {
        w[b].abs().total_cmp(&w[a].abs()).then_with(|| a.cmp(&b))
    });
    for &j in &idx[keep..] {
        w[j] = 0.0;
    }
}

/// Keep the `keep` largest-magnitude entries of `v` ([`hard_select_keep`]),
/// then scale `v` to unit norm. Returns the norm it was divided by, or
/// `None`, leaving `v` selected but unscaled, when that norm is below
/// `floor`.
///
/// The one select-and-normalize step of every hard-threshold iteration:
/// the `w` step of `pls1_component_loop`, the `u` and `v` steps of
/// `pls3::spls3_component`, and the Gram mirror of the `v` step in
/// `dual_route::pls3_split_zbars_columns`. Sharing it is what keeps those
/// in the same float sequence: select, then norm, then one reciprocal, then
/// multiply. `v *= 1/nv` and `v /= nv` round differently, and on the Gram
/// route a one-ulp split can move the stopping sweep.
///
/// `keep >= v.nrows()` is a literal skip of the selection, not a no-op
/// call, so a dense endpoint's float sequence is provably untouched.
pub(crate) fn select_and_normalize(v: &mut Col<f64>, keep: usize, floor: f64) -> Option<f64> {
    let d = v.nrows();
    if keep < d {
        hard_select_keep(v, keep);
    }
    // faer's `norm_l2` is slower at these lengths; scaling by a power of two
    // keeps its range, so entries near 1e±300 still give a finite norm.
    let max_abs = (0..d).map(|j| v[j].abs()).fold(0.0_f64, f64::max);
    let (s, inv_s) = crate::linalg::pow2_scale(max_abs);
    let nv = s
        * (0..d)
            .map(|j| (v[j] * inv_s) * (v[j] * inv_s))
            .sum::<f64>()
            .sqrt();
    if nv < floor {
        return None;
    }
    let inv_nv = 1.0 / nv;
    for j in 0..d {
        v[j] *= inv_nv;
    }
    Some(nv)
}

/// Absolute floor of the PLS1 kernel (`pls1_component_loop`): a component
/// stops when `w_norm` or `t't` falls below it. The gates that mirror those
/// exits name this constant rather than restate the literal.
pub(crate) const NIPALS_ABS_FLOOR: f64 = 1e-14;

/// Relative floor on `w_norm = ‖X_a'y_a‖` for every PLS1 component:
/// `max(n, d)·ε·‖X‖_F·‖y‖`, where `X` (`n × d`) and `y` are the kernel's
/// inputs (`Xs`, `ys` in `pls1_kernel`), undeflated.
///
/// The kernel never forms `X_a'y_a` from a deflated matrix: it carries
/// `s_a = X_a'y_a` as a d-vector, `s_{a+1} = s_a − (q_a·t_a't_a)·p_a` (see
/// `pls1_kernel`). The `1e-14` absolute floor does not catch `y` running
/// out. Once the part of `y` that `X` can explain has been fitted, the exact
/// `s_a` is zero and the computed one is rounding noise, which sits at a
/// small multiple of `ε·‖X‖_F·‖y‖`: every step that feeds `s_a` (the initial
/// `X'y` sum, the scores and loadings read from the undeflated `X`, the
/// updates of `s`) carries absolute errors proportional to the entries of
/// the undeflated inputs, not the deflated ones. Standardizing with ddof 0
/// makes `‖X‖_F·‖y‖ = n·√d`, so that noise clears `1e-14` as soon as `n` is
/// in the hundreds. Normalizing that noise and continuing on it loses
/// orthogonality, and `w_norm` can climb back up over the following
/// components, so the floor must fire at the first noise component rather
/// than wait for a monotone decline. Measured at `n = 2000`, `d = 40` (rank
/// 39) with the explicit-deflation oracle kernel (`fit::kernel_tests`,
/// `nipals_pls1_reference`):
/// `w_norm` fell from 339 to about `1.4e-13` by the 18th component, regrew
/// to 40, and all 40 components were kept. With the current kernel and the
/// floor disabled, `w_norm` falls to about `4.6e-13` by component 19,
/// reaches at most `7.207e2` after that, and 40 of the 40 components are
/// kept.
///
/// The first component is no exception: when `y` is orthogonal to the
/// columns of `X` (for instance an outcome residualized on a set of
/// covariates that spans `X`), `X'y` is itself rounding noise, and without
/// the floor the fit keeps one component whose weight vector `w` is a
/// normalized noise vector pointing nowhere in particular. With it such a fit
/// returns `k_used = 0`, the same zero model a constant `y` gives.
///
/// The reference scale is `‖X‖_F·‖y‖` rather than the first `w_norm`
/// (`sigma_rel_floor` in `pls3.rs`, the PLS3 counterpart, likewise uses the
/// block norms `‖X̃‖_F·‖Ỹ‖_F` rather than `σ₁`). The
/// first `w_norm` is only a lower bound on the rounding scale and can be
/// arbitrarily far below it when `y` is nearly orthogonal to `X`; a floor
/// on `max(n, d)·ε·‖X'y‖` kept noise components on such designs. The
/// current residual `‖y_a‖` is no reference either: when `y` lies in the
/// span of `X`, `y_a` is itself rounding noise, and `‖X_a'y_a‖` relative to
/// `‖X_a‖·‖y_a‖` is then of order one.
///
/// `max(n, d)` is the factor `sigma_rel_floor` uses: `n·ε` bounds the
/// rounding of a length-`n` dot product relative to the product of the
/// norms, and at most `d` updates of `s` feed into any component. On a
/// sweep of rank-deficient, `y`-exhausted, `y`-orthogonal, weak-signal,
/// `p ≫ n`, sparse, weighted and tiny designs (`n` from 5 to 2e5;
/// `fit::kernel_tests::floor_calibration_sweep`, run on the current
/// kernel), the first noise component, located from each design's
/// construction (`y` in the span of `m` singular directions, `y`
/// orthogonal to `X`, or `X` deflated to zero at its rank), sat at most at
/// `0.020×` this floor (`first_noise_max`), while every
/// component contributing more than `1e-9` of `‖y‖` to the fit sat at least
/// `38.807×` above it. The real components it drops, which have
/// decayed into rounding, contributed at most about `4.4e-12` of `‖y‖`
/// to the fitted values of the unfloored fit. For the first component in
/// particular (880 designs over the same families, `n` from 5 to 1e5,
/// weighted and not; measured on the
/// explicit-deflation oracle kernel, whose first component is the same float
/// sequence as the current kernel's, so the numbers carry over), a `y`
/// orthogonal to the standardized `X` up to rounding sat at most at `0.33×`
/// the floor, a true effect of `1e-9` of `‖y‖` added to it at least `14×`
/// above it (the margin shrinks as `1/n`), and an unstructured random `y`
/// at least `2.7e7×` above it.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn w_rel_floor(n: usize, d: usize, x_fro: f64, y_norm: f64) -> f64 {
    (n.max(d) as f64) * f64::EPSILON * x_fro * y_norm
}

/// What a [`ComponentBackend`] returns for one rotation vector `r`.
pub(crate) struct Scored {
    /// `t't` on the X backend, `r'Cr` on the Gram backend.
    pub(crate) tt: f64,
    /// The scores `t = Xs r`; `None` on a backend that never forms them.
    pub(crate) t: Option<Col<f64>>,
}

/// Backend control after a gate hook of [`ComponentBackend`].
pub(crate) enum Gate {
    /// The backend decides this component the way the X backend would.
    Continue,
    /// It cannot; the caller recomputes the whole fit on the X backend.
    // Constructed only by a Gram backend; the X backend never returns it.
    Unresolved,
}

/// One way to form scores, loadings and `q` from a rotation vector `r`.
///
/// Every rule that decides numbers (the weight step, `keep` selection, the
/// floors, truncation, the rotation, the update of `s`) lives in
/// [`pls1_component_loop`], once, and not in a backend.
pub(crate) trait ComponentBackend {
    /// Called at every `a` from 1 up to `min(k, k_used + 1)` (1-based),
    /// before selection, on the current `s`, including the component whose
    /// selection stops the loop. The X backend always continues; the Gram
    /// backend checks `‖s‖` bands and the keep-boundary gap, both of which
    /// it only evaluates starting at `a = 2`.
    fn gate_s(&mut self, a: usize, s: &Col<f64>, keep: Option<usize>) -> Gate;
    /// `t` (X backend) and `tt = ‖t‖²` or `r'Cr`.
    fn score(&mut self, r: &Col<f64>) -> Scored;
    /// After `tt` is known, before the `tt` floor test.
    fn gate_tt(&mut self, a: usize, r: &Col<f64>, tt: f64) -> Gate;
    /// `p = Xs't·inv_tt` (X) or `C r·inv_tt` (Gram).
    fn loading(&mut self, r: &Col<f64>, t: Option<&Col<f64>>, inv_tt: f64) -> Col<f64>;
    /// `q`: X backend `(Σ ys[i]·t[i])·inv_tt`; Gram `(s'w)·inv_tt`.
    fn q(&mut self, s: &Col<f64>, w: &Col<f64>, t: Option<&Col<f64>>, inv_tt: f64) -> f64;
}

/// Result of [`pls1_component_loop`].
#[allow(clippy::large_enum_variant)] // one per fit, moved, never copied
pub(crate) enum LoopOutcome {
    /// `k_used = w.ncols()` components. `t` is `None` when the backend forms
    /// no scores, or when no component was kept.
    Done {
        t: Option<Mat<f64>>,
        p: Mat<f64>,
        w: Mat<f64>,
        q: Col<f64>,
    },
    /// A gate returned [`Gate::Unresolved`].
    Unresolved,
}

/// `r_a = w_a − R_{<a}(P_{<a}'w_a)`, the rotation that makes `Xs r_a` the
/// deflated score `X_a w_a` (see [`pls1_kernel`]). `a` is the 0-based index
/// of the component being built, so `R_{<a}`, `P_{<a}` are the first `a`
/// columns. Two small GEMVs, sequential, in ascending order: first
/// `c_j = Σ_i p_{ij}·w_i`, then `r_i = w_i − Σ_j r_{ij}·c_j`. At `a = 0` it
/// is a literal copy of `w`, which keeps every K = 1 fit the float sequence
/// of the explicit-deflation kernel.
#[allow(clippy::many_single_char_names)]
fn rotation(w: &Col<f64>, r_mat: &Mat<f64>, p_mat: &Mat<f64>, a: usize) -> Col<f64> {
    if a == 0 {
        return w.clone();
    }
    let d = w.nrows();
    let c: Vec<f64> = (0..a)
        .map(|j| (0..d).map(|i| p_mat[(i, j)] * w[i]).sum::<f64>())
        .collect();
    Col::<f64>::from_fn(d, |i| {
        w[i] - (0..a).map(|j| r_mat[(i, j)] * c[j]).sum::<f64>()
    })
}

/// One fit's running state through the component steps: the deflated `s`,
/// the recorded `R`, `P`, `W` and `q`, and the count of components kept.
/// The step functions hold every rule that decides numbers (selection, the
/// floors, the rotation, the update of `s`, the truncation); a driver only
/// supplies the scores, loadings and `q`.
pub(crate) struct LaneState {
    s: Col<f64>,
    r_mat: Mat<f64>,
    p_mat: Mat<f64>,
    w_mat: Mat<f64>,
    q_vec: Col<f64>,
    k_used: usize,
    floor: f64,
    k: usize,
}

/// The stop on `tt < NIPALS_ABS_FLOOR`.
pub(crate) fn tt_stops(tt: f64) -> bool {
    tt < NIPALS_ABS_FLOOR
}

impl LaneState {
    /// Starts from `s0 = Xs'ys`, for up to `k` components.
    pub(crate) fn new(s0: Col<f64>, w_floor: f64, k: usize) -> Self {
        let d = s0.nrows();
        Self {
            s: s0,
            r_mat: Mat::<f64>::zeros(d, k),
            p_mat: Mat::<f64>::zeros(d, k),
            w_mat: Mat::<f64>::zeros(d, k),
            q_vec: Col::<f64>::zeros(k),
            k_used: 0,
            // `w_norm < max(a, b)` is `w_norm < a || w_norm < b`, NaN included
            // (`f64::max` ignores a NaN operand, and a NaN `w_norm` fails either test).
            floor: NIPALS_ABS_FLOOR.max(w_floor),
            k,
        }
    }

    /// The current `s = X_a'y_a`.
    pub(crate) fn s(&self) -> &Col<f64> {
        &self.s
    }

    /// `w = s` selected and normalized by [`select_and_normalize`], and the
    /// rotation `r` of component `k_used`; `None` stops the lane. Sparse
    /// keep-count selection sits between `s` and the norm guard, so an
    /// all-zero surviving set truncates via the absolute floor. `keep =
    /// None` and `keep = d` are a literal skip of the selection (dense
    /// bit-parity at `keep = n_features`).
    pub(crate) fn weight(&self, keep: Option<usize>) -> Option<(Col<f64>, Col<f64>)> {
        let mut w = self.s.clone();
        select_and_normalize(&mut w, keep.unwrap_or(self.s.nrows()), self.floor)?;
        let r = rotation(&w, &self.r_mat, &self.p_mat, self.k_used);
        Some((w, r))
    }

    /// Records component `k_used` and deflates: `s ← s − (q·tt)·p` in
    /// ascending `j`.
    #[allow(clippy::many_single_char_names, clippy::needless_pass_by_value)] // owned by the driver, copied into the record
    pub(crate) fn accept(&mut self, w: Col<f64>, r: Col<f64>, p: Col<f64>, q: f64, tt: f64) {
        let a = self.k_used;
        let qtt = q * tt;
        for j in 0..self.s.nrows() {
            self.s[j] -= qtt * p[j];
        }
        self.r_mat.col_mut(a).copy_from(&r);
        self.p_mat.col_mut(a).copy_from(&p);
        self.w_mat.col_mut(a).copy_from(&w);
        self.q_vec[a] = q;
        self.k_used += 1;
    }

    /// `(p, w, q)` truncated to the components kept.
    pub(crate) fn finish(self) -> (Mat<f64>, Mat<f64>, Col<f64>) {
        let k_used = self.k_used;
        if k_used == self.k {
            (self.p_mat, self.w_mat, self.q_vec)
        } else {
            (
                self.p_mat.subcols(0, k_used).to_owned(),
                self.w_mat.subcols(0, k_used).to_owned(),
                Col::<f64>::from_fn(k_used, |i| self.q_vec[i]),
            )
        }
    }
}

/// The shared PLS1 component loop.
///
/// Starts from `s0 = Xs'ys` and carries `s = X_a'y_a` as a d-vector in a
/// [`LaneState`]. Per component `a`: the backend's `gate_s`;
/// [`LaneState::weight`] (`w = s`, selected and normalized against
/// `max(NIPALS_ABS_FLOOR, w_floor)`, and the rotation `r = w − R(P'w)`; a
/// `None` stops the loop); the backend's `score` and `gate_tt`; the stop on
/// [`tt_stops`]; the backend's `loading` and `q` with the same
/// `inv_tt = 1/tt`; then [`LaneState::accept`]. Outputs are truncated to the
/// components kept, as the explicit-deflation kernel truncated them.
///
/// Call order is a guarantee, not an implementation detail: `gate_s` runs
/// at every `a` up to `min(k, k_used + 1)`, `a = 1` included, including the
/// component whose selection stops the loop; a `tt` stop is preceded by
/// `gate_tt` at that same `a`; after either gate returns `Unresolved`, no
/// backend method is called for the rest of the fit.
///
/// `pls1_fit_bytes` counts what this allocates (change together).
#[allow(clippy::many_single_char_names, clippy::similar_names)]
pub(crate) fn pls1_component_loop<B: ComponentBackend>(
    backend: &mut B,
    s0: Col<f64>,
    w_floor: f64,
    k: usize,
    keep: Option<usize>,
) -> LoopOutcome {
    let mut lane = LaneState::new(s0, w_floor, k);
    let mut t_cols: Vec<Col<f64>> = Vec::with_capacity(k);

    for a in 0..k {
        if matches!(backend.gate_s(a + 1, lane.s(), keep), Gate::Unresolved) {
            return LoopOutcome::Unresolved;
        }
        let Some((w, r)) = lane.weight(keep) else {
            break;
        };
        let scored = backend.score(&r);
        let tt = scored.tt;
        if matches!(backend.gate_tt(a + 1, &r, tt), Gate::Unresolved) {
            return LoopOutcome::Unresolved;
        }
        if tt_stops(tt) {
            break;
        }
        let inv_tt = 1.0 / tt;
        let p = backend.loading(&r, scored.t.as_ref(), inv_tt);
        let q = backend.q(lane.s(), &w, scored.t.as_ref(), inv_tt);
        lane.accept(w, r, p, q, tt);
        if let Some(t) = scored.t {
            t_cols.push(t);
        }
    }

    let k_actual = lane.k_used;
    let t = if t_cols.is_empty() {
        None
    } else {
        let n = t_cols[0].nrows();
        Some(Mat::<f64>::from_fn(n, k_actual, |i, j| t_cols[j][i]))
    };
    let (p, w, q) = lane.finish();
    LoopOutcome::Done { t, p, w, q }
}

/// Bytes of the large buffers one PLS1 fit on `p` columns to `k`
/// components holds at once in [`pls1_component_loop`]: `r_mat`, `p_mat`
/// and `w_mat` (`p × k` each), the copies of `p_mat` and `w_mat` it
/// returns when it stops short of `k`, and one component's p-vectors `s`,
/// `w`, `r` and the loading. Mirrors `pls1_component_loop` (change
/// together). Mirrored by `pls1_fit_bytes` in `tests/common/budget_cases.rs`
/// (change together). The scores (`n × k`) and the `k`-vectors are not counted.
pub(crate) fn pls1_fit_bytes(p: usize, k: usize) -> usize {
    use crate::resample::copy_bytes;
    copy_bytes(p, k)
        .saturating_mul(5)
        .saturating_add(copy_bytes(p, 1).saturating_mul(4))
}

/// `alpha·Xs'·v`, `(d,)`. A row-major `Xs` at least
/// `linalg::ROW_BLOCK_MIN_COLS` wide (a Rust caller's row-major view, or the
/// row-major copy `FitX::materialize` writes) reads its rows in blocks
/// (`linalg::row_major_t_mul`), as `ImplicitXBackend::xs_t_mul` does, and
/// each entry is then multiplied by `alpha`; any other `Xs` goes to faer
/// (GEMV with `alpha`, as in the explicit-deflation kernel).
fn xs_t_mul(xs: MatRef<'_, f64>, v: ColRef<'_, f64>, alpha: f64, par: Par) -> Col<f64> {
    if let Some(mut g) = crate::linalg::row_major_t_mul(xs, v, par) {
        for j in 0..g.nrows() {
            g[j] *= alpha;
        }
        return g;
    }
    let mut g = Col::<f64>::zeros(xs.ncols());
    matmul(
        g.as_mut().as_mat_mut(),
        Accum::Replace,
        xs.transpose(),
        v.as_mat(),
        alpha,
        par,
    );
    g
}

/// Algorithm 1 of Dayal and MacGregor (1997): scores and loadings from the
/// undeflated `Xs`, two reads per component, no writes, no copy.
#[allow(clippy::doc_markdown)] // "MacGregor"
struct XBackend<'a> {
    xs: MatRef<'a, f64>,
    ys: ColRef<'a, f64>,
    par: Par,
}

impl ComponentBackend for XBackend<'_> {
    fn gate_s(&mut self, _a: usize, _s: &Col<f64>, _keep: Option<usize>) -> Gate {
        Gate::Continue
    }

    fn score(&mut self, r: &Col<f64>) -> Scored {
        // t = Xs r  (GEMV)
        let mut t = Col::<f64>::zeros(self.xs.nrows());
        matmul(
            t.as_mut().as_mat_mut(),
            Accum::Replace,
            self.xs,
            r.as_ref().as_mat(),
            1.0,
            self.par,
        );
        let tt = t.squared_norm_l2();
        Scored { tt, t: Some(t) }
    }

    fn gate_tt(&mut self, _a: usize, _r: &Col<f64>, _tt: f64) -> Gate {
        Gate::Continue
    }

    fn loading(&mut self, _r: &Col<f64>, t: Option<&Col<f64>>, inv_tt: f64) -> Col<f64> {
        // p = Xs' t / (t't)
        let t = t.expect("the X backend always forms t");
        xs_t_mul(self.xs, t.as_ref(), inv_tt, self.par)
    }

    fn q(&mut self, _s: &Col<f64>, _w: &Col<f64>, t: Option<&Col<f64>>, inv_tt: f64) -> f64 {
        // The explicit-deflation kernel's expression, on the undeflated ys:
        // a scalar sum in ascending i, times the same 1/tt.
        // `ImplicitXBackend::q` repeats it; change the two together.
        let t = t.expect("the X backend always forms t");
        let n = self.ys.nrows();
        (0..n).map(|i| self.ys[i] * t[i]).sum::<f64>() * inv_tt
    }
}

/// Largest `max_j |mean_j| / scale_j` for which `pls1_fit` forms the
/// standardized products from the raw X ([`ImplicitXBackend`]); above it the
/// standardized X is written into a copy. The implicit products lose about
/// `log10(1 + |mean_j| / scale_j)` digits to cancellation: on Gaussian designs
/// with a common offset the coefficients moved by about `ratio·1e-16`, at most
/// `6·ratio·1e-16` at `k = 5`, so this bound keeps them within about `1e-12` of
/// the copy's. The PLS3 fits take the same bound for their own implicit
/// products (`pls3::Prepared`).
pub(crate) const IMPLICIT_MAX_MEAN_RATIO: f64 = 1e3;

/// `‖select(s)‖`: the norm of the `keep` largest-magnitude entries of `s`
/// ([`hard_select_keep`]); dense (`None`, or `keep` at least the length of
/// `s`), the norm of all of `s`. Both branches use faer's `norm_l2`, so they
/// cover the same range. It is the norm [`select_and_normalize`] tests
/// against the floor, to rounding, since that function sums in its own order.
/// The gate needs no exact bits: it compares against a band of at least twice
/// the floor ([`pls1_fit_implicit`]'s `band`).
fn selected_norm(s: &Col<f64>, keep: Option<usize>) -> f64 {
    match keep {
        Some(kp) if kp < s.nrows() => {
            let mut v = s.clone();
            hard_select_keep(&mut v, kp);
            v.norm_l2()
        }
        _ => s.norm_l2(),
    }
}

/// The X backend on `Xs = diag(sqw)·(X − 1·mean')·diag(1/scale)` formed from
/// the raw `X` on the fly, so the standardized matrix is never written: the
/// kernel only multiplies by `Xs` and `Xs'`, and each product is the raw one
/// plus a rank-one correction,
///
/// - `Xs·v  = sqw ⊙ (X·u − 1·(mean'u))`, with `u = v / scale`;
/// - `Xs'·t = (X'·t̃ − mean·Σt̃) / scale`, with `t̃ = sqw ⊙ t`.
///
/// Same float model as [`XBackend`] on a materialized `Xs` up to rounding, but
/// the correction subtracts two terms of size `|mean_j|·|u_j|`, so column `j`
/// loses about `log10(1 + |mean_j| / scale_j)` digits to cancellation that
/// the materialized `x − mean` does not.
///
/// That cancellation also lifts the rounding noise in `s`, which the
/// truncation floor ([`w_rel_floor`]) is sized for on the materialized `Xs`:
/// with `y` orthogonal to `X`, the implicit first `‖s‖` reached `27×` the floor
/// (`n = 8`, ratio 900), where the materialized one stays below it; without a
/// gate the fit keeps a noise component. So `gate_s` hands every stop decision
/// on the `w` floor to the caller: a selected `‖s‖` below `band =
/// 2·(1 + max_j |mean_j| / scale_j)·floor` is `Unresolved`. On orthogonal-`y`
/// sweeps (`n` from 8 to 1000, ratio up to 900) the implicit noise sat at most
/// at `0.03·(1 + ratio)·floor` and the materialized at `0.58·floor`, so the
/// factor 2 also covers the copy's own rounding at a ratio near 0.
struct ImplicitXBackend<'a> {
    x: MatRef<'a, f64>,
    ys: ColRef<'a, f64>,
    mean: &'a [f64],
    scale: &'a [f64],
    sqw: Option<&'a [f64]>,
    par: Par,
    band: f64,
}

impl ImplicitXBackend<'_> {
    /// `Xs·v`, `(n,)`.
    fn xs_mul(&self, v: &Col<f64>) -> Col<f64> {
        let d = self.x.ncols();
        let u = Col::<f64>::from_fn(d, |j| v[j] / self.scale[j]);
        let c: f64 = (0..d).map(|j| self.mean[j] * u[j]).sum();
        let mut xu = Col::<f64>::zeros(self.x.nrows());
        matmul(
            xu.as_mut().as_mat_mut(),
            Accum::Replace,
            self.x,
            u.as_ref().as_mat(),
            1.0,
            self.par,
        );
        match self.sqw {
            None => {
                for i in 0..xu.nrows() {
                    xu[i] -= c;
                }
            }
            Some(sqw) => {
                for i in 0..xu.nrows() {
                    xu[i] = (xu[i] - c) * sqw[i];
                }
            }
        }
        xu
    }

    /// `alpha·Xs'·t`, `(d,)`.
    fn xs_t_mul(&self, t: ColRef<'_, f64>, alpha: f64) -> Col<f64> {
        let n = self.x.nrows();
        let tw_owned: Option<Col<f64>> = self
            .sqw
            .map(|sqw| Col::<f64>::from_fn(n, |i| t[i] * sqw[i]));
        let tw = tw_owned.as_ref().map_or(t, Col::as_ref);
        let sum_tw: f64 = (0..n).map(|i| tw[i]).sum();
        // A row-major X at least `linalg::ROW_BLOCK_MIN_COLS` wide (a Rust
        // caller's row-major view) reads its rows in blocks; any other X
        // goes to faer.
        let mut g = crate::linalg::row_major_t_mul(self.x, tw, self.par).unwrap_or_else(|| {
            let mut g = Col::<f64>::zeros(self.x.ncols());
            matmul(
                g.as_mut().as_mat_mut(),
                Accum::Replace,
                self.x.transpose(),
                tw.as_mat(),
                1.0,
                self.par,
            );
            g
        });
        for j in 0..g.nrows() {
            g[j] = (g[j] - self.mean[j] * sum_tw) / self.scale[j] * alpha;
        }
        g
    }
}

impl ComponentBackend for ImplicitXBackend<'_> {
    fn gate_s(&mut self, _a: usize, s: &Col<f64>, keep: Option<usize>) -> Gate {
        if selected_norm(s, keep) < self.band {
            Gate::Unresolved
        } else {
            Gate::Continue
        }
    }

    fn score(&mut self, r: &Col<f64>) -> Scored {
        let t = self.xs_mul(r);
        let tt = t.squared_norm_l2();
        Scored { tt, t: Some(t) }
    }

    fn gate_tt(&mut self, _a: usize, _r: &Col<f64>, _tt: f64) -> Gate {
        Gate::Continue
    }

    fn loading(&mut self, _r: &Col<f64>, t: Option<&Col<f64>>, inv_tt: f64) -> Col<f64> {
        let t = t.expect("the implicit X backend always forms t");
        self.xs_t_mul(t.as_ref(), inv_tt)
    }

    fn q(&mut self, _s: &Col<f64>, _w: &Col<f64>, t: Option<&Col<f64>>, inv_tt: f64) -> f64 {
        // As in `XBackend::q`; change the two together.
        let t = t.expect("the implicit X backend always forms t");
        let n = self.ys.nrows();
        (0..n).map(|i| self.ys[i] * t[i]).sum::<f64>() * inv_tt
    }
}

/// The PLS1 kernel: the NIPALS PLS1 model, computed by Improved Kernel PLS
/// (Dayal and MacGregor 1997, Algorithm 1) through [`pls1_component_loop`]
/// with the X backend. Same inputs and outputs (`T`, `P`, `W`, `Q`) as the
/// explicit-deflation oracle kernel in `fit::kernel_tests`.
///
/// # Derivation
/// NIPALS deflates `X_{a+1} = X_a − t_a p_a'` and `y_{a+1} = y_a − q_a t_a`
/// and takes `w_a ∝ X_a'y_a`, `t_a = X_a w_a`, `p_a = X_a't_a / t_a't_a`,
/// `q_a = y_a't_a / t_a't_a`. This kernel forms none of `X_a`, `y_a` and
/// keeps `s_a = X_a'y_a`. In exact arithmetic, by induction on `a`:
///
/// 1. `s_{a+1} = s_a − p_a·(q_a·t_a't_a)`. `X_{a+1}'t_a = 0` gives
///    `X_{a+1}'y_{a+1} = X_{a+1}'y_a = X_a'y_a − p_a·(t_a'y_a)`, and
///    `t_a'y_a = q_a·t_a't_a`.
/// 2. `t_a = Xs r_a` with `r_a = w_a − R_{<a}(P_{<a}'w_a)`. Writing
///    `X_a = Xs·M_a`, the deflation gives `M_{a+1} = M_a − r_a p_a'`, so
///    `M_a = I − Σ_{j<a} r_j p_j'` and `r_a = M_a w_a`.
/// 3. `p_a = Xs't_a / t_a't_a`. `X_a't_a = Xs't_a − Σ_{j<a} p_j (t_j't_a)`,
///    and the scores are mutually orthogonal (`t_a = X_a w_a` lies in the
///    complement of `t_1 … t_{a−1}`), so the sum vanishes.
/// 4. `q_a = ys't_a / t_a't_a = s_a'w_a / t_a't_a`. `ys − y_a` is a
///    combination of `t_1 … t_{a−1}`, orthogonal to `t_a`, and
///    `y_a't_a = y_a'X_a w_a = s_a'w_a`.
///
/// So the floors test `‖s_a‖`, the undeflated `‖Xs‖_F` (passed in as
/// `x_fro`), `‖ys‖`, and `t't`, and at `a = 1` every step is the
/// same operation on the same inputs as the explicit-deflation kernel: a
/// K = 1 fit on a column-major `Xs` is bit-identical to it. At `a ≥ 2` the
/// two differ in rounding only; `fit::kernel_tests` bounds the difference.
///
/// Each component reads `Xs` twice (`t = Xs r`, `p = Xs't/tt`) and never
/// writes it; `s_1 = Xs'ys` is one more read per fit. On a row-major `Xs` at
/// least `linalg::ROW_BLOCK_MIN_COLS` wide the `Xs'·v` products read its rows
/// in blocks (`linalg::row_major_t_mul`). `par`, resolved by the caller
/// (`resolve_par`), threads the three products: `t = Xs r` is a faer GEMV,
/// and `Xs'ys` and `Xs't` are the blocked row product or a faer GEMV.
#[allow(clippy::doc_markdown)] // "MacGregor"
#[allow(clippy::many_single_char_names)]
pub(crate) fn pls1_kernel(
    xs: MatRef<'_, f64>,
    ys: ColRef<'_, f64>,
    k: usize,
    keep: Option<usize>,
    par: Par,
    x_fro: f64,
) -> (Mat<f64>, Mat<f64>, Mat<f64>, Col<f64>) {
    let n = xs.nrows();
    let d = xs.ncols();
    // s_1 = Xs' ys, formed as the loadings are (`xs_t_mul`)
    let s0 = xs_t_mul(xs, ys, 1.0, par);
    // `x_fro` is `‖xs‖_F`, supplied by the caller: `pls1_fit` forms it from
    // its standardization moments or its sum-of-squares screen,
    // `signal_test::prepare_split` from the column moments of its halves,
    // the other callers take `xs.norm_l2()`. It only ever gates a `break`, so
    // no output bit depends on it unless it truncates.
    let w_floor = w_rel_floor(n, d, x_fro, ys.norm_l2());
    let mut backend = XBackend { xs, ys, par };
    match pls1_component_loop(&mut backend, s0, w_floor, k, keep) {
        LoopOutcome::Done { t, p, w, q } => (t.unwrap_or_else(|| Mat::<f64>::zeros(n, 0)), p, w, q),
        LoopOutcome::Unresolved => unreachable!("the X backend never returns Gate::Unresolved"),
    }
}

/// Regression coefficient using first `k` PLS components.
/// Formula: coef = W (P'W)^{-1} Q.
///
/// `par` threads the two GEMMs (`P'W` and `W·z`) explicitly: operator-`*`
/// would dispatch on faer's global parallelism (default `Rayon` at the
/// pool's size), which leaks into the global pool when this runs inside a
/// resampler's Rayon worker and rounds differently per pool size. The K×K
/// solve is `linalg::lu_solve_in_place`, sequential for the same reason.
#[allow(clippy::many_single_char_names)]
pub(crate) fn pls1_coef_at_k(
    w: &Mat<f64>,
    p: &Mat<f64>,
    q: &Col<f64>,
    k: usize,
    par: Par,
) -> Col<f64> {
    let d = w.nrows();
    let wk = w.subcols(0, k);
    let pk = p.subcols(0, k);
    let qk = q.subrows(0, k);
    // P' W is (k, k); solve (P'W) z = Q via faer's LU, then coef = W z.
    let mut pwk = Mat::<f64>::zeros(k, k);
    matmul(pwk.as_mut(), Accum::Replace, pk.transpose(), wk, 1.0, par);
    let mut z: Col<f64> = qk.to_owned();
    crate::linalg::lu_solve_in_place(pwk.as_ref(), z.as_mut().as_mat_mut());
    let mut coef = Col::<f64>::zeros(d);
    matmul(
        coef.as_mut().as_mat_mut(),
        Accum::Replace,
        wk,
        z.as_ref().as_mat(),
        1.0,
        par,
    );
    coef
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // test code: oracles and designs may use faer's global-parallelism APIs
mod tests {
    use super::*;
    use crate::test_support::orthogonal_y;
    use approx::assert_relative_eq;

    /// The explicit-deflation PLS1 kernel (textbook NIPALS deflation), kept
    /// verbatim as the oracle for the IKPLS kernel (`pls1_kernel`). Do not
    /// edit: every equivalence test in `fit::kernel_tests` measures against
    /// this exact float sequence.
    #[allow(clippy::many_single_char_names)]
    #[allow(clippy::similar_names)]
    #[allow(clippy::type_complexity)]
    #[allow(clippy::unnecessary_wraps)] // reserved for future variants that may return Err
    pub(super) fn nipals_pls1_reference(
        x: MatRef<'_, f64>,
        y: ColRef<'_, f64>,
        k: usize,
        keep: Option<usize>,
        par: Par,
    ) -> PlsKitResult<(Mat<f64>, Mat<f64>, Mat<f64>, Col<f64>)> {
        let n = x.nrows();
        let d = x.ncols();
        // Owned working copies — deflated in place across components.
        // The caller resolves `par` from `FitOpts::par` (see `resolve_par`).
        let mut xk: Mat<f64> = x.to_owned();
        let mut yk: Col<f64> = y.to_owned();
        // From the undeflated inputs, once. It only ever gates a `break`, so no
        // output bit depends on it unless it truncates.
        let w_floor = w_rel_floor(n, d, x.norm_l2(), y.norm_l2());

        // Pre-allocate output matrices (truncated at end if convergence stops short).
        let mut t_mat = Mat::<f64>::zeros(n, k);
        let mut p_mat = Mat::<f64>::zeros(d, k);
        let mut w_mat = Mat::<f64>::zeros(d, k);
        let mut q_vec = Col::<f64>::zeros(k);
        let mut k_actual = 0usize;

        for a in 0..k {
            // w = X' y  (GEMV)
            let mut w: Col<f64> = Col::<f64>::zeros(d);
            matmul(
                w.as_mut().as_mat_mut(),
                Accum::Replace,
                xk.as_ref().transpose(),
                yk.as_ref().as_mat(),
                1.0,
                par,
            );
            // Sparse keep-count selection (spls1 family) sits between the GEMV
            // and the norm guard, so an all-zero surviving set truncates via the
            // absolute floor. `keep = None` and the dense endpoint (keep ==
            // n_features) are a LITERAL skip of the selection, so the dense
            // float sequence is provably untouched (bit-parity tripwire), not
            // merely value-preserving. `w_norm < max(a, b)` is `w_norm < a ||
            // w_norm < b`, NaN included (`f64::max` ignores a NaN operand, and
            // a NaN `w_norm` fails either test).
            if select_and_normalize(&mut w, keep.unwrap_or(d), NIPALS_ABS_FLOOR.max(w_floor))
                .is_none()
            {
                break;
            }
            // t = X w  (GEMV)
            let mut t: Col<f64> = Col::<f64>::zeros(n);
            matmul(
                t.as_mut().as_mat_mut(),
                Accum::Replace,
                xk.as_ref(),
                w.as_ref().as_mat(),
                1.0,
                par,
            );
            let tt = t.squared_norm_l2();
            if tt < NIPALS_ABS_FLOOR {
                break;
            }
            let inv_tt = 1.0 / tt;
            // p = X' t / (t't)  (GEMV)
            let mut p: Col<f64> = Col::<f64>::zeros(d);
            matmul(
                p.as_mut().as_mat_mut(),
                Accum::Replace,
                xk.as_ref().transpose(),
                t.as_ref().as_mat(),
                inv_tt,
                par,
            );
            // q = y' t / (t't) — small dot, scalar is fine
            let q: f64 = (0..n).map(|i| yk[i] * t[i]).sum::<f64>() * inv_tt;

            // Rank-1 deflation: Xk -= t · p'  (GER via matmul with alpha=-1)
            matmul(
                xk.as_mut(),
                Accum::Add,
                t.as_ref().as_mat(),
                p.as_ref().as_mat().transpose(),
                -1.0,
                par,
            );
            // y -= q · t  (AXPY; scalar n-pass is fine)
            for i in 0..n {
                yk[i] -= q * t[i];
            }

            t_mat.col_mut(a).copy_from(&t);
            p_mat.col_mut(a).copy_from(&p);
            w_mat.col_mut(a).copy_from(&w);
            q_vec[a] = q;
            k_actual = a + 1;
        }

        if k_actual == k {
            Ok((t_mat, p_mat, w_mat, q_vec))
        } else {
            // Truncate to actually-fitted columns.
            let t_out = t_mat.subcols(0, k_actual).to_owned();
            let p_out = p_mat.subcols(0, k_actual).to_owned();
            let w_out = w_mat.subcols(0, k_actual).to_owned();
            let q_out = Col::<f64>::from_fn(k_actual, |i| q_vec[i]);
            Ok((t_out, p_out, w_out, q_out))
        }
    }

    fn linear_data(n: usize, d: usize, k_true: usize, seed: u64) -> (Mat<f64>, Col<f64>) {
        use rand::RngExt;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let x = Mat::<f64>::from_fn(n, d, |_, _| rng.random_range(-1.0..1.0));
        let beta_true = Col::<f64>::from_fn(d, |j| if j < k_true { 1.0 } else { 0.0 });
        let noise = Col::<f64>::from_fn(n, |_| rng.random_range(-0.1..0.1));
        let y_signal: Col<f64> = &x * &beta_true;
        let y = Col::<f64>::from_fn(n, |i| y_signal[i] + noise[i]);
        (x, y)
    }

    #[test]
    fn fit_pre_standardized_skips_centering() {
        // Uncentred, unscaled input: the fit is the kernel's on the arrays
        // as given, with no back-projection. A fit that standardized anyway
        // would return the coefficients of the standardized arrays.
        let (x, y) = linear_data(50, 8, 3, 1);
        let x = Mat::<f64>::from_fn(50, 8, |i, j| x[(i, j)] + 3.0);
        let y = Col::<f64>::from_fn(50, |i| 2.0 * y[i] + 10.0);
        let m = pls1_fit(
            x.as_ref(),
            y.as_ref(),
            3,
            None,
            None,
            FitOpts {
                pre_standardized: true,
                ..FitOpts::default()
            },
        )
        .unwrap();
        assert!(m.pre_standardized);
        let (_, p_ref, w_ref, q_ref) =
            nipals_pls1_reference(x.as_ref(), y.as_ref(), 3, None, Par::Seq).unwrap();
        let raw = pls1_coef_at_k(&w_ref, &p_ref, &q_ref, 3, Par::Seq);
        for j in 0..m.coef.nrows() {
            assert_relative_eq!(m.coef[j], raw[j], max_relative = 1e-10);
            assert_eq!(m.beta[j].to_bits(), m.coef[j].to_bits(), "beta[{j}]");
        }
        assert_eq!(m.intercept.to_bits(), 0.0f64.to_bits());
    }

    // ── spls1 sparse kernel ──────────────────────────────────────────

    #[test]
    #[allow(clippy::float_cmp)] // powers of two: the norm is exact
    fn selected_norm_keeps_range() {
        // Four entries of magnitude 2^e and a smaller one that `keep = 4`
        // drops: the norm is √(4·4^e) = 2^(e+1) exactly, where the squares
        // overflow to inf (e = 600) or underflow to 0 (e = -600).
        for e in [600, -600] {
            let t = 2f64.powi(e);
            let s = Col::<f64>::from_fn(5, |j| [t, -t, 0.5 * t, t, -t][j]);
            assert_eq!(selected_norm(&s, Some(4)), 2f64.powi(e + 1), "e = {e}");
        }
    }

    #[test]
    fn spls1_keep_eq_n_features_is_bit_identical_to_dense() {
        // THE bit-parity tripwire (`docs/rust/api.md`): keep = n_features must be a literal
        // skip — exact equality, not approx.
        let (x, y) = linear_data(50, 8, 3, 1);
        let dense = pls1_fit(x.as_ref(), y.as_ref(), 3, None, None, FitOpts::default()).unwrap();
        let sparse =
            spls1_fit(x.as_ref(), y.as_ref(), 3, 8, None, None, FitOpts::default()).unwrap();
        for j in 0..8 {
            assert_eq!(
                dense.coef[j].to_bits(),
                sparse.coef[j].to_bits(),
                "coef[{j}]"
            );
            assert_eq!(
                dense.beta[j].to_bits(),
                sparse.beta[j].to_bits(),
                "beta[{j}]"
            );
        }
        assert_eq!(dense.intercept.to_bits(), sparse.intercept.to_bits());
        assert_eq!(dense.k_used, sparse.k_used);
        assert_eq!(sparse.keep, Some(8));
        assert_eq!(dense.keep, None);
    }

    #[test]
    fn spls1_exact_nonzero_count_per_component() {
        // At keep = m, exactly m nonzeros per w column — always exact
        // (ties break by lowest index, so never m±1).
        let (x, y) = linear_data(50, 10, 3, 7);
        for keep in [1usize, 3, 7] {
            let m = spls1_fit(
                x.as_ref(),
                y.as_ref(),
                3,
                keep,
                None,
                None,
                FitOpts::default(),
            )
            .unwrap();
            for a in 0..m.k_used {
                let nnz = (0..10).filter(|&j| m.w_star[(j, a)] != 0.0).count();
                assert_eq!(nnz, keep, "component {a} at keep={keep}");
            }
        }
    }

    #[test]
    #[allow(clippy::float_cmp)] // asserting exact-zero from hard_select_keep — intentional bit-equality
    fn spls1_tie_break_lowest_index_wins() {
        // X built so that |X'y| has exact ties: column 1 duplicates column 0
        // and column 3 duplicates column 2. keep=1 must select column 0;
        // keep=3 must select {0, 1, 2} (not {0, 1, 3}).
        use rand::RngExt;
        use rand::SeedableRng;
        let n = 16;
        let mut x = Mat::<f64>::zeros(n, 4);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(5);
        for i in 0..n {
            let a = rng.random_range(-1.0..1.0);
            let b = rng.random_range(-1.0..1.0);
            x[(i, 0)] = a;
            x[(i, 1)] = a; // exact duplicate → exact |w| tie with col 0
            x[(i, 2)] = b;
            x[(i, 3)] = b; // exact duplicate → exact |w| tie with col 2
        }
        let y = Col::<f64>::from_fn(n, |i| x[(i, 0)] + 0.5 * x[(i, 2)]);
        let m1 = pls1_fit_impl(
            x.as_ref(),
            y.as_ref(),
            1,
            None,
            None,
            FitOpts {
                pre_standardized: true,
                ..FitOpts::default()
            },
            InternalFitOpts {
                check_n_eff: false,
                keep: Some(1),
            },
        )
        .unwrap();
        assert!(
            m1.w_star[(0, 0)] != 0.0,
            "keep=1 must keep column 0 (lowest index in tie)"
        );
        for j in 1..4 {
            assert_eq!(m1.w_star[(j, 0)], 0.0, "col {j} must be zeroed");
        }
        let m3 = pls1_fit_impl(
            x.as_ref(),
            y.as_ref(),
            1,
            None,
            None,
            FitOpts {
                pre_standardized: true,
                ..FitOpts::default()
            },
            InternalFitOpts {
                check_n_eff: false,
                keep: Some(3),
            },
        )
        .unwrap();
        assert!(
            m3.w_star[(2, 0)] != 0.0,
            "col 2 (higher |w|-rank than its duplicate at idx 3 via index tie-break path) must survive"
        );
        assert_eq!(m3.w_star[(3, 0)], 0.0, "col 3 loses the tie against col 2");
    }

    // ── relative floor on w_norm (y exhausted) ──────────────────────

    pub(super) fn uniform_mat(n: usize, d: usize, seed: u64) -> Mat<f64> {
        use rand::RngExt;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        Mat::<f64>::from_fn(n, d, |_, _| rng.random_range(-1.0..1.0))
    }

    #[test]
    fn select_and_normalize_norm_keeps_range() {
        let base = uniform_mat(50, 1, 3);
        for scale in [1e300, 1e-300, 1.0] {
            let mut v = Col::<f64>::from_fn(50, |i| base[(i, 0)] * scale);
            let expect = v.norm_l2();
            let nv = select_and_normalize(&mut v, 50, 1e-14 * scale.min(1.0)).expect("non-zero");
            assert!(nv.is_finite() && nv > 0.0, "scale {scale}: norm {nv}");
            assert_relative_eq!(nv, expect, max_relative = 1e-14);
            assert_relative_eq!(v.norm_l2(), 1.0, max_relative = 1e-14);
        }
        let mut z = Col::<f64>::zeros(50);
        assert!(select_and_normalize(&mut z, 50, 1e-14).is_none());
        let mut v = Col::<f64>::from_fn(50, |i| base[(i, 0)]);
        v[7] = f64::NAN;
        assert!(select_and_normalize(&mut v, 50, 1e-14).is_some_and(f64::is_nan));
    }

    /// `n × 40`, last column the sum of the first two (rank 39), `y`
    /// independent of `X`. At `n = 2000`, `seed = 11`, `y`'s projection onto
    /// the span of `X` is fitted within about 17 components, after which
    /// `‖X_a'y_a‖` is rounding noise near `1e-13`, above the absolute floor.
    pub(super) fn exhausted_design(n: usize, seed: u64) -> (Mat<f64>, Col<f64>) {
        let d = 40;
        let mut x = uniform_mat(n, d, seed);
        for i in 0..n {
            x[(i, d - 1)] = x[(i, 0)] + x[(i, 1)];
        }
        let yc = uniform_mat(n, 1, seed + 1);
        (x, Col::<f64>::from_fn(n, |i| yc[(i, 0)]))
    }

    /// A `y` orthogonal to the standardized `X` plus `signal` times `X b`:
    /// with `signal = 0` the first component is already noise, with
    /// `signal = 1e-6` the real components are weak (the design of
    /// `nipals_floor_is_relative_*`).
    #[allow(clippy::many_single_char_names)]
    pub(super) fn orthogonal_design(
        n: usize,
        d: usize,
        signal: f64,
        seed: u64,
    ) -> (Mat<f64>, Col<f64>) {
        let x = uniform_mat(n, d, seed);
        let (xs, _, _) = crate::linalg::standardize(x.as_ref());
        let svd = xs.thin_svd().expect("svd");
        let u = svd.U();
        let e = uniform_mat(n, 1, seed + 1);
        let mut e_perp = Col::<f64>::from_fn(n, |i| e[(i, 0)]);
        for j in 0..d {
            let c: f64 = (0..n).map(|i| u[(i, j)] * e_perp[i]).sum();
            for i in 0..n {
                e_perp[i] -= c * u[(i, j)];
            }
        }
        let b = uniform_mat(d, 1, seed + 2);
        let y = Col::<f64>::from_fn(n, |i| {
            e_perp[i] + signal * (0..d).map(|j| xs[(i, j)] * b[(j, 0)]).sum::<f64>()
        });
        (x, y)
    }

    #[test]
    fn nipals_drops_noise_components_once_y_is_exhausted() {
        let (x, y) = exhausted_design(2000, 11);
        let m = pls1_fit(x.as_ref(), y.as_ref(), 40, None, None, FitOpts::default()).unwrap();
        // Without the relative floor all 40 components were kept, the
        // last ~20 of them fitted to rounding noise.
        assert!(
            (5..=20).contains(&m.k_used),
            "expected truncation near the numerical rank, got k_used={}",
            m.k_used
        );
        assert!(m.coef.norm_l2().is_finite());
    }

    #[test]
    fn spls1_drops_noise_components_once_y_is_exhausted() {
        // Same design through the sparse path: selection runs before the
        // floor, so the floor sees the selected norm. The selected norm
        // crosses the relative floor after 28 components; with the floor
        // disabled the fit keeps 39.
        let (x, y) = exhausted_design(2000, 11);
        let m = spls1_fit(
            x.as_ref(),
            y.as_ref(),
            40,
            10,
            None,
            None,
            FitOpts::default(),
        )
        .unwrap();
        assert!(
            (20..=34).contains(&m.k_used),
            "expected truncation before the noise tail, got k_used={}",
            m.k_used
        );
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn nipals_stops_at_m_when_y_is_in_span_of_m_directions() {
        // y is an exact combination of two left singular vectors of the
        // standardized X, so exactly two components are real and the
        // third `‖X_a'y_a‖` is rounding noise (about 1e-13 here, above the
        // absolute floor).
        let (n, d) = (500, 30);
        let x = uniform_mat(n, d, 21);
        let (xs, _, _) = crate::linalg::standardize(x.as_ref());
        let svd = xs.thin_svd().unwrap();
        let u = svd.U();
        let y = Col::<f64>::from_fn(n, |i| 1.3 * u[(i, 0)] - 0.7 * u[(i, 5)]);
        let m = pls1_fit(x.as_ref(), y.as_ref(), 10, None, None, FitOpts::default()).unwrap();
        assert_eq!(m.k_used, 2);
    }

    #[test]
    fn nipals_floor_is_relative_to_x_and_y_not_to_the_first_component() {
        // y = (component orthogonal to X) + 1e-6 · (signal in X): the first
        // `‖X'y‖` is about 1e-6 of `‖X‖_F·‖y‖`, so a floor relative to it
        // would sit below the rounding noise and keep all 20 components.
        let (x, y) = orthogonal_design(1000, 20, 1e-6, 31);
        let m = pls1_fit(x.as_ref(), y.as_ref(), 20, None, None, FitOpts::default()).unwrap();
        assert!(m.k_used < 20, "got k_used={}", m.k_used);
    }

    /// `n = 200`, `d = 10`, and a `y` orthogonal to the span of `1` and the
    /// standardized columns of `X` (Gram-Schmidt, each projection applied
    /// twice), plus the standardized `X`. `X'y` is then rounding noise near
    /// `1e-14`: under the relative floor but, for this `y`, not always under
    /// the absolute one.
    fn orthogonal_y_design(seed: u64) -> (Mat<f64>, Mat<f64>, Col<f64>) {
        let (n, d) = (200, 10);
        let x = uniform_mat(n, d, seed);
        let (xs, _, _) = crate::linalg::standardize(x.as_ref());
        let y = orthogonal_y(&xs, seed + 1);
        (x, xs, y)
    }

    #[test]
    fn nipals_drops_the_first_component_when_y_is_orthogonal_to_x() {
        // Without the floor on the first component the fit kept one
        // component whose `w` is normalized rounding noise.
        for seed in [41, 43, 45] {
            let (x, _, y) = orthogonal_y_design(seed);
            let m = pls1_fit(x.as_ref(), y.as_ref(), 3, None, None, FitOpts::default()).unwrap();
            assert_eq!(m.k_used, 0, "seed {seed}");
            assert_eq!(m.w_star.ncols(), 0);
            assert_eq!(m.q_loadings.nrows(), 0);
            assert!((0..10).all(|j| m.coef[j] == 0.0 && m.beta[j] == 0.0));
            // The zero model predicts the mean of y.
            let y_mean = (0..y.nrows()).map(|i| y[i]).sum::<f64>() / y.nrows() as f64;
            assert_relative_eq!(m.intercept, y_mean, epsilon = 1e-12);
            // Same decision on the sparse path.
            let s =
                spls1_fit(x.as_ref(), y.as_ref(), 1, 4, None, None, FitOpts::default()).unwrap();
            assert_eq!(s.k_used, 0, "seed {seed} (spls1)");
        }
    }

    #[test]
    fn orthogonal_y_errors_when_strict_and_is_a_zero_model_when_internal() {
        // pre_standardized=true: a top-level call reports the truncation,
        // a per-iteration internal call gets the zero model.
        let (_, xs, y) = orthogonal_y_design(41);
        let (ys, _, _) = crate::linalg::standardize1(y.as_ref());
        let strict = pls1_fit(
            xs.as_ref(),
            ys.as_ref(),
            1,
            None,
            None,
            FitOpts {
                pre_standardized: true,
                ..FitOpts::default()
            },
        );
        assert!(
            matches!(strict, Err(PlsKitError::InvalidInput(_))),
            "expected InvalidInput, got {strict:?}"
        );
        let internal = pls1_fit_impl(
            xs.as_ref(),
            ys.as_ref(),
            1,
            None,
            None,
            FitOpts {
                pre_standardized: true,
                ..FitOpts::default()
            },
            InternalFitOpts {
                check_n_eff: false,
                ..InternalFitOpts::default()
            },
        )
        .unwrap();
        assert_eq!(internal.k_used, 0);
        assert!((0..10).all(|j| internal.coef[j] == 0.0));
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn nipals_keeps_a_tiny_real_first_component() {
        // Guard against an over-eager first-component floor: the orthogonal
        // y plus a true effect of 1e-9 of ‖y‖ along X keeps its component.
        let (x, xs, y_perp) = orthogonal_y_design(41);
        let (n, d) = (xs.nrows(), xs.ncols());
        let b = uniform_mat(d, 1, 47);
        let s = Col::<f64>::from_fn(n, |i| (0..d).map(|j| xs[(i, j)] * b[(j, 0)]).sum::<f64>());
        let scale = 1e-9 * y_perp.norm_l2() / s.norm_l2();
        let y = Col::<f64>::from_fn(n, |i| y_perp[i] + scale * s[i]);
        let m = pls1_fit(x.as_ref(), y.as_ref(), 1, None, None, FitOpts::default()).unwrap();
        assert_eq!(m.k_used, 1);
    }

    #[test]
    fn nipals_floor_keeps_every_component_of_full_signal_data() {
        // Guard against an over-eager floor: ordinary data with signal in
        // every direction keeps all requested components, weighted or not.
        let (x, y) = linear_data(100, 20, 20, 3);
        let m = pls1_fit(x.as_ref(), y.as_ref(), 20, None, None, FitOpts::default()).unwrap();
        assert_eq!(m.k_used, 20);
        let w = Col::<f64>::from_fn(100, |i| 0.5 + (i % 7) as f64 * 0.25);
        let mw = pls1_fit(
            x.as_ref(),
            y.as_ref(),
            20,
            Some(w.as_ref()),
            None,
            FitOpts::default(),
        )
        .unwrap();
        assert_eq!(mw.k_used, 20);
    }

    #[test]
    fn memory_budget_parses_every_documented_form() {
        let g2 = 2_u64 << 30;
        for s in [
            "2147483648",
            "2048M",
            "2G",
            "2g",
            "2GB",
            "2gb",
            "2GiB",
            "2gib",
        ] {
            assert_eq!(parse_memory_budget(Some(s)).unwrap(), g2, "{s}");
        }
        assert_eq!(parse_memory_budget(Some("0")).unwrap(), 0);
        assert_eq!(parse_memory_budget(Some("512K")).unwrap(), 512 << 10);
        assert_eq!(parse_memory_budget(Some("1T")).unwrap(), 1 << 40);
        assert_eq!(parse_memory_budget(Some("7B")).unwrap(), 7);
        assert_eq!(parse_memory_budget(None).unwrap(), DEFAULT_MEMORY_BUDGET);
        for bad in [
            "",
            " 2G",
            "2G ",
            "2.5G",
            "-1",
            "2X",
            "G",
            "2GiBB",
            "99999999999T",
            "2iB",
            "B",
            "iB",
            "18446744073709551616",
        ] {
            let e = parse_memory_budget(Some(bad)).unwrap_err().to_string();
            assert!(
                e.contains(MEMORY_BUDGET_ENV) && e.contains(&format!("{bad:?}")),
                "{bad}: {e}"
            );
        }
    }

    #[test]
    fn default_budget_is_4_gib() {
        assert_eq!(parse_memory_budget(None).unwrap(), 4 * 1024 * 1024 * 1024);
    }

    #[test]
    fn memory_budget_override_wins_and_is_restored() {
        let before = memory_budget();
        with_budget(123, || {
            assert_eq!(memory_budget(), 123);
            with_budget(0, || assert_eq!(memory_budget(), 0));
            assert_eq!(memory_budget(), 123);
        });
        assert_eq!(memory_budget(), before);
    }

    /// `with_thread_limit` reads the budget once: a change of the variable
    /// while the call runs does not reach the call, and does reach the next
    /// read outside it. The value is process-wide, so the test re-runs
    /// itself in a child process (as `tests/num_threads_env.rs` does).
    #[test]
    fn with_thread_limit_reads_the_budget_once() {
        const CHILD: &str = "PLSKIT_TEST_CHILD";
        const RESULT: &str = "PLSKIT_RESULT ";
        let set = |v: &str| *MEMORY_BUDGET_VAR.lock().unwrap() = Some(v.to_owned());
        if std::env::var_os(CHILD).is_some() {
            set("1M");
            let inside = with_thread_limit(|| {
                set("2M");
                Ok(memory_budget())
            })
            .unwrap();
            println!("\n{RESULT}{inside} {}", memory_budget());
            return;
        }
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "fit::tests::with_thread_limit_reads_the_budget_once",
                "--exact",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            out.status.success(),
            "{stdout}\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let result = stdout.lines().find_map(|l| l.strip_prefix(RESULT));
        assert_eq!(result, Some("1048576 2097152"), "{stdout}");
    }

    #[test]
    fn parse_num_threads_follows_the_documented_table() {
        use super::parse_num_threads as p;
        assert_eq!(p(None).unwrap(), None);
        assert_eq!(p(Some("0")).unwrap(), None);
        assert_eq!(p(Some("1")).unwrap(), Some(1));
        assert_eq!(p(Some("4")).unwrap(), Some(4));
        for bad in ["", " 4 ", "-1", "2.5", "abc", "+4"] {
            match p(Some(bad)) {
                Err(PlsKitError::InvalidArgument(msg)) => {
                    assert!(msg.contains("PLSKIT_NUM_THREADS"), "{msg}");
                    assert!(msg.contains(&format!("{bad:?}")), "{msg}");
                }
                other => panic!("{bad:?}: {other:?}"),
            }
        }
    }

    #[test]
    fn run_with_threads_caps_the_callers_pool() {
        let outer = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for (n, want) in [
            (Some(1), 1),
            (Some(3), 3),
            (Some(4), 4),
            (Some(100_000), 4),
            (None, 4),
        ] {
            let got =
                outer.install(|| super::run_with_threads(n, rayon::current_num_threads).unwrap());
            assert_eq!(got, want, "cap {n:?}");
        }
    }

    #[test]
    fn run_with_threads_concurrent_sizes_each_get_their_own_pool() {
        // Each thread calls from inside its own 4-thread pool, so every cap
        // below is under the caller's pool size on any host, a one-core one
        // included, and each call runs on a plskit pool of exactly that size.
        let handles: Vec<_> = [1usize, 3, 1, 3, 2, 2]
            .into_iter()
            .map(|n| {
                std::thread::spawn(move || {
                    let outer = rayon::ThreadPoolBuilder::new()
                        .num_threads(4)
                        .build()
                        .unwrap();
                    (0..20)
                        .map(|_| {
                            outer.install(|| {
                                super::run_with_threads(Some(n), rayon::current_num_threads)
                                    .unwrap()
                            })
                        })
                        .all(|got| got == n)
                })
            })
            .collect();
        assert!(handles.into_iter().all(|h| h.join().unwrap()));
    }

    #[test]
    fn n_eff_does_not_depend_on_the_scale_of_the_weights() {
        use crate::preprocess::{PreprocessInput, preprocess};
        let (x, y) = crate::test_support::synth(10, 3, 1, 1.0, 21);
        let n_effs = |s: f64| {
            #[allow(clippy::cast_precision_loss)]
            let w = Col::<f64>::from_fn(10, |i| (i + 1) as f64 * s);
            let fit = pls1_fit(
                x.as_ref(),
                y.as_ref(),
                1,
                Some(w.as_ref()),
                None,
                FitOpts::default(),
            )
            .unwrap()
            .n_eff;
            let pre = preprocess(PreprocessInput {
                x: Some(x.as_ref()),
                y: Some(y.as_ref()),
                weights: Some(w.as_ref()),
            })
            .unwrap()
            .n_eff
            .unwrap();
            [fit, pre]
        };
        let base = n_effs(1.0);
        for s in [1e200, 1e-170] {
            for (v, b) in n_effs(s).into_iter().zip(base) {
                assert!(v.is_finite(), "n_eff at scale {s} is {v}");
                assert_relative_eq!(v, b, max_relative = 1e-12);
            }
        }
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn weights_whose_sum_overflows_fit_as_their_ratios() {
        // Each weight is finite, their sum is past f64::MAX. Times a power
        // of two, so the ratios are exactly those of `v` and the fit is too.
        let (x, y) = crate::test_support::synth(20, 3, 1, 1.0, 21);
        let v = Col::<f64>::from_fn(20, |i| [0.5, 1.0, 1.5, 2.0][i % 4]);
        let huge = Col::<f64>::from_fn(20, |i| v[i] * 2f64.powi(1020));
        let fit = |w: &Col<f64>| {
            pls1_fit(
                x.as_ref(),
                y.as_ref(),
                2,
                Some(w.as_ref()),
                None,
                FitOpts::default(),
            )
            .unwrap()
        };
        let (a, b) = (fit(&huge), fit(&v));
        assert_eq!(a.n_eff.to_bits(), b.n_eff.to_bits(), "n_eff {}", a.n_eff);
        assert_eq!(a.intercept.to_bits(), b.intercept.to_bits());
        for j in 0..3 {
            assert_eq!(a.coef[j].to_bits(), b.coef[j].to_bits(), "coef[{j}]");
        }
        // Two equal weights at the top of the range are no weights.
        let two = Col::<f64>::from_fn(20, |_| 1e308);
        let m = fit(&two);
        let none = pls1_fit(x.as_ref(), y.as_ref(), 2, None, None, FitOpts::default()).unwrap();
        assert!(m.weights.is_none());
        assert_eq!(m.n_eff.to_bits(), none.n_eff.to_bits());
        for j in 0..3 {
            assert_eq!(m.coef[j].to_bits(), none.coef[j].to_bits(), "coef[{j}]");
        }
        // The sum is finite but `w·n` is not: still the error the ratios
        // call for (one dominant weight, n_eff ≈ 1), not `NonFiniteInput`.
        let mut one = Col::<f64>::from_fn(20, |_| 1.0);
        one[0] = 1e308;
        let err = pls1_fit(
            x.as_ref(),
            y.as_ref(),
            1,
            Some(one.as_ref()),
            None,
            FitOpts::default(),
        )
        .unwrap_err();
        assert!(matches!(err, PlsKitError::InvalidWeights { .. }), "{err:?}");
    }
}

#[cfg(test)]
mod kernel_tests;

/// `pls1_fit` at its public boundary: layout invariance (to rounding), input
/// screening, error codes and precedence, and the prepared tail the
/// resamplers use.
#[cfg(test)]
#[allow(
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::type_complexity
)]
mod boundary_tests {
    use super::*;
    use crate::linalg::{
        normalize_weights, standardize_weighted, standardize1, standardize1_weighted,
    };
    use crate::test_support::{
        Agree, Family, assert_agree, assert_bits_eq, assert_layout_agree, col_vals,
        copy_free_families, mat_vals, signal_data,
    };

    /// Layouts of X agree to rounding, not bit for bit: `pls1_fit` forms its
    /// products from X in X's own layout (CHANGELOG 0.6.1), so each layout
    /// sums in its own order. `Agree::Corpus(1e-10)` is the corpus array
    /// tolerance, `1e-10 + 1e-14 · |value|`, which `docs/python/api.md`
    /// (Conventions) promises across layouts.
    const FIT_LAYOUT_TOL: f64 = 1e-10;

    /// Every numeric output of a fit, flattened; `k_used` as an `f64` fails
    /// `Agree::Corpus(FIT_LAYOUT_TOL)` on any difference.
    fn model_vals(m: &Pls1Model) -> Vec<f64> {
        let mut v = mat_vals(m.t_scores.as_ref());
        v.extend(mat_vals(m.p_loadings.as_ref()));
        v.extend(mat_vals(m.w_star.as_ref()));
        v.extend(col_vals(m.q_loadings.as_ref()));
        v.extend(col_vals(m.coef.as_ref()));
        v.extend(col_vals(m.beta.as_ref()));
        #[allow(clippy::cast_precision_loss)]
        v.extend([m.intercept, m.n_eff, m.k_used as f64]);
        v
    }

    /// `pls1_fit` on every layout of `x` agrees with the owned column-major
    /// run to `FIT_LAYOUT_TOL`; returns the owned run's values. A `keep`
    /// runs `spls1_fit`.
    fn fit_layouts(
        what: &str,
        x: &Mat<f64>,
        y: &Col<f64>,
        w: Option<ColRef<'_, f64>>,
        k: usize,
        (opts, keep): (FitOpts, Option<usize>),
    ) -> Vec<f64> {
        assert_layout_agree(x, what, Agree::Corpus(FIT_LAYOUT_TOL), |xv| {
            match keep {
                Some(kp) => spls1_fit(xv, y.as_ref(), k, kp, w, None, opts),
                None => pls1_fit(xv, y.as_ref(), k, w, None, opts),
            }
            .map(|m| model_vals(&m))
        })
        .unwrap_or_else(|e| panic!("{what}: {e}"))
    }

    fn offset(x: &Mat<f64>, c: f64) -> Mat<f64> {
        Mat::<f64>::from_fn(x.nrows(), x.ncols(), |i, j| x[(i, j)] + c)
    }

    #[test]
    fn pls1_fit_is_layout_invariant_to_rounding() {
        let opts = |pre, keep, par| {
            (
                FitOpts {
                    pre_standardized: pre,
                    par,
                },
                keep,
            )
        };
        for f in copy_free_families() {
            let w = f.w.as_ref().map(Col::as_ref);
            for pre in [false, true] {
                let (x, y) = f.inputs(pre);
                for k in [1_usize, 3] {
                    for keep in [None, Some(3)] {
                        for par in [ParChoice::Seq, ParChoice::Auto] {
                            let what = format!("{} pre={pre} k={k} keep={keep:?} {par:?}", f.name);
                            fit_layouts(&what, &x, &y, w, k, opts(pre, keep, par));
                        }
                    }
                }
            }
        }
        // Sizes where `ParChoice::Auto` resolves to the Rayon path (at k = 3):
        // the owned Seq and Auto runs agree too (nothing else runs the Rayon
        // implicit products against Seq at this size). At 2100 columns
        // (`linalg::ROW_BLOCK_MIN_COLS` and up) the row-major view forms
        // `Xs'·t`, raw or pre-standardized, with its rows in blocks
        // (`linalg::row_major_t_mul`); 203 rows leave a partial block, also
        // in each Rayon piece. Its moments run over strips of 1,288 and 812
        // columns, in blocks of rows too. At 2000 it keeps faer's product.
        for (n, d, seed) in [(200, 2000, 9), (203, 2100, 10)] {
            let (x, y) = signal_data(n, d, seed);
            // The row-major view of `Layouts` (the transpose of a stored
            // transpose) takes the blocked product exactly at 2100 columns,
            // under both arms.
            let x_t = x.transpose().to_owned();
            for par in [Par::Seq, crate::fit::par_fixed()] {
                assert_eq!(
                    crate::linalg::row_major_t_mul(x_t.transpose(), y.as_ref(), par).is_some(),
                    d == 2100,
                    "{n}x{d} {par:?}: left its row-major route"
                );
            }
            let (xs, _, _) = crate::linalg::standardize(x.as_ref());
            let (ys, _, _) = standardize1(y.as_ref());
            for k in [1_usize, 3] {
                for keep in [None, Some(3)] {
                    let what = format!("large {n}x{d} k={k} keep={keep:?}");
                    let seq = fit_layouts(
                        &format!("{what} Seq"),
                        &x,
                        &y,
                        None,
                        k,
                        opts(false, keep, ParChoice::Seq),
                    );
                    let auto = fit_layouts(
                        &format!("{what} Auto"),
                        &x,
                        &y,
                        None,
                        k,
                        opts(false, keep, ParChoice::Auto),
                    );
                    assert_agree(
                        &auto,
                        &seq,
                        Agree::Corpus(FIT_LAYOUT_TOL),
                        &format!("{what} Auto vs Seq"),
                    );
                    for par in [ParChoice::Seq, ParChoice::Auto] {
                        fit_layouts(
                            &format!("{what} pre {par:?}"),
                            &xs,
                            &ys,
                            None,
                            k,
                            opts(true, keep, par),
                        );
                    }
                }
            }
        }
        // The raw fallback on a row-major X at least
        // `linalg::ROW_BLOCK_MIN_COLS` wide: +2e3 puts the column means past
        // `IMPLICIT_MAX_MEAN_RATIO`, so the row-major view is standardized into
        // a row-major copy (`FitX::materialize`) whose `Xs'·t` reads its rows
        // in blocks, under both arms.
        let (x, y) = signal_data(203, 2100, 10);
        let x = offset(&x, 2e3);
        let x_t = x.transpose().to_owned();
        let mut fx = crate::linalg::fit_x_moments(x_t.transpose(), None, Par::Seq);
        assert!(
            fx.max_mean_ratio() > IMPLICIT_MAX_MEAN_RATIO,
            "203x2100 + 2e3: left the standardized copy"
        );
        fx.materialize(x_t.transpose(), None);
        for par in [Par::Seq, crate::fit::par_fixed()] {
            assert!(
                crate::linalg::row_major_t_mul(fx.xs(), y.as_ref(), par).is_some(),
                "203x2100 + 2e3 {par:?}: the copy left its row-major route"
            );
        }
        for k in [1_usize, 3] {
            let what = format!("large 203x2100 + 2e3 k={k}");
            let seq = fit_layouts(
                &format!("{what} Seq"),
                &x,
                &y,
                None,
                k,
                opts(false, None, ParChoice::Seq),
            );
            let auto = fit_layouts(
                &format!("{what} Auto"),
                &x,
                &y,
                None,
                k,
                opts(false, None, ParChoice::Auto),
            );
            assert_agree(
                &auto,
                &seq,
                Agree::Corpus(FIT_LAYOUT_TOL),
                &format!("{what} Auto vs Seq"),
            );
        }
        // Offset X on both sides of `IMPLICIT_MAX_MEAN_RATIO`: +300 on the
        // dense family stays on the implicit products, +1e6 on the weighted
        // family takes the standardized copy.
        let fams = copy_free_families();
        for (f, c, implicit) in [(&fams[0], 300.0, true), (&fams[1], 1e6, false)] {
            let x = offset(&f.x, c);
            let w = f.w.as_ref().map(Col::as_ref);
            let ratio = crate::linalg::fit_x_moments(x.as_ref(), w, Par::Seq).max_mean_ratio();
            let what = format!("{} + {c:e} (ratio {ratio:.0})", f.name);
            assert_eq!(
                ratio > 100.0 && ratio < IMPLICIT_MAX_MEAN_RATIO,
                implicit,
                "{what}: left its route"
            );
            assert!(ratio > 100.0, "{what}: not mean-heavy");
            for k in [1_usize, 3] {
                fit_layouts(
                    &format!("{what} k={k}"),
                    &x,
                    &f.y,
                    w,
                    k,
                    opts(false, None, ParChoice::Seq),
                );
            }
        }
    }

    /// A pre-standardized fit on a row-major view at least
    /// `linalg::ROW_BLOCK_MIN_COLS` wide forms `Xs'·ys` and `Xs'·t` with the
    /// blocked row product, as the implicit backend does: the first weight
    /// vector is `row_major_t_mul(Xs, ys)` scaled to unit norm, and the first
    /// loading is `row_major_t_mul` of the first score times `1/t't`, bit for
    /// bit, on both arms. At 203 × 5000 and k = 1, `Auto` takes the Rayon arm.
    #[test]
    fn pre_standardized_row_major_loading_is_the_row_product() {
        let (x, y) = signal_data(203, 5000, 12);
        let (xs, _, _) = crate::linalg::standardize(x.as_ref());
        let (ys, _, _) = standardize1(y.as_ref());
        let xs_t = xs.transpose().to_owned();
        let view = xs_t.transpose();
        assert!(matches!(
            resolve_par(ParChoice::Auto, 203, 5000, 1),
            Par::Rayon(_)
        ));
        for (choice, par) in [
            (ParChoice::Seq, Par::Seq),
            (ParChoice::Auto, crate::fit::par_fixed()),
        ] {
            let m = pls1_fit(
                view,
                ys.as_ref(),
                1,
                None,
                None,
                FitOpts {
                    pre_standardized: true,
                    par: choice,
                },
            )
            .unwrap();
            let mut w =
                crate::linalg::row_major_t_mul(view, ys.as_ref(), par).expect("row-major and wide");
            select_and_normalize(&mut w, view.ncols(), 0.0).expect("s_1 is not zero");
            assert_bits_eq(
                &col_vals(m.w_star.col(0)),
                &col_vals(w.as_ref()),
                &format!("{choice:?} w"),
            );
            let t = m.t_scores.col(0);
            let inv_tt = 1.0 / t.squared_norm_l2();
            let g = crate::linalg::row_major_t_mul(view, t, par).expect("row-major and wide");
            let want: Vec<f64> = (0..g.nrows()).map(|j| g[j] * inv_tt).collect();
            assert_bits_eq(
                &col_vals(m.p_loadings.col(0)),
                &want,
                &format!("{choice:?}"),
            );
        }
    }

    /// A non-finite entry is found on every layout, on both paths, weighted
    /// or not, also when its row has weight zero.
    #[test]
    fn non_finite_x_is_found_on_every_layout() {
        let (xl, yl) = signal_data(200, 2000, 9);
        let mut families = copy_free_families();
        families.push(Family {
            name: "large",
            x: xl,
            y: yl,
            w: None,
        });
        for f in &families {
            let n = f.x.nrows();
            let mut bad = f.x.clone();
            bad[(n - 1, f.x.ncols() - 1)] = f64::INFINITY;
            let zero_last = Col::<f64>::from_fn(n, |i| if i + 1 == n { 0.0 } else { 1.0 });
            for pre in [false, true] {
                for w in [None, Some(zero_last.as_ref())] {
                    let what = format!("{} pre={pre} weighted={}", f.name, w.is_some());
                    let opts = FitOpts {
                        pre_standardized: pre,
                        ..FitOpts::default()
                    };
                    let r = assert_layout_agree(&bad, &what, Agree::Bits, |xv| {
                        pls1_fit(xv, f.y.as_ref(), 1, w, None, opts).map(|_| vec![])
                    });
                    assert!(
                        matches!(r, Err(PlsKitError::NonFiniteInput)),
                        "{what}: {r:?}"
                    );
                }
            }
        }
    }

    /// A `y` orthogonal to an offset X, below `IMPLICIT_MAX_MEAN_RATIO`: two
    /// offset columns (largest `|mean_j| / scale_j` about 270), and a centered
    /// column beside the offset column `a + 999.3` (ratio 999.3; a constant
    /// column's ratio is at most 1, so it does not count). The implicit products
    /// would put the first `‖X'y‖` above the floor and keep one component;
    /// the copy puts it below, and the fit decides on the copy: `k_used = 0`.
    #[test]
    #[allow(clippy::float_cmp)] // the zero model's intercept is mean(y) exactly
    fn orthogonal_y_on_offset_x_keeps_no_component() {
        use rand::{RngExt, SeedableRng};
        let designs = [[0.37, 99.7, -1.3, 89.73], [0.37, 0.0, 1.0, 999.3]];
        for (c, n) in designs
            .iter()
            .flat_map(|c| [8_usize, 12, 16].map(|n| (c, n)))
        {
            // Column j is `c[2j]·a + c[2j + 1]`, `a = ±1` alternating.
            let x = Mat::<f64>::from_fn(n, 2, |i, j| {
                let a = if i % 2 == 0 { 1.0 } else { -1.0 };
                c[2 * j] * a + c[2 * j + 1]
            });
            let ones = Col::<f64>::from_fn(n, |_| 1.0);
            let basis = crate::test_support::orthonormal_basis(ones.as_ref(), x.as_ref(), 1e-12);
            for seed in 0..20_u64 {
                let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
                let mut y = Col::<f64>::from_fn(n, |_| rng.random_range(-1.0..1.0));
                crate::test_support::project_off(&basis, &mut y);
                let what = format!("{c:?} n={n} seed={seed}");
                let m =
                    pls1_fit(x.as_ref(), y.as_ref(), 1, None, None, FitOpts::default()).unwrap();
                assert_eq!(m.k_used, 0, "{what}");
                assert_eq!(
                    (m.t_scores.ncols(), m.w_star.ncols(), m.q_loadings.nrows()),
                    (0, 0, 0),
                    "{what}"
                );
                assert!(
                    (0..2).all(|j| m.coef[j] == 0.0 && m.beta[j] == 0.0),
                    "{what}"
                );
                let (_, y_mean, _) = standardize1(y.as_ref());
                assert_eq!(
                    m.intercept, y_mean,
                    "{what}: the zero model predicts mean(y)"
                );
            }
        }
    }

    /// A pre-standardized X outside the band (`2^±600`, whose products
    /// overflow or underflow, and `2^±30`, where the kernel's absolute floors
    /// would cut components) is fitted divided by a power of two and mapped
    /// back: the fit of `X·s` is the fit of X with `coef`, `beta` and `q`
    /// divided by `s` and the scores times `s`, to the bit.
    #[test]
    fn pre_standardized_fit_far_from_unit_scale_is_the_rescaled_fit() {
        let (x, y) = signal_data(40, 6, 3);
        let (ys, _, _) = standardize1(y.as_ref());
        let opts = FitOpts {
            pre_standardized: true,
            ..FitOpts::default()
        };
        let want = pls1_fit(x.as_ref(), ys.as_ref(), 2, None, None, opts).unwrap();
        for s in [
            2f64.powi(600),
            2f64.powi(-600),
            2f64.powi(30),
            2f64.powi(-30),
        ] {
            let xs = Mat::<f64>::from_fn(40, 6, |i, j| x[(i, j)] * s);
            let got = pls1_fit(xs.as_ref(), ys.as_ref(), 2, None, None, opts).unwrap();
            let what = format!("s = {s:e}");
            let back = |c: &Col<f64>| {
                col_vals(c.as_ref())
                    .iter()
                    .map(|v| v * s)
                    .collect::<Vec<_>>()
            };
            assert_bits_eq(&back(&got.coef), &col_vals(want.coef.as_ref()), &what);
            assert_bits_eq(&back(&got.beta), &col_vals(want.beta.as_ref()), &what);
            assert_bits_eq(
                &back(&got.q_loadings),
                &col_vals(want.q_loadings.as_ref()),
                &what,
            );
            let t: Vec<f64> = mat_vals(got.t_scores.as_ref())
                .iter()
                .map(|v| v / s)
                .collect();
            assert_bits_eq(&t, &mat_vals(want.t_scores.as_ref()), &what);
            assert_bits_eq(
                &mat_vals(got.p_loadings.as_ref()),
                &mat_vals(want.p_loadings.as_ref()),
                &what,
            );
            assert_bits_eq(
                &mat_vals(got.w_star.as_ref()),
                &mat_vals(want.w_star.as_ref()),
                &what,
            );
        }
    }

    /// Weighted, a row of weight 0 does not pick that power of two: the fit
    /// of X·2^-600 is the same with the zero-weight rows at X·2^600, which
    /// divided the other rows below the `f64` range.
    #[test]
    #[allow(clippy::float_cmp)]
    fn weighted_pre_standardized_fit_ignores_far_zero_weight_rows() {
        let (x, y) = signal_data(40, 6, 3);
        let (ys, _, _) = standardize1(y.as_ref());
        let w = crate::test_support::test_weights(40);
        assert_eq!(w[10], 0.0);
        let opts = FitOpts {
            pre_standardized: true,
            ..FitOpts::default()
        };
        let fit = |far: f64| {
            let xs = Mat::<f64>::from_fn(40, 6, |i, j| {
                x[(i, j)] * if w[i] == 0.0 { far } else { 2f64.powi(-600) }
            });
            pls1_fit(xs.as_ref(), ys.as_ref(), 2, Some(w.as_ref()), None, opts).unwrap()
        };
        let (want, got) = (fit(2f64.powi(-600)), fit(2f64.powi(600)));
        assert_eq!(got.k_used, 2);
        assert_bits_eq(
            &col_vals(got.beta.as_ref()),
            &col_vals(want.beta.as_ref()),
            "beta",
        );
        assert_bits_eq(
            &col_vals(got.q_loadings.as_ref()),
            &col_vals(want.q_loadings.as_ref()),
            "q",
        );
    }

    /// Each fault gets its documented code (and, for `InvalidWeights`, its
    /// reason); with several faults at once the first check in `pls1_fit`'s
    /// order wins.
    #[test]
    fn pls1_fit_errors_have_documented_codes_in_precedence_order() {
        let (x, y) = signal_data(20, 5, 7);
        let mut bad_x = x.clone();
        bad_x[(3, 2)] = f64::NAN;
        let short_y = Col::<f64>::zeros(19);
        let zero_w = Col::<f64>::zeros(20);
        let neg_w = Col::<f64>::from_fn(20, |i| if i == 4 { -1.0 } else { 1.0 });
        let mut nan_y = y.clone();
        nan_y[6] = f64::NAN;
        let nan_w = Col::<f64>::from_fn(20, |i| if i == 8 { f64::NAN } else { 1.0 });
        let long_w = Col::<f64>::from_fn(21, |_| 1.0);
        // n_eff about 1: k = 2 needs n_eff >= 3.
        let one_row_w = Col::<f64>::from_fn(20, |i| if i == 0 { 1.0 } else { 1e-6 });
        let (x3, y3) = signal_data(3, 5, 7);
        let tiny_x = Mat::<f64>::from_fn(30, 4, |i, j| {
            1e-9 * if (i + j) % 2 == 0 { 1.0 } else { -1.0 }
        });
        let tiny_y = Col::<f64>::from_fn(30, |i| 1e-20 * (i as f64 - 15.0));
        let strict_pre = FitOpts {
            pre_standardized: true,
            ..FitOpts::default()
        };
        let d = FitOpts::default();
        let (ia, nf, iw) = ("invalid_argument", "non_finite_input", "invalid_weights");
        let cases: Vec<(
            &str,
            MatRef<'_, f64>,
            ColRef<'_, f64>,
            usize,
            Option<ColRef<'_, f64>>,
            (FitOpts, Option<usize>),
            &str,
            Option<&str>,
        )> = vec![
            (
                "k = 0",
                x.as_ref(),
                y.as_ref(),
                0,
                None,
                (d, None),
                ia,
                None,
            ),
            (
                "k > d",
                x.as_ref(),
                y.as_ref(),
                6,
                None,
                (d, None),
                "k_exceeds_max",
                None,
            ),
            (
                "keep = 0",
                x.as_ref(),
                y.as_ref(),
                2,
                None,
                (d, Some(0)),
                ia,
                None,
            ),
            (
                "keep > d",
                x.as_ref(),
                y.as_ref(),
                2,
                None,
                (d, Some(6)),
                ia,
                None,
            ),
            (
                "NaN in X",
                bad_x.as_ref(),
                y.as_ref(),
                2,
                None,
                (d, None),
                nf,
                None,
            ),
            (
                "short y",
                x.as_ref(),
                short_y.as_ref(),
                2,
                None,
                (d, None),
                "dimension_mismatch",
                None,
            ),
            (
                "all-zero weights",
                x.as_ref(),
                y.as_ref(),
                2,
                Some(zero_w.as_ref()),
                (d, None),
                iw,
                Some("all_zero"),
            ),
            (
                "negative weight",
                x.as_ref(),
                y.as_ref(),
                2,
                Some(neg_w.as_ref()),
                (d, None),
                iw,
                Some("negative"),
            ),
            (
                "NaN in y",
                x.as_ref(),
                nan_y.as_ref(),
                2,
                None,
                (d, None),
                nf,
                None,
            ),
            (
                "NaN weight",
                x.as_ref(),
                y.as_ref(),
                2,
                Some(nan_w.as_ref()),
                (d, None),
                nf,
                None,
            ),
            (
                "wrong-length weights",
                x.as_ref(),
                y.as_ref(),
                2,
                Some(long_w.as_ref()),
                (d, None),
                iw,
                Some("length_mismatch"),
            ),
            (
                "weight on one row",
                x.as_ref(),
                y.as_ref(),
                2,
                Some(one_row_w.as_ref()),
                (d, None),
                iw,
                Some("insufficient_effective_n"),
            ),
            // Unweighted, n < k + 1. (All-equal weights raise the same:
            // `equal_weights_are_absent_weights_at_every_entry`.)
            (
                "n < k + 1",
                x3.as_ref(),
                y3.as_ref(),
                3,
                None,
                (d, None),
                ia,
                None,
            ),
            // Several faults at once: the first check in `pls1_fit`'s order wins.
            (
                "NaN X + k = 0",
                bad_x.as_ref(),
                y.as_ref(),
                0,
                None,
                (d, None),
                nf,
                None,
            ),
            (
                "short y + NaN X",
                bad_x.as_ref(),
                short_y.as_ref(),
                2,
                None,
                (d, None),
                "dimension_mismatch",
                None,
            ),
            (
                "NaN y + all-zero weights",
                x.as_ref(),
                nan_y.as_ref(),
                2,
                Some(zero_w.as_ref()),
                (d, None),
                nf,
                None,
            ),
            (
                "NaN y + keep = 0",
                x.as_ref(),
                nan_y.as_ref(),
                2,
                None,
                (d, Some(0)),
                nf,
                None,
            ),
            // With `pre_standardized = true`, X is rescaled and y is used as
            // given: y is so far below unit scale that ‖X_a'y_a‖ < 1e-14
            // stops the first component. A top-level call reports the
            // truncation instead of returning a k_used = 0 model.
            (
                "strict truncation",
                tiny_x.as_ref(),
                tiny_y.as_ref(),
                3,
                None,
                (strict_pre, None),
                "invalid_input",
                None,
            ),
        ];
        for (what, xv, yv, k, w, (opts, keep), code, reason) in cases {
            let e = match keep {
                Some(kp) => spls1_fit(xv, yv, k, kp, w, None, opts),
                None => pls1_fit(xv, yv, k, w, None, opts),
            }
            .expect_err(what);
            assert_eq!(e.code(), code, "{what}: {e}");
            if let Some(want) = reason {
                assert!(
                    matches!(&e, PlsKitError::InvalidWeights { reason } if *reason == want),
                    "{what}: {e:?}"
                );
            }
        }
    }

    /// Every public entry that takes weights and a component count, as its
    /// whole output's `Debug` (an `f64` prints its shortest round-trip form,
    /// so equal strings are equal bits) or its error code and message.
    /// `confirmatory` runs one call per listed method.
    fn entry_outcomes(
        x: MatRef<'_, f64>,
        y: ColRef<'_, f64>,
        k: usize,
        w: Option<ColRef<'_, f64>>,
        confirmatory: &[crate::signal_test::ConfirmatoryArgs],
    ) -> Vec<(String, String)> {
        use crate::find_k::{
            FindKOptimalOpts, FindKSequenceOpts, FindKeepOptimalOpts, Selector,
            pls1_find_k_optimal, pls1_find_k_sequence, spls1_find_keep_optimal,
        };
        use crate::perm_null::{PermNullOpts, pls1_perm_null};
        use crate::signal_test::{
            ConfirmatoryTestInput, ConfirmatoryTestOpts, pls1_confirmatory_test,
        };
        fn show<T: std::fmt::Debug>(r: PlsKitResult<T>) -> String {
            match r {
                Ok(v) => format!("ok: {v:?}"),
                Err(e) => format!("{}: {e}", e.code()),
            }
        }
        let seed = Some(5);
        let mut out = vec![
            (
                "pls1_fit".to_owned(),
                show(pls1_fit(x, y, k, w, None, FitOpts::default())),
            ),
            (
                "perm_null".to_owned(),
                show(pls1_perm_null(
                    x,
                    y,
                    k,
                    w,
                    None,
                    PermNullOpts {
                        n_perm: 100,
                        ..Default::default()
                    },
                    seed,
                )),
            ),
            (
                "find_k_sequence".to_owned(),
                show(pls1_find_k_sequence(
                    x,
                    y,
                    k,
                    w,
                    None,
                    FindKSequenceOpts {
                        n_splits: 10,
                        seed,
                        ..Default::default()
                    },
                )),
            ),
            (
                "find_keep_optimal".to_owned(),
                show(spls1_find_keep_optimal(
                    x,
                    y,
                    k,
                    w,
                    None,
                    FindKeepOptimalOpts {
                        seed,
                        ..Default::default()
                    },
                )),
            ),
        ];
        for selector in [Selector::R2Se, Selector::Bic] {
            let r = pls1_find_k_optimal(
                x,
                y,
                k,
                w,
                None,
                FindKOptimalOpts {
                    selector,
                    seed,
                    ..Default::default()
                },
            );
            out.push((format!("find_k_optimal {selector:?}"), show(r)));
        }
        for &args in confirmatory {
            let r = pls1_confirmatory_test(
                ConfirmatoryTestInput::Raw {
                    x,
                    y,
                    k: Some(&[k]),
                    weights: w,
                    covariates: None,
                    null_y: None,
                },
                ConfirmatoryTestOpts {
                    args,
                    seed,
                    ..Default::default()
                },
            );
            out.push((format!("confirmatory {:?}", args.method()), show(r)));
        }
        out
    }

    /// All-equal weights are no weights, at every top-level entry that
    /// takes a component count: at n < k + 1 they raise the unweighted
    /// `invalid_argument` (not `insufficient_effective_n`), and at a
    /// feasible n every output is bit-identical to the call without
    /// weights: `rho_hat`, `n_eff`, the replicate route (n = 40, p = 60
    /// puts `perm_null` on the n-space route) and the BIC penalty included.
    /// 0.3 is there because Kish's ratio of equal non-unit weights rounds
    /// an ulp below n (2.999999999999999 at n = 3).
    #[test]
    fn equal_weights_are_absent_weights_at_every_entry() {
        use crate::signal_test::ConfirmatoryArgs;
        let every_method = [
            ConfirmatoryArgs::RawPerm {
                n_perm: 19,
                n_folds: 5,
            },
            ConfirmatoryArgs::SplitNb {
                n_splits: 10,
                force: false,
            },
            ConfirmatoryArgs::SplitExact {
                n_perm: 19,
                n_splits: 10,
            },
            ConfirmatoryArgs::Score { n_perm: None },
            ConfirmatoryArgs::E,
        ];
        let tiny = signal_data(3, 5, 7);
        let feasible = signal_data(40, 60, 8);
        // The n < k + 1 case runs one method: `raw_perm`'s own n_folds < n
        // floor would fire first and pass for the wrong reason.
        for ((x, y), k, methods, fails) in [
            (&tiny, 3, &every_method[2..3], true),
            (&feasible, 2, &every_method[..], false),
        ] {
            let n = x.nrows();
            let absent = entry_outcomes(x.as_ref(), y.as_ref(), k, None, methods);
            for (entry, got) in &absent {
                let ok = if fails {
                    got.starts_with("invalid_argument: ") && got.contains("insufficient n for k=")
                } else {
                    got.starts_with("ok: ")
                };
                assert!(ok, "{entry}: {got}");
            }
            for c in [1.0, 0.3, 1e6] {
                let w = Col::<f64>::from_fn(n, |_| c);
                let equal = entry_outcomes(x.as_ref(), y.as_ref(), k, Some(w.as_ref()), methods);
                for ((entry, a), (_, e)) in absent.iter().zip(&equal) {
                    assert!(
                        a == e,
                        "n = {n}, w = {c}: {entry}\nabsent: {a}\nequal:  {e}"
                    );
                }
            }
        }
    }

    /// The public entries that take X, besides the PLS1 fits, the PLS3
    /// family (`pls3::tests::pls3_family_agrees_across_layouts`) and the
    /// engines whose own tests pin layout agreement. The Python wrapper
    /// hands each a column-major X: read in place when F-ordered, copied
    /// otherwise.
    /// `preprocess` / `preprocess_block` and the `split_nb` gate give the
    /// same bits on every layout. `pls1_predict`, `factors`, `pca_basis` and
    /// `profiles_find_n` agree to `FIT_LAYOUT_TOL`.
    #[test]
    fn x_taking_entries_agree_across_layouts() {
        use crate::preprocess::{
            PreprocessBlockInput, PreprocessInput, preprocess, preprocess_block,
        };
        use crate::signal_test::split_nb_gate;
        let agree = Agree::Corpus(FIT_LAYOUT_TOL);
        let bits = Agree::Bits;
        for f in copy_free_families() {
            let w = f.w.as_ref().map(Col::as_ref);
            let n = f.x.nrows();
            let yy = Mat::<f64>::from_fn(n, 3, |i, j| f.y[i] * (j + 1) as f64 + f.x[(i, j)]);
            let what = |e: &str| format!("{e} {}", f.name);
            assert_layout_agree(&f.x, &what("preprocess"), bits, |xv| {
                let r = preprocess(PreprocessInput {
                    x: Some(xv),
                    y: Some(f.y.as_ref()),
                    weights: w,
                })?;
                let (xs, m, sd) = r.x_std.unwrap();
                let mut v = mat_vals(xs.as_ref());
                v.extend(col_vals(m.as_ref()));
                v.extend(col_vals(sd.as_ref()));
                Ok(v)
            })
            .unwrap();
            assert_layout_agree(&f.x, &what("preprocess_block"), bits, |xv| {
                let r = preprocess_block(PreprocessBlockInput {
                    x: Some(xv),
                    y: Some(yy.as_ref()),
                    weights: w,
                })?;
                Ok(mat_vals(r.x_std.unwrap().0.as_ref()))
            })
            .unwrap();
            assert_layout_agree(&f.x, &what("split_nb_gate"), bits, |xv| {
                let g = split_nb_gate(xv, w, false)?;
                Ok(vec![f64::from(u8::from(g.fires)), g.stable_rank, g.n_eff])
            })
            .unwrap();
            let model =
                pls1_fit(f.x.as_ref(), f.y.as_ref(), 2, w, None, FitOpts::default()).unwrap();
            assert_layout_agree(&f.x, &what("pls1_predict"), agree, |xv| {
                Ok(col_vals(
                    crate::predict::pls1_predict(&model, xv, None)?.as_ref(),
                ))
            })
            .unwrap();
            assert_layout_agree(&f.x, &what("factors"), agree, |xv| {
                let out = crate::factors::factors(model.coef.as_ref(), xv, 2, false, None)?;
                let mut v = mat_vals(out.f.as_ref());
                v.extend(col_vals(out.size.as_ref()));
                Ok(v)
            })
            .unwrap();
            assert_layout_agree(&f.x, &what("pca_basis"), agree, |xv| {
                let out = crate::pca::pca_basis(xv, false)?;
                let mut v = mat_vals(out.v.as_ref());
                v.extend(col_vals(out.s.as_ref()));
                Ok(v)
            })
            .unwrap();
            assert_layout_agree(&f.x, &what("profiles_find_n"), agree, |xv| {
                let opts = crate::factors::ProfilesFindNOpts {
                    seed: Some(3),
                    ..Default::default()
                };
                let out = crate::factors::profiles_find_n(xv, f.y.as_ref(), 1, 9, opts)?;
                let mut v = out.r2_mean;
                v.extend(out.r2_se);
                #[allow(clippy::cast_precision_loss)]
                v.extend([out.n_star as f64, out.k_star as f64]);
                Ok(v)
            })
            .unwrap();
        }
    }

    /// A zero count is `invalid_argument` at every public entry, with one
    /// message per argument: `"{arg} must be >= 1"` (`k`, `k_max`, and the
    /// keep-counts, which add why). `k_exceeds_max` is for a count above its
    /// maximum only. The `find_k` rows of `entry_outcomes` pass their `k` as
    /// `k_max`; `pls3_confirmatory_test` says this rather than its own
    /// `k = 1` restriction. `score` is rejected too although its statistic
    /// never reads k; `split_nb` is forced so the gate cannot reroute it.
    #[test]
    fn zero_count_is_invalid_argument_at_every_entry() {
        use crate::find_k::{
            FindKOptimalOpts, FindKSequenceOpts, spls1_find_k_optimal, spls1_find_k_sequence,
        };
        use crate::pls3::{Pls3FitOpts, pls3_fit, plssvd_fit, spls3_fit};
        use crate::pls3_signal_test::{Pls3ConfirmatoryTestOpts, pls3_confirmatory_test};
        use crate::signal_test::ConfirmatoryArgs;
        fn show<T>(r: PlsKitResult<T>) -> String {
            match r {
                Ok(_) => "ok".to_owned(),
                Err(e) => format!("{}: {e}", e.code()),
            }
        }
        let (x, y) = signal_data(40, 6, 3);
        let (x, y) = (x.as_ref(), y.as_ref());
        let yy = Mat::<f64>::from_fn(40, 3, |i, j| x[(i, j)] + x[(i, 5 - j)]);
        let yy = yy.as_ref();
        let every_method = [
            ConfirmatoryArgs::RawPerm {
                n_perm: 19,
                n_folds: 5,
            },
            ConfirmatoryArgs::SplitNb {
                n_splits: 10,
                force: true,
            },
            ConfirmatoryArgs::SplitExact {
                n_perm: 19,
                n_splits: 10,
            },
            ConfirmatoryArgs::Score { n_perm: None },
            ConfirmatoryArgs::E,
        ];
        let seed = Some(5);
        let fk = || FindKOptimalOpts {
            seed,
            ..Default::default()
        };
        let fs = || FindKSequenceOpts {
            n_splits: 10,
            seed,
            ..Default::default()
        };
        let p3 = Pls3FitOpts::default;
        let p3c = |keep_x, keep_y| Pls3ConfirmatoryTestOpts {
            args: ConfirmatoryArgs::SplitExact {
                n_perm: 19,
                n_splits: 6,
            },
            seed,
            keep_x,
            keep_y,
            ..Default::default()
        };
        let mut rows: Vec<(String, &str, String)> = entry_outcomes(x, y, 0, None, &every_method)
            .into_iter()
            .map(|(entry, got)| {
                let arg = if entry.starts_with("find_k_") {
                    "k_max"
                } else {
                    "k"
                };
                (entry, arg, got)
            })
            .collect();
        let d = FitOpts::default();
        let mut row = |entry: &str, arg: &'static str, got: String| {
            rows.push((entry.to_owned(), arg, got));
        };
        row("spls1_fit", "k", show(spls1_fit(x, y, 0, 3, None, None, d)));
        row(
            "spls1_fit keep",
            "keep",
            show(spls1_fit(x, y, 2, 0, None, None, d)),
        );
        row(
            "spls1_find_k_optimal",
            "k_max",
            show(spls1_find_k_optimal(x, y, 0, 3, None, None, fk())),
        );
        row(
            "spls1_find_k_optimal keep",
            "keep",
            show(spls1_find_k_optimal(x, y, 2, 0, None, None, fk())),
        );
        row(
            "spls1_find_k_sequence",
            "k_max",
            show(spls1_find_k_sequence(x, y, 0, 3, None, None, fs())),
        );
        row(
            "spls1_find_k_sequence keep",
            "keep",
            show(spls1_find_k_sequence(x, y, 2, 0, None, None, fs())),
        );
        row("pls3_fit", "k", show(pls3_fit(x, yy, 0, None, p3())));
        row("plssvd_fit", "k", show(plssvd_fit(x, yy, 0, None, p3())));
        row(
            "spls3_fit",
            "k",
            show(spls3_fit(x, yy, 0, 2, 2, None, p3())),
        );
        row(
            "spls3_fit keep_x",
            "keep_x",
            show(spls3_fit(x, yy, 1, 0, 2, None, p3())),
        );
        row(
            "spls3_fit keep_y",
            "keep_y",
            show(spls3_fit(x, yy, 1, 2, 0, None, p3())),
        );
        row(
            "pls3_confirmatory_test",
            "k",
            show(pls3_confirmatory_test(x, yy, 0, p3c(None, None))),
        );
        row(
            "pls3_confirmatory_test keep_x",
            "keep_x",
            show(pls3_confirmatory_test(x, yy, 1, p3c(Some(0), None))),
        );
        row(
            "pls3_confirmatory_test keep_y",
            "keep_y",
            show(pls3_confirmatory_test(x, yy, 1, p3c(None, Some(0)))),
        );
        for (entry, arg, got) in rows {
            let want = format!("invalid_argument: invalid argument: {arg} must be >= 1");
            assert!(got.starts_with(&want), "{entry}: {got}");
        }
    }

    #[test]
    fn prepared_tail_on_hoisted_scaling_is_pls1_fit() {
        for f in copy_free_families() {
            let Some(w_raw) = f.w.as_ref() else {
                continue;
            };
            // The caller's weights as `perm_null` and the CV folds hold them.
            let wn = normalize_weights(w_raw.as_ref()).unwrap();
            let (xs, _, _) = standardize_weighted(f.x.as_ref(), Some(wn.as_ref()));
            let (ys, _, _) = standardize1_weighted(f.y.as_ref(), Some(wn.as_ref()));
            let sqw = fit_row_scale(wn.as_ref());
            let xs_fit = scale_rows(xs.as_ref(), sqw.as_ref());
            let ys_fit = scale_col(ys.as_ref(), sqw.as_ref());
            for i in 0..xs.nrows() {
                assert_eq!(xs_fit[(i, 1)].to_bits(), (sqw[i] * xs[(i, 1)]).to_bits());
                assert_eq!(ys_fit[i].to_bits(), (sqw[i] * ys[i]).to_bits());
            }
            // Taken once for the block, as `perm_null` and the CV folds do.
            let x_fro = xs_fit.norm_l2();
            for k in [1_usize, 3] {
                for keep in [None, Some(3)] {
                    let opts = FitOpts {
                        pre_standardized: true,
                        par: ParChoice::Seq,
                    };
                    let direct = pls1_fit_impl(
                        xs.as_ref(),
                        ys.as_ref(),
                        k,
                        Some(wn.as_ref()),
                        None,
                        opts,
                        InternalFitOpts {
                            check_n_eff: false,
                            keep,
                        },
                    )
                    .unwrap();
                    let hoisted = pls1_fit_prepared_fro(
                        xs_fit.as_ref(),
                        ys_fit.as_ref(),
                        k,
                        keep,
                        ParChoice::Seq,
                        x_fro,
                    );
                    let what = format!("{} k={k} keep={keep:?}", f.name);
                    assert_eq!(direct.k_used, hoisted.k_used, "{what}");
                    assert_bits_eq(
                        &mat_vals(direct.t_scores.as_ref()),
                        &mat_vals(hoisted.t_scores.as_ref()),
                        &what,
                    );
                    assert_bits_eq(
                        &mat_vals(direct.w_star.as_ref()),
                        &mat_vals(hoisted.w_star.as_ref()),
                        &what,
                    );
                    assert_bits_eq(
                        &col_vals(direct.coef.as_ref()),
                        &col_vals(hoisted.coef.as_ref()),
                        &what,
                    );
                    // pre_standardized: beta is coef.
                    assert_bits_eq(
                        &col_vals(direct.beta.as_ref()),
                        &col_vals(hoisted.coef.as_ref()),
                        &what,
                    );
                }
            }
        }
    }
}

/// `pls1_fit` and `spls1_fit` with covariates: the fit on the adjusted data,
/// the raw-scale coefficients, and the covariate errors.
#[cfg(test)]
#[allow(clippy::many_single_char_names)]
mod covariate_tests {
    use super::*;
    use crate::covariates::adjust;
    use crate::test_support::{
        assert_close_rel, assert_mat_bits_eq, signal_data, test_weights, unif, unif_col,
    };

    /// `(x, y, z)`: X offset from zero, y driven by X and by the first
    /// covariate, Z with two iid columns.
    fn data(n: usize) -> (Mat<f64>, Col<f64>, Mat<f64>) {
        let (x, y) = signal_data(n, 8, 3);
        let x = Mat::<f64>::from_fn(n, 8, |i, j| x[(i, j)] + 3.0);
        let z = unif(n, 2, 4);
        let y = Col::<f64>::from_fn(n, |i| 2.0 * y[i] + 1.5 * z[(i, 0)] + 10.0);
        (x, y, z)
    }

    fn fit(
        x: MatRef<'_, f64>,
        y: ColRef<'_, f64>,
        w: Option<&Col<f64>>,
        z: Option<MatRef<'_, f64>>,
        pre_standardized: bool,
    ) -> PlsKitResult<Pls1Model> {
        pls1_fit(
            x,
            y,
            2,
            w.map(Col::as_ref),
            z,
            FitOpts {
                pre_standardized,
                ..FitOpts::default()
            },
        )
    }

    #[test]
    fn fit_with_covariates_equals_fit_on_adjusted_data() {
        let n = 60;
        let (x, y, z) = data(n);
        for w in [None, Some(test_weights(n))] {
            let m = fit(x.as_ref(), y.as_ref(), w.as_ref(), Some(z.as_ref()), false).unwrap();
            let (wn, _) = validate_and_normalize_weights(w.as_ref().map(Col::as_ref), n).unwrap();
            let adj = adjust(
                x.as_ref(),
                y.as_ref(),
                z.as_ref(),
                wn.as_ref().map(Col::as_ref),
                false,
            )
            .unwrap();
            // The weighted fit scales the adjusted rows in place: the same bits.
            let o = fit(adj.x.as_ref(), adj.y.as_ref(), w.as_ref(), None, true).unwrap();
            assert_mat_bits_eq(m.coef.as_mat(), o.coef.as_mat(), "coef");
            assert_mat_bits_eq(m.t_scores.as_ref(), o.t_scores.as_ref(), "t_scores");
            assert_mat_bits_eq(m.w_star.as_ref(), o.w_star.as_ref(), "w_star");
            assert_mat_bits_eq(m.q_loadings.as_mat(), o.q_loadings.as_mat(), "q_loadings");
            assert_eq!(m.weights, o.weights);
            assert_eq!(m.n_eff.to_bits(), o.n_eff.to_bits());
            assert_eq!(m.n_covariates, Some(2));
            assert_eq!(m.covariate_coef.as_ref().map(Col::nrows), Some(2));
            assert!(!m.pre_standardized);
            assert!(o.covariate_coef.is_none() && o.n_covariates.is_none());
        }
    }

    #[test]
    fn fit_with_aliased_columns_equals_fit_without() {
        let n = 60;
        let (x, y, z) = data(n);
        // [1, z0, 5, z0, z1]: intercept, constant and duplicate dropped.
        let z_alias = Mat::<f64>::from_fn(n, 5, |i, j| match j {
            0 => 1.0,
            1 | 3 => z[(i, 0)],
            2 => 5.0,
            _ => z[(i, 1)],
        });
        for w in [None, Some(test_weights(n))] {
            let clean = fit(x.as_ref(), y.as_ref(), w.as_ref(), Some(z.as_ref()), false).unwrap();
            let alias = fit(
                x.as_ref(),
                y.as_ref(),
                w.as_ref(),
                Some(z_alias.as_ref()),
                false,
            )
            .unwrap();
            assert_close_rel(alias.beta.as_mat(), clean.beta.as_mat(), 1e-12, "beta");
            assert_close_rel(
                alias.t_scores.as_ref(),
                clean.t_scores.as_ref(),
                1e-12,
                "t_scores",
            );
            assert!((alias.intercept - clean.intercept).abs() <= 1e-12 * clean.intercept.abs());
            let (ca, cc) = (alias.covariate_coef.unwrap(), clean.covariate_coef.unwrap());
            let want = Col::<f64>::from_fn(5, |j| match j {
                1 => cc[0],
                4 => cc[1],
                _ => 0.0,
            });
            assert_close_rel(ca.as_mat(), want.as_mat(), 1e-12, "covariate_coef");
            assert_eq!((ca[0], ca[2], ca[3]), (0.0, 0.0, 0.0));
            assert_eq!(alias.n_covariates, Some(2));
        }
    }

    #[test]
    fn spls1_fit_takes_covariates() {
        let n = 60;
        let (x, y, z) = data(n);
        let m = spls1_fit(
            x.as_ref(),
            y.as_ref(),
            2,
            3,
            None,
            Some(z.as_ref()),
            FitOpts::default(),
        )
        .unwrap();
        assert_eq!(m.keep, Some(3));
        assert_eq!(m.n_covariates, Some(2));
    }

    /// A constant covariate is only the intercept, which the plain fit
    /// already removes: an independent check of `adjust`'s standardization.
    #[test]
    #[allow(clippy::float_cmp)] // an aliased covariate gets a coefficient of exactly 0
    fn fit_with_a_constant_covariate_equals_plain_fit() {
        let n = 60;
        let (x, y, _) = data(n);
        let z = Mat::<f64>::from_fn(n, 1, |_, _| 4.0);
        for w in [None, Some(test_weights(n))] {
            let m = fit(x.as_ref(), y.as_ref(), w.as_ref(), Some(z.as_ref()), false).unwrap();
            let o = fit(x.as_ref(), y.as_ref(), w.as_ref(), None, false).unwrap();
            assert_close_rel(m.beta.as_mat(), o.beta.as_mat(), 1e-12, "beta");
            assert_close_rel(m.t_scores.as_ref(), o.t_scores.as_ref(), 1e-12, "t_scores");
            assert!((m.intercept - o.intercept).abs() <= 1e-12 * o.intercept.abs());
            assert_eq!(m.n_covariates, Some(0));
            assert_eq!(m.covariate_coef.unwrap()[0], 0.0);
        }
    }

    /// `n − rank = k + 1` is accepted, `n − rank = k` refused.
    #[test]
    fn check_df_boundary() {
        let k = 2;
        let z = unif(10, 7, 14);
        let ok = crate::covariates::CovariateBasis::new(z.as_ref().subcols(0, 6), None).unwrap();
        assert_eq!(ok.rank(), 7);
        assert!(ok.check_df(k).is_ok());
        let tight = crate::covariates::CovariateBasis::new(z.as_ref(), None).unwrap();
        assert_eq!(tight.rank(), 8);
        assert!(matches!(
            tight.check_df(k),
            Err(PlsKitError::InsufficientResidualDf {
                n: 10,
                rank: 8,
                k: 2
            })
        ));
    }

    /// Zero-weight rows add no residual degrees of freedom: 10 rows weighted
    /// 1 and 40 weighted 0 are refused as the 10 rows alone are.
    #[test]
    fn check_df_counts_rows_with_positive_weight() {
        let z = unif(50, 9, 15);
        let w = Col::<f64>::from_fn(50, |i| if i < 10 { 1.0 } else { 0.0 });
        let weighted =
            crate::covariates::CovariateBasis::new(z.as_ref(), Some(w.as_ref())).unwrap();
        let alone =
            crate::covariates::CovariateBasis::new(z.as_ref().subrows(0, 10), None).unwrap();
        for basis in [weighted, alone] {
            assert!(matches!(
                basis.check_df(2),
                Err(PlsKitError::InsufficientResidualDf {
                    n: 10,
                    rank: 10,
                    k: 2
                })
            ));
        }
    }

    /// y in the span of `[1, Z]` gives what a constant y gives without
    /// covariates: the zero model, and `split_exact` with p = 1, statistic 0.
    #[test]
    fn y_in_the_span_of_the_covariates_is_a_constant_y() {
        use crate::signal_test::{
            ConfirmatoryArgs, ConfirmatoryTestInput, ConfirmatoryTestOpts, pls1_confirmatory_test,
        };
        let n = 200;
        let x = unif(n, 30, 16);
        let z = unif(n, 2, 17);
        let y_span = Col::<f64>::from_fn(n, |i| 3.0 + 2.0 * z[(i, 0)] - z[(i, 1)]);
        let y_const = Col::<f64>::from_fn(n, |_| 3.0);
        for (y, z) in [(&y_span, Some(z.as_ref())), (&y_const, None)] {
            let m = fit(x.as_ref(), y.as_ref(), None, z, false).unwrap();
            assert_eq!(m.k_used, 0);
            let t = pls1_confirmatory_test(
                ConfirmatoryTestInput::Raw {
                    x: x.as_ref(),
                    y: y.as_ref(),
                    k: Some(&[1]),
                    weights: None,
                    covariates: z,
                    null_y: None,
                },
                ConfirmatoryTestOpts {
                    args: ConfirmatoryArgs::SplitExact {
                        n_perm: 19,
                        n_splits: 4,
                    },
                    seed: Some(18),
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!((t.pvalue, t.statistic.unwrap()), (1.0, 0.0));
        }
    }

    /// Under `pre_standardized` the fit takes X and y as given: on X
    /// standardized by hand it predicts what the plain fit predicts on raw X.
    #[test]
    fn pre_standardized_fit_with_covariates_predicts_as_the_plain_fit() {
        let n = 60;
        let (x, y, z) = data(n);
        let y = Col::<f64>::from_fn(n, |i| 5.0 * y[i] + 3.0);
        let xs = Mat::<f64>::from_fn(n, x.ncols(), |i, j| {
            let c = x.col(j);
            let mean = (0..n).map(|r| c[r]).sum::<f64>() / n as f64;
            let sd = ((0..n).map(|r| (c[r] - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt();
            (c[i] - mean) / sd
        });
        let zr = Some(z.as_ref());
        let flagged = fit(xs.as_ref(), y.as_ref(), None, zr, true).unwrap();
        let plain = fit(x.as_ref(), y.as_ref(), None, zr, false).unwrap();
        let got = crate::predict::pls1_predict(&flagged, xs.as_ref(), zr).unwrap();
        let want = crate::predict::pls1_predict(&plain, x.as_ref(), zr).unwrap();
        for i in 0..n {
            assert!(
                (got[i] - want[i]).abs() <= 1e-10 * want[i].abs().max(1.0),
                "row {i}: {} vs {}",
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn covariate_errors() {
        let n = 40;
        let (x, y, z) = data(n);
        let short = z.as_ref().subrows(0, n - 1);
        assert!(matches!(
            fit(x.as_ref(), y.as_ref(), None, Some(short), false),
            Err(PlsKitError::ShapeMismatch(_))
        ));
        let mut bad = z.clone();
        bad[(5, 1)] = f64::NAN;
        assert!(matches!(
            fit(x.as_ref(), y.as_ref(), None, Some(bad.as_ref()), false),
            Err(PlsKitError::NonFiniteInput)
        ));
        let x10 = unif(10, 5, 11);
        let y10 = unif_col(10, 12);
        let z10 = unif(10, 8, 13);
        let r = fit(x10.as_ref(), y10.as_ref(), None, Some(z10.as_ref()), false);
        assert!(
            matches!(
                r,
                Err(PlsKitError::InsufficientResidualDf {
                    n: 10,
                    rank: 9,
                    k: 2
                })
            ),
            "{r:?}"
        );
    }
}
