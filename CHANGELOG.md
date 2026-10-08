# Changelog

All notable changes to this project will be documented here.

## [Unreleased]

- Changed (performance): the `split_nb` auto-gate takes the stable rank
  from the Gram matrix on the shorter side of the standardized `X`, not
  from an SVD of it, so the gate holds one copy of `X` where it held two.
  `stable_rank` differs from the previous release in the last bits. The
  testdata corpus is unchanged.
- Changed (performance): `pls3_fit`, `plssvd_fit` and `spls3_fit` no longer
  write a standardized copy of `X`: `X'Y` and the X scores are formed from
  the raw `X`, read in its own memory layout (a C-ordered NumPy array in
  place), with the centering and scaling applied inline, as `pls1_fit` /
  `spls1_fit` do. When a column's |mean| exceeds 1000 times its scale, `X`
  is standardized into a copy as before. A fit in which one of the `k`
  requested components has its `σ` below `2·(1 + max |mean| / scale)` times
  the truncation floor is refitted on a standardized copy, which costs
  about two fits and the copy's memory. The truncation floor takes `‖X‖_F`
  from the standardization moments. `Y` is standardized into a copy as
  before, and a `pre_standardized_X=True` fit is unchanged to the bit.
- Changed (numerics): `pls3_fit`, `plssvd_fit` and `spls3_fit` without
  `pre_standardized_X` are no longer bit-identical across memory layouts of
  `X`; row- and column-major inputs agree to rounding (the corpus
  tolerance). Their results also differ from the previous release in the
  last bits. The testdata corpus is unchanged.
- Added: `test_method="auto"`, now the default of `pls1_confirmatory_test`,
  `pls3_confirmatory_test`, `pls1_find_k_sequence` and
  `spls1_find_k_sequence`, in Rust, Python, R and Julia. It runs
  `split_exact` or `split_nb`, chosen once per call from `X` and the weights,
  and `result.test_method` reports the method that ran. The two confirmatory
  tests no longer require `test_method`.
- Changed (breaking, Rust): the `Default` of `ConfirmatoryTestOpts` and
  `Pls3ConfirmatoryTestOpts` changes from `SplitExact` to `Auto`, so large
  designs now run `split_nb` by default.
- Changed (breaking): the default `test_method` of `pls1_find_k_sequence` and
  `spls1_find_k_sequence` changes from `split_nb` to `"auto"`, which can
  change the method that runs by default. `pls1_fit(k="sequence")` inherits
  it.
- Changed: `spls1_find_k_sequence` runs `split_exact` by default on designs
  where `"auto"` picks it (`n_eff < 250`, or the `split_nb` auto-gate
  fires). That is a refit route, about `(n_perm + 1) * n_splits` fits per step
  instead of `n_splits`.
- Changed (breaking): `disable_parallelism` is removed from every function,
  in Rust, Python, R and Julia. Set `PLSKIT_NUM_THREADS` to cap the thread
  count; `PLSKIT_NUM_THREADS=1` runs on one core. Results do not depend on
  the setting. The testdata manifest drops the `disable_parallelism` key;
  numerical outputs are unchanged.
- Changed (breaking): `pls1_confirmatory_test` and `pls3_confirmatory_test`
  take `test_method=` instead of `method=`, and their result reports
  `result.test_method` instead of `result.method`, in Python, R and Julia.
  The Rust `ConfirmatoryTestOutput.method` field is now `test_method`. This
  matches the `test_method` argument of `pls1_find_k_sequence` and
  `spls1_find_k_sequence`. There is no alias for the old name. Error
  messages that name the argument say `test_method=` too. The testdata
  corpus renames the `method` output entry and manifest kwargs key of the
  confirmatory cases to `test_method`; numerical outputs are unchanged.
- Changed (breaking, Rust): `linalg::standardize_weighted` and
  `linalg::standardize1_weighted` panic when `weights` is `Some` and its
  length differs from the row count. They used to read the first `n`
  entries of a longer `weights`.
- Added: the `plskit-bind` workspace crate, a language-neutral binding
  layer over the plskit engine. It exposes the whole Python surface
  through one `call(fn_name, Record)` entry point. The R wrapper calls the
  engine through it; the Julia wrapper uses only its registry dump, to
  render its stubs.
- Changed (Python, stricter input): `pls1_fit`, `spls1_fit`,
  `spls1_find_keep_optimal`, `spls1_find_k_optimal` and
  `spls1_find_k_sequence` reject a `k` or `keep` that is not a non-negative
  whole number with `PlsKitError(code="invalid_argument")` instead of
  coercing it: a fractional float (`k=2.7` used to fit 2 components) or a
  `bool` now raises where it used to coerce; a negative value already
  raised, but as a PyO3 `OverflowError` rather than a `PlsKitError`, and now
  raises `PlsKitError(code="invalid_argument")` instead. A whole float such
  as `2.0` is still accepted. `pls1_fit` with an integer `k` rejects a
  `k_max` or `find_k_args` argument instead of ignoring it; for
  `spls1_find_k_optimal` and `spls1_find_k_sequence`, only `keep` is
  checked this way, not `k_max`. All of this matches the rules the R
  wrapper applies through `plskit-bind`; the Julia wrapper inherits them
  from the Python package.
- Changed (Python): the extension calls the engine through `plskit-bind`.
  Newly accepted: a plain `dict` holding a model's fields as `model` (a
  dict missing a field raises `invalid_argument`); a decimal
  string as `seed` (`seed="7"` is seed 7); a 0-D array where a scalar is
  expected (`pre_standardized=np.array(False)`,
  `test_method=np.array("score")`), also inside `args` / `rotation_args`.
  Newly rejected, with
  `PlsKitError(code="invalid_argument")`: a number type with no plain
  int / float form (`Decimal`, `Fraction`) inside `args` / `rotation_args`;
  `rotate(model)` with a `P` whose row count differs from `W`'s, or with a
  `None` or mistyped field the rotation does not use (`coef=None`);
  `pls1_predict` and `pls3_transform` on a model with a mistyped field
  (`pre_standardized=1`, `intercept="3.5"`) or an unconvertible value in a
  field they do not use (`selection_result=object()`); a subclass of a
  result class as `model`; a model field that is a nested list
  (`T=m.T.tolist()`) or a flat list holding a float (`Q=m.Q.tolist()`) in
  `rotate`, `pls1_predict` and `pls3_transform`; an int of `2**64` or
  more, or below `-2**63`, as a float argument (`tol=2**64`,
  `tol=-2**63 - 1`). Error codes: a wrong-typed value inside
  `find_k_args` is `invalid_args` (was `invalid_argument`); non-string keys
  in `args` / `rotation_args` / `find_k_args`, and any other value in
  `args` / `rotation_args` that cannot be converted (`object()`,
  `{'n_perm': 2**64}`), are `invalid_argument` (was `invalid_args`);
  `rotate(model)` with `T`, `P` or `Q` not matching `W`'s column count
  raises `PlsKitError` with `invalid_argument` (was numpy's `ValueError`);
  a Rust panic raises `PlsKitError` with `internal` (was `PanicException`).
  Several error messages are now worded as `plskit-bind` words them.
  `rotate(model)` forms `T·R`, `P·R` and `Rᵀ·Q` in faer instead of numpy,
  so results agree to rounding, not bit for bit. `rotate(model)` returns
  new, equal objects for the fields it does not rotate
  (`selection_result`, `coef`, `beta`) instead of the caller's objects.
  The reroute warning from `pls1_find_k_optimal`, `pls1_find_k_sequence`
  and `pls1_fit(k="optimal" | "sequence")` now points at the caller's
  line.
- Added (R): the R package `plskit` (`plskit-r/`) with the whole Python
  surface: the same 20 functions, argument names and result fields,
  through the `plskit-bind` layer. Results are classed named lists
  (`c("pls1_result", "plskit_result")`, ...), seeds are decimal strings,
  and errors are `plskit_error` conditions carrying the Python error
  codes. It passes the shared `testdata/` corpus. Until `plskit-bind` is
  on crates.io it builds only from a checkout, with `PLSKIT_CARGO_CONFIG`
  naming a cargo config that patches in the in-repo crates
  (`_docs/r/installation.md`). The package version moves from 0.0.1 to
  0.7.0, the engine version. The placeholder `version()` export, which
  masked `base::version`, is removed.
- Added (Julia): the Julia package `PLSKit.jl` (`plskit-jl/`) with the
  whole Python surface: it runs the Python `plskit` package through
  PythonCall.jl, so every function, argument name and result field is the
  Python one. Results are `PlsKitResult{T}` values with one alias per
  result type (`PLS1Result`, ...), errors are a single `PlsKitError`
  carrying Python's `code` as a `Symbol`, and Python warnings are
  re-emitted with `@warn`. Its tests check one `testdata/` fixture per
  function at the corpus tolerances; the numbers come from the Python
  wheel, which passes the whole corpus. The
  package and module are renamed from `plskit` to `PLSKit` (UUID kept),
  version 0.7.0, pinning PyPI `plskit==0.7.0` through `CondaPkg.toml`.
  The placeholder `version()` export is removed.
  - The Julia package version, its `CondaPkg.toml` pin and
    `PLSKIT_PY_VERSION` are always equal and name the `plskit-py` release
    the package runs on; Julia may lag Python, never lead it. CI builds
    `plskit-py` from source when the checkout matches the pin and installs
    the pinned wheel from PyPI when the checkout is newer
    (`scripts/julia-python-pin.py`); the `-jl` release job tests against
    PyPI only.
- Added (CI): `scripts/check-profile-sync.py` keeps the R crate's
  `[profile.release]`, `[workspace]` table and dependency versions in
  step with the workspace; `scripts/render-r-stubs.py --check` and
  `scripts/render-jl-stubs.py --check` fail when the committed R stubs or
  Julia stubs and aliases drift from the `plskit-bind` registry;
  `scripts/check-docs-drift.py` also checks the R `NAMESPACE` exports and
  the Julia `export`s against the Python surface. The R package is checked
  with `R CMD check` on Linux, macOS and Windows, and the Julia tests run
  on the same three OSes (Julia LTS and current).
- Fixed: `pls1_confirmatory_test` confidence intervals (normal-theory
  leverage CI and NB-Wald `holdout_corr` CI) used a wrong inverse normal.
  Below `level` 0.85 the critical value was off by up to 20%
  (Φ⁻¹(0.9) gave 1.0253, not 1.2816): CIs were too narrow near level 0.8
  and too wide near level 0.5. At the default level 0.95 the bounds move by
  about 2e-10 relative. `standard_normal_inv` now matches Wichura's AS241
  in every branch, including a far-tail coefficient typo that the public
  API cannot reach.
- Fixed: a column of `X` (or `y`, or a column of `Y`) that is constant to
  rounding now gets scale `max(1, |mean|)` instead of `1`, so it
  standardizes to zeros up to rounding at any magnitude. Before, a constant
  column near `1e300` truncated `pls1_fit` to `k_used = 0` or gave NaN
  coefficients and made `pls3_fit` fail with "SVD of X'Y failed to
  converge", and one near `1e20` silently shifted the other `pls1_fit`
  coefficients. `X_scale` / `Y_scale` (from `preprocess` and PLS3 fits) now
  report `|mean|` for such a column when `|mean| > 1`; constants of
  magnitude at most 1 keep scale 1. Such columns no longer force
  `pls1_fit` onto its materialized-copy route.
- Fixed: a column whose standard deviation underflows (zero or subnormal,
  e.g. a single `5e-324` among zeros) is treated as constant (only
  centered) instead of getting scale `0` or a subnormal scale and a silent
  all-NaN `pls1_fit` or a failed `pls3_fit`.
- Fixed: all-equal observation weights are now identical to absent weights
  at every entry: results are bit-identical to `weights=None` (including
  `rho_hat` on `split_nb`, the `split_nb` auto-gate, the replicate route,
  BIC, and `n_eff = n` exactly), and an `n_eff < k + 1` failure raises
  `invalid_argument` rather than `invalid_weights` /
  `insufficient_effective_n`. Before, equal non-unit weights could round
  `n_eff` just below `n`, rejecting a feasible `n = k + 1` call or firing
  the `split_nb` gate at `n = 25`. The same holds inside
  `pls1_confirmatory_test`: a `raw_perm` fold or a `split_exact` /
  `split_nb` half whose own weights are all equal (or all zero) trains
  unweighted, as `pls1_fit` does on those rows.
- Fixed: `pls1_confirmatory_test` rejects `k = 0` with `invalid_argument`
  ("k must be >= 1") for every method; `split_exact`, `split_nb` and
  `score` used to return a p-value.
- Fixed (Python): `preprocess` with a 2-D `Y` containing NaN or infinity
  raises `non_finite_input`, like a 1-D `y`. Validation and
  standardization of a multi-column `Y` moved into the core
  (`plskit::preprocess::preprocess_block`); outputs for finite input are
  bit-identical.
- Fixed (Python): an unknown `method`, `selector`, `diagnostic` or
  `test_method` raises `PlsKitError(code="invalid_args")` (the `code` used
  to be empty). Model-dict fields missing on `pls1_predict` /
  `pls3_transform` raise `invalid_argument` instead of an uncoded error or
  a panic.
- Fixed (Python): an unusable argument value no longer escapes as a raw
  `TypeError` / `OverflowError`: a negative, fractional or non-numeric
  count, a seed outside `[0, 2^64)`, a non-number float option, a non-bool
  flag, a non-string method name or a non-dict `args`. Inside an `args` /
  `rotation_args` dict the error is `invalid_args`, for a top-level
  argument `invalid_argument`; messages match `plskit-bind`.
- Changed (Python): argument rules now follow `plskit-bind`. `None` for an
  optional argument or an args-dict key means its default, a whole float
  such as `100.0` is accepted as a count, and a `bool` is rejected where a
  number is expected. `pls1_rotation_stability` parses `rotation_args` with
  `rotate`'s parser, and `pls1_fit` validates `seed` when `k` is an int.
- Changed (Python): `RotationSpec.args` records the varimax arguments the
  engine resolved (defaults filled, counts as `int`) instead of echoing the
  caller's dict. The private `_plskit.VARIMAX_DEFAULTS` is removed.
- Changed (Python): `ConfirmatoryTestResult` fields are ordered `n_eff,
  rho_hat, stable_rank`, matching `results.md` and the R/Julia wrappers;
  `scripts/check-docs-drift.py` now fails when a Python result dataclass
  orders its fields differently from `results.md`.
- Changed (testing): corpus comparisons (Rust, Python, R, Julia, bind and
  the generator's settle step) add a relative term, `|a - e| <= atol +
  1e-14 * |e|` for finite `e` (an infinity equals only itself), so
  cross-host drift of an ulp or two in large values (the ~6126 score
  statistic) no longer sits at the edge of `atol = 1e-12`.
- Testdata: regenerated the three `pls1_confirmatory_test` `split_nb` CI
  fixtures (AS241 fix) and added `pls1_confirmatory_split_nb_ci_level80`,
  the first CI fixture off the default level, which pins the level-0.8
  critical value and fails a wrapper that drops `level`.
  `producing_version` is 0.7.0. The manifest drops the per-case `tolerance`
  field, which no corpus reader used; the tolerances are in
  `testdata/README.md` ("Tolerance").
- Docs: `length_mismatch` is listed as an `invalid_weights` reason; the
  `split_nb` docstrings state the actual auto-gate rule (at most 4 columns,
  `n_eff < 25`, or stable rank < 3);
  `RotationStabilityResult.degenerate_baseline` no longer claims
  `variance_ratio.point` is always NaN when set (only the `V_unrot = 0`
  trigger gives NaN).
- Fixed: `k = 0` and `k_max = 0` raise `invalid_argument` ("k must be >= 1"
  / "k_max must be >= 1") at every entry, in every wrapper.
  `pls1_rotation_stability`, the `find_k` family, `spls1_find_keep_optimal`
  and `pls1_fit(k="optimal"|"sequence")` raised `k_exceeds_max`,
  `pls1_perm_null` used a different message, and `pls3_confirmatory_test`
  reported its `k = 1` restriction. `k_exceeds_max` now means only a count
  above its maximum.
- Fixed: `preprocess` reports `n_eff` exactly `n` for all-equal weights, as
  every fit does (Kish's ratio could round an ulp below `n`).
  `weights_normalized` is still returned.
- Fixed: `rotate` on a one-column `W` (K = 1) reports `V_converged` for the
  target the K >= 2 path starts from: `L` if given, else `W`, row-normalized
  under `kaiser_normalize`. It used to skip the row normalization, so with
  the default `kaiser_normalize=True` it reported the criterion of the raw
  column. `W_rot`, `R` and `sweeps` are unchanged.
- Fixed (Python): an array argument numpy cannot read as real numbers
  (strings, including numeric strings, ragged nested lists, non-numeric
  objects, complex, datetime or structured dtypes) raises
  `PlsKitError(code="invalid_argument")` naming the argument, as
  `plskit-bind` does, instead of numpy's `ValueError` / `TypeError`; a
  complex array used to be cast silently to its real part. `None` and
  `pd.NA` in an object array read as NaN (`non_finite_input`). A 0-d
  `weights` or `Y` is rejected as not 1-D / 2-D.
- Changed (Python, performance and numerics): a C- or F-ordered float64 `X`
  reaches the engine without a copy at every entry that takes one. This
  reverses the 0.6.1 rule that the Python API passes every `X` on in C
  order: an F-ordered `X` no longer pays a full copy (about 25 to 40 ms per
  fit at 1e7 entries), and results for different memory layouts of the same
  values agree to rounding (corpus tolerance) rather than bit for bit.
  `pls1_fit`, `spls1_fit`, `pls1_predict`, `pls1_rotation_stability` and,
  without `pre_standardized_X`, `pls3_fit`, `plssvd_fit` and `spls3_fit` can
  differ in the last bits between C and F order; the same array in the same
  layout still gives byte-identical results across runs and thread counts.
  Julia arrays are column-major and now read in place, so Julia
  `pls1_fit` / `spls1_fit` results can differ from 0.6.2 in the last bits
  (they now match the Rust core's column-major bits).
- Changed (performance): the permutation and split loops take `‖X̃‖_F` once
  per block instead of once per replicate on their primal route
  (`pls1_perm_null`, the `raw_perm` CV folds, the `split_exact` refit
  splits). On one core at 1000 × 1000 with 1000 permutations, a k = 1
  `pls1_perm_null` drops from 1.41 s to 0.66 s and a k = 1 `raw_perm` from
  5.86 s to 2.88 s, so k = 1 is again faster than k = 2 at that shape.
  Results are bit-identical.
- Changed (internal): the p-space Gram backend's per-replicate diagnostics
  are compiled only into tests; production replicates no longer fill or
  allocate them. Results are unchanged.
- Fixed: `pls3_fit`, `plssvd_fit`, `spls3_fit`, `preprocess` and
  `preprocess_block` raise `invalid_argument` ("insufficient n: need
  n >= 1") on zero rows, as `pls1_fit` does. They used to return NaN
  standardization moments, the PLS3 fits with `k_used = 0`. One row is
  still not an error for the PLS3 family: its `k` is not bounded by `n`, so
  the fit truncates.
- Fixed (numerics): `pls3_fit`, `plssvd_fit` and `spls3_fit` give
  bit-identical results for every memory layout of `Y`, and of an `X`
  passed with `pre_standardized_X`, also with `pre_standardized_Y`. A
  pre-standardized row-major or negative-stride block used to move the
  scores in the last bits (up to about 4e-14 for a reversed row order) and
  could move `k_used` on the truncation floor; such a block is now copied
  column-major. Column-major pre-standardized
  results are unchanged to the bit. In Python, a default C-ordered `X` with
  `pre_standardized_X=True` now gives the same bits as its F-ordered copy.
- Changed (R, plskit-bind): bool data is read as 1/0 in every numeric
  array argument, as numpy reads a bool array in Python and Julia a `Bool`
  array, with the same bits as the equal 0/1 doubles. In R a logical matrix
  keeps its shape, and logical vectors and logical data.frame columns are
  accepted; `NA` in logical data raises `non_finite_input`, as in numeric
  data. Flags and integer arguments stay strict.
- Fixed (Python): `pls1_predict`, `pls3_transform`, `plssvd_transform` and
  `rotate` given a model of the wrong type (a `PLS3Result` where a
  `PLS1Result` is expected, `None`, ...) raise
  `PlsKitError(code="invalid_argument")` in `plskit-bind`'s words
  ("model must be a PLS1Result, got a PLS3Result"), as R does, instead of
  `AttributeError` or `TypeError`. `rotate` reads a nested-list `W` like any
  other array argument.
- Changed (Python, performance): an aligned F-ordered float64 `Y`, `Y_new`,
  `W` or `L` reaches the engine without a copy. C-ordered and strided ones
  are still copied column-major, so results are bit-identical to before for
  every layout.
- Changed (performance): `pls1_fit` / `spls1_fit` on a row-major
  (C-ordered, the Python default) `X` at least 2048 columns wide read the
  standardization moments and the `X'·t` products in blocks of 8 rows. On
  one M4 core a 1000 × 10000 C-ordered k = 1 fit drops from about 24 ms to
  about 13 ms (it was about 1.4x slower than its column-major copy, and is
  now no slower). Moments are
  bit-identical. Fits of such inputs that are not pre-standardized move in
  the last bits, agreeing with 0.6.2 to corpus tolerance but not bit for
  bit, and stay byte-identical across thread counts. Column-major `X`,
  pre-standardized fits and narrower `X` are unchanged bit for bit. The
  reference fit inside `pls1_rotation_stability`, `pls1_confirmatory_test`
  with `ci`, and the final fit of `pls1_fit(k="optimal"|"sequence")` take
  the same path.
- Changed (performance): the rule that sends a `perm_null`, `raw_perm` or
  `split_exact` resampling loop to the p-space Gram route (build
  `C = X'X` once, then one `C·r` per replicate) now weighs the build and
  the per-replicate product at their measured costs:
  `p·(0.33·n_tr + 1.3·B·k) < 2·B·k·n_tr`, was `p·(n_tr + B·k) <
  2·B·k·n_tr`. Shapes near the boundary that stayed primal although the
  Gram route was faster now take it, and break-even shapes that took it
  stay primal (over a 164-shape sweep: 91.3 s to 85.5 s in total;
  `split_exact` at 1600 × 640, k = 2, 199 permutations: about 1.2x faster
  on one M4 core). The
  coefficients were fitted on an Apple M4. No corpus fixture changes
  route. On shapes that do, results agree with 0.6.2 to corpus tolerance
  (observed at most 1.6e-14), not bit for bit.
- Changed (performance): `split_exact` at k = 2 on the n-space Gram route
  (`p` well above the training size) runs its kernel over runs of 16
  outcome columns as matrix products instead of one matrix-vector product
  per column: 1000 × 1000, 1000 permutations, k = 2 drops from about 12.2 s
  to about 6.0 s on one M4 core. k = 1 is unchanged. The statistic and
  p-value were unchanged in every case tested; the per-split statistics can
  move by about 1e-16 when the last run of columns is partial
  (`n_perm + 1` not a multiple of 16) at small training sizes, within
  corpus tolerance. Results stay byte-identical across thread counts, and
  splits that fall back to the primal refit are unchanged bit for bit.
- Changed (release): a `vX.Y.Z` tag now publishes `plskit-bind` to
  crates.io after `plskit`, for the R package to build against. The
  workspace pins its `plskit` dependency exactly (`=X.Y.Z`), so a version
  bump that misses it fails to build instead of drifting. The `-r` release
  job installs the R tarball against crates.io with the committed lock file
  before uploading it.
- Changed (dependencies): `chacha20` (through `rand`) moves from the yanked
  0.10.0 to 0.10.2, so a freshly resolved R lock file gets the same version
  as the workspace. No numeric change: corpus, byte-parity and
  thread-count tests pass unchanged.

## [0.6.1] - 2026-09-29

- Changed (performance): `pls1_fit` / `spls1_fit` no longer write a
  standardized copy of `X`: the kernel forms its products from the raw `X`,
  read in its own memory layout (a C-ordered NumPy array in place), with the
  centering and scaling applied inline. When a column's |mean| exceeds 1000
  times its scale, `X` is standardized into a copy as before, since the
  inline form would lose digits to cancellation. A fit that stops before `k`
  components on its truncation floor, or keeps one within
  `2·(1 + max |mean| / scale)` times that floor, is refitted on a
  standardized copy, which costs about two fits and the copy's memory.
  `pre_standardized=True` no longer copies `X` into column-major order (a
  weighted fit still forms its √w-scaled `X`). The truncation floor takes `‖X‖_F` from the standardization
  moments, and the finiteness check from the same pass.
- Changed (numerics): `pls1_fit` / `spls1_fit` results from Rust are no
  longer bit-identical across memory layouts of `X`; row- and column-major
  inputs agree to rounding, except that a fit sitting on its truncation
  floor (typically a `pre_standardized` `X` outside the scale contract) can
  stop at a different `k_used`. The Python API passes every `X` on in C
  order, so its results do not depend on the input's layout; a
  `pre_standardized=True` fit of a C-ordered array is now read row-major
  rather than copied column-major, so its last bits (and, on the floor, its
  `k_used`) can differ from 0.6.0. A standardizing fit forms its products from
  the raw `X`, so its coefficients differ from 0.6.0 by about
  `max |mean| / scale · 1e-16` relative; a stop on the truncation floor
  (`k_used`) is decided on the standardized copy whenever it is close enough
  that the raw-`X` products could decide it differently.

## [0.6.0] - 2026-09-27

- Added: `spls3_fit`, sparse PLS3 / PLSSVD with a keep-count per side
  (`keep_X` non-zeros per column of `U`, `keep_Y` per column of `V`, ties
  broken toward the lower index). `keep_Y < n_targets` forces each latent
  dimension onto a few outcomes. Full keep-counts delegate to `pls3_fit`
  (bit-identical); otherwise saliences are unit-norm but not orthogonal.
  `Pls3Model` / `PLS3Result` gain `keep_X`, `keep_Y` (`None` on a dense
  fit), `converged` and `n_iter`; reaching `max_iter` is not an error.
- Added (Rust only): `keep_x` / `keep_y` on `Pls3ConfirmatoryTestOpts`,
  applied inside each training half (the wrappers do not expose them);
  `PermNullOpts: Default`; public `SPLIT_NB_REROUTE_N_PERM`; error variants
  `PlsKitError::OptimalNoComponent` and `SequenceNoRejection { alpha }`
  (an exhaustive `match` needs two new arms); `k_to_fit()` on the find-K
  outputs.
- Changed (behaviour): these inputs now raise `invalid_argument` instead of
  returning a degenerate result. Check for them before upgrading.
  - `pls1_confirmatory_test(method="raw_perm")` with `n_folds >= n` (was
    statistic `0`, `p = 1`).
  - `raw_perm` as `test_method` / `diagnostic` at `n <= 5`, through the
    `*_find_k_*` functions and `pls1_fit(k="sequence")`.
  - The CV selectors (`r2_se`, `r2_max`) at `n <= 2`, through
    `*_find_k_optimal`, `spls1_find_keep_optimal` and
    `pls1_fit(k="optimal")`. `bic` is unaffected.
  - `preprocess` with a 2-D `Y` whose row count does not match `X` or
    `weights` now raises `dimension_mismatch` / `invalid_weights`.
- Changed (behaviour): truncation floors are relative to the data. Every
  PLS1 component must clear `max(n, p)·ε·‖X‖_F·‖y‖`, every PLS3 component
  `max(n, p, q)·ε·‖X̃‖_F·‖Ỹ‖_F`, instead of an absolute `1e-14`. Fits no
  longer keep noise components once `y` is exhausted or on a
  rank-deficient `X'Y`, so `k_used` can be smaller (`n = 2000`, `p = 40` of
  rank 39, `k = 40`: 40 → 13). A `y` or `Y` orthogonal to `X` up to
  rounding gives `k_used = 0`, and then `*_find_k_optimal` returns
  `k_star = 0` and `spls1_find_keep_optimal` returns `keep_star = 0`. No
  corpus fixture changes.
- Changed: PLS1 fits use Improved Kernel PLS (Dayal and MacGregor 1997)
  instead of explicit deflation. Same model; one-component fits are
  bit-identical, multi-component fits may differ in the last bits.
- Changed: `pls1_fit(k="optimal" | "sequence")` raises
  `optimal_no_component` / `sequence_no_rejection` from the engine, not
  the Python wrapper. Codes unchanged, wording slightly different.
- Changed (Python): resampling defaults (`n_boot`, `m_rate`, `level`,
  `max_failure_rate`, `max_skip_rate`, `n_perm`, `alpha`, and `spls3_fit`'s
  `max_iter` / `tol`) are now `None`, resolving to the engine default.
  Values and results are unchanged.
- Changed (Rust only): `ConfirmatoryTestOpts::default()` selects
  `split_exact` (`n_perm = 1000`, `n_splits = 50`) instead of `split_nb`.
- Fixed: results for a fixed seed no longer depend on the Rayon pool size.
  faer split parallel products by pool size, so large fits (`n·d·k >= 1e6`
  for PLS1, `n·p·q >= 1e6` for PLS3) and everything built on them moved in
  the last bits with `RAYON_NUM_THREADS`. Every faer call now uses a fixed
  8-way split or `Par::Seq`; `tests/thread_count_parity.rs` requires
  bit-identical output at 1, 3 and 5 threads.
- Fixed: `ConfirmatoryCI` (the subsampling CI of `pls1_confirmatory_test`).
  - `beta_sign_z` is `|β_ref[j]| / beta_se[j]`, using the finite-population
    rate `√(m/(n − m))` and a correction for subsample shrinkage `κ̂`
    (`beta_sign_z_signed` keeps the sign). The old z flagged 63–92% of
    noise variables.
  - `beta_ci_*` / `beta_se` carry the same corrections: coverage at nominal
    95% rose from 77–89% to 91–96% at `K = 1`. β remains uncalibrated at
    `K ≥ 2`.
  - `leverage_*` come from an n-out-of-n bootstrap, with a normal interval
    clamped to `[0, 1]`: signal-variable coverage 0.92–0.97 at `K = 1`,
    0.92–0.98 at `K = 2, 3` (was 40–65% at `D = 20`).
  - The `pls1_confirmatory_*_ci` fixtures are regenerated.
- Fixed: scale invariance. Standardization's constant-column test and the
  split-half correlation (`split_exact`, `split_nb`,
  `pls3_confirmatory_test`) were absolute (`sd <= 1e-12`, `ss < 1e-15`)
  and overflowed or underflowed past about `1e±154`, so rescaling the data
  could zero a column or return statistic `0`. Both are now relative and
  exactly rescaled by a power of two; wherever the old sums stayed in
  range, outputs are bit-identical.
- Fixed: weighted `pls1_rotation_stability` standardized each resample with
  unweighted moments. Weighted results change; unweighted are identical.
- Fixed: the `split_nb` auto-gate uses one `n_eff` everywhere, so
  `*_find_k_sequence` and the confirmatory test agree at the floor.
- Fixed: the `split_exact` no-refit route at `k = 1`, the `raw_perm` Gram
  route at `k = 1`, and the sparse-Y Gram route of `pls3_confirmatory_test`
  now recompute on the primal route any case that sits within rounding of
  a decision threshold, so the route choice is again invisible.
- Fixed (Python): misaligned `float64` arrays were read in place
  (undefined behaviour in Rust); they are now copied.
- Performance: Gram routes for resampling loops, chosen from shape alone,
  with a primal refit for any replicate they cannot certify. A p-space
  `X'X` route serves `pls1_perm_null`, `raw_perm` and `split_exact` refits
  on tall data (dense, weighted, sparse); an n-space `X X'` route serves
  the same loops at `k <= 2` when `p ≫ n`. At a fixed seed, p-values are
  unchanged except where a null statistic lies within `1e-10` of the
  observed one.
- Performance: IKPLS reads `X` twice per component and never writes it;
  resampling loops no longer copy `X` per replicate; standardization is
  faster and bit-identical; Python no longer copies a C-ordered `X` before
  fitting. `pls1_fit` from Python now beats scikit-learn on all four rows
  of `scripts/bench_pls1_fit.py`'s default grid.
- Tests and docs: `tests/corpus.rs` checks every fixture family from Rust;
  three `spls3_fit` fixtures added. README examples use `k=1` with
  `split_exact`.

## [0.5.0] - 2026-09-06

- Added: `pls3_fit` (alias `plssvd_fit`) — SVD-PLS / PLSC. One SVD of the
  standardized cross-covariance `X'Y`, no deflation, so all `k ≤ min(p, q)`
  components come out of a single decomposition and are orthogonal by
  construction. The SVD acts on a `p × q` matrix and never on a `p × p`
  one, so `p ≫ n` is the ordinary case for this family. Salience signs are
  pinned — the largest-magnitude entry of each `U` column is positive and
  the matching `V` column flips with it — so repeated and cross-platform
  fits agree.
- Added: `pls3_transform` (alias `plssvd_transform`) — project new X and/or
  Y onto a fitted PLS3's latent-variable scores, selected by
  `which="x_scores" | "y_scores" | "both"`. There is no `pls3_predict`:
  PLS3 is symmetric, so neither block is the outcome.
- Added: `pls3_confirmatory_test` — omnibus test at LV1 on the held-out
  latent-variable correlation `r = cor(X_te @ u1, Y_te @ v1)`, Fisher-z
  averaged across splits and reported as `tanh(z_bar)`. Two methods over
  that one statistic: `method="split_exact"` calibrates it by permuting the
  rows of Y against X with the splits held fixed, and `method="split_nb"`
  compares it against a t reference instead, costing `n_splits` fits in
  total rather than `n_perm * n_splits`. `split_exact` is the
  recommendation — it holds its level on any design. Both sides of the
  correlation are estimated on the training half, but that costs the t
  reference nothing: conditional on the training half the two held-out
  score vectors are fixed linear combinations of independent test-half
  rows, so under the null `r` follows the ordinary null correlation law,
  and on Gaussian, heavy-tailed, low-stable-rank and real two-block designs
  `split_nb` measured conservative rather than anti-conservative.
  `raw_perm` needs a CV statistic a method without `predict` does not have,
  and `score` and `e` have no symmetric formulation. `k=1` only: above LV1
  the training-half component ordering need not survive to the test half.
- Added: the `split_nb` auto-gate applies to `pls3_confirmatory_test` on
  the same terms as `pls1_confirmatory_test`, on X only. A flagged design
  runs `split_exact` instead, `result.method` says so, Python warns, and
  `args={"force": True}` overrides. Y never enters the gate: `q` is small
  by construction in PLSC, so a stable-rank floor on Y would flag almost
  every design. The thresholds are the PLS1 ones and have not been
  re-derived for a two-block design.
- Added: `PLS3Result` and `PLS3Scores` result objects. `pls3_confirmatory_test`
  reuses `ConfirmatoryTestResult`, with `ci` always `None` and `n_eff` equal
  to `n`. `rho_hat` is populated for `split_nb` only, `stable_rank` whenever
  `split_nb` was requested, and `n_perm` is `None` for `split_nb`.
- Added: six `pls3_*` reference-corpus cases (`testdata/`), regenerated
  from the Rust core.
- Note: observation weights are not implemented anywhere in the PLS3
  family. `pls3_fit(weights=...)` raises rather than silently ignoring the
  argument.
- Changed: `pls1_confirmatory_test(method="raw_perm")` at `k=1` and
  `pls3_confirmatory_test` now pick between two execution routes internally.
  The new Gram route builds the training Gram `X_tr X_tr'` and the
  train-to-test map `X_te X_tr'` once per fold or split, so each
  permutation replicate afterwards costs nothing in the feature count — the
  win is concentrated where `p` is large and `n` is small (voxel- or
  embedding-width data at small sample size). The route is chosen from
  `(n_tr, p, n_perm, q)` alone and is taken only when it is estimated
  faster; there is no knob and nothing is reported on the result, because it
  is an execution strategy and not a statistic. Both routes see the same
  splits, folds and permutations at the same seed, and every replicate is
  checked against an honest refit, agreeing within the project's numerical
  tolerance. Weighted input, `k >= 2` and sparse `keep` always take the
  original route.

## [0.4.0] - 2026-08-03

- Changed: `split_perm` and `split_perm_nr` are merged into one method,
  `split_exact` — the permutation-calibrated split-half test. The engine
  picks the no-refit route at K = 1 (dense input) or the refit route
  otherwise; there is no route knob, and `split_perm` / `split_perm_nr`
  are no longer valid `method` values. This changes the refit route's
  numbers versus 0.3.0's `split_perm`: splits are now drawn once and held
  fixed across permutation replicates instead of redrawn per replicate —
  redrawing folded split-to-split scatter into the null and miscalibrated
  `split_perm`'s p-values — and the reported statistic moved from mean-r
  to `tanh(z̄)` to match. Re-running a 0.3.0 `split_perm` analysis under
  0.4.0 will produce different numbers. The no-refit route's numbers are
  unchanged from 0.3.0's `split_perm_nr` (bit-identical), but it now also
  accepts weighted input at K = 1, which `split_perm_nr` used to reject.
- Added: `split_nb` auto-gate. `split_nb`'s Fisher-z correction drifts
  off level when `n_eff < 25` or the stable rank of the standardized `X`
  is `< 3`. Stable rank can never exceed the column count, so `X` with 4
  columns or fewer is rerouted outright without consulting the computed
  rank. A request that trips any of the three clauses now reroutes to
  `split_exact` (at `n_perm=1000`) and `result.method` reports the
  method actually run. Pass `args={'force': True}` to run `split_nb`
  anyway. Python raises a `UserWarning` when a reroute happens.
- Added: `stable_rank` on `ConfirmatoryTestResult`, `FindKOptimalResult`
  and `FindKSequenceResult` — the stable rank the `split_nb` auto-gate
  saw, populated whenever `split_nb` was requested (fired or not,
  including under `force`); `None` for every other method. On the two
  `find_k` results it is the sequence-level gate's value, read off the
  undeflated `X`, so a rerouted run can say which clause fired.
- Added: `split_nb_gate(X, weights=None)` — ask whether the auto-gate
  flags a design without running a test. Returns `fires`, `stable_rank`
  and `n_eff`. It evaluates the same rule the test functions apply
  internally, so it cannot drift from them.
- Changed: the recommended default for `pls1_confirmatory_test` is now
  K = 1 with `method="split_exact"` (previously `split_nb`).

## [0.3.0] - 2026-07-31

- Added: `split_perm_nr` confirmatory test method — the same statistic as
  `split_nb` (mean Fisher-z of held-out correlations, reported as
  `tanh(z̄)`), compared against a permutation reference instead of the t
  approximation. K = 1 and unweighted input only; raises rather than
  degrading on ineligible input.
- Added: `rho_hat` on `ConfirmatoryTestResult` — reported for the `split_nb`
  arm (`None` for every other method, and `None` for `split_nb` itself when
  the input is weighted or the test half is too small).
- Changed: `pls1_find_k_optimal` and `pls1_find_k_sequence` name the offending
  method when it has no sequential variant, instead of always reporting
  `score`. `split_perm_nr` is the second such method.
- Fixed: pinned `time` to 0.3.41 and `deflate64` to 0.1.9 in `Cargo.lock`.
  Dependency bumps had pulled in `time-core` 0.1.8 (requires rustc 1.88) and
  `deflate64` 0.1.12 (uses `unbounded_shifts`, stable in 1.87), both past the
  declared MSRV of 1.85. Dev-dependency-only, reached via `ndarray-npy` →
  `zip`; nothing in the published crate or wheel is affected.

## [0.2.1] - 2026-06-21

- Fixed: the `coverage_mc` test's oracle for `leverage_ci_*` coverage now
  estimates the population value of the same finite-n estimand each
  per-dataset CI targets — Monte Carlo over `N_ORACLE = 200` freshly drawn
  size-`n` datasets run through the identical engine path — instead of a
  single asymptotic 50,000-row fit. Per-coordinate leverage coverage is no
  longer asserted above `k = 1` (the synthetic DGP's true signal rank): the
  centered-scaled leverage CI is anti-conservative outside the low-`d`/
  large-`n` regime (measured between-dataset SD / reported SE ≈ 1.2, rising
  to ≈ 2.1 at `d=20, n=100`), so those numbers are now printed for
  diagnostic monitoring only. `holdout_corr` remains the sole asserted
  calibration guarantee. Test-only; no change to library behavior.

## [0.2.0] - 2026-06-12

- Added: `spls1_*` sparse PLS1 family (`spls1_fit`, `spls1_find_keep_optimal`, `spls1_find_k_optimal`, `spls1_find_k_sequence`) — hard keep-count NIPALS selection with dense bit-parity at `keep = n_features`.
- Changed (numerical — outputs shift from 0.1.0; pin the version for
  reproducibility):
  - `pls1_find_k_sequence` now standardizes and deflates with the supplied
    observation weights (0.1.0 ran the incremental steps unweighted and only
    weighted the final per-step test).
  - universal-inference e-value (`method="e"`) fixes σ²_alt on the training
    half instead of the test half.
  - χ²(df) survival function computed via the upper incomplete gamma directly
    (no 1 − lower complement round-trip).
  - subsample subspace leverage computed without Procrustes alignment (the
    hat matrix is rotation-invariant).
  - varimax final `w_rot` matmul via faer (`Par::Seq`) instead of a scalar loop.
  - rotation-stability paired-bootstrap seed derived from the parent RNG
    stream instead of a fixed `0xB007` offset.
- R and Julia wrappers remain at 0.0.1 (no `spls1` surface yet).

## [0.1.0] - 2026-05-09

Initial release of plskit. PLS1 with modern inference (canonical
percentile CIs, `split_nb`, `split_perm`). Ships the Rust engine
(`plskit` on crates.io) and the Python wrapper (`plskit` on PyPI);
R and Julia wrappers are not yet at feature parity and ship at a
lower version.
