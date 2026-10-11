# Threads

`PLSKIT_NUM_THREADS` caps how many threads plskit uses, and
`PLSKIT_MEMORY_BUDGET` ([Memory budget](#memory-budget)) bounds the memory
its resampling loops use. Both work the same way in every language: Rust,
Python, R and Julia.

## What `PLSKIT_NUM_THREADS` does

Every function in the API reference that can run in parallel reads
`PLSKIT_NUM_THREADS` once per call.

| Value | Behaviour |
|---|---|
| unset or `0` | no cap: runs on the current Rayon pool. In Python, R and Julia that is Rayon's global pool: `RAYON_NUM_THREADS` threads if set, otherwise all logical cores. |
| a positive integer `N` | runs on at most `N` threads: inside a plskit-owned pool of `N` threads when `N` is smaller than the current pool (outside any pool: the number of logical cores, even when `RAYON_NUM_THREADS` is set), otherwise on the current pool |
| anything else (`""`, a negative number, a non-integer, `"abc"`, `" 4 "`) | `InvalidArgument` (Python: `PlsKitError` with code `invalid_argument`). The message names the variable and quotes the value. |

The value is not trimmed, so `" 4 "` is an error.

`pls1_predict`, `pls3_transform`, `plssvd_transform`, `preprocess` and `rotate` run on one thread and do not read the variable.

## Setting it

| Language | Call |
|---|---|
| shell | `export PLSKIT_NUM_THREADS=4` |
| Python | `os.environ["PLSKIT_NUM_THREADS"] = "4"` |
| R | `Sys.setenv(PLSKIT_NUM_THREADS = 4)` |
| Julia | `ENV["PLSKIT_NUM_THREADS"] = "4"` |

All four write the process environment. `PLSKIT_NUM_THREADS` is read once
per call, so a change takes effect on the next call.

## Results do not depend on the thread count

A run with `PLSKIT_NUM_THREADS=1` and a run with all cores give the same
results.

## Inside parallel workers

When you run plskit inside worker processes (joblib processes,
`parallel::makePSOCKcluster`, Julia `Distributed`), set
`PLSKIT_NUM_THREADS=1` in each worker. Otherwise every worker starts a pool
of all logical cores and the machine is oversubscribed.

## Relation to `RAYON_NUM_THREADS`

When `PLSKIT_NUM_THREADS` is unset, `RAYON_NUM_THREADS` still sizes Rayon's
global pool. As with `RAYON_NUM_THREADS`, a `0` here means "use the
default".

## Rust

When `PLSKIT_NUM_THREADS` is unset, a caller's own
`rayon::ThreadPool::install` sets the thread count for the plskit calls
made inside it.

## Memory budget

`PLSKIT_MEMORY_BUDGET` sets the bytes of working memory, beyond the
inputs, that plskit gives to the buffers that grow with the data: copies of
`X` or of its halves, Gram matrices, and the blocks in which `X` is read.
`X` itself, the results, the outcome and permutation buffers and the
per-column vectors are not counted.

The functions below repeat a computation over units (splits, folds or
resamples) and estimate the memory one unit needs from the shape of `X`.
They run the units in batches of `budget / unit memory` units, at least one
and at most all of them: in parallel inside a batch, one batch after
another. At most one batch of units holds its buffers at once, whatever the
thread count.

- `pls1_confirmatory_test` with `split_nb`, `split_exact` or `raw_perm`,
  and its confidence intervals (`ci`);
- `pls3_confirmatory_test` with `split_nb` or `split_exact`;
- the cross-validated selection of `pls1_find_k_optimal` and
  `spls1_find_k_optimal` (selector `r2_se` or `r2_max`) and of
  `spls1_find_keep_optimal`;
- `pls1_rotation_stability`;
- the functions that run these tests or this selection inside them: the
  steps of `pls1_find_k_sequence` and `spls1_find_k_sequence`, the
  `diagnostic` of `pls1_find_k_optimal` and `spls1_find_k_optimal`, and
  `pls1_fit` with `k="optimal"` or `k="sequence"` (Python, R and Julia).

`pls1_perm_null` holds at most one standardized copy of `X`, shared by all
its permutations, and is not batched. `pls1_perm_alpha` holds one copy of
`gram`, shared by all its columns, and is not batched.

| Value | Behaviour |
|---|---|
| unset | the default, `2G` (2 GiB) |
| a whole number of bytes, optionally followed by `K`, `M`, `G` or `T` (powers of 1024), optionally followed by `B`, or by `iB` after a unit letter, in any case | that many bytes: `2147483648`, `2048M`, `2G`, `2gb` and `2GiB` are the same value |
| `0` | one unit at a time |
| anything else (`""`, `"2X"`, `"1.5G"`, `"2iB"`, `" 2G "`), or a value of 2^64 bytes or more | `InvalidArgument` (Python: `PlsKitError` with code `invalid_argument`). The message names the variable and quotes the value. |

The value is not trimmed. Every function that reads `PLSKIT_NUM_THREADS`
also checks `PLSKIT_MEMORY_BUDGET` at the start of each call and raises the
error before any work, including the functions that do not batch anything.
`pls1_predict`, `pls3_transform`, `plssvd_transform`, `preprocess` and
`rotate` do not read it. It is set like `PLSKIT_NUM_THREADS` (for example
`export PLSKIT_MEMORY_BUDGET=8G`, or
`os.environ["PLSKIT_MEMORY_BUDGET"] = "8G"` in Python) and applies to each
call separately, so `N` worker processes can together use `N` times the
budget.

The budget is an estimate of the large buffers, not an exact ceiling on the
memory of the process. One unit always runs, even when its buffers alone
exceed the budget, since it cannot be computed in parts: with `0`, or when
one unit's buffers take more than half the budget, the peak is about `X`
plus one unit's buffers.

Results do not depend on the budget: any value gives the same results, as
any thread count does.

### When to change it

At the default, an `X` whose copy is small next to 2 GiB (for example 200 ×
60,000, about 92 MiB per copy) already runs its units in parallel, so small
data on many cores is fast without changing anything. On an `X` whose copy
is more than half the budget (for example 254 × 1,447,336, about 2.7 GiB per
copy), the units that copy `X` run one at a time; units that hold only Gram
matrices, which are small when `X` has far more columns than rows, still
run in parallel. Raise the budget to run more copies of a large `X` at
once: more memory, more units in parallel. Lower it, or set `0`, to keep
the peak down on a machine shared with other jobs.
