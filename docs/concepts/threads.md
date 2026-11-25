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
| anything else (`""`, a negative number, a non-integer, `"abc"`, `" 4 "`, `"+4"`) | `InvalidArgument` (Python: `PlsKitError` with code `invalid_argument`). The message names the variable and quotes the value. |

The value is not trimmed, so `" 4 "` is an error.

`pls1_predict`, `pls3_transform`, `plssvd_transform`, `preprocess`, `rotate` and `profiles` run on one thread and do not read the variable.

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

`PLSKIT_MEMORY_BUDGET` sets the memory that a call into the Rust engine may use beyond its inputs. At a nonzero budget, the peak memory of a call is about the budget plus the data, where the data are the inputs at float64 size: `8·n·p` bytes for `X`, plus `Y`, the outcome, the weights and the covariates. "About" means that the call is meant to complete in a process whose memory is capped at its memory before the call, plus the budget, plus a margin of the larger of 32 MiB and 5% of the budget. For example, an `X` of 10,000 × 20,000 float64 values is 1.6 GB, so at the default budget of 4 GiB (4.3 GB) such a call peaks at about 5.9 GB at most. The Python, R and Julia wrappers may copy or cast the inputs on top of this.

The budget is a best-effort bound, not a hard limit: the engine computes the bytes of its buffers from the shapes of the inputs and does not watch the process. It counts the large buffers that grow with the data: copies of `X` or of its parts, Gram matrices, the blocks in which `X` is read, the buffers of the model fits that run at once, and outcome and permutation buffers. It leaves out small per-call buffers (such as the basis of the covariates and the indices of the folds) and a few copies written only on rare inputs (such as the standardized copy of `X` that the cross-validated selection writes for a column whose mean is over 1000 times its spread), so a call can pass the cap by a little, and on those inputs by up to one copy of `X`. The `split_nb` gate holds the Gram matrix of `X` on its shorter side (and on tall `X` the standardized copy of `X`) and takes its largest eigenvalue with an iterative solver whose buffers grow with that side, not its square; under `force` it computes no stable rank and holds only the Gram matrix of wide `X`, which the test then reads. On x86_64 it also counts the matrix-multiply buffer of faer, the linear-algebra library, at about 57.5 MB per thread whatever the size of the data, plus one more for the calling thread when that thread is not in a Rayon pool. That size was measured on one machine: the buffer is twice the L3 cache of the CPU that runs the call, so on a CPU with a larger cache a call can pass the cap by the difference on each thread. A small budget with many threads can therefore raise `MemoryBudget` even on small data: lower `PLSKIT_NUM_THREADS` or raise the budget. The rule covers the functions this section lists. A single fit (`pls1_fit` at an integer `k`, `spls1_fit`, `pls3_fit` and the like) does not check its buffers against the budget, nor do `gram_matrix` and the `e` test, which write a copy of the data (the `n × n` Gram matrix, the halves of one split) without checking it.

The functions below repeat a computation over units (splits, folds, permutations or resamples). Where a loop can take more than one route (a Gram matrix of the rows or of the columns, sums over the cells of the split design, moments, or a refit on `X` read from a standardized copy or in strips of 64 columns standardized as they are read), it picks the fastest route that fits: for each route it counts how many units fit the budget at once and estimates the time from that. The route depends on the shapes and the budget, not on the thread count, so a different budget can give a different route. One exception: where a refit on `X` can read it from a standardized copy or in strips, the thread count can decide between the two, since the strips run more units at once but gain from that only up to the number of threads; the results are the same either way. The loop then runs its units in batches of as many as fit, in parallel inside a batch, one batch after another.

- `pls1_confirmatory_test` with `split_nb`, `split_exact` or `raw_perm`, and its confidence intervals (`ci`);
- `pls3_confirmatory_test` with `split_nb` or `split_exact`;
- the cross-validated selection of `pls1_find_k_optimal` and `spls1_find_k_optimal` (selector `r2_se` or `r2_max`) and of `spls1_find_keep_optimal`;
- the functions that run these tests or this selection inside them: the steps of `pls1_find_k_sequence` and `spls1_find_k_sequence`, the `diagnostic` of `pls1_find_k_optimal` and `spls1_find_k_optimal`, and `pls1_fit` with `k="optimal"` or `k="sequence"` (Python, R and Julia);
- the column shuffles of the parallel analysis and the folds of `profiles_find_n`, and of the `profiles_find_n` step of `profiles_search` (Python, R and Julia);
- the permutations of `pls1_perm_null`, which holds the `n_perm × p` matrix of all permuted coefficients only with `return_perm_matrix`, and otherwise folds each batch into the mean and standard deviation.

`sv_perm` in `pls3_confirmatory_test` picks its route the same way and runs as one unit. `pls1_perm_alpha`, `factors`, `pca_basis` and `score` are not batched; they check their buffers against the budget before they write them. `profiles` runs on one thread and is not batched.

At a nonzero budget, when no route of a loop fits one unit, or when the route it picks does not fit one unit at the real thread count (with faer's buffers counted), the call raises `MemoryBudget` (Python: `PlsKitError` with code `memory_budget`; Rust: `PlsKitError::MemoryBudget { budget, needed }`) before the loop writes its buffers. The message names the variable, the budget and `needed`, and says to set the budget to at least `needed`, or to `0`. Read `needed` as "at least": a call with several loops (`ci`, the steps of a `k` sequence) checks each loop when it reaches it, so a later loop can name a larger value, and a run at `needed` can need another round, since a larger budget can admit a different route. The checks on the arguments and the data come first, so a bad argument gives its own error at any budget. The exceptions are the checks that read a copy of the data: the constant-column, singular-value and `n` checks of `factors`, and the entry checks of the centred Gram matrix (diagonal, symmetry and Cauchy-Schwarz) on a `GramMatrixResult`, in `pls1_perm_alpha` and in the Gram form of `pls1_confirmatory_test` and `pls3_confirmatory_test`.

| Value | Behaviour |
|---|---|
| unset | the default, `4G` (4 GiB) |
| a whole number of bytes, optionally followed by `K`, `M`, `G` or `T` (powers of 1024), optionally followed by `B`, or by `iB` after a unit letter, in any case | that many bytes: `2147483648`, `2048M`, `2G`, `2gb` and `2GiB` are the same value |
| `0` | the least memory possible: each loop takes the route that needs the least memory and runs one unit at a time (units that hold no buffers of their own run all at once). The call never raises `MemoryBudget`; its peak is about the data plus the buffers the loop holds and those of one unit. |
| anything else (`""`, `"2X"`, `"1.5G"`, `"2iB"`, `" 2G "`), or a value of 2^64 bytes or more | `InvalidArgument` (Python: `PlsKitError` with code `invalid_argument`). The message names the variable and quotes the value. |

The value is not trimmed. Every function that reads `PLSKIT_NUM_THREADS` also checks `PLSKIT_MEMORY_BUDGET` at the start of each call and raises `InvalidArgument` for a bad value before any work, including the functions that do not batch anything. `pls1_predict`, `pls3_transform`, `plssvd_transform`, `preprocess`, `rotate` and `profiles` do not read it. It is set like `PLSKIT_NUM_THREADS` (for example `export PLSKIT_MEMORY_BUDGET=8G`, or `os.environ["PLSKIT_MEMORY_BUDGET"] = "8G"` in Python) and applies to each call separately, so `N` worker processes can together use `N` times the budget.

Results do not depend on how many units run at once, so a call that keeps its route gives the same results bit for bit at every budget; the copy of `X` and the strips give the same bits too. A budget that changes the route changes the results to rounding on most inputs, and by up to O(1) on a few ([limitations](limitations.md#numerical-limits)).

### When to change it

At the default 4 GiB, an `X` whose copy is small next to the budget (for example 200 × 60,000, about 92 MiB per copy) already runs its units in parallel, so small data on many cores is fast without changing anything. On an `X` whose copy is more than half the budget (for example 254 × 1,447,336, about 2.7 GiB per copy), at most one unit can hold a copy of `X` at a time, so the engine weighs running the copies one at a time against reading `X` in strips, which holds no copy and runs the units together but costs about 1.5 times as much per unit, and takes the faster by its cost rule; units that hold only Gram matrices, which are small when `X` has far more columns than rows, and the folds of a selection by cross-validation, which fit `X` in place, still run in parallel. Raise the budget to run more copies of a large `X` at once: more memory, more units in parallel. Lower it, or set `0` for the least memory, to keep the peak down on a machine shared with other jobs.
