# Threads

`PLSKIT_NUM_THREADS` caps how many threads plskit uses. It works the same
way in every language: Rust, Python, R and Julia.

## What the variable does

Every function in the API reference that can run in parallel reads the
variable once per call.

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

All four write the process environment. The variable is read once per call,
so a change takes effect on the next call.

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

When the variable is unset, a caller's own `rayon::ThreadPool::install`
sets the thread count for the plskit calls made inside it.
