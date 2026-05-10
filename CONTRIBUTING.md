# Contributing

File all PRs and issues against the monorepo at https://github.com/pawlenartowicz/plskit. Every wrapper (`plskit-rs`, `plskit-py`, `plskit-r`, `plskit-jl`) lives in this repo as a subdirectory; there are no per-language mirror repos.

Versions are per-artifact (RULES.md RULE 7); each artifact tags independently as `vX.Y.Z` (engine), `vX.Y.Z-py`, `vX.Y.Z-r`, or `vX.Y.Z-jl`. Wrappers may lag the engine; CI verifies the artifact↔tag match on each tag push.

## Pre-release: slow MC tests

Before tagging a release, run the gated Monte Carlo coverage test:

```
cargo test -p plskit --release -- --ignored coverage_mc
```

Asserts two-sided empirical coverage at `level=0.95` on `holdout_corr`
and on per-coordinate `leverage_ci_*` (signal coordinates only) across
the cell grid `n ∈ {100, 200, 500} × d ∈ {6, 20} × K ∈ {1, 2, 3} × SNR ∈ {1, 4}`,
200 datasets per cell. Empirical coverage must lie in
`[level − 0.05, level + 0.05] = [0.90, 1.00]`. Takes minutes-to-hours.
Over-coverage near 1.00 is expected for the NB-Wald path on
`holdout_corr` (Nadeau–Bengio 2003 is conservative by design); the
two-sided band tolerates that. Under-coverage on either metric is a
release-gate signal — investigate before tagging.
