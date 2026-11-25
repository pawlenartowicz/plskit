# Contributing

File all PRs and issues against the monorepo at https://github.com/pawlenartowicz/plskit. Every wrapper (`plskit-rs`, `plskit-py`, `plskit-r`, `plskit-jl`) lives in this repo as a subdirectory; there are no per-language mirror repos.

Versions are per-artifact; each artifact tags independently as `vX.Y.Z` (engine), `vX.Y.Z-py`, `vX.Y.Z-r`, or `vX.Y.Z-jl`. Wrappers may lag the engine; CI verifies the artifact↔tag match on each tag push.

## Pre-release: slow MC tests

Before tagging a release, run the Monte Carlo gate, the same command as `.github/workflows/mc-gate.yml`:

```
cargo test -p plskit --release -- --ignored --nocapture --skip cap_kills_an_over_budget_child --skip peak_within_cap --skip batched_primal_precision_sweep
```

Asserts two-sided empirical coverage at `level=0.95` on `holdout_corr` and on per-coordinate `leverage_ci_*` (signal coordinates only) across the cell grid `n ∈ {100, 200, 500} × d ∈ {6, 20} × K ∈ {1, 2, 3} × SNR ∈ {1, 4}`, 200 datasets per cell. Coverage near 0.95 is expected. Empirical coverage must lie in `[level − 0.05, level + 3·√(level·(1 − level)/200)] = [0.9000, 0.9962]`. The upper edge is three binomial standard errors above the level, so a cell fails high only when all 200 datasets cover. Coverage above the upper edge fails like coverage below the lower one: either is a release-gate signal, so investigate before tagging. The test prints the band. Takes minutes to hours.
