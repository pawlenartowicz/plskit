# Deterministic test data from R's own RNG (the corpus test covers
# numerical parity; these tests cover the R seam).

make_xy <- function(n = 60L, p = 6L, seed = 1L) {
  set.seed(seed)
  X <- matrix(stats::rnorm(n * p), n, p)
  y <- 4 * (X[, 1L] + X[, 2L]) + 0.5 * stats::rnorm(n)
  list(X = X, y = y)
}

# The plskit_error condition an expression signals, or NULL.
catch_plskit <- function(expr) {
  tryCatch({
    expr
    NULL
  }, plskit_error = function(e) e)
}
