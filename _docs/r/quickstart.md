# R quickstart

```r
library(plskit)

set.seed(1)
X <- matrix(rnorm(60 * 6), 60, 6)
y <- 4 * (X[, 1] + X[, 2]) + 0.5 * rnorm(60)

# Fit and predict
fit <- pls1_fit(X, y, k = 3, seed = 42)
fit                      # compact summary: field names and shapes
yhat <- pls1_predict(fit, X[1:5, ])

# Confirmatory test; method-specific options go in `args`
test <- pls1_confirmatory_test(X, y, k = 1, method = "split_exact",
                               args = list(n_perm = 1000, n_splits = 50))
test$pvalue
test$seed                # the drawn seed, as a decimal string

# Reproduce the run exactly
again <- pls1_confirmatory_test(X, y, k = 1, method = "split_exact",
                                args = list(n_perm = 1000, n_splits = 50),
                                seed = test$seed)
identical(again, test)   # TRUE

# Choose k, then rotate the model (the selection result travels along)
fit2 <- pls1_fit(X, y, k = "optimal", k_max = 4,
                 find_k_args = list(selector = "bic"))
rot <- rotate(fit2, method = "varimax")
rot$rotation_spec$args

# Errors are conditions carrying the Python error codes
tryCatch(pls1_fit(X, y, k = 2.5), plskit_error = function(e) e$code)
# "invalid_argument"
```

Every argument name, default and result field is documented once, in the
[Python API](../python/api.md) and [result objects](../python/results.md)
pages; `?pls1_fit` in R links there. R-specific behaviour is in
[Differences from Python](differences.md).
