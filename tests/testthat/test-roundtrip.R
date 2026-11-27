# Results passed back into the package.

d <- make_xy()

test_that("predict and transform take returned results", {
  fit <- pls1_fit(d$X, d$y, k = 3)
  expect_s3_class(fit, c("pls1_result", "plskit_result"), exact = TRUE)
  pred <- pls1_predict(fit, d$X[1:5, ])
  expect_type(pred, "double")
  expect_length(pred, 5L)
  Y <- cbind(d$y, d$y + stats::rnorm(60))
  p3 <- pls3_fit(d$X, Y, k = 2)
  sc <- pls3_transform(p3, X_new = d$X, which = "x_scores")
  expect_s3_class(sc, c("pls3_scores", "plskit_result"), exact = TRUE)
  expect_equal(sc$x_scores, p3$x_scores, tolerance = 1e-10)
  expect_null(sc$y_scores)
})

test_that("rotate dispatches on class: a matrix gives a rotate_result", {
  fit <- pls1_fit(d$X, d$y, k = 2)
  rw <- rotate(fit$W)
  expect_s3_class(rw, c("rotate_result", "plskit_result"), exact = TRUE)
})

test_that("a one-component model round-trips (scalar Q)", {
  fit <- pls1_fit(d$X, d$y, k = 1)
  expect_length(fit$Q, 1L)
  expect_length(pls1_predict(fit, d$X[1:2, , drop = FALSE]), 2L)
})

test_that("an unclassed model still predicts; a broken one raises", {
  fit <- pls1_fit(d$X, d$y, k = 2)
  expect_identical(pls1_predict(unclass(fit), d$X), pls1_predict(fit, d$X))
  broken <- fit
  broken$W <- "x"
  expect_identical(catch_plskit(pls1_predict(broken, d$X))$code, "invalid_argument")
  broken <- fit
  class(broken) <- "lm"
  expect_identical(catch_plskit(pls1_predict(broken, d$X))$code, "invalid_argument")
})

test_that("a result with a prepended unknown class still round-trips", {
  fit <- pls1_fit(d$X, d$y, k = 2)
  fit_prepended <- fit
  class(fit_prepended) <- c("my_fit", class(fit_prepended))
  expect_identical(pls1_predict(fit_prepended, d$X), pls1_predict(fit, d$X))
})

test_that("nested results and lists keep their R types", {
  cf <- pls1_confirmatory_test(d$X, d$y, 2, test_method = "raw_perm", args = list(n_perm = 50), ci = TRUE, n_boot = 100, seed = 1)
  expect_s3_class(cf$ci$holdout_corr, c("ci_scalar", "plskit_result"), exact = TRUE)
  kp <- spls1_find_keep_optimal(d$X, d$y, 1, seed = 1)
  expect_type(kp$keep_grid, "integer")
  s3 <- spls3_fit(d$X, cbind(d$y, d$y + stats::rnorm(60), stats::rnorm(60)), 2, 3, 2)
  expect_type(s3$converged, "logical")
  expect_type(s3$n_iter, "integer")
})

# Input coercion and malformed passed-back records.

test_that("literal duplicate int-map names raise invalid_argument", {
  fit <- pls1_fit(d$X, d$y, k = "optimal", k_max = 4, seed = 7)
  broken <- fit
  names(broken$selection_result$cv_scores) <- c("1", "1", "3", "4")
  err <- catch_plskit(profiles(broken, d$X, 1L))
  expect_identical(err$code, "invalid_argument")
  expect_match(err$message, "cv_scores", fixed = TRUE)
})

test_that("a non-canonical int-map name ('01') is not silently accepted as a key", {
  fit <- pls1_fit(d$X, d$y, k = "optimal", k_max = 4, seed = 7)
  broken <- fit
  names(broken$selection_result$cv_scores) <- c("1", "01", "3", "4")
  # Not all names are canonical integer keys, so the field is read as an
  # ordinary vector (values preserved, in order) rather than becoming an
  # int map keyed 1, 1, 3, 4.
  rot <- profiles(broken, d$X, 1L)$model
  expect_equal(
    unname(rot$selection_result$cv_scores),
    unname(fit$selection_result$cv_scores)
  )
})

test_that("a duplicate key in a passed-back model names the field", {
  err <- catch_plskit(pls1_predict(list(a = 1, a = 2), d$X))
  expect_identical(err$code, "invalid_argument")
  expect_match(err$message, "duplicate key 'a'", fixed = TRUE)
})

test_that("a byte string that is not valid UTF-8 raises invalid_argument", {
  bad <- "\xff"
  Encoding(bad) <- "bytes"
  err <- catch_plskit(rotate(pls1_fit(d$X, d$y, k = 1)$W, method = bad))
  expect_identical(err$code, "invalid_argument")
})

test_that("a byte-string name that is not valid UTF-8 raises invalid_argument", {
  bad <- "\xff"
  Encoding(bad) <- "bytes"
  bad_named <- list(seed = 1L)
  names(bad_named) <- bad
  err <- catch_plskit(pls1_fit(d$X, d$y, k = "optimal", k_max = 3, find_k_args = bad_named))
  expect_identical(err$code, "invalid_argument")
  expect_match(err$message, "not valid UTF-8", fixed = TRUE)
})

test_that("factors and profiles results round-trip through rotate", {
  coef <- drop(pls1_fit(d$X, d$y, k = 2L)$coef)
  f <- factors(coef, d$X, 3L)
  expect_s3_class(f, c("factors_result", "plskit_result"), exact = TRUE)
  expect_equal(unname(rowSums(f$F)), unname(f$coef), tolerance = 1e-10)
  # a named numeric vector is a vector, not an integer-keyed map
  expect_identical(factors(setNames(coef, seq_along(coef)), d$X, 3L)$F, f$F)
  expect_identical(catch_plskit(factors(rep(NA, ncol(d$X)), d$X, 3L))$code, "non_finite_input")
  r <- rotate(f, d$X)
  expect_s3_class(r, c("factors_result", "plskit_result"), exact = TRUE)
  expect_equal(unname(rowSums(r$F)), unname(r$coef), tolerance = 1e-10)
  m <- pls1_fit(d$X, d$y, k = 3L)
  expect_s3_class(factors(m, d$X, 3L), c("factors_result", "plskit_result"), exact = TRUE)
  p <- profiles(m, d$X, 2L)
  Xs <- scale(d$X)
  ps <- profiles(coef, Xs, 3L, pre_standardized = TRUE)
  expect_s3_class(ps, c("profiles_result", "plskit_result"), exact = TRUE)
  expect_equal(unname(ps$scores), unname(Xs %*% ps$factors$F), tolerance = 1e-10)
  expect_s3_class(p, c("profiles_result", "plskit_result"), exact = TRUE)
  expect_s3_class(p$model, c("pls1_result", "plskit_result"), exact = TRUE)
  expect_s3_class(p$factors, c("factors_result", "plskit_result"), exact = TRUE)
  s <- profiles_search(m, d$X, d$y, seed = 1L)
  expect_s3_class(s, c("profiles_search_result", "plskit_result"), exact = TRUE)
})

test_that("pre_standardized overrides the result and bad combinations are refused", {
  Xu <- sweep(d$X, 2L, seq_len(ncol(d$X)), "*")
  mu <- pls1_fit(Xu, d$y, k = 2L)
  po <- profiles(mu, Xu, 2L, pre_standardized = TRUE)
  expect_equal(unname(po$scores), unname(Xu %*% po$factors$F), tolerance = 1e-8)
  expect_false(isTRUE(all.equal(unname(po$scores), unname(profiles(mu, Xu, 2L)$scores))))
  m1 <- pls1_fit(d$X, d$y, k = 2L)
  f <- factors(m1, d$X, 2L)
  Y <- cbind(d$y, d$y + stats::rnorm(60))
  m3 <- pls3_fit(d$X, Y, k = 2L)
  expect_identical(catch_plskit(rotate(m1))$code, "invalid_argument")
  expect_identical(catch_plskit(rotate(f, d$X, L = f$W))$code, "invalid_argument")
  expect_identical(catch_plskit(rotate(f, pre_standardized = TRUE))$code, "invalid_argument")
  expect_identical(catch_plskit(factors(m3, d$X, 2L))$code, "invalid_argument")
})

test_that("rotate keeps the class of a PLS3 result and warns on a sparse one", {
  Y <- cbind(d$y, d$y + stats::rnorm(60))
  m3 <- pls3_fit(d$X, Y, k = 2L)
  r3 <- rotate(m3, d$X)
  expect_s3_class(r3, c("pls3_result", "plskit_result"), exact = TRUE)
  expect_false(is.null(r3$rotation_spec))
  s3 <- spls3_fit(d$X, Y, k = 2L, keep_X = 3L, keep_Y = 1L)
  expect_condition(rotate(s3), class = "plskit_rotated_sparse")
  rs <- suppressWarnings(rotate(s3, d$X))
  expect_type(rs$converged, "logical")
  expect_type(rs$n_iter, "integer")
})

test_that("a length-1 coef reaches the engine as a vector", {
  x1 <- matrix(stats::rnorm(20), 20, 1)
  f <- factors(0.5, x1, 1L)
  expect_equal(unname(drop(f$F)), 0.5, tolerance = 1e-12)
})

test_that("subset() on a gram_matrix_result takes 1-based rows and checks them", {
  g <- gram_matrix(d$X, pre_standardized = TRUE)
  expect_s3_class(g, c("gram_matrix_result", "plskit_result"), exact = TRUE)
  rows <- c(8, 1, 4, 4, 60)
  s <- subset(g, rows)
  expect_s3_class(s, c("gram_matrix_result", "plskit_result"), exact = TRUE)
  expect_identical(s$gram, g$gram[rows, rows])
  expect_identical(s[c("n_features", "pre_standardized")], g[c("n_features", "pre_standardized")])
  for (bad in list(c(0, 2), c(1, 61), c(-1, 2), c(1.5, 2), 3, numeric(0), c(1, NA), c(TRUE, TRUE))) {
    expect_identical(catch_plskit(subset(g, bad))$code, "invalid_argument")
  }
  expect_identical(catch_plskit(subset(g, 1:2, drop = TRUE))$code, "invalid_argument")
  for (bad in list(c(1, 61), c(1, Inf))) {
    e <- catch_plskit(subset(g, bad))
    expect_identical(e$code, "invalid_argument")
    expect_identical(conditionMessage(e), "rows must index the 60 rows of the Gram matrix")
  }
  for (bad in list(c(0, 2), c(1.5, 2))) {
    e <- catch_plskit(subset(g, bad))
    expect_identical(e$code, "invalid_argument")
    expect_identical(conditionMessage(e), "rows must be whole row numbers, at least 1")
  }
})

test_that("gram_matrix_blocks matches gram_matrix on the joined X and checks its blocks", {
  X <- matrix(stats::rnorm(40 * 30), 40, 30)
  cuts <- c(0, 5, 13, ncol(X))
  i <- 0L
  feed <- function() {
    i <<- i + 1L
    if (i >= length(cuts)) return(NULL)
    X[, (cuts[i] + 1):cuts[i + 1], drop = FALSE]
  }
  b <- gram_matrix_blocks(feed)
  g <- gram_matrix(X)
  expect_s3_class(b, c("gram_matrix_result", "plskit_result"), exact = TRUE)
  expect_equal(b$gram, g$gram, tolerance = 1e-12 * max(abs(g$gram)))
  expect_identical(b$n_features, g$n_features)
  i <- 0L
  bs <- gram_matrix_blocks(feed, pre_standardized = TRUE)
  gs <- gram_matrix(X, pre_standardized = TRUE)
  expect_true(bs$pre_standardized)
  expect_equal(bs$gram, gs$gram, tolerance = 1e-12 * max(abs(gs$gram)))
  Xs <- matrix(stats::rnorm(30 * 80), 30, 80)
  ys <- Xs[, 1] + stats::rnorm(30)
  j <- 0L
  half <- function() {
    j <<- j + 1L
    if (j > 2L) NULL else Xs[, (40 * (j - 1) + 1):(40 * j)]
  }
  p1 <- pls1_confirmatory_test(gram = gram_matrix_blocks(half), y = ys, seed = 1)
  p2 <- pls1_confirmatory_test(Xs, ys, seed = 1)
  expect_equal(p1$pvalue, p2$pvalue, tolerance = 1e-12)
  e <- catch_plskit(gram_matrix_blocks(function() NULL))
  expect_identical(e$code, "invalid_input")
  expect_identical(conditionMessage(e), "blocks holds no block")
  m <- 5L
  short <- function() {
    m <<- m - 1L
    if (m < 3L) NULL else X[seq_len(m), 1:3]
  }
  e <- catch_plskit(gram_matrix_blocks(short))
  expect_identical(e$code, "shape_mismatch")
  expect_identical(conditionMessage(e), "block has 3 rows, the first block has 4")
  k <- 0L
  uneven <- function() {
    k <<- k + 1L
    if (k > 2L) NULL else X[seq_len(40 - k), 1:3]
  }
  expect_identical(catch_plskit(gram_matrix_blocks(uneven))$code, "shape_mismatch")
})

test_that("gram= takes a gram_matrix_result, its subset, or a matrix with n_features", {
  X <- matrix(stats::rnorm(30 * 80), 30, 80)
  y <- X[, 1] + stats::rnorm(30)
  a <- pls1_confirmatory_test(X, y, k = 1L, test_method = "split_exact", args = list(n_perm = 19, n_splits = 8), seed = 3)
  g <- gram_matrix(X)
  b <- pls1_confirmatory_test(gram = g, y = y, k = 1L, test_method = "split_exact", args = list(n_perm = 19, n_splits = 8), seed = 3)
  expect_equal(b$statistic, a$statistic, tolerance = 1e-12)
  expect_identical(b$pvalue, a$pvalue)
  c2 <- pls1_confirmatory_test(gram = g$gram, n_features = 80, y = y, k = 1L, test_method = "split_exact", args = list(n_perm = 19, n_splits = 8), seed = 3)
  expect_identical(c2$statistic, b$statistic)
  r <- pls1_perm_alpha(subset(g, 1:25), matrix(y[1:25], ncol = 1), k = 2)
  expect_identical(dim(r$alpha), c(25L, 1L))
  expect_identical(catch_plskit(pls1_confirmatory_test(X, y, gram = g))$code, "invalid_argument")
  expect_identical(catch_plskit(pls1_confirmatory_test(gram = g, y = y, n_features = 80))$code, "invalid_argument")
})

test_that("k takes a set: the tested k and the per-k p-values come back", {
  d <- make_xy()
  run <- function(k) {
    pls1_confirmatory_test(d$X, d$y, k = k, test_method = "split_exact",
                           args = list(n_perm = 19L, n_splits = 5L), seed = 1)
  }
  a <- run(1:3)
  expect_identical(a$k, 1:3)
  expect_type(a$pvalues_by_k, "double")
  expect_identical(names(a$pvalues_by_k), c("1", "2", "3"))
  expect_identical(names(a$statistics_by_k), c("1", "2", "3"))
  expect_null(a$statistic)
  expect_identical(run(c(1, 2, 3)), a)
})

test_that("failed split counts reach the result", {
  # One nonzero row leaves one half of every split with a constant outcome.
  X <- matrix(stats::rnorm(120), 30, 4)
  y <- c(1, rep(0, 29))
  r <- pls1_confirmatory_test(X, y, 1, test_method = "split_exact", args = list(n_perm = 19, n_splits = 8), seed = 1)
  expect_identical(r$n_failed_splits, 8L)
  expect_identical(r$n_failed_perms, 0L)
  expect_null(r$n_failed_folds)
})
