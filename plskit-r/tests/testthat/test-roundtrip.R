# Results passed back into the package.

d <- make_xy()

test_that("predict, rotate and transform take returned results", {
  fit <- pls1_fit(d$X, d$y, k = 3)
  expect_s3_class(fit, c("pls1_result", "plskit_result"), exact = TRUE)
  pred <- pls1_predict(fit, d$X[1:5, ])
  expect_type(pred, "double")
  expect_length(pred, 5L)
  rot <- rotate(fit)
  expect_s3_class(rot, c("pls1_result", "plskit_result"), exact = TRUE)
  expect_s3_class(rot$rotation_spec, c("rotation_spec", "plskit_result"), exact = TRUE)
  expect_equal(pls1_predict(rot, d$X[1:5, ]), pred, tolerance = 1e-10)
  Y <- cbind(d$y, d$y + stats::rnorm(60))
  p3 <- pls3_fit(d$X, Y, k = 2)
  sc <- pls3_transform(p3, X_new = d$X, which = "x_scores")
  expect_s3_class(sc, c("pls3_scores", "plskit_result"), exact = TRUE)
  expect_equal(sc$x_scores, p3$x_scores, tolerance = 1e-10)
  expect_null(sc$y_scores)
})

test_that("rotate(fit) carries selection_result through unchanged", {
  fit <- pls1_fit(d$X, d$y, k = "optimal", k_max = 4, seed = 7)
  expect_s3_class(fit$selection_result, c("find_k_optimal_result", "plskit_result"), exact = TRUE)
  expect_identical(names(fit$selection_result$cv_scores), c("1", "2", "3", "4"))
  expect_identical(rotate(fit)$selection_result, fit$selection_result)
})

test_that("rotate dispatches on class: a matrix gives a rotate_result", {
  fit <- pls1_fit(d$X, d$y, k = 2)
  rw <- rotate(fit$W)
  expect_s3_class(rw, c("rotate_result", "plskit_result"), exact = TRUE)
  expect_equal(rw$W_rot, rotate(fit)$W, tolerance = 1e-12)
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
  broken$T <- NULL
  expect_identical(catch_plskit(pls1_predict(broken, d$X))$code, "invalid_argument")
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
  rs <- pls1_rotation_stability(d$X, d$y, 2, n_boot = 100, seed = 1)
  expect_s3_class(rs$variance_ratio, c("ci_scalar", "plskit_result"), exact = TRUE)
  expect_type(rs$variance_ratio_per_axis, "list")
  expect_null(names(rs$variance_ratio_per_axis))
  expect_length(rs$variance_ratio_per_axis, 2L)
  kp <- spls1_find_keep_optimal(d$X, d$y, 1, seed = 1)
  expect_type(kp$keep_grid, "integer")
  s3 <- spls3_fit(d$X, cbind(d$y, d$y + stats::rnorm(60), stats::rnorm(60)), 2, 3, 2)
  expect_type(s3$converged, "logical")
  expect_type(s3$n_iter, "integer")
})

test_that("every result's fields follow the registry order", {
  types <- plskit:::.plskit_registry()$result_types
  fields <- lapply(types, function(t) vapply(t$fields, function(f) f$name, ""))
  names(fields) <- vapply(types, function(t) t$r_class, "")
  check <- function(x) {
    if (inherits(x, "plskit_result")) {
      expect_identical(names(x), fields[[class(x)[[1L]]]], label = class(x)[[1L]])
    }
    if (is.list(x)) for (v in x) check(v)
  }
  check(pls1_fit(d$X, d$y, k = "optimal", k_max = 3, seed = 1))
  check(pls1_confirmatory_test(d$X, d$y, test_method = "split_exact", ci = TRUE, n_boot = 100, seed = 1))
  check(pls1_rotation_stability(d$X, d$y, 2, n_boot = 100, seed = 1))
})

# Input coercion and malformed passed-back records.

test_that("a logical data vector is promoted to double, like numpy's bool cast", {
  y_logical <- d$y > 0
  fit_logical <- pls1_fit(d$X, y_logical, k = 2)
  fit_numeric <- pls1_fit(d$X, as.numeric(y_logical), k = 2)
  expect_identical(fit_logical, fit_numeric)
})

test_that("a scalar logical flag is never promoted (it stays a flag, not data)", {
  fit_true <- pls1_confirmatory_test(d$X, d$y, test_method = "split_exact", ci = TRUE, n_boot = 100, seed = 1)
  fit_false <- pls1_confirmatory_test(d$X, d$y, test_method = "split_exact", ci = FALSE, n_boot = 100, seed = 1)
  expect_false(isTRUE(all.equal(fit_true, fit_false)))
})

test_that("a 1-D array is a plain vector, not rejected", {
  fit_vec <- pls1_fit(d$X, d$y, k = 2)
  fit_arr <- pls1_fit(d$X, array(d$y), k = 2)
  expect_identical(fit_vec, fit_arr)
})

test_that("literal duplicate int-map names raise invalid_argument", {
  fit <- pls1_fit(d$X, d$y, k = "optimal", k_max = 4, seed = 7)
  broken <- fit
  names(broken$selection_result$cv_scores) <- c("1", "1", "3", "4")
  err <- catch_plskit(rotate(broken))
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
  rot <- rotate(broken)
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
  err <- catch_plskit(rotate(pls1_fit(d$X, d$y, k = 1), method = bad))
  expect_identical(err$code, "invalid_argument")
})

# Smoke tests for functions no other test calls.

test_that("plssvd_fit, plssvd_transform and split_nb_gate work", {
  Y <- cbind(d$y, d$y + stats::rnorm(60))
  fit <- plssvd_fit(d$X, Y, k = 2)
  expect_s3_class(fit, c("pls3_result", "plskit_result"), exact = TRUE)
  expect_identical(dim(fit$U), c(6L, 2L))

  sc <- plssvd_transform(fit, X_new = d$X, which = "x_scores")
  expect_s3_class(sc, c("pls3_scores", "plskit_result"), exact = TRUE)
  expect_identical(dim(sc$x_scores), c(60L, 2L))

  gate <- split_nb_gate(d$X)
  expect_s3_class(gate, c("split_nb_gate_result", "plskit_result"), exact = TRUE)
  expect_type(gate$n_eff, "double")
})

test_that("a byte-string name that is not valid UTF-8 raises invalid_argument", {
  bad <- "\xff"
  Encoding(bad) <- "bytes"
  bad_named <- list(seed = 1L)
  names(bad_named) <- bad
  err <- catch_plskit(pls1_fit(d$X, d$y, k = 1, find_k_args = bad_named))
  expect_identical(err$code, "invalid_argument")
})
