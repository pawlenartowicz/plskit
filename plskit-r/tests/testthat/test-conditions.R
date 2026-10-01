# Errors and warnings as R conditions.

codes <- plskit:::.plskit_registry()$error_codes
d <- make_xy()

test_that("every error code maps to its condition classes", {
  expect_length(codes, 17L)
  for (code in codes) {
    e <- tryCatch(
      plskit:::.plskit_signal_error(code, "boom", list()),
      condition = function(e) e
    )
    specific <- if (code %in% c("invalid_weights", "resampling_degenerate")) paste0("plskit_", code)
    expect_identical(class(e), c(specific, "plskit_error", "error", "condition"), label = code)
    expect_identical(e$code, code)
    expect_identical(conditionMessage(e), "boom")
  }
})

test_that("engine and bind errors arrive with their codes", {
  X <- d$X
  y <- d$y
  expect_identical(catch_plskit(pls1_fit(X, y, k = 2.5))$code, "invalid_argument")
  expect_identical(
    catch_plskit(pls1_confirmatory_test(X, y, test_method = "split_exact", args = list(bogus = 1)))$code,
    "invalid_args"
  )
  expect_identical(catch_plskit(pls1_confirmatory_test(X, y, test_method = "split_perm"))$code, "invalid_args")
  expect_identical(catch_plskit(pls1_fit(X, y[-1]))$code, "dimension_mismatch")
  X_na <- X
  X_na[1, 1] <- NA
  expect_identical(catch_plskit(pls1_fit(X_na, y))$code, "non_finite_input")
  expect_identical(catch_plskit(pls1_perm_null(X[, 1:4], y, k = 5, n_perm = 200, seed = 7))$code, "k_exceeds_max")
  expect_identical(catch_plskit(rotate(matrix(0, 10, 0)))$code, "invalid_input")
  W <- matrix(stats::rnorm(30), 10, 3)
  expect_identical(catch_plskit(rotate(W, L = matrix(stats::rnorm(80), 40, 2)))$code, "shape_mismatch")
  fit <- pls1_fit(X, y, k = 2)
  expect_identical(catch_plskit(rotate(fit, method = "promax"))$code, "rotation_method_not_implemented")
  expect_identical(catch_plskit(rotate(rotate(fit)))$code, "already_rotated")
})

test_that("the condition call is the user's call", {
  e <- catch_plskit(pls1_fit(d$X, d$y, k = 2.5))
  expect_identical(conditionCall(e), quote(pls1_fit(d$X, d$y, k = 2.5)))
  expect_match(conditionMessage(e), "k must be a non-negative whole number, got 2.5", fixed = TRUE)
})

test_that("invalid_weights carries its reason", {
  w <- rep(1, 60)
  w[3] <- -1
  reason <- tryCatch(
    pls1_fit(d$X, d$y, weights = w),
    plskit_invalid_weights = function(e) e$reason,
    plskit_error = function(e) "wrong class"
  )
  expect_identical(reason, "negative")
})

test_that("resampling_degenerate carries its counts", {
  set.seed(13)
  X <- matrix(stats::rnorm(300), 60, 5)
  y <- X[, 1] + 0.2 * stats::rnorm(60)
  w <- c(rep(1, 5), rep(1e-8, 55))
  e <- tryCatch(
    pls1_rotation_stability(X, y, k = 3, weights = w, n_boot = 500, seed = 0),
    plskit_resampling_degenerate = function(e) e
  )
  expect_s3_class(e, "plskit_resampling_degenerate")
  expect_identical(e$code, "resampling_degenerate")
  expect_identical(e$total, 500L)
  expect_gt(e$skipped, 0L)
  expect_identical(e$threshold, 0.01)
  expect_gt(e$skip_rate, 0.01)
})

test_that("sequence_no_rejection surfaces from pls1_fit", {
  set.seed(13)
  X <- matrix(stats::rnorm(300), 60, 5)
  e <- catch_plskit(pls1_fit(
    X, stats::rnorm(60), k = "sequence", k_max = 3, seed = 1,
    find_k_args = list(
      test_method = "split_exact", alpha = 1e-12,
      args = list(n_perm = 100, n_splits = 10)
    )
  ))
  expect_identical(e$code, "sequence_no_rejection")
})

test_that("a rerouted split_nb warns with class plskit_rerouted", {
  X3 <- d$X[, 1:3]
  w <- NULL
  res <- withCallingHandlers(
    pls1_confirmatory_test(X3, d$y, test_method = "split_nb", seed = 3),
    plskit_rerouted = function(cond) {
      w <<- cond
      invokeRestart("muffleWarning")
    }
  )
  expect_identical(res$test_method, "split_exact")
  expect_identical(class(w), c("plskit_rerouted", "plskit_warning", "warning", "condition"))
  expect_identical(w$requested, "split_nb")
  expect_identical(w$actual, "split_exact")
  expect_identical(w$n_perm, 1000L)
  expect_match(conditionMessage(w), "^'split_nb' was rerouted to 'split_exact'")
  expect_silent(suppressWarnings(
    pls1_confirmatory_test(X3, d$y, test_method = "split_nb", seed = 3),
    classes = "plskit_rerouted"
  ))
})
