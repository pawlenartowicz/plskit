# Errors and warnings as R conditions.

d <- make_xy()

test_that("an engine error arrives with its code and class vector", {
  e <- catch_plskit(pls1_fit(d$X, d$y[-1]))
  expect_identical(e$code, "dimension_mismatch")
  expect_identical(class(e), c("plskit_error", "error", "condition"))
  # A 10 x 0 matrix crosses the seam; the engine is what rejects it.
  expect_identical(catch_plskit(rotate(matrix(0, 10, 0)))$code, "invalid_input")
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
})
