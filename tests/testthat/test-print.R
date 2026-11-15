d <- make_xy()

test_that("print shows a compact summary and returns invisibly", {
  fit <- pls1_fit(d$X, d$y, k = "optimal", k_max = 3, seed = 1)
  out <- capture.output(res <- print(fit))
  expect_identical(res, fit)
  expect_identical(out[[1L]], "<PLS1Result>")
  expect_true(any(grepl("^  T +matrix 60 x [0-9]+$", out)))
  expect_true(any(grepl("^  selection_result +<FindKOptimalResult>$", out)))
  expect_true(any(grepl("^  weights +NULL$", out)))
  expect_output(print(fit$selection_result), "cv_scores +numeric \\[3\\], named")
})
