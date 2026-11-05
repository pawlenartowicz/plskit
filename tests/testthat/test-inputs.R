# Input coercion (data frames, logicals, integers, arrays).

d <- make_xy()

test_that("whole doubles are accepted where an integer is expected", {
  expect_identical(pls1_fit(d$X, d$y, k = 3)$k_used, 3L)
  expect_identical(pls1_fit(d$X, d$y, k = 3L)$k_used, 3L)
})

test_that("a numeric data.frame is converted, a non-numeric one rejected", {
  expect_equal(pls1_fit(as.data.frame(d$X), d$y, k = 2)$beta, pls1_fit(d$X, d$y, k = 2)$beta)
  bad <- data.frame(a = rep(c("u", "v"), 30), b = d$X[, 1])
  e <- catch_plskit(pls1_fit(bad, d$y))
  expect_identical(e$code, "invalid_argument")
  expect_match(conditionMessage(e), "not numeric: a", fixed = TRUE)
  expect_identical(conditionCall(e), quote(pls1_fit(bad, d$y)))
})

test_that("an integer matrix is promoted to double", {
  Xi <- matrix(as.integer(round(d$X * 10)), 60, 6)
  expect_identical(pls1_fit(Xi, d$y, k = 2)$beta, pls1_fit(Xi * 1.0, d$y, k = 2)$beta)
})

test_that("logical data arguments are read as 1/0 and keep their shape", {
  # A logical X in pls1_fit is covered by "a logical X matrix fits like its
  # 0/1 double matrix" below; this covers the other data arguments.
  Xl <- d$X > 0
  X01 <- Xl * 1
  yl <- d$y > median(d$y)
  expect_identical(pls1_fit(d$X, yl, k = 2)$beta, pls1_fit(d$X, yl * 1, k = 2)$beta)
  expect_identical(pls1_predict(pls1_fit(d$X, d$y, k = 2), Xl), pls1_predict(pls1_fit(d$X, d$y, k = 2), X01))
  Yl <- cbind(yl, d$X[, 3] > 0)
  expect_identical(pls3_fit(d$X, Yl)$U, pls3_fit(d$X, Yl * 1)$U)
  expect_identical(
    pls1_fit(d$X, d$y, k = 2, weights = rep(TRUE, 60))$beta,
    pls1_fit(d$X, d$y, k = 2, weights = rep(1, 60))$beta
  )
  # A data.frame with logical columns is converted like a numeric one.
  df <- data.frame(a = Xl[, 1], b = d$X[, 2])
  expect_identical(pls1_fit(df, d$y)$beta, pls1_fit(cbind(X01[, 1], d$X[, 2]), d$y)$beta)
  # NA in a logical data argument is non-finite input, as in a numeric one.
  Xl[2, 3] <- NA
  expect_identical(catch_plskit(pls1_fit(Xl, d$y))$code, "non_finite_input")
  yl[4] <- NA
  expect_identical(catch_plskit(pls1_fit(d$X, yl))$code, "non_finite_input")
  # A logical flag stays a flag.
  expect_identical(
    pls1_fit(d$X, d$y, k = 2, pre_standardized = FALSE)$beta,
    pls1_fit(d$X, d$y, k = 2)$beta
  )
  expect_identical(catch_plskit(pls1_fit(d$X, d$y, pre_standardized = 1))$code, "invalid_argument")
})

test_that("names on data vectors are ignored", {
  y_named <- stats::setNames(d$y, seq_along(d$y))
  expect_identical(pls1_fit(d$X, y_named, k = 2)$beta, pls1_fit(d$X, d$y, k = 2)$beta)
  w <- stats::setNames(rep(1, 60), seq_len(60))
  expect_identical(pls1_fit(d$X, d$y, k = 2, weights = w)$beta, pls1_fit(d$X, d$y, k = 2, weights = rep(1, 60))$beta)
})

test_that("a vector y is 1-D and a one-column Y stays 2-D", {
  expect_length(preprocess(d$X, d$y)$Y_std, 60L)
  expect_identical(dim(preprocess(d$X, matrix(d$y))$Y_std), c(60L, 1L))
})

test_that("NA values reach the engine as non-finite input", {
  y_na <- d$y
  y_na[5] <- NA
  expect_identical(catch_plskit(pls1_fit(d$X, y_na))$code, "non_finite_input")
  y_int <- as.integer(round(d$y))
  expect_identical(pls1_fit(d$X, y_int, k = 2)$beta, pls1_fit(d$X, y_int * 1.0, k = 2)$beta)
  y_int[5] <- NA
  expect_identical(catch_plskit(pls1_fit(d$X, y_int))$code, "non_finite_input")
})

test_that("unsupported R values raise invalid_argument", {
  expect_identical(catch_plskit(pls1_fit(d$X, factor(d$y > 0)))$code, "invalid_argument")
  expect_identical(catch_plskit(pls1_fit(d$X, d$y, k = c(1, 2)))$code, "invalid_argument")
  expect_identical(catch_plskit(pls1_fit(d$X, d$y, pre_standardized = NA))$code, "invalid_argument")
  expect_identical(
    catch_plskit(pls1_confirmatory_test(d$X, d$y, test_method = "split_exact", args = list(1, 2)))$code,
    "invalid_argument"
  )
})

test_that("an empty args list and NULL args entries mean the defaults", {
  a <- pls1_confirmatory_test(d$X, d$y, test_method = "split_exact", args = list(), seed = 1)
  b <- pls1_confirmatory_test(d$X, d$y, test_method = "split_exact", args = list(n_perm = NULL), seed = 1)
  c <- pls1_confirmatory_test(d$X, d$y, test_method = "split_exact", seed = 1)
  expect_identical(a, c)
  expect_identical(b, c)
})

test_that("an empty args list nested in find_k_args means the defaults", {
  a <- pls1_fit(d$X, d$y, k = "optimal", k_max = 3, find_k_args = list(args = list()), seed = 1)
  b <- pls1_fit(d$X, d$y, k = "optimal", k_max = 3, seed = 1)
  expect_identical(a, b)
})

test_that("bit64::integer64 seed converts exactly; elsewhere it is rejected", {
  skip_if_not_installed("bit64")
  s <- "4611686018427387904"
  r64 <- pls1_confirmatory_test(d$X, d$y, test_method = "split_exact", seed = bit64::as.integer64(s))
  rchr <- pls1_confirmatory_test(d$X, d$y, test_method = "split_exact", seed = s)
  expect_identical(r64, rchr)
  expect_identical(r64$seed, s)

  Xi64 <- as.data.frame(d$X)
  Xi64[[1L]] <- bit64::as.integer64(round(Xi64[[1L]] * 10))
  err <- catch_plskit(pls1_fit(Xi64, d$y))
  expect_identical(err$code, "invalid_argument")

  err_vec <- catch_plskit(pls1_fit(d$X, bit64::as.integer64(round(d$y))))
  expect_identical(err_vec$code, "invalid_argument")
})

test_that("a 1-D array (length-1 dim) behaves like a plain vector", {
  # An integer data argument: array(y_int) reaches from_r.rs::integers()
  # with a length-1 dim, same as the y_int vector used above.
  y_int <- as.integer(round(d$y))
  expect_identical(pls1_fit(d$X, array(y_int), k = 2)$beta, pls1_fit(d$X, y_int, k = 2)$beta)

  # A logical scalar flag: pre_standardized has kind "bool", so it is
  # never promoted to double by .plskit_call and reaches
  # from_r.rs::logicals() directly; array(TRUE) has a length-1 dim.
  expect_identical(
    pls1_fit(d$X, d$y, k = 2, pre_standardized = array(TRUE))$beta,
    pls1_fit(d$X, d$y, k = 2, pre_standardized = TRUE)$beta
  )

  # A logical data vector (kind "vec"): promoted to double by .plskit_call,
  # so array(y_logical) and y_logical must fit identically.
  y_logical <- d$y > median(d$y)
  expect_identical(
    pls1_fit(d$X, array(y_logical), k = 2)$beta,
    pls1_fit(d$X, y_logical, k = 2)$beta
  )
})

test_that("a logical X matrix fits like its 0/1 double matrix", {
  # kind "mat": promoted to double by .plskit_call, keeping its dim.
  X_logical <- d$X > 0
  expect_identical(
    pls1_fit(X_logical, d$y, k = 2)$beta,
    pls1_fit(X_logical + 0, d$y, k = 2)$beta
  )
})
