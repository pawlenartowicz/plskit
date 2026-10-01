# The seed is a decimal string on results and accepts a string, an
# integer or a whole double up to 2^53 (spec section 4.3).

d <- make_xy()
test_fn <- function(seed) {
  pls1_confirmatory_test(d$X, d$y, method = "split_exact",
                         args = list(n_perm = 100, n_splits = 5), seed = seed)
}

test_that("a drawn seed is a decimal string and reproduces the run", {
  first <- test_fn(NULL)
  expect_type(first$seed, "character")
  expect_match(first$seed, "^[0-9]+$")
  expect_identical(test_fn(first$seed), first)
})

test_that("integer, double and string seeds are the same seed", {
  expect_identical(test_fn(42L), test_fn(42))
  expect_identical(test_fn("42"), test_fn(42))
  expect_identical(test_fn(42)$seed, "42")
  expect_identical(test_fn(2^53)$seed, "9007199254740992")
})

test_that("seeds above 2^63 survive exactly", {
  expect_identical(test_fn("18446744073709551615")$seed, "18446744073709551615")
  expect_identical(test_fn("13835058055282163712")$seed, "13835058055282163712")
})

test_that("invalid seeds raise invalid_argument", {
  for (bad in list(2^53 + 2, -1, 1.5, "abc", "18446744073709551616", NA)) {
    expect_identical(catch_plskit(test_fn(bad))$code, "invalid_argument", label = format(bad))
  }
})

test_that("a selection result's seed reproduces the selection", {
  fit <- pls1_fit(d$X, d$y, k = "optimal", k_max = 4)
  again <- pls1_fit(d$X, d$y, k = "optimal", k_max = 4, seed = fit$selection_result$seed)
  expect_identical(again, fit)
})
