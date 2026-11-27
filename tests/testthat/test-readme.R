# Runs the R examples in README.md as written. From the first ```r block
# that loads plskit, every later ```r block runs in one fresh environment,
# so a block may use names an earlier one defined; blocks before it
# (install lines) are skipped. Same rule as plskit-py/tests/test_readme.py:
# change together.
#
# .Rbuildignore keeps README.md out of the built package, so the README is
# read from the checkout, found next to testdata/ (see corpus_root()).

test_that("the README examples run", {
  readme <- file.path(dirname(corpus_root()), "plskit-r", "README.md")
  skip_if_not(file.exists(readme), "plskit-r/README.md not found next to testdata/")
  text <- paste(readLines(readme, encoding = "UTF-8"), collapse = "\n")
  blocks <- regmatches(text, gregexpr("(?ms)^```r\n.*?^```", text, perl = TRUE))[[1]]
  blocks <- sub("(?s)^```r\n(.*)```$", "\\1", blocks, perl = TRUE)
  start <- which(grepl("library(plskit)", blocks, fixed = TRUE))[1]
  expect_false(is.na(start))
  env <- new.env(parent = globalenv())
  for (block in blocks[start:length(blocks)]) {
    expect_no_error(eval(parse(text = block), envir = env))
  }
})
