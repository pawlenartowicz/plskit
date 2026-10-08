# Every testdata/ case through the R surface, compared with the frozen
# outputs at the corpus tolerances (scalars 1e-12, arrays 1e-10, plus
# rtol 1e-14 for finite expected values; integers and strings exact). Each case makes
# the call plskit-py/tests/test_corpus.py makes, and a fixture key maps
# onto the result by name: exact, then ASCII case-insensitive, then into
# nested named lists, recursively. `{field}__keys` / `{field}__values`
# encode integer-keyed maps (named numeric vectors here) and
# `{field}_{point|lower|upper|sd}` (`{field}_k{i}_{part}` for list items)
# encode CIScalars.

# PLSKIT_CORPUS_EXACT=1 compares with atol 0 (an exact-equality
# canary): values are compared with `==`, so -0 equals 0 and NaN payload
# bits are not compared. This is not a bit-identity check, and it is only
# meaningful on Linux x86_64, the host the fixtures come from.
EXACT <- env_flag("PLSKIT_CORPUS_EXACT")
ATOL_SCALAR <- if (EXACT) 0 else 1e-12
ATOL_ARRAY <- if (EXACT) 0 else 1e-10
RTOL <- if (EXACT) 0 else 1e-14

describes_data <- function(fn, key) {
  key %in% c("d", "n", "n_new", "n_train", "seed_new", "seed_train") ||
    (key == "seed" && fn %in% c("spls1_fit", "pls3_fit"))
}

plain <- function(a) {
  attr(a, "npy_kind") <- NULL
  attr(a, "npy_shape") <- NULL
  a
}

call_fn <- function(fn, args) do.call(getExportedValue("plskit", fn), args)

run_case <- function(case, inputs) {
  fn <- case[["function"]]
  kw <- case[["kwargs"]]
  arr <- function(name) {
    if (is.null(inputs[[name]])) stop(case[["name"]], ": input ", name, " missing")
    plain(inputs[[name]])
  }
  has_weights <- identical(kw[["weights"]], "nonuniform")
  stopifnot(has_weights == !is.null(inputs[["weights"]]))
  switch(fn,
    pls1_predict = {
      fit <- pls1_fit(arr("X_train"), arr("y_train"), k = kw[["k"]])
      fit$y_pred <- pls1_predict(fit, arr("X_new"))
      fit
    },
    rotate = {
      fit <- pls1_fit(arr("X"), arr("y"), k = kw[["k"]])
      rotate(fit$W, method = kw[["method"]])
    },
    preprocess = {
      preprocess(arr("X"), arr("y"), weights = if (has_weights) arr("weights"))
    },
    pls3_transform = {
      model <- pls3_fit(arr("X"), arr("Y"), k = kw[["k"]])
      pls3_transform(model, arr("X_new"), arr("Y_new"), which = kw[["which"]])
    },
    {
      args <- list(X = arr("X"))
      if (!is.null(inputs[["y"]])) args[["y"]] <- arr("y")
      if (!is.null(inputs[["Y"]])) args[["Y"]] <- arr("Y")
      if (has_weights) args["weights"] <- list(arr("weights"))
      for (key in names(kw)) {
        if (key != "weights" && !describes_data(fn, key)) args[key] <- list(kw[[key]])
      }
      call_fn(fn, args)
    }
  )
}

# A found value is list(value); NULL means "no such field".
lookup <- function(rec, key) {
  if (!is.list(rec) || is.null(names(rec))) return(NULL)
  if (key %in% names(rec)) return(list(rec[[key]]))
  i <- match(tolower(key), tolower(names(rec)))
  if (!is.na(i)) return(list(rec[[i]]))
  for (v in rec) {
    if (is.list(v) && !is.null(names(v))) {
      found <- lookup(v, key)
      if (!is.null(found)) return(found)
    }
  }
  NULL
}

resolve <- function(rec, key) {
  if (endsWith(key, "__keys")) {
    found <- lookup(rec, sub("__keys$", "", key))
    if (is.null(found) || is.null(names(found[[1L]]))) return(NULL)
    return(list(as.integer(names(found[[1L]]))))
  }
  if (endsWith(key, "__values")) {
    found <- lookup(rec, sub("__values$", "", key))
    if (is.null(found) || is.null(names(found[[1L]]))) return(NULL)
    return(list(unname(found[[1L]])))
  }
  found <- lookup(rec, key)
  if (!is.null(found)) return(found)
  part <- sub(".*_", "", key)
  if (!part %in% c("point", "lower", "upper", "sd")) return(NULL)
  base <- sub("_[^_]*$", "", key)
  ci <- lookup(rec, base)
  if (is.null(ci)) {
    m <- regmatches(base, regexec("^(.*)_k([0-9]+)$", base))[[1L]]
    if (length(m) != 3L) return(NULL)
    items <- lookup(rec, m[[2L]])
    i <- as.integer(m[[3L]]) + 1L
    if (is.null(items) || length(items[[1L]]) < i) return(NULL)
    ci <- list(items[[1L]][[i]])
  }
  if (!is.list(ci[[1L]]) || is.null(ci[[1L]][[part]])) return(NULL)
  list(ci[[1L]][[part]])
}

compare <- function(got, want) {
  kind <- attr(want, "npy_kind")
  shape <- attr(want, "npy_shape")
  want <- plain(want)
  if (kind == "str") {
    if (identical(got, want)) return(NULL)
    return(sprintf("%s vs %s", format(got), want))
  }
  if (kind == "i64") {
    if (!(is.integer(got) || is.logical(got) || is.character(got))) {
      return(sprintf(
        "R class %s, expected integer/logical/character for an i64 field",
        paste(class(got), collapse = ",")
      ))
    }
    got_num <- if (is.character(got)) suppressWarnings(as.numeric(got)) else as.numeric(unname(got))
    ok <- length(got_num) == length(want) &&
      isTRUE(all(!is.na(got_num) & got_num == as.vector(want)))
    if (ok) return(NULL)
    return(sprintf("[%s] vs [%s]", toString(got_num), toString(want)))
  }
  if (!is.double(got)) {
    return(sprintf("R type %s, expected double for an f64 field", typeof(got)))
  }
  if (length(shape) == 2L) {
    if (!is.matrix(got) || !identical(dim(got), as.integer(shape))) {
      return(sprintf("shape %s vs %s", toString(dim(got)), toString(shape)))
    }
  } else if (is.matrix(got) || length(got) != length(want)) {
    return(sprintf("shape %s vs length %d", toString(dim(got)), length(want)))
  }
  # Mirrors plskit-testdata-gen/src/settle.rs::close, which owns the rule:
  # change together.
  atol <- if (length(shape) == 0L) ATOL_SCALAR else ATOL_ARRAY
  got <- as.vector(got)
  want <- as.vector(want)
  ok <- (is.nan(got) & is.nan(want)) | got == want |
    (is.finite(want) & abs(got - want) <= atol + RTOL * abs(want))
  ok[is.na(ok)] <- FALSE
  if (all(ok)) return(NULL)
  bad <- which(!ok)[[1L]]
  sprintf("[%d] %.17g vs %.17g", bad, got[[bad]], want[[bad]])
}

test_that("every testdata case matches through the R surface", {
  root <- corpus_root()
  manifest <- jsonlite::fromJSON(file.path(root, "manifest.json"), simplifyVector = FALSE)
  cases <- manifest[["cases"]]
  expect_gt(length(cases), 0L)
  failures <- character()
  n_compared <- 0L
  for (case in cases) {
    inputs <- read_npz(file.path(root, case[["inputs"]]))
    expected <- read_npz(file.path(root, case[["outputs"]]))
    # Some split_nb cases reroute by design; the warning is tested elsewhere.
    actual <- suppressWarnings(run_case(case, inputs), classes = "plskit_rerouted")
    for (key in sort(names(expected))) {
      found <- resolve(actual, key)
      if (is.null(found)) {
        failures <- c(failures, sprintf("%s.%s: no matching result field", case[["name"]], key))
        next
      }
      n_compared <- n_compared + 1L
      problem <- compare(found[[1L]], expected[[key]])
      if (!is.null(problem)) failures <- c(failures, sprintf("%s.%s: %s", case[["name"]], key, problem))
    }
  }
  expect_gt(n_compared, 0L)
  expect(length(failures) == 0L, paste(c("corpus mismatches:", failures), collapse = "\n"))
})
