# The one path from R into Rust. Every generated stub in stubs.R calls
# .plskit_call(); inputs are prepared in R and converted in Rust, and
# engine errors and warnings come back as R conditions.

.plskit_env <- new.env(parent = emptyenv())

# The registry (functions, result types), read once from Rust.
.plskit_registry <- function() {
  if (is.null(.plskit_env$registry)) {
    .plskit_env$registry <- .plskit_registry_impl()
  }
  .plskit_env$registry
}

# The registry kind of one top-level parameter, or NA if fn or arg is not
# found (the caller then treats it as "not a numeric-array kind").
.plskit_kind_of <- function(fn, arg) {
  reg <- .plskit_registry()
  f <- Find(function(f) identical(f$name, fn), reg$functions)
  if (is.null(f)) return(NA_character_)
  p <- Find(function(p) identical(p$name, arg), f$params)
  if (is.null(p)) NA_character_ else p$kind
}

# Kinds whose R argument is a numeric array (a 1-D or 2-D float array on
# the Rust side). A logical value under one of these is promoted to
# double before conversion, the same way numpy casts bool to float; a
# scalar flag (kind "bool", e.g. `pre_standardized`, `ci`) is never
# touched, because R has no way to tell a length-1 data vector from a
# scalar flag except by looking up the declared kind. The strings mirror
# `ParamKind::as_str` in plskit-bind's registry — change together.
.plskit_numeric_array_kinds <- c("mat", "vec", "vec_or_mat")

# Raise invalid_argument (through the same condition path as every other
# seam error) when a string is not valid UTF-8 after re-encoding: this
# catches bytes-encoded input enc2utf8() cannot fix up.
.plskit_check_utf8 <- function(s, call) {
  if (!all(validUTF8(s))) {
    .plskit_signal_error("invalid_argument", "a string is not valid UTF-8", call = call)
  }
  s
}

# bit64::integer64 stores its data as raw double bits with class
# "integer64"; read as a plain double it silently gives the wrong number
# (or NaN). Detected by class alone, so the package does not need to
# depend on bit64. The `seed` argument is the one place an integer64
# value has an exact, lossless R representation (as.character(), the
# same decimal string the seam already accepts); everywhere else there
# is no safe conversion, so it raises invalid_argument.
.plskit_check_integer64 <- function(x, is_seed, call) {
  if (!inherits(x, "integer64")) return(x)
  if (is_seed) return(as.character(x))
  .plskit_signal_error(
    "invalid_argument",
    "bit64::integer64 is not supported here; convert it with as.numeric()",
    call = call
  )
}

# R-side input preparation, applied recursively: a data.frame of numeric
# (or logical) columns becomes a matrix, an integer or logical matrix
# becomes double (the engine takes float64 input; logical vectors are promoted at the top level in
# .plskit_call, where the registry kind is known), strings and names are
# re-encoded as UTF-8 and checked. Everything else is read by the Rust
# seam. `is_seed` is TRUE only for the top-level `seed` argument (there,
# bit64::integer64 means an exact-but-huge seed, see
# .plskit_check_integer64).
.plskit_prep <- function(x, call, is_seed = FALSE) {
  x <- .plskit_check_integer64(x, is_seed, call)
  if (is.data.frame(x)) {
    is64_col <- vapply(x, inherits, logical(1), what = "integer64")
    if (any(is64_col)) {
      .plskit_signal_error(
        "invalid_argument",
        paste0(
          "bit64::integer64 is not supported here; convert with as.numeric(): ",
          paste(names(x)[is64_col], collapse = ", ")
        ),
        call = call
      )
    }
    # A logical column counts as numeric (TRUE/FALSE become 1/0 below),
    # the same way a bool column of a pandas DataFrame does in Python.
    numeric_col <- vapply(x, function(col) is.numeric(col) || is.logical(col), logical(1))
    if (!all(numeric_col)) {
      .plskit_signal_error(
        "invalid_argument",
        paste0(
          "a data.frame argument must have only numeric columns; not numeric: ",
          paste(names(x)[!numeric_col], collapse = ", ")
        ),
        call = call
      )
    }
    x <- as.matrix(x)
  }
  if (is.matrix(x) && (is.integer(x) || is.logical(x))) storage.mode(x) <- "double"
  if (is.character(x)) x <- .plskit_check_utf8(enc2utf8(x), call)
  nm <- names(x)
  if (!is.null(nm)) names(x) <- .plskit_check_utf8(enc2utf8(nm), call)
  if (is.list(x) && length(x) > 0L) x[] <- lapply(x, .plskit_prep, call = call)
  x
}

.plskit_call <- function(fn, inputs) {
  caller <- sys.call(-1L)
  arg_names <- names(inputs)
  prepped <- lapply(arg_names, function(nm) {
    value <- inputs[[nm]]
    if (is.logical(value) && .plskit_kind_of(fn, nm) %in% .plskit_numeric_array_kinds) {
      # storage.mode<- keeps dim (a logical matrix stays 2-D) and turns NA
      # into NA_real_, so the engine answers non_finite_input for it, as
      # for NA in a numeric argument.
      storage.mode(value) <- "double"
    }
    .plskit_prep(value, call = caller, is_seed = identical(nm, "seed"))
  })
  names(prepped) <- arg_names
  out <- .plskit_call_impl(fn, prepped)
  if (!isTRUE(out$ok)) {
    .plskit_signal_error(out$code, out$message, out$details, call = caller)
  }
  for (w in out$warnings) .plskit_signal_warning(w, call = caller)
  out$result
}

# c("plskit_<code>", "plskit_error", ...) for the two codes that carry
# structured details, c("plskit_error", ...) for every other code.
.plskit_error_class <- function(code) {
  specific <- if (code %in% c("invalid_weights", "resampling_degenerate")) {
    paste0("plskit_", code)
  }
  c(specific, "plskit_error", "error", "condition")
}

.plskit_signal_error <- function(code, message, details = list(), call = NULL) {
  cond <- structure(
    c(list(message = message, call = call, code = code), details),
    class = .plskit_error_class(code)
  )
  stop(cond)
}

# A structured warning record from plskit-bind: `kind` picks the class,
# `message` is the pre-rendered sentence, every other field rides along.
.plskit_signal_warning <- function(w, call = NULL) {
  fields <- w[setdiff(names(w), "message")]
  cond <- structure(
    c(list(message = w$message, call = call), fields),
    class = c(paste0("plskit_", w$kind), "plskit_warning", "warning", "condition")
  )
  warning(cond)
}
