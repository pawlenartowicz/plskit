# Base-R reader for the testdata/ corpus (spec section 7): .npz files are
# zip archives of .npy v1.0 arrays, little-endian float64 ('<f8'), int64
# ('<i8') or uint8 strings ('|u1'), C order. No test dependency.

# An env switch is "on" only for "1" or "true" (case-insensitive); anything
# else, including unset, is off.
env_flag <- function(name) {
  tolower(Sys.getenv(name)) %in% c("1", "true")
}

read_npy <- function(path) {
  con <- file(path, "rb")
  on.exit(close(con))
  magic <- readBin(con, "raw", 6L)
  if (!identical(magic, as.raw(c(0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59)))) {
    stop(path, ": not a .npy file")
  }
  major <- readBin(con, "integer", 1L, size = 1L, signed = FALSE)
  if (!major %in% c(1L, 2L, 3L)) stop(path, ": unsupported .npy version ", major)
  readBin(con, "integer", 1L, size = 1L, signed = FALSE) # minor version
  header_len <- if (major == 1L) {
    readBin(con, "integer", 1L, size = 2L, signed = FALSE, endian = "little")
  } else {
    readBin(con, "integer", 1L, size = 4L, endian = "little")
  }
  header <- rawToChar(readBin(con, "raw", header_len))
  descr <- sub(".*'descr': *'([^']+)'.*", "\\1", header)
  fortran <- grepl("'fortran_order': *True", header)
  shape_txt <- gsub(" ", "", sub(".*'shape': *\\(([^)]*)\\).*", "\\1", header))
  shape <- as.integer(strsplit(shape_txt, ",")[[1L]])
  if (length(shape) > 2L) stop(path, ": unsupported ndim ", length(shape))
  n <- prod(shape)
  values <- switch(descr,
    "<f8" = {
      v <- readBin(con, "double", n, size = 8L, endian = "little")
      if (length(v) != n) stop(path, ": truncated f8 data (", length(v), " of ", n, ")")
      v
    },
    "<i8" = {
      raw_bytes <- readBin(con, "raw", 8L * n)
      if (length(raw_bytes) != 8L * n) {
        stop(path, ": truncated i8 data (", length(raw_bytes), " of ", 8L * n, " bytes)")
      }
      b <- matrix(as.numeric(raw_bytes), nrow = 8L)
      negative <- b[8L, ] >= 128
      # Two's complement, exact for magnitudes below 2^53.
      v <- ifelse(negative, -(colSums((255 - b) * 256^(0:7)) + 1), colSums(b * 256^(0:7)))
      if (any(abs(v) >= 2^53)) {
        stop(path, ": i64 magnitude >= 2^53, not exactly representable as double")
      }
      v
    },
    "|u1" = {
      raw_bytes <- readBin(con, "raw", n)
      if (length(raw_bytes) != n) {
        stop(path, ": truncated u1 data (", length(raw_bytes), " of ", n, " bytes)")
      }
      rawToChar(raw_bytes)
    },
    stop(path, ": unsupported dtype ", descr)
  )
  if (descr == "|u1") {
    return(structure(values, npy_kind = "str", npy_shape = shape))
  }
  if (length(shape) == 2L) {
    values <- matrix(values, shape[[1L]], shape[[2L]], byrow = !fortran)
  }
  structure(values, npy_kind = if (descr == "<f8") "f64" else "i64", npy_shape = shape)
}

read_npz <- function(path) {
  dir <- tempfile("npz")
  dir.create(dir)
  on.exit(unlink(dir, recursive = TRUE))
  files <- utils::unzip(path, exdir = dir)
  out <- lapply(files, read_npy)
  names(out) <- sub("\\.npy$", "", basename(files))
  out
}

# The testdata/ directory: $PLSKIT_TESTDATA, else the first ancestor of the
# test directory that holds testdata/manifest.json. Skips when neither is
# found, unless PLSKIT_REQUIRE_CORPUS is set (CI), which makes it an error.
corpus_root <- function() {
  env <- Sys.getenv("PLSKIT_TESTDATA")
  if (nzchar(env)) return(normalizePath(env, mustWork = TRUE))
  dir <- normalizePath(testthat::test_path("."), mustWork = FALSE)
  repeat {
    candidate <- file.path(dir, "testdata")
    if (file.exists(file.path(candidate, "manifest.json"))) return(candidate)
    parent <- dirname(dir)
    if (identical(parent, dir)) break
    dir <- parent
  }
  if (env_flag("PLSKIT_REQUIRE_CORPUS")) {
    stop("testdata/ not found; set PLSKIT_TESTDATA")
  }
  testthat::skip("testdata/ not found; set PLSKIT_TESTDATA to run the corpus test")
}
