# Build the plskit Rust static library. Run by src/Makevars and
# src/Makevars.win from the src/ directory:
#
#   Rscript ../tools/cargo-build.R [--target=<rust target triple>] [--clippy]
#
# --clippy runs `cargo clippy ... -- -D warnings` on the same manifest
# instead of building (the lint gate for this crate).
#
# Dev mode: when PLSKIT_CARGO_CONFIG names a cargo config file holding a
# [patch.crates-io] block for plskit and plskit-bind (absolute paths), the
# crate manifest is copied to rust/target/dev-crate/ with both requirements
# relaxed to "*" (a [patch] applies only when the patched version satisfies
# the requirement), cargo builds that copy with --config <file>, and the
# lock file is checked to confirm both crates came from the patch. The
# committed rust/Cargo.toml is never edited. Without PLSKIT_CARGO_CONFIG
# the build uses the committed rust/Cargo.lock with --locked.
#
# Release mode: the release tarball (r-universe branch, CRAN) carries every
# crate in rust/vendor.tar.xz, written by the vX.Y.Z-r release job. When
# that file is present the build unpacks it and runs offline against it.

args <- commandArgs(trailingOnly = TRUE)
target <- sub("^--target=", "", grep("^--target=", args, value = TRUE))
clippy <- "--clippy" %in% args

fail <- function(...) {
  message("plskit: ", ...)
  quit(status = 1, save = "no")
}

crate_dir <- normalizePath("rust", winslash = "/", mustWork = TRUE)
target_dir <- file.path(crate_dir, "target")
manifest <- file.path(crate_dir, "Cargo.toml")
config <- Sys.getenv("PLSKIT_CARGO_CONFIG")

cargo <- Sys.which("cargo")
if (!nzchar(cargo)) {
  home <- Sys.getenv(if (.Platform$OS.type == "windows") "USERPROFILE" else "HOME")
  exe <- if (.Platform$OS.type == "windows") "cargo.exe" else "cargo"
  cargo <- file.path(home, ".cargo", "bin", exe)
  if (!file.exists(cargo)) fail("cargo not found; install Rust from https://rustup.rs")
}

flags <- c(if (clippy) "clippy" else "build", "--lib", "--release", "--target-dir", target_dir)
if (length(target) == 1L) {
  flags <- c(flags, paste0("--target=", target))
  if (grepl("windows-gnu$", target)) {
    # rustc passes -lgcc_eh, which Rtools' GCC lacks; an empty archive
    # satisfies the linker (rextendr does the same).
    mock <- file.path(target_dir, "libgcc_mock")
    dir.create(mock, recursive = TRUE, showWarnings = FALSE)
    file.create(file.path(mock, "libgcc_eh.a"))
    Sys.setenv(LIBRARY_PATH = paste(Sys.getenv("LIBRARY_PATH"), mock, sep = ";"))
  }
}

if (nzchar(config)) {
  if (!file.exists(config)) fail("PLSKIT_CARGO_CONFIG names a missing file: ", config)
  dev_dir <- file.path(target_dir, "dev-crate")
  dir.create(dev_dir, recursive = TRUE, showWarnings = FALSE)
  toml <- readLines(manifest)
  toml <- sub('^(plskit|plskit-bind) = .*$', '\\1 = "*"', toml)
  lib <- which(toml == "[lib]")
  if (length(lib) != 1L) fail("rust/Cargo.toml must have exactly one [lib] table")
  toml <- append(toml, sprintf("path = '%s/src/lib.rs'", crate_dir), after = lib)
  writeLines(toml, file.path(dev_dir, "Cargo.toml"))
  manifest <- file.path(dev_dir, "Cargo.toml")
  # Start from the workspace lock file, so the dev build compiles the
  # dependency versions the workspace CI tests; cargo keeps
  # them and adds only extendr. Refreshed whenever the workspace lock is
  # newer than the dev one.
  cfg <- readLines(config)
  bind_line <- grep("^ *plskit-bind *=", cfg, value = TRUE)
  if (length(bind_line) == 0L) {
    message(
      "plskit: lock seeding skipped: ", config,
      " has no inline 'plskit-bind = ...' line to find the workspace ",
      "Cargo.lock from; the dev build resolves its own dependency versions"
    )
  } else {
    bind_path <- sub("^.*path *= *[\"']([^\"']+)[\"'].*$", "\\1", bind_line)
    ws_lock <- file.path(dirname(bind_path), "Cargo.lock")
    dev_lock <- file.path(dev_dir, "Cargo.lock")
    if (length(ws_lock) == 1L && file.exists(ws_lock) &&
        (!file.exists(dev_lock) || file.mtime(ws_lock) > file.mtime(dev_lock))) {
      file.copy(ws_lock, dev_lock, overwrite = TRUE)
    }
  }
  flags <- c(flags, "--config", normalizePath(config, winslash = "/"))
  message("plskit: dev build against the crates patched in ", config)
} else {
  if (!file.exists(file.path(crate_dir, "Cargo.lock"))) {
    fail(
      "rust/Cargo.lock is not committed yet, so plskit builds only in dev ",
      "mode: set PLSKIT_CARGO_CONFIG to a cargo config file that patches ",
      "plskit and plskit-bind to a plskit checkout (see _docs/r/installation.md)"
    )
  }
  flags <- c(flags, "--locked")
  vendor_tar <- file.path(crate_dir, "vendor.tar.xz")
  if (file.exists(vendor_tar)) {
    if (utils::untar(vendor_tar, exdir = crate_dir) != 0L) fail("could not unpack ", vendor_tar)
    # A config file rather than inline --config 'key="value"' strings:
    # system2 passes arguments through the shell unquoted, which would
    # strip the TOML quotes. The absolute directory avoids cargo's rules
    # for resolving relative paths in a --config file.
    vendor_cfg <- file.path(target_dir, "vendor-config.toml")
    dir.create(target_dir, recursive = TRUE, showWarnings = FALSE)
    writeLines(c(
      "[source.crates-io]",
      'replace-with = "vendored-sources"',
      "[source.vendored-sources]",
      sprintf('directory = "%s"', file.path(crate_dir, "vendor"))
    ), vendor_cfg)
    flags <- c(flags, "--offline", "--config", vendor_cfg)
    message("plskit: offline build against the crates in ", vendor_tar)
  }
}
flags <- c(flags, "--manifest-path", manifest)
if (clippy) flags <- c(flags, "--", "-D", "warnings")

status <- system2(cargo, flags)
if (!identical(status, 0L)) fail("cargo ", flags[[1L]], " failed (status ", status, ")")

if (nzchar(config)) {
  lock <- readLines(file.path(dirname(manifest), "Cargo.lock"))
  if ("[[patch.unused]]" %in% lock) {
    fail("the [patch.crates-io] entries in ", config, " were not used; see Cargo.lock")
  }
  for (pkg in c("plskit", "plskit-bind")) {
    starts <- which(lock == sprintf('name = "%s"', pkg))
    if (length(starts) == 0L) fail(pkg, " is missing from the dev Cargo.lock")
    for (start in starts) {
      end <- start
      while (end < length(lock) && nzchar(lock[end + 1L])) end <- end + 1L
      if (any(startsWith(lock[start:end], "source = "))) {
        fail(pkg, " resolved from a registry, not from the patch in ", config)
      }
    }
  }
}
