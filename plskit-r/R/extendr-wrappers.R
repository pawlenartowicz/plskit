# .Call bindings to the Rust seam (src/rust/src/lib.rs). Written by hand:
# the three entry points are internal and their shapes do not change.

#' @useDynLib plskit, .registration = TRUE
NULL

.plskit_call_impl <- function(name, inputs) .Call(wrap__plskit_call_impl, name, inputs)

.plskit_registry_impl <- function() .Call(wrap__plskit_registry_impl)

.plskit_engine_version_impl <- function() .Call(wrap__plskit_engine_version_impl)
