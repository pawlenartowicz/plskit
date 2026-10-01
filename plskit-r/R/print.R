# Compact display of plskit results (spec 4.4): one line per field, shapes
# instead of values for arrays, the type name for nested results.

.plskit_type_name <- function(x) {
  cls <- class(x)[[1L]]
  for (t in .plskit_registry()$result_types) {
    if (identical(t$r_class, cls)) return(t$name)
  }
  cls
}

.plskit_describe <- function(v) {
  if (is.null(v)) return("NULL")
  if (inherits(v, "plskit_result")) return(sprintf("<%s>", .plskit_type_name(v)))
  if (is.matrix(v)) return(sprintf("matrix %d x %d", nrow(v), ncol(v)))
  if (is.list(v)) {
    if (is.null(names(v))) return(sprintf("list of %d", length(v)))
    return(sprintf("list(%s)", paste(names(v), collapse = ", ")))
  }
  if (length(v) == 1L && is.null(names(v))) {
    if (is.character(v)) return(encodeString(v, quote = "\""))
    return(format(v, digits = 6L))
  }
  kind <- if (is.logical(v)) "logical" else if (is.integer(v)) "integer" else "numeric"
  sprintf("%s [%d]%s", kind, length(v), if (is.null(names(v))) "" else ", named")
}

#' @export
#' @noRd
print.plskit_result <- function(x, ...) {
  cat("<", .plskit_type_name(x), ">\n", sep = "")
  fields <- names(x)
  width <- max(nchar(fields), 0L)
  for (f in fields) {
    cat("  ", formatC(f, width = -width), "  ", .plskit_describe(x[[f]]), "\n", sep = "")
  }
  invisible(x)
}
