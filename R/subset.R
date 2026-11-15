# subset() on a gram_matrix_result: the Gram matrix of a row subsample, as
# Python's GramMatrixResult.subset(rows), with 1-based row numbers.

#' @rdname gram_matrix
#' @param x A `gram_matrix_result`.
#' @param rows Row numbers of `x$gram` to keep (1-based), in the order given,
#'   at least 2. The rows keep the all-row centring and scale.
#' @param ... Not used; must be empty.
#' @export
subset.gram_matrix_result <- function(x, rows, ...) {
  call <- sys.call()
  if (...length() > 0L) {
    .plskit_signal_error("invalid_argument", "subset() takes only x and rows", call = call)
  }
  n <- nrow(x$gram)
  if (!is.numeric(rows) || !is.null(dim(rows)) || anyNA(rows) ||
    any(rows != trunc(rows)) || any(rows < 1) || any(rows > n)) {
    .plskit_signal_error(
      "invalid_argument",
      sprintf("rows must be whole row numbers in 1..%d", n),
      call = call
    )
  }
  if (length(rows) < 2L) {
    .plskit_signal_error(
      "invalid_argument",
      sprintf("rows must hold at least 2 rows, got %d", length(rows)),
      call = call
    )
  }
  x$gram <- x$gram[rows, rows, drop = FALSE]
  x
}
