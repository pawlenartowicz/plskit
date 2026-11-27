#' Gram matrix from column blocks
#'
#' Builds the `gram_matrix` of an X too wide to hold at once, one column block
#' at a time. Equal to `gram_matrix` of the joined X to rounding. Under
#' `pre_standardized`, data beyond 2^+-256 must use `gram_matrix` on the whole X.
#' The calls that take the result rescale it themselves, by a power of four,
#' when its scale would reach their absolute floors, and map their outputs back.
#'
#' @param blocks A function called with no arguments that returns the next
#'   block (a numeric matrix with the same rows as the first) or `NULL` when no
#'   blocks are left. At least one block is required.
#' @param pre_standardized As in [gram_matrix()].
#' @return A list of class `c("gram_matrix_result", "plskit_result")`.
#' @examples
#' set.seed(1)
#' X <- matrix(rnorm(40 * 20), 40, 20)
#' parts <- list(X[, 1:10], X[, 11:20])
#' i <- 0
#' g <- gram_matrix_blocks(function() {
#'   i <<- i + 1
#'   if (i <= length(parts)) parts[[i]]
#' })
#' all.equal(g$gram, gram_matrix(X)$gram)
#' @export
gram_matrix_blocks <- function(blocks, pre_standardized = FALSE) {
  call <- sys.call()
  if (!is.function(blocks)) {
    .plskit_signal_error("invalid_argument", "blocks must be a function", call = call)
  }
  out <- NULL
  repeat {
    b <- blocks()
    if (is.null(b)) break
    g <- gram_matrix(b, pre_standardized = pre_standardized)
    if (is.null(out)) {
      out <- g
    } else {
      if (nrow(g$gram) != nrow(out$gram)) {
        .plskit_signal_error(
          "shape_mismatch",
          sprintf("block has %d rows, the first block has %d", nrow(g$gram), nrow(out$gram)),
          call = call
        )
      }
      out$gram <- out$gram + g$gram
      out$n_features <- out$n_features + g$n_features
    }
  }
  if (is.null(out)) {
    .plskit_signal_error("invalid_input", "blocks holds no block", call = call)
  }
  out
}
