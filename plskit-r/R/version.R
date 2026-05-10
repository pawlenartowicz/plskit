#' plskit version
#'
#' Returns the version string of the underlying plskit-core engine.
#'
#' @return Character scalar.
#' @export
version <- function() {
  .Call("wrap__version", PACKAGE = "plskit")
}
