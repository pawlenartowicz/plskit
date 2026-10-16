#' plskit: PLS regression with modern inference
#'
#' R wrapper of the Rust plskit engine. Every function has the name and the
#' arguments of its Python counterpart; the canonical reference is
#' <https://github.com/pawlenartowicz/plskit/blob/main/_docs/python/api.md>.
#' Results are named lists with an S3 class (`c("pls1_result",
#' "plskit_result")`, ...); errors are conditions of class `plskit_error`.
#'
#' @keywords internal
"_PACKAGE"
