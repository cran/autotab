

#' Get TensorFlow Addons module safely
#' @keywords internal
get_tfa <- function() {
  if (!reticulate::py_module_available("tensorflow_addons")) {
    stop("Python module 'tensorflow_addons' not found. Install it in your TensorFlow environment.", call. = FALSE)
  }
  reticulate::import("tensorflow_addons", delay_load = TRUE)
}
