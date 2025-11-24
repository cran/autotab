#' @importFrom magrittr %>%
#' @importFrom R6 R6Class

#' @keywords internal
.onLoad <- function(libname, pkgname) {
  options(AutoTab.startup_shown = FALSE)
}

#' @keywords internal
.onAttach <- function(libname, pkgname) {
  if (!interactive() || Sys.getenv("NOT_CRAN") != "true") return(invisible())

  if (isTRUE(getOption("AutoTab.startup_shown"))) return(invisible())
  options(AutoTab.startup_shown = TRUE)

  packageStartupMessage(
    "AutoTab loaded successfully!\n",
    "------------------------------------------------------------\n",
    "Before using AutoTab, ensure your Python environment is active:\n",
    " Example: reticulate::use_condaenv('r-reticulate', required = TRUE)\n",
    "The environment must include TensorFlow 2.10.0 and tensorflow-addons.\n",
    "------------------------------------------------------------"
  )

  if (!reticulate::py_available(initialize = FALSE)) {
    packageStartupMessage("No active Python environment detected.")
    return(invisible())
  }

  tf_version <- tryCatch({
    tfmod <- reticulate::import("tensorflow", delay_load = TRUE)
    as.character(tfmod$`__version__`)
  }, error = function(e) NULL)

  if (is.null(tf_version)) {
    packageStartupMessage("TensorFlow not detected in the active Python environment.")
  } else {
    tf_version <- trimws(tf_version)
    expected_tf <- "2.10.0"
    if (!grepl(expected_tf, tf_version)) {
      packageStartupMessage(
        paste0("AutoTab expects TensorFlow ", expected_tf,
               " (Python backend), but found ", tf_version, ".")
      )
    }
  }

  check_pkg_version <- function(pkg, expected) {
    if (as.character(utils::packageVersion(pkg)) != expected) {
      packageStartupMessage(
        paste0("AutoTab expects ", pkg, " ", expected,
               " but found ", utils::packageVersion(pkg), ".")
      )
    }
  }
  check_pkg_version("keras", "2.15.0")
  check_pkg_version("reticulate", "1.41.0")
  check_pkg_version("tensorflow", "2.16.0")
}
