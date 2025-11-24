loss_history <- NULL

utils::globalVariables(c("%>%", "loss_history"))

#' Min–max scale continuous variables
#'
#' Scales numeric vectors to the \[0, 1\] range using the formula:
#' \deqn{(x - \min(x)) / (\max(x) - \min(x))}
#'
#' This is the recommended preprocessing step for continuous variables
#' prior to VAE training with AutoTab, ensuring all inputs are on
#' comparable scales to binary and categorical features.
#'
#' @param x Numeric vector. Continuous variable(s) to scale.
#'
#' @return Numeric vector of the same length as `x`, scaled to \[0, 1\].
#'
#' @details
#' - The transformation is **performed column-wise** when applied
#'   to data frames.
#'
#' @examples
#' x <- c(10, 20, 30)
#' min_max_scale(x)
#'
#' # Apply to multiple columns
#' data <- data.frame(age = c(20, 40, 60), income = c(3000, 5000, 7000))
#' Continuous_MinMaxScaled = as.data.frame(lapply(data, min_max_scale))
#'
#'
#' @seealso [extracting_distribution()], [set_feat_dist()], [VAE_train()]
#' @export
min_max_scale = function(x) {
  (x - min(x)) / (max(x) - min(x))
}
