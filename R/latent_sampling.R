#######################
#Sampling latent space#
#######################
#' Sample from the latent space
#'
#' Draws a stochastic sample from the latent space of a trained VAE given
#' the mean (`z_mean`) and log-variance (`z_log_var`) outputs of the encoder.
#' This operation implements the **reparameterization trick**:
#' \deqn{z = \mu + \sigma \odot \epsilon}
#' where \eqn{\epsilon \sim \mathcal{N}(0, I)}.
#'
#' The function is used internally within `VAE_train()` but can also be
#' called directly to sample latent points and decode synthetic output.
#' Typically, `z_mean` and `z_log_var` are obtained via [`encoder_latent()`]
#' and the corresponding weights extracted using [`Encoder_weights()`].
#'
#' @details
#' - The log-variance (`z_log_var`) is clamped between -10 and 10 to prevent
#'   numerical overflow or vanishing variance during training.
#' - The standard deviation is lower-bounded by `1e-3` for stability.
#'
#' This function returns a TensorFlow tensor representing the sampled latent
#' points. Use `as.matrix()` or `as.data.frame()` to convert to an R matrix or data frame before passing to
#' [`decoder_model()`] or other R functions.
#'
#' @param z_mean TensorFlow tensor or R matrix. The mean values of the latent space.
#' @param z_log_var TensorFlow tensor or R matrix. The log-variances of the latent space.
#'
#' @return A TensorFlow tensor of latent samples with the same shape as `z_mean`.
#'
#' @examples
#' # Suppose encoder_latent() returns z_mean and z_log_var
#' z_mean    <- matrix(rnorm(10), ncol = 5)
#' z_log_var <- matrix(rnorm(10), ncol = 5)
#'
#' \donttest{
#' if (reticulate::py_module_available("tensorflow")) {
#'   # Sample from latent space
#'   z_sample <- Latent_sample(z_mean, z_log_var)
#'
#'   # Convert to R matrix for decoder prediction
#'   z_mat <- as.matrix(z_sample)
#'
#'   # Suppose the computational graph was rebuilt using `decoder_model()`
#'   # and assigned to an object named `decoder`:
#'   # decoder_output <- predict(decoder, z_mat)
#' }
#' }
#'
#' @seealso [VAE_train()], [encoder_latent()], [Encoder_weights()], [decoder_model()]
#' @export
Latent_sample = function(z_mean, z_log_var){
  tf = tensorflow::tf
  z_mean = tf$cast(z_mean, dtype = tf$float32)
  z_log_var = tf$cast(z_log_var, dtype = tf$float32)

  z_log_var_clamped = tf$clip_by_value(z_log_var, clip_value_min = -10.0, clip_value_max = 10.0)
  z_std = tf$maximum(tf$exp(0.5*z_log_var_clamped), 1e-3)
  e = tf$random$normal(shape = tf$shape(z_mean))
  return(z_mean + z_std*e )}
