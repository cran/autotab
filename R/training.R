
##############
#VAE Training#
##############
#' Train an AutoTab VAE on mixed-type tabular data
#'
#' Runs the full AutoTab training loop (encoder + decoder + latent space),
#' with optional Beta-annealing (linear or cyclical), optional Gumbel-softmax
#' temperature warming for categorical outputs, and options for the prior.
#'
#' **Prerequisite:** call [set_feat_dist()] once before training to register the
#' per-feature distributions and parameter counts (see [extracting_distribution()]
#' and [feat_reorder()]).
#'
#' @param data Matrix/data.frame. **Preprocessed** training data (columns match
#'   the order in `feat_dist`).
#' @param encoder_info,decoder_info Lists describing layer stacks. Each element
#'   is e.g. `list("dense", units, "activation", L2_flag, L2_lambda, BN_flag, BN_momentum, BN_learn)` or
#'   `list("dropout", rate)`.
#' @param Lip_en,lip_dec Integer (0/1). Use spectral normalization (Lipschitz)
#'   in encoder/decoder.
#' @param pi_enc,pi_dec Integer. Power-iteration counts for spectral normalization.
#' @param latent_dim Integer. Latent dimensionality.
#' @param epoch Integer. Max training epochs.
#' @param beta Numeric. Beta-VAE weight on the KL term in the ELBO.
#' @param kl_warm Logical. Enable Beta-annealing.
#' @param kl_cyclical Logical. Enable **cyclical** Beta-annealing (requires `kl_warm = TRUE`).
#' @param n_cycles Integer. Number of cycles when `kl_cyclical = TRUE`.
#' @param ratio Numeric from range 0 to 1. Fraction of each cycle used for warm-up (rise from 0→Beta).
#' @param beta_epoch Integer. Warm-up length (epochs) for **linear** Beta-annealing; when
#'   `kl_cyclical = TRUE`, the cycle length is `(beta_epoch / n_cycles)`.
#' @param temperature Numeric. Gumbel-softmax temperature (used for categorical heads).
#' @param temp_warm Logical. Enable temperature warm-up.
#' @param temp_epoch Integer. Warm-up length (epochs) for temperature when `temp_warm = TRUE`.
#' @param batchsize Integer. Mini-batch size.
#' @param wait Integer. Early-stopping patience (epochs) on validation reconstruction loss.
#' @param min_delta Numeric. Minimum improvement to reset patience (early stopping).
#' @param lr Numeric. Learning rate (Adam).
#' @param max_std,min_val Numerics. Decoder constraints for Gaussian heads
#'   (max SD; minimum variance surrogate).
#' @param weighted Integer (0/1). If 1, weight reconstruction terms by type.
#' @param recon_weights Numeric length-3. Weights for (continuous, binary, categorical);
#'   **required** when `weighted = 1`.
#' @param seperate Integer (0/1). If 1, logs per-group reconstruction losses as metrics
#'   (`cont_loss`, `bin_loss`, `cat_loss`) in addition to total `recon_loss`.
#' @param prior Character. `"single_gaussian"` or `"mixture_gaussian"`.
#' @param K Integer. Number of mixture components when `prior = "mixture_gaussian"`.
#' @param learnable_mog Logical. If TRUE, MoG prior parameters are trainable.
#' @param mog_means,mog_log_vars,mog_weights Optional initial values for the MoG prior
#'   (ignored unless `prior = "mixture_gaussian"`; when `learnable_mog = FALSE` they must be provided).
#'
#' @return A list with:
#' \itemize{
#'   \item `trained_model` — the compiled Keras model (encoder→decoder) with KL and recon losses added.
#'   \item `loss_history` — numeric vector of per-epoch total loss (as tracked during training).
#' }
#'
#' @details
#' **Metrics exposed during training:** `loss`, `recon_loss`, `kl_loss`, and,
#' when `seperate = 1`, `cont_loss`, `bin_loss`, `cat_loss`, and, `beta`, `temperature`
#' when annealed.
#'
#' **Early stopping:** monitored on `val_recon_loss` with `patience = wait`.
#'
#' **Reproducibility:** set seeds via your own workflow or the helper `reset_seeds()`.
#'
#' **Expected Warning:** When running AutoTab the user will receive the following warning from tensorflow:
#'  "WARNING:tensorflow:The following Variables were used in a Lambda layer's call (tf.math.multiply_3),
#'  but are not present in its tracked objects:   <tf.Variable 'beta:0' shape=() dtype=float32>.
#'   This is a strong indication that the Lambda layer should be rewritten as a subclassed Layer."
#'
#'   This is merely a warning and should not effect the computation of AutoTab.
#'   This occurs because tensorflow does not see beta, (the weight on the regularization part of the ELBO)
#'   until after the first iteration of training and the first computation of the loss is initiated.
#'   Therefore it is not an internally tracked object. However, it  is being tracked and updated outside
#'    of the model graph which can be seen in the KL loss plots and in the training printout in the R console.
#'
#'
#' @seealso [set_feat_dist()], [extracting_distribution()], [feat_reorder()],
#'   [Encoder_weights()], [encoder_latent()], [Decoder_weights()], [Latent_sample()]
#' @export

VAE_train = function(data,encoder_info, decoder_info,Lip_en, pi_enc=1,lip_dec, pi_dec=1, latent_dim, epoch, beta,kl_warm=FALSE,kl_cyclical = FALSE, n_cycles, ratio, beta_epoch=15, temperature, temp_warm = FALSE,temp_epoch,batchsize, wait, min_delta=1e-3, lr,max_std=10.0,min_val=1e-3,weighted=0, recon_weights, seperate = 0,prior="single_gaussian",K =3,learnable_mog=FALSE,mog_means=NULL, mog_log_vars=NULL, mog_weights=NULL){

  #Helpful errors for users
  feat_dist <- get_feat_dist()
  if (is.null(feat_dist)) {
    stop("`feat_dist` is not set. Call set_feat_dist(feat_reorder(extracting_distribution(raw_data), data)) before VAE_train().")
  }
  if (weighted == 1) {
    if (missing(recon_weights) || length(recon_weights) != 3) {
      stop("When weighted = 1, `recon_weights` must be a numeric vector of length 3: (continuous, binary, categorical).")
    }
  }
  if (kl_cyclical && !kl_warm) {
    stop("kl_cyclical = TRUE requires kl_warm = TRUE.")
  }
  output_count = sum(feat_dist$num_params)
  lastdencoder = decoder_info[[length(decoder_info)]]
  lastdencoder_count = lastdencoder[[2]]
  if (output_count > lastdencoder_count) {
    message ("The node count of the last layer must be >= number of output layers. To get the number of ouput layers look at the sum of num_params in feat_dist.")
  }



  EarlyStop = keras::callback_early_stopping(monitor='val_recon_loss', patience=wait, min_delta=min_delta,restore_best_weights = TRUE)

  # Setup beta (fixed or dynamic)
  if (kl_warm) {
    if (kl_cyclical){
      beta_dynamic = cyclical_beta_callback(beta_max = beta, total_epochs = epoch, n_cycles = n_cycles, ratio = ratio)
    } else {
      beta_dynamic = beta_callback(beta_max = beta, warmup_epochs = beta_epoch)
    }
    beta_used = beta_dynamic$beta_var
    beta_callback_list = list(beta_dynamic$callback)
  } else {
    beta_used = keras::k_variable(beta, dtype = "float32", name = "beta_fixed")
    beta_callback_list = list()
  }

  # Setup temperature (fixed or dynamic)
  if (temp_warm) {
    temp_dynamic = temperature_callback(temperature = temperature, warmup_epochs = temp_epoch)
    temp_used = temp_dynamic$temp_var
    temp_callback_list = list(temp_dynamic$callback)
  } else {
    temp_used = keras::k_variable(temperature, dtype = "float32", name = "temp_fixed")
    temp_callback_list = list()
  }

  #pull global feat_dist into local package enviorment
  feat_dist <- get_feat_dist()
  run_vae = model_VAE(data=data, encoder_info=encoder_info, decoder_info=decoder_info,Lip_en=Lip_en, pi_enc=pi_enc,lip_dec=lip_dec, pi_dec=pi_dec, latent_dim=latent_dim, feat_dist=feat_dist, lr=lr , beta=beta_used,max_std=max_std, min_val=min_val,temperature=temp_used,weighted=weighted, recon_weights=recon_weights, seperate=seperate,prior=prior,K =K,learnable_mog=learnable_mog,mog_means=mog_means, mog_log_vars=mog_log_vars, mog_weights=mog_weights)

  #Tracking loss as we go
  loss_history <-list()
  loss_tracked = keras::callback_lambda(on_epoch_end = function(epoch,logs){
    loss_history[[epoch+1]] <<- logs$loss
 })

  if (seperate == 1){callbacks = c(list(EarlyStop, loss_tracked),beta_callback_list,temp_callback_list,LossPrinterCallback$new())}
  else if (seperate == 0) { callbacks = c(list(EarlyStop, loss_tracked),beta_callback_list,temp_callback_list)}

  input_data <- as.matrix(data)

  run_vae %>% keras::fit(input_data, input_data, epochs = epoch, batch_size = batchsize, validation_split = 0.2, callbacks=callbacks)

  return(list(trained_model = run_vae, loss_history = loss_history))
}


#' Reset all random seeds across R, TensorFlow, and Python
#'
#' Ensures reproducibility by synchronizing random seeds across:
#' \itemize{
#'   \item R's random number generator (`set.seed()`),
#'   \item TensorFlow's random state (`tf$random$set_seed()`),
#'   \item Python's built-in `random` module.
#' }
#'
#' This also clears the current Keras/TensorFlow graph and session before
#' reseeding, preventing residual state from prior model builds.
#'
#' @param spec_seed Integer. The seed value to apply across R, TensorFlow, and Python.
#'
#' @details
#' - This function is **not** called automatically within AutoTab.
#'   Use it before training runs for reproducibility.
#' - Equivalent results still require identical environments
#'   (same TensorFlow, CUDA/cuDNN, and library versions).
#'
#' @return No return value but will print a confirmation message.
#'
#' @examples
#' \donttest{
#' if (reticulate::py_module_available("tensorflow")) {
#' reset_seeds(1234)
#' }
#' }
#'
#'
#' @seealso [VAE_train()], [set_feat_dist()]
#' @export
reset_seeds <- function(spec_seed) {
  tf = tensorflow::tf
  # Reset TensorFlow/Keras session and clear the graph
  tf$compat$v1$reset_default_graph()
  keras::k_clear_session()  # clears the Keras session
  # Set R random seed
  set.seed(spec_seed)  # seed value is an input option
  # Set TensorFlow random seed
  tf$random$set_seed(spec_seed)
  # Import and set Python's random seed (via reticulate)
  py_random <- reticulate::import("random")  # Import Python's random module
  py_random$seed(spec_seed)  # Set Python's random seed
  message("Random seeds reset\n")
}
