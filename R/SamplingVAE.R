#################################
#Function to get encoder weights#
#################################
#' Extract encoder-only weights from a trained Keras model
#'
#' Pulls just the **encoder** weights from `keras::get_weights(trained_model)`,
#' skipping any parameters introduced by batch normalization (BN) or spectral
#' normalization (SN). The split index is computed from the number of encoder
#' layers and whether BN/SN were used.
#'
#' @param encoder_layers Integer. Number of encoder layers (used to compute split index).
#' @param trained_model Keras model. Typically `training$trained_model` from [VAE_train()].
#' @param lip_enc Integer (0/1). Whether spectral normalization was used in the encoder.
#' @param pi_enc Integer. Power iteration count if spectral normalization was used.
#' @param BNenc_layers Integer. Number of encoder layers that had batch normalization.
#' @param learn_BN Integer (0/1). Whether BN layers learned scale and center.
#'
#' @return A `list()` of encoder weight tensors in order, suitable for `set_weights()`.
#'
#' @details
#' - The index arithmetic assumes AutoTab's standard Dense/BN/SN layout. If you
#'   substantially change layer ordering or introduce new per-layer parameters,
#'   re-check the split index.
#' - All model weights can always be accessed directly using
#'   `keras::get_weights(trained_model)`. This function is provided as a
#'   convenience tool within AutoTab to streamline encoder reconstruction but
#'   is not the only method available.
#'
#' @seealso [encoder_latent()], [Decoder_weights()], [VAE_train()], [Latent_sample()]
#'
#' @examples
#' encoder_info <- list(
#'   list("dense", 100, "relu"),
#'   list("dense",  80, "relu")
#' )
#'
#'  \donttest{
#' if (reticulate::py_module_available("tensorflow") &&
#'     exists("training")) {
#' weights_encoder <- Encoder_weights(
#'   encoder_layers = 2,
#'   trained_model  = training$trained_model, #where training = VAE_train(...)
#'   lip_enc        = 0,
#'   pi_enc         = 0,
#'   BNenc_layers   = 0,
#'   learn_BN       = 0
#' )
#' }
#' }
#'
#' @export
Encoder_weights = function(encoder_layers, trained_model,lip_enc, pi_enc, BNenc_layers,learn_BN ){
  if (lip_enc == 1){
    if (learn_BN == 1){  stop = (encoder_layers*(2+pi_enc))+(4*(BNenc_layers+1))   }
    if (learn_BN == 0){  stop = (encoder_layers*(2+pi_enc))+(4*(BNenc_layers+1)-(2*BNenc_layers))}
  }
  if (lip_enc == 0){
    if (learn_BN == 1){ stop = (encoder_layers*2)+(4*(BNenc_layers+1))  }
    if (learn_BN == 0){ stop = (encoder_layers*2)+(4*(BNenc_layers+1))-(2*BNenc_layers)}}
  weights = list()
  for(i in 1:stop){
    grab = keras::get_weights(trained_model)[[i]]
    weights[[i]] = grab
    final_weights = Filter(Negate(is.null), weights)}
  return(final_weights)}


###############################################
#Build the decoder so we can apply the weights#
###############################################
#' Rebuild the encoder graph to export z_mean and z_log_var
#'
#' Constructs the encoder computation graph (matching your original `encoder_info`)
#' so that weights extracted by [Encoder_weights()] can be applied and the encoder
#' to produce `z_mean` and `z_log_var`.
#'
#' @param encoder_input Data frame or matrix of the **preprocessed** variables (used for shape only).
#' @param encoder_info List defining encoder architecture.
#' @param latent_dim Integer. Latent dimension.
#' @param Lip_en Integer (0/1). Whether spectral normalization was used in the encoder.
#' @param power_iterations Integer. Power iterations for spectral normalization (if used).
#'
#' @return A Keras model whose outputs are `list(z_mean, z_log_var)`.
#'
#' @details
#' - Spectral normalization is sourced from TensorFlow Addons via [get_tfa()].
#' - `encoder_input` provides shape; the data are not consumed at build time.
#' - Apply weights with `set_weights()` using the output of [Encoder_weights()].
#'
#' @seealso [Encoder_weights()], [Latent_sample()], [Decoder_weights()]
#'
#' @examples
#' encoder_info <- list(
#'   list("dense", 100, "relu"),
#'   list("dense",  80, "relu")
#' )
#' \donttest{
#' if (reticulate::py_module_available("tensorflow") &&
#'     exists("training")) {
#' weights_encoder <- Encoder_weights(
#'   encoder_layers = 2,
#'   trained_model  = training$trained_model,  #where training = VAE_train(...)
#'   lip_enc        = 0,
#'   pi_enc         = 0,
#'   BNenc_layers   = 0,
#'   learn_BN       = 0
#' )
#'
#' latent_encoder <- encoder_latent(
#'   encoder_input    = data,
#'   encoder_info     = encoder_info,
#'   latent_dim       = 5,
#'   Lip_en           = 0,
#'   power_iterations = 0
#' )
#' latent_encoder %>% keras::set_weights(weights_encoder)
#' }
#' }
#'
#' @export
encoder_latent = function(encoder_input,encoder_info,latent_dim,Lip_en, power_iterations){
  tf = tensorflow::tf
  tfa <- get_tfa()
  #Creating spectral normalization option
  sn <- tfa$layers$SpectralNormalization
  power_iterations = as.integer(power_iterations)
  #Pulling in layer and Lipschitz option
  layer_list = list()
  for (i in 1:length(encoder_info)) {
    if (length(encoder_info[[i]]) < 4) {encoder_info[[i]][[4]] = 0}
    if (length(encoder_info[[i]]) < 5) {encoder_info[[i]][[5]] = 0.01}
    if (length(encoder_info[[i]]) < 6) {encoder_info[[i]][[6]] = FALSE}
    if (length(encoder_info[[i]]) < 7) {encoder_info[[i]][[7]] = 0.99 }
    if (length(encoder_info[[i]]) < 8) {encoder_info[[i]][[8]] = TRUE }

    if (encoder_info[[i]][[1]] == "dense") {
      layer_list[[i]] = function(x) { layer <- keras::layer_dense( units = encoder_info[[i]][[2]],  activation = encoder_info[[i]][[3]],
                                                            kernel_regularizer = if (encoder_info[[i]][[4]] == 1)
                                                              keras::regularizer_l2(encoder_info[[i]][[5]])
                                                            else NULL  )
      if (Lip_en == 1) {
        layer <- sn(layer, power_iterations = power_iterations)      }
      layer(x)      }
    } else if (encoder_info[[i]][[1]] == "dropout") {      layer_list[[i]] = function(x) {
      keras::layer_dropout(rate = encoder_info[[i]][[2]])(x)      }    }  }

  encoder_input = keras::layer_input(shape = c(ncol(encoder_input)))
  encoder_hidden = encoder_input  # Start with the input layer
  # Add layers dynamically
  for (i in 1:length(layer_list)) {
    if (encoder_info[[i]][[6]]==TRUE){encoder_hidden = encoder_hidden %>% layer_list[[i]]() %>% keras::layer_batch_normalization(momentum = encoder_info[[i]][[7]],scale = encoder_info[[i]][[8]],  center = encoder_info[[i]][[8]]  ) }
    else {encoder_hidden = encoder_hidden %>% layer_list[[i]]()}

  }
  z_mean = encoder_hidden %>% keras::layer_dense(units = latent_dim, name = "z_mean")
  z_log_var = encoder_hidden %>% keras::layer_dense(units = latent_dim, name = "z_log_var")

  model = keras::keras_model(encoder_input, list(z_mean,z_log_var), name= "encoder")
  return(model)}


#################################
#Function to get decoder weights#
#################################
#' Extract decoder-only weights from a trained Keras model
#'
#' Pulls just the **decoder** weights from `keras::get_weights(trained_model)`,
#' skipping encoder parameters and (if used) the final trainable tensors from a
#' learnable mixture-of-Gaussians (MoG) prior (means, log_vars, and weight logits).
#'
#' @param encoder_layers Integer. Number of encoder layers (used to compute split index).
#' @param trained_model Keras model. Typically `training$trained_model`.
#' @param lip_enc Integer (0/1). Whether spectral normalization was used in the encoder.
#' @param pi_enc Integer. Power iterations used in encoder spectral normalization.
#' @param prior_learn Character. `"fixed"` for fixed prior; any other value implies learnable MoG.
#' @param BNenc_layers Integer. Number of encoder BN layers (affects split index).
#' @param learn_BN Integer (0/1). Whether BN layers learned scale and center.
#'
#' @return A `list()` of decoder weight tensors in order, suitable for `set_weights()`.
#'
#' @details
#' - When `prior_learn != "fixed"`, the final **three** tensors are assumed to belong to
#'   the learnable MoG prior (`mog_means`, `mog_log_vars`, `mog_weights_logit`) and are excluded.
#' - The split index math mirrors [Encoder_weights()] and assumes the standard AutoTab graph wiring.
#' - All model weights can always be accessed directly using
#'   `keras::get_weights(trained_model)`. This function is provided as a
#'   convenience tool within AutoTab to streamline decoder reconstruction but
#'   is not the only method available.
#'
#' @seealso [decoder_model()], [Encoder_weights()], [VAE_train()]
#'
#' @examples
#' decoder_info <- list(
#'   list("dense", 80, "relu"),
#'   list("dense", 100, "relu")
#' )
#' \donttest{
#' if (reticulate::py_module_available("tensorflow") &&
#'     exists("training")) {
#' weights_decoder <- Decoder_weights(
#'   encoder_layers = 2,
#'   trained_model  = training$trained_model,  #where training = VAE_train(...)
#'   lip_enc        = 0,
#'   pi_enc         = 0,
#'   prior_learn    = "fixed",
#'   BNenc_layers   = 0,
#'   learn_BN       = 0
#' )
#' }
#' }
#' @export
Decoder_weights = function(encoder_layers,trained_model,lip_enc,pi_enc , prior_learn,BNenc_layers,learn_BN){
  all = keras::get_weights(trained_model)
  if (lip_enc == 1){
    if (learn_BN == 1){  start = (encoder_layers*(2+pi_enc))+(4*(BNenc_layers+1))+1   }
    if (learn_BN == 0){  start = ((encoder_layers*(2+pi_enc))+(4*(BNenc_layers+1))+1)-(2*BNenc_layers)}}
  if (lip_enc == 0){
    if (learn_BN == 1){ start = (encoder_layers*2)+(4*(BNenc_layers+1))+1  }
    if (learn_BN == 0){ start = (encoder_layers*2)+(4*(BNenc_layers+1))-(2*BNenc_layers)+1}}
  weights = list()
  for(i in start:length(all)){
    grab = all[[i]]
    # If using MoG prior, remove last 3 weights *outside* the loop
    if (prior_learn != "fixed" && i > (length(all) - 3)) {
      next  # skip the final 4 weights if they're from the MoG prior
    }
    weights[[i]] = grab
    final_weights = Filter(Negate(is.null), weights)}
  return(final_weights)}
