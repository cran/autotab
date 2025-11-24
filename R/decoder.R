#Add option for temperature warming
#' @keywords internal
temperature_callback = function(temperature = 1.5, temp_min = 0.5, warmup_epochs = 500) {
  current_temp = keras::k_variable(temperature, dtype = "float32", name = "temperature")

  callback = keras::callback_lambda(
    on_epoch_begin = function(epoch, logs = NULL) {
      new_temp <- max(temp_min, (1 - epoch / warmup_epochs) * temperature)
      keras::k_set_value(current_temp, new_temp)
      message(paste("Temperature updated to:", round(new_temp, 4)))
    }  )
  return(list(callback = callback, temp_var = current_temp)) }



####################################
#Create custom decoder output layer#
####################################

#########################################
#Make a keras layer so it can be dynamic!!
#' @keywords internal
gumbel_softmax = R6::R6Class("Gumbel_layer", inherit = keras::KerasLayer,
                         public = list(
                           initialize= function(){},#initialize the keras layer

                           call = function(inputs, mask=NULL){ #create the function we want within the layer action (aka the call)
                             logits = inputs[[1]]
                             temperature = inputs[[2]]
                             noise = -keras::k_log(-keras::k_log(keras::k_random_uniform(shape = keras::k_shape(logits), minval = 0, maxval = 1)+1e-20)+1e-20)
                             logits_scaled = (logits + noise)/ temperature
                             output = keras::k_softmax(logits_scaled)

                             return(output)
                           }
                         ))

#' @keywords internal
layer_gumbel_softmax <- function(object) {
  keras::create_layer(gumbel_softmax, object, list())
}

#scale factor no longer learnable
#' @keywords internal
decoder_activation = function(input, feat_dist, min_val=1e-3, max_std=10.0,guas_scale=1, temperature=0.5){
  tf = tensorflow::tf
  #making a place for the output
  n_feats = nrow(feat_dist)
  decoder_out = vector("list", n_feats)
  index_x = 0 #initialize the first index we look at as the first input of the tensor

  #loop through all our features so we get the correct number of outputs
  for (i in seq_len(n_feats)) {
    feature      = feat_dist$column_name[i]
    distribution = feat_dist$distribution[i]
    num_params   = feat_dist$num_params[i]

  # TRACKING: Print the current distribution and num_params and index
    message(paste("Output- Processing feature", feature, "with distribution:", distribution, "and num_params:", num_params, "with index starting at", index_x))

    if (distribution=="gaussian"){
      mean_slice = tf$slice(input, begin=list(0L,as.integer(index_x)), size=list(tf$shape(input)[1], 1L))
      mean_temp = tf$nn$tanh(mean_slice)
      mean = (mean_temp+1)/2

      SD_raw = tf$slice(input, begin=list(0L, as.integer(index_x+1)), size=list(tf$shape(input)[1], 1L))
      SD_raw2 = tf$nn$softplus(SD_raw)
      SD = tf$clip_by_value(SD_raw2, clip_value_min = min_val, clip_value_max = max_std)

      decoder_out[[i]]= tf$concat(list(mean,SD),axis=1L)
      index_x = index_x + 2L # we used up two of the indexes for this gaussian so the next one starts at 2 over
    }

    else   if (distribution =="bernoulli"){
      bern_lin = tf$slice(input, begin=list(0L,as.integer(index_x)), size=list(tf$shape(input)[1], 1L))
      gain = tf$constant(5.0, dtype = bern_lin$dtype)
      logit = bern_lin*gain
      bernoulli = tf$keras$activations$sigmoid(logit)

      decoder_out[[i]] = bernoulli

      index_x = index_x + 1L # only used one position so move over by 1
    }
    else if (distribution =="categorical"){
      logit = tf$keras$activations$tanh(tf$slice(input, begin = list(0L, as.integer(index_x)), size = list(tf$shape(input)[1], as.integer(num_params))))*10 #this will take the current space up to the num_params -1
      decoder_out[[i]] = list(logit, temperature) %>% layer_gumbel_softmax()
      index_x = index_x + num_params # We move over based on the number of params used for that given categorical variable
    }
    else {
      stop("Unknown Distribution:", distribution)  #A measure to catch errors
    }
  }
  #combine all variable outputs together into one "row"
  decoder_output = keras::layer_concatenate(decoder_out, axis = 1) #this is a keras function to merge together dense output layers by column

  return(decoder_output) }

#########
#Decoder#
#########
#' Builds the decoder graph for an AutoTab VAE
#'
#' Reconstructs the **decoder** computational graph used during training. This is used
#' internally by `VAE_train()` and externally when you want to load the trained
#' decoder weights and generate new samples by sampling the latent space.
#'
#' @details
#' The final output layer of an AutoTab decoder slices outputs by feature distribution in `feat_dist`:
#' Gaussian heads output mean/SD (with `min_val`/`max_std` constraints),
#' Bernoulli heads output logits passed through sigmoid to extract probabilities,
#' and Categorical heads use Gumbel–Softmax with the given `temperature`.
#'
#' If `lip_dec = 1`, dense hidden layers are wrapped with
#' #' spectral normalization using `pi_dec` power iterations.
#'
#' @param decoder_input Ignored; pass `NULL`. No input is needed when building the compitational graph.
#' @param decoder_info List defining the decoder architecture, e.g.
#'   `list(list("dense", 80, "relu"), list("dropout", 0.1), list("dense", 100, "relu"))`.
#'   Each `dense` entry is `list("dense", units, activation)`. Each `dropout`
#'   entry is `list("dropout", rate)`. Optional elements:
#'   `[[4]]` L2 flag (0/1), `[[5]]` L2 value, `[[6]]` BN flag (FALSE/TRUE),
#'   `[[7]]` BN momentum, `[[8]]` BN scale/center (TRUE/FALSE).
#' @param latent_dim Integer. Latent dimension used during training.
#' @param feat_dist Data frame with columns `column_name`, `distribution`, `num_params`
#'   (created by `extracting_distribution()` and set via `set_feat_dist()`).
#' @param lip_dec 0/1 (logical). Use spectral normalization on dense hidden layers.
#' @param pi_dec Integer. Power-iteration count for spectral normalization.
#' @param max_std Numeric. Upper bound for Gaussian SD heads (default `10.0`).
#' @param min_val Numeric. Lower bound (epsilon) for Gaussian SD heads (default `1e-3`).
#' @param temperature Numeric. Gumbel–Softmax temperature for categorical heads (default `0.5`).
#'
#' @return A compiled **Keras model** representing the decoder computational graph. You can
#'   load trained decoder weights with `Decoder_weights()` + `set_weights()`, then
#'   call `predict(decoder, Z)` where `Z` is an `n x latent_dim` matrix (typically a sample from your latent space).
#'
#' @examples
#' \donttest{
#' if (reticulate::py_module_available("tensorflow") &&
#'     exists("training") &&
#'     exists("feat_dist")) {
#'
#'   # Assume you already have feat_dist set via set_feat_dist(feat_dist)
#'   decoder_info <- list(
#'     list("dense", 80, "relu"),
#'     list("dense", 100, "relu")
#'   )
#'
#'   # Rebuild and apply decoder
#'   weights_decoder <- Decoder_weights(
#'     encoder_layers = 2,
#'     trained_model  = training$trained_model,
#'     lip_enc        = 0,
#'     pi_enc         = 0,
#'     prior_learn    = "fixed",
#'     BNenc_layers   = 0,
#'     learn_BN       = 0
#'   )
#'
#'   decoder <- decoder_model(
#'     decoder_input = NULL,
#'     decoder_info  = decoder_info,
#'     latent_dim    = 5,
#'     feat_dist     = feat_dist,
#'     lip_dec       = 0,
#'     pi_dec        = 0
#'   )
#'
#'   decoder %>% keras::set_weights(weights_decoder)
#' }
#' }
#'
#' @seealso [VAE_train()], [Decoder_weights()], [encoder_latent()], [Latent_sample()], [extracting_distribution()]
#' @export

decoder_model = function(decoder_input, decoder_info, latent_dim, feat_dist,lip_dec, pi_dec,max_std=10.0,min_val=1e-3, temperature=0.5){
  tf = tensorflow::tf
  #Creating spectral normalization option
  tfa = get_tfa()
  sn = tfa$layers$SpectralNormalization
  power_iterations = as.integer(pi_dec)
  #Pulling in layer and Lipschitz option
  layer_list = list()

  # Make default no regularization

  for (i in 1:length(decoder_info)) {
    if (length(decoder_info[[i]]) < 4) {decoder_info[[i]][[4]] = 0}
    if (length(decoder_info[[i]]) < 5) {decoder_info[[i]][[5]] = 1e-4}
    if (length(decoder_info[[i]]) < 6) {decoder_info[[i]][[6]] = FALSE}
    if (length(decoder_info[[i]]) < 7) {decoder_info[[i]][[7]] = 0.99 }
    if (length(decoder_info[[i]]) < 8) {decoder_info[[i]][[8]] = TRUE }


    if (decoder_info[[i]][[1]] == "dense") {
      layer_list[[i]] = function(x) { layer = keras::layer_dense( units = decoder_info[[i]][[2]],  activation = decoder_info[[i]][[3]],
                                                            kernel_regularizer = if (decoder_info[[i]][[4]] == 1)
                                                              keras::regularizer_l2(decoder_info[[i]][[5]])
                                                            else NULL
      )
      if (lip_dec == 1) {
        layer = sn(layer, power_iterations = power_iterations)
      }
      layer(x)
      }
    } else if (decoder_info[[i]][[1]] == "dropout") {
      layer_list[[i]] = function(x) {
        keras::layer_dropout(rate = decoder_info[[i]][[2]])(x)
      }
    }
  }

  decoder_input = keras::layer_input(shape=c(latent_dim))
  decoder_hidden = decoder_input
  # Add layers dynamically
  for (i in 1:length(layer_list)) {
    if (decoder_info[[i]][[6]]==TRUE){decoder_hidden = decoder_hidden %>% layer_list[[i]]() %>% keras::layer_batch_normalization(momentum = decoder_info[[i]][[7]],scale = decoder_info[[i]][[8]],  center = decoder_info[[i]][[8]]  ) }
    else {decoder_hidden = decoder_hidden %>% layer_list[[i]]()}

  }

  decoder_output = decoder_activation(input=decoder_hidden, feat_dist=feat_dist,max_std=max_std, min_val=min_val,temperature=temperature)
  model = keras::keras_model(inputs=decoder_input, outputs= decoder_output)
  return(model)}

