

#########
#Encoder#
#########
#' @keywords internal
encoder_model = function(encoder_input,encoder_info,latent_dim, beta,Lip_en=0, pi_enc=1,prior,K =3,learnable_mog=FALSE,mog_means=NULL, mog_log_vars=NULL, mog_weights=NULL){
  tf = tensorflow::tf
  #Creating spectral normalization option
  tfa = get_tfa()
  sn <- tfa$layers$SpectralNormalization
  power_iterations = as.integer(pi_enc)

  #Pulling in layer and Lipschitz option
  layer_list = list()
  for (i in 1:length(encoder_info)) {
    if (length(encoder_info[[i]]) < 4) {encoder_info[[i]][[4]] = 0}
    if (length(encoder_info[[i]]) < 5) {encoder_info[[i]][[5]] = 1e-4}
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
  z_sample= Latent_sample(z_mean,z_log_var)
  #The above clamped z_log_var, we should also clamp it here so the divergences use the correct one
  z_log_var_clamped = tf$clip_by_value(z_log_var, clip_value_min = -10.0, clip_value_max = 10.0)

  #Get KL Divergence
  if (prior == "single_gaussian"){
    kl_loss_layer = KL_divergenceLayer$new()
    kl_loss = kl_loss_layer$call(list(z_mean,z_log_var_clamped,beta))
  }
  else if (prior == "mixture_gaussian") {
    kl_loss = list(z_sample, z_mean, z_log_var_clamped, beta) %>%
      layer_kl_mog(
        K            = K,   latent_dim   = latent_dim, mog_means    = mog_means,
        mog_log_vars = mog_log_vars,   mog_weights  = mog_weights,   learnable    = learnable_mog,name         = "kl_mog"
      ) }

  #Quality Check
  shape = tf$shape(z_sample)
  if (length(shape) != 2){
    stop(paste("ERROR: The shape of the latent space is not 2; it is", shape))}
  model = keras::keras_model(encoder_input, list(z_sample,kl_loss), name= "encoder")
  return(model)}


