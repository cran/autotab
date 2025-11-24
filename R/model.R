
###############
#VAE Framework#
###############
#' @keywords internal
model_VAE = function(data, encoder_info, decoder_info,Lip_en, pi_enc=1,lip_dec, pi_dec=1,latent_dim, feat_dist, lr, beta,max_std=10.0,min_val, temperature=0.5,weighted=0, recon_weights, seperate = 0,prior="single_gaussian",K =3,learnable_mog=FALSE,mog_means=NULL, mog_log_vars=NULL, mog_weights=NULL ){
  #prep data
  input_data <-  as.matrix(data)
  #encoder
  encoder = encoder_model(encoder_input=input_data, encoder_info=encoder_info, latent_dim=latent_dim, beta=beta,Lip_en=Lip_en, pi_enc=pi_enc,prior=prior,K=K,learnable_mog=learnable_mog,mog_means=mog_means, mog_log_vars=mog_log_vars, mog_weights=mog_weights )
  #Decoder
  decoder = decoder_model(encoder$output[[1]], decoder_info=decoder_info,latent_dim=latent_dim, feat_dist=feat_dist,lip_dec=lip_dec, pi_dec=pi_dec,max_std=max_std , min_val=min_val,temperature = temperature )

  #Define VAE model
  vae = keras::keras_model(inputs = encoder$input, outputs= decoder(encoder$output[[1]]))

  #make the KL that is calculated in the encoder as part of the loss
  kl_loss = encoder$output[[2]]
  vae$add_loss(kl_loss)
  #add reconstruction loss as part of the loss
  reconstruction_loss = lossbasedondist(vae$output, feat_dist , encoder$input,weighted=weighted, recon_weights=recon_weights)
  vae$add_loss(reconstruction_loss[[1]])

  #Looking at losses separate
  vae$add_metric(kl_loss, name = "kl_loss", aggregation = "mean")
  vae$add_metric(reconstruction_loss[[1]], name = "recon_loss", aggregation = "mean")
  if (seperate == 1) {
    vae$add_metric(reconstruction_loss[[2]], name = "cont_loss", aggregation = "mean")
    vae$add_metric(reconstruction_loss[[3]], name = "bin_loss", aggregation = "mean")
    vae$add_metric(reconstruction_loss[[4]], name = "cat_loss", aggregation = "mean")}

  #compile
  optimizer = keras::optimizer_adam(learning_rate = lr, clipnorm = 0.1)


  vae %>% keras::compile(optimizer=optimizer, loss = NULL)
  return(vae)
}


#' @keywords internal
LossPrinterCallback <- R6::R6Class(
  "LossPrinterCallback",
  inherit = keras::KerasCallback,

  public = list(
    on_epoch_end = function(epoch, logs = list()) {
      cat(sprintf("Epoch %d: Continuous Loss = %.4f | Binary Loss = %.4f | Categorical Loss = %.4f \n",
                  epoch + 1,
                  logs[["cont_loss"]],
                  logs[["bin_loss"]],
                  logs[["cat_loss"]]))
    }  ))
