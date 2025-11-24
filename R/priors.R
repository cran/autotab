########################################
#Regularizer   (aka the priors)        #
#This needs to be its own custom layer #
#so that way it can be a part of the   #
#computational graph                   #
########################################

###########Regular KL Divergence where the prior is a singular Gaussian prior
#' @keywords internal
KL_divergenceLayer = R6::R6Class("KL_layer", inherit = keras::KerasLayer,
                             public = list(
                               initialize= function(){},#initialize the keras layer

                               call = function(inputs){ #create the function we want within the layer action (aka the call)
                                 z_mean = inputs[[1]]
                                 z_log_var = inputs[[2]]
                                 beta = inputs[[3]]
                                 kl_loss <- beta*(-0.5 * keras::k_sum(1 + z_log_var - keras::k_square(z_mean) - keras::k_exp(z_log_var), axis = -1))
                                 kl_loss <- keras::k_mean(kl_loss)
                                 return(kl_loss)
                               }
                             ))

############# MoG prior on the regularization
#' @keywords internal
KL_MoG_Layer = R6::R6Class("KL_MoG_Layer", inherit = keras::KerasLayer,
                       public = list(
                         num_components = NULL,
                         latent_dim = NULL,
                         learnable = FALSE,
                         #Parameters for the prior
                         mog_means = NULL,
                         mog_log_vars = NULL,
                         mog_weights = NULL,
                         mog_weights_logit = NULL,

                         initialize = function(K, latent_dim, learnable = FALSE,mog_means = NULL, mog_log_vars = NULL, mog_weights = NULL) {

                           self$num_components = K
                           self$latent_dim = latent_dim
                           self$learnable = learnable

                           self$mog_means    = mog_means
                           self$mog_log_vars = mog_log_vars
                           self$mog_weights  = mog_weights


                           if (!learnable) {
                             if (is.null(mog_means) || is.null(mog_log_vars) || is.null(mog_weights)) {
                               stop("Model will not learn MoG parameters when learnable = FALSE. mog_means,mog_log_vars, and mog_weights mys be provided. ")
                             }
                             self$mog_means = keras::k_constant(mog_means)
                             self$mog_log_vars = keras::k_constant(mog_log_vars)
                             self$mog_weights = keras::k_constant(mog_weights)
                           }        },
                         build = function(input_shape) {
                           if (self$learnable) {
                             #mog_means
                             init_means = if (!is.null(self$mog_means)) #If the initial is not null then use it
                               keras::initializer_constant(self$mog_means)
                             else
                               keras::initializer_random_normal() #otherwise initialize it with normal dist
                             self$mog_means = self$add_weight(name  = "mog_means", shape  = c(self$num_components, self$latent_dim),initializer = init_means, trainable   = TRUE )

                             #mog_vars
                             init_logv = if (!is.null(self$mog_log_vars))
                               keras::initializer_constant(self$mog_log_vars)
                             else
                               keras::initializer_zeros()
                             self$mog_log_vars = self$add_weight(name  = "mog_log_vars", shape = c(self$num_components, self$latent_dim),initializer = init_logv, trainable   = TRUE )

                             #Weights
                             init_w = if (!is.null(self$mog_weights))
                               keras::initializer_constant(log(self$mog_weights))
                             else
                               keras::initializer_zeros()
                             self$mog_weights_logit = self$add_weight(name  = "mog_weights_logit", shape = c(self$num_components),initializer = init_w, trainable   = TRUE )

                           }
                           super$build(input_shape)
                         },


                         call = function(inputs, mask = NULL) {
                           tf = tensorflow::tf
                           z_sample     = inputs[[1]]
                           z_mean       = inputs[[2]]
                           z_log_var    = inputs[[3]]
                           beta         = inputs[[4]]

                           # Log probability under q(z|x) — getting Gaussian distribution of the latent space through a sample
                           # Formula: -0.5 * ∑ (log σ² + (z - μ)² / σ² + log(2π))
                           log_qzx = -0.5 * keras::k_sum(
                             z_log_var + keras::k_square(z_sample - z_mean) / keras::k_exp(z_log_var) + keras::k_log(2 * pi),
                             axis = -1   )

                           # Make this sample as many times as the number of k mixtures using built in keras function
                           z_for_mog = keras::k_expand_dims(z_sample, axis = 2)
                           mog_means_exp = keras::k_expand_dims(self$mog_means, axis = 1)
                           mog_log_vars_exp = keras::k_expand_dims(self$mog_log_vars, axis = 1)


                           # Log p_k(z)- getting Gaussian distribution of the prior that were either chosen or learned
                           # Formula: -0.5 * ∑ (log σ² + (z - μ)² / σ² + log(2π))
                           log_probs = -0.5 * keras::k_sum(
                             (z_for_mog - mog_means_exp)^2 / keras::k_exp(mog_log_vars_exp) +
                               mog_log_vars_exp + keras::k_log(2 * pi),
                             axis = -1  # sum over latent dimensions
                           )

                           # Get mixture weights — softmax if learnable, otherwise use fixed
                           mog_weights = if (self$learnable)  keras::k_softmax(self$mog_weights_logit)
                           else self$mog_weights

                           # Compute log p(z) = log sum_k π_k * N_k(z) - this is an approximation through monte carlo
                           log_mix = log_probs + keras::k_log(mog_weights)  #TRICK: get the inside portion of the prior and put it to a log
                           log_pz = tf$reduce_logsumexp(log_mix, axis = -1L)  #log gets undone by taking log(sum(e^log_mix)), since internal logs cancel it works out

                           #Monte Carlo average average over the batch (this is L)
                           kl_loss = beta * keras::k_mean(log_qzx - log_pz)
                           return(kl_loss)
                         }               ))

#' @keywords internal
layer_kl_mog <- function(object,
                         K,
                         latent_dim,
                         mog_means    = NULL,
                         mog_log_vars = NULL,
                         mog_weights  = NULL,
                         learnable    = FALSE,
                         name         = NULL) {
  keras::create_layer(
    KL_MoG_Layer, object,
    list(
      K           = as.integer(K),
      latent_dim  = as.integer(latent_dim),
      mog_means   = mog_means,
      mog_log_vars= mog_log_vars,
      mog_weights = mog_weights,
      learnable   = learnable,
      name        = name    )  ) }


#Create option for KL warming:
#' @keywords internal
beta_callback <- function(beta_max = 0.01, warmup_epochs = 15) {
  current_beta <- keras::k_variable(0.0, dtype = "float32", name = "beta")

  callback <- keras::callback_lambda(
    on_epoch_begin = function(epoch, logs = NULL) {
      new_beta <- min(1, epoch / warmup_epochs) * beta_max
      keras::k_set_value(current_beta, new_beta)
      message(paste("Beta updated to:", new_beta))
    }
  )
  return(list(callback = callback, beta_var = current_beta))  }

#Make option for cyclical warming
#' @keywords internal
cyclical_beta_callback = function(beta_max = 0.01, total_epochs = 1000, n_cycles = 4, ratio = 0.5) {
  current_beta = keras::k_variable(0.0, dtype = "float32", name = "beta")

  callback = keras::callback_lambda(
    on_epoch_begin = function(epoch, logs = NULL) {
      cycle_length = total_epochs / n_cycles
      cycle_pos = epoch %% cycle_length
      warmup_length = ratio * cycle_length

      if (cycle_pos <= warmup_length) {
        beta = (cycle_pos / warmup_length) * beta_max
      } else {
        beta = beta_max
      }
      keras::k_set_value(current_beta, beta)
      message(sprintf("Cyclical Beta updated to: %.5f (epoch %d)", beta, epoch))
    }
  )
  return(list(callback = callback, beta_var = current_beta))
}
