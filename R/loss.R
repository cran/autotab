
#############################
#Create custom loss function#
#############################
#' @keywords internal
lossbasedondist = function(input, feat_dist, target,weighted=0, recon_weights){ #here the "input" data set contains the output of the VAE, target is the real data (dummy coded)
  tf = tensorflow::tf

  index_x = 0L #start at zero just as we did when defining the activation functions since tensors start at 0
  index_y = 0L
  cont_loss_LL = list()
  bin_loss_LL = list()
  cat_loss_LL = list()
  eps = 1e-7  # epsilon for stabilizing logs so no logs of 0

  for(index_type in 1:nrow(feat_dist)){
    variable = feat_dist[index_type,] #[i,] extracts a row; [i] extracts a column for the distribution data
    distribution = variable$distribution
    num_params = variable$num_params

    # TRACKING: Print the current distribution and num_params and index
    message(paste("Loss - Processing feature", index_type,
                  "with distribution:", distribution,
                  "and num_params:", num_params,
                  "with index starting at", index_x))

    if (distribution=="gaussian"){
      mean = tf$slice(input, begin=list(0L,as.integer(index_x)), size=list(tf$shape(input)[1], 1L))
      SD_raw = tf$slice(input, begin=list(0L,as.integer(index_x+1)), size=list(tf$shape(input)[1], 1L))
      SD = tf$math$softplus(SD_raw) + 1e-3 #ensure SD will be positive so continuous recon does not come out negative

      ll = 0.5*tf$math$log(2*pi)+tf$math$log(SD)+0.5*tf$square(((tf$slice(target, begin=list(0L,as.integer(index_y)), size=list(tf$shape(input)[1], 1L)) - mean)/ SD))
      #This is the typical negative log likelihood. It pulls the true target mean, and uses the VAE output SD and mean

      cont_loss_LL[[index_type]]= tf$reduce_mean(ll)
      index_x = index_x + 2L # we used up two of the indexes for this Gaussian so the next one starts at 2 over
      index_y = index_y + 1L
    }
    else   if (distribution =="bernoulli"){
      prob_crude = tf$slice(input, begin=list(0L,as.integer(index_x)), size=list(tf$shape(input)[1], 1L))
      prob = tf$clip_by_value(prob_crude, eps, 1 - eps)
      y_true= tf$slice(target, begin=list(0L,as.integer(index_y)), size=list(tf$shape(input)[1], 1L))

      ll = -(y_true*tf$math$log(prob)+(1-y_true)*tf$math$log(1-prob))
      #This is the typical negative log likelihood of the Bernoulli distribution! We have the target probability and the predicted probability , sum is inherntly handled by how tensors work
      #ll based on logist ont probs

      bin_loss_LL[[index_type]]= tf$reduce_mean(ll)
      index_x = index_x + 1L # only used one position so move over by 1
      index_y = index_y + 1L
    }
    else if (distribution =="categorical"){

      logit_crude = tf$slice(input, begin = list(0L, as.integer(index_x)), size = list(tf$shape(input)[1], as.integer(num_params))) #this will take the current space up to the num_params -1
      logit = tf$clip_by_value(logit_crude, eps, 1 - eps)

      ll = -tf$math$log(logit)*tf$slice(target, begin = list(0L, as.integer(index_y)), size = list(tf$shape(input)[1], as.integer(num_params)))
      #This is the typical negative log liklihood for categorical,

      #Now take sum
      sum_across_rows = tf$reduce_sum(ll, axis = 1L)
      cat_loss_LL[[index_type]]= tf$reduce_mean(sum_across_rows)
      index_x = index_x + as.integer(num_params) # We move over based on the number of params used for that given categorical variable
      index_y = index_y + as.integer(num_params)
    }
    else {
      stop("Unknown:", distribution)  #A measure to catch errors
    }}

  #Get rid of nulls so we do not have nulls when the indiex was not of that type
  cont_loss_LL <- cont_loss_LL[!sapply(cont_loss_LL, is.null)]
  bin_loss_LL  <- bin_loss_LL[!sapply(bin_loss_LL,  is.null)]
  cat_loss_LL  <- cat_loss_LL[!sapply(cat_loss_LL,  is.null)]

  cont_group_loss <- tf$reduce_mean(tf$stack(cont_loss_LL, axis = 0L),axis = 0L  )
  bin_group_loss  <- tf$reduce_mean(tf$stack(bin_loss_LL,  axis = 0L),axis = 0L  )
  cat_group_loss  <- tf$reduce_mean(tf$stack(cat_loss_LL,  axis = 0L),  axis = 0L )

  if (weighted == 0) {
    w_cont <- tf$constant(1.0, dtype = cont_group_loss$dtype)
    w_bin  <- tf$constant(1.0, dtype = bin_group_loss$dtype)
    w_cat  <- tf$constant(1.0, dtype = cat_group_loss$dtype)
  }
  else if (weighted == 1) {
    w_cont <-  tf$constant(recon_weights[[1]], dtype = cont_group_loss$dtype)
    w_bin  <-  tf$constant(recon_weights[[2]], dtype = bin_group_loss$dtype)
    w_cat  <-  tf$constant(recon_weights[[3]], dtype = cat_group_loss$dtype)}

  group_losses <- tf$stack(list(cont_group_loss, bin_group_loss, cat_group_loss),  axis = 0L )

  group_weights <- tf$stack(list(w_cont, w_bin,  w_cat),axis = 0L)

  weighted_groups   <- tf$multiply(group_weights, group_losses)
  total_loss        <- tf$reduce_sum(weighted_groups, axis = 0L)

  message(paste("Loss - total", total_loss))
  return(list(total_loss,cont_group_loss,bin_group_loss,cat_group_loss) )  }

