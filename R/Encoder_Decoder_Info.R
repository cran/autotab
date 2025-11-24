#' Specifying Encoder and Decoder Architectures for `VAE_train()`
#'
#' @section Encoder and Decoder configuration:
#' The arguments `encoder_info` and `decoder_info` define the architecture of
#' the encoder and decoder networks used in `VAE_train()`. Each is a list in
#' which every element describes one layer in sequence.
#'
#' AutoTab currently supports two layer types: `"dense"` and `"dropout"`.
#'
#' **Dense layers**
#'
#' When `input1 = "dense"`, the layer specification takes the form:
#'
#' \itemize{
#'   \item `input2`: *Numeric.* Number of units (nodes).
#'   \item `input3`: *Character.* Activation function
#'         (any TensorFlow/Keras activation name).
#'   \item `input4`: *Integer (0/1).* L2 regularization flag. Default: `0`.
#'   \item `input5`: *Numeric.* L2 regularization strength (`lambda`). Default: `1e-4`.
#'   \item `input6`: *Logical.* Apply batch normalization. Default: `FALSE`.
#'   \item `input7`: *Numeric.* Batch normalization momentum. Default: `0.99`.
#'   \item `input8`: *Logical.* Whether batch normalization scale and center
#'         parameters are trainable. Default: `TRUE`.
#' }
#'
#' **Dropout layers**
#'
#' When `input1 = "dropout"`, the layer specification is:
#'
#' \itemize{
#'   \item `input2`: *Numeric.* Dropout rate.
#' }
#'
#' Together, these lists fully specify the encoder and decoder architectures
#' used during VAE training.
#'
#' @seealso [VAE_train()]
#' @name encoder_decoder_information
NULL

