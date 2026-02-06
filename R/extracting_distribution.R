###############################
#Getting Feature Distributions#
###############################
#' Build the `feat_dist` data frame for AutoTab
#'
#' Creates one row per original variable with columns:
#' - `column_name`: variable name
#' - `distribution`: one of `"gaussian"`, `"bernoulli"`, or `"categorical"`
#' - `num_params`: number of decoder outputs the VAE should produce for that variable
#'
#' A variable is classified as:
#' - **bernoulli** if it has exactly 2 unique values (any type)
#' - **categorical** if it is a character/factor with more than 2 unique values
#' - **gaussian** otherwise (e.g., numeric with >2 distinct values)
#'
#' AutoTab is not built to handle missing data. A message will prompt the user if the data has NA values.
#'
#' @details
#' In AutoTab, the decoder outputs **distribution-specific parameters** for each variable,
#' not reconstructed values directly. Therefore:
#'
#' - **Continuous (Gaussian)** variables output **two parameters** per feature:
#'   the mean (\eqn{\mu}) and the standard deviation (\eqn{\sigma}).
#' - **Binary (Bernoulli)** variables output **one parameter**:
#'   the probability (`p`) of observing a 1.
#' - **Categorical** variables output **one parameter per category level**:
#'   the probabilities corresponding to each possible class.
#'
#' As a result, the **decoder output matrix** will typically have **more columns**
#' than the original training data.
#'
#' For example, if your original dataset has:
#' ```
#' 1 continuous variable   →  2 decoder parameters
#' 1 binary variable       →  1 decoder parameter
#' 1 categorical variable with 3 levels → 3 decoder parameters
#' ```
#' The total number of decoder outputs will be **2 + 1 + 3 = 6**, even though the
#' input data has only 3 original variables.
#'
#' AutoTab keeps track of this mapping internally through the `feat_dist` object,
#' ensuring that the reconstruction loss and sampling functions correctly handle
#' each distributional head.
#'
#' @param data Data frame of the **original (preprocessed)** variables.
#' @return A data frame with columns `column_name`, `distribution`, and `num_params`. Note: refer to [feat_reorder()].
#'
#' @examples
#' data_example <- data.frame(
#'   cont = rnorm(5),
#'   bin  = c(0,1,0,1,1),
#'   cat  = factor(c("A","B","C","A","C"))
#' )
#'
#' feat_dist <- extracting_distribution(data_example)
#' print(feat_dist)
#' # column_name distribution num_params
#' # 1        cont      gaussian          2
#' # 2         bin     bernoulli          1
#' # 3         cat    categorical          3
#'
#' # The decoder will therefore output 6 total columns (2+1+3)
#'
#'
#' @seealso [feat_reorder()], [set_feat_dist()]
#' @export

extracting_distribution = function(data){
  # Create a data set that the following information will fill in
  feat_dist = data.frame(
    column_name = colnames(data), # keep the same column name
    distribution = character(length(colnames(data))), # distribution type as a string
    num_params = integer(length(colnames(data)))  ) # number of parameters as an integer
  for (i in 1:ncol(data)) {
    variable = data[[i]] #This will loop through each column
    name = colnames(data)[i]

    if (any(is.na(variable))) {
      feat_dist$distribution[i] <- "Missing data - cannot use column"
      feat_dist$num_params[i] <- 0
      message(
        "STOP: The dataset has one or more columns with missing values. ",
        "Please use imputation or other methods to prep data for AutoTab."
      )
      next
    }

    if(is.numeric(variable) && length(unique(variable))>2){ #A numeric column with more than 2 distinct values
      feat_dist$distribution[i] = "gaussian"
      feat_dist$num_params[i] = 2     } #mean and SD
    else if (length(unique(variable))==2){ # character column with only 2 distinct values (binary)
      feat_dist$distribution[i] = "bernoulli"
      feat_dist$num_params[i] = 1     }#just the probability of 1
    else if ((is.character(variable) || is.factor(variable)) && length(unique(variable))>2){ #added option for it to be character or factor
      feat_dist$distribution[i] = "categorical"
      feat_dist$num_params[i] = length(unique(variable))  }}
  return(feat_dist)}

#Make sure order matches the data
#' Reorder `feat_dist` rows to match preprocessed data
#'
#' Ensures row order in `feat_dist` matches the **column prefix order** in the
#' preprocessed (dummy-coded) training data. This assumes dummy columns are
#' named as `<original_name>_<level>` and therefore start with the original
#' variable name.
#'
#' @param feat_dist Data frame from [extracting_distribution()].
#' @param data Data frame of the **original (not preprocessed)** variables.
#' @return The input `feat_dist`, reordered to align with `data`.
#'
#' @examples
#' # Small toy dataset
#' data_example <- data.frame(
#'   cont = rnorm(5),
#'   bin  = c(0, 1, 0, 1, 1),
#'   cat  = factor(c("A", "B", "C", "A", "C"))
#' )
#'
#' # Extract feature distributions in original column order
#' feat_dist <- extracting_distribution(data_example)
#'
#' # Suppose preprocessing (e.g., dummy coding) reordered the columns
#' data_reordered <- data_example[, c("cat", "cont", "bin")]
#'
#' # Reorder feat_dist rows to match the preprocessed data columns
#' feat_dist_reordered <- feat_reorder(feat_dist, data_reordered)
#' feat_dist_reordered
#'
#' @seealso [extracting_distribution()], [set_feat_dist()]
#' @export
feat_reorder = function(feat_dist,data){
  #Reorder to match the order in the data
  feat_names <- feat_dist$column_name  # These are the categorical feature names

  # For each dummy column, figure out which original variable it came from
  get_original_var <- function(colname, vars) {
    match <- vars[which(startsWith(colname, vars))]
    if (length(match) > 0) return(match[1]) else return(NA)}

  # Apply to all columns# Extract column names from the dummy-coded data
  dummy_cols <- colnames(data)
  dummy_to_original <- sapply(dummy_cols, get_original_var, vars = feat_names)
  # Get the order in which original variables appear (first time they show up in dummy_cols)
  ordered_original_vars <- unique(dummy_to_original)
  # Reorder feature_dist to match this order
  feat_dist <- feat_dist[match(ordered_original_vars, feat_dist$column_name), ]
  return(feat_dist)}

