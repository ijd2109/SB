##########################################################
# Function to compute an AUC-based permutation importance
# for imbalanced data, based on (combining) Janitza et al. 
# (2013) & Janitza et al. (2015) describing AUC-based 
# and test-set VIM, respectively. Compatible with ranger 
# random forest objects generated from ranger::ranger()
# Ian J. Douglas - March 2020
# version 0.0.1                     
#

testSetAUC_VIM <- function(
  forest.ranger, test.data = NULL, xtest=NULL, ytest=NULL, target.class, 
  random.seed = NULL, nCores=NULL
) {
  # require packages
  require(dplyr); require(purrr); require(MLmetrics)
  # check that the random forest model and function args have required attributes
  if (class(forest.ranger) != "ranger") {
    stop("Function is currently only implemented for ranger objects")
  }
  if (forest.ranger$treetype != "Probability estimation") {
    stop("To estimate ROC-AUC, recreate forest with ranger::ranger(..., type='probability')")
  }
  if (length(forest.ranger$forest$levels) > 2) {
    stop("More than two classes detected; multinomial AUC not currently supported")
  }
  if (all(sapply(list(test.data, xtest, ytest), is.null))) {
    stop("Supply either test.data or both xtest and ytest. Supplying training data to any will yield meaningless results")
  }
  # Define some variables to insert into the below functions
  if (is.null(test.data)) {
    test_df <- data.frame(xtest, ytest)
  } else test_df <- test.data
  factor.i = which(
    sapply(test_df, function(x) {identical(levels(as.factor(x)), forest.ranger$forest$levels)})
  )
  target.i = which(forest.ranger$forest$levels == target.class)
  .seed = ifelse(is.null(random.seed), 1, random.seed)
  
  # 1. Compute the forest predictions for the test data
  prediction_obj = predict(
    object = forest.ranger, 
    data = test_df, 
    seed = .seed,
    predict.all = T # get the predictions from each individual tree
  )
  # 2. Extract the predited probabilities (for each test subject & both classes) for each tree
  ### This is a subject x class x tree 3-D array, so now convert it to a list (length == num.trees)
  treewise_result = purrr::array_branch(array = prediction_obj$predictions, margin = 3)
  # 3. Compute the actual teset-set AUC for each individual tree; add this to each tree's results inplace
  treewise_result = map(.x = treewise_result, .f = function(x) {
    target_proba = x[, target.i] # extract the predicted probability of being in the target class
    .auc <- MLmetrics::AUC(y_pred = target_proba,
                           y_true = ifelse(test_df[,factor.i] == target.class, 1, 0))
    list('proba' = x, "auc" = .auc, "truetest" = test_df[factor.i])
  })
  # For convenience, vectorize the AUC values for each tree for comparison with permuted AUCs later
  treewise_auc = sapply(treewise_result, function(L) L$auc)
  
  # 4. Compute the AUC-based permutation importances as follows (adapted from Janitza et al., 2013)
  # For each predictor i in the test data {1 ... p}:
  #### i. Shuffle (permute) that variable's values in place
  ### ii. Regenerate a predicted probability from each tree in the forest model with this permuted data
  ## iii. Use these probabilities to compute a new AUC for each tree in the forest; simplify to a vector
  ### iv. Subtract the real AUC vector (above: "treewise_auc") from the new vector obtained in (iii)
  #### v. Average the values in the vector obtained in (iv) to calculate the i-th variable's importance
  
  # NOTES: The procedure also requires deleting any trees for which the test set did not sample both classes
  # - This is redundant beacuse test data is required by the function to be supplied by the user (rather than OOB)
  # - Also, the average values will be returned in addition to the importances to individual trees
  Importances = mclapply(mc.silent = F, mc.cores = ifelse(is.null(nCores), 1, nCores),
    X = names(test_df)[-factor.i], # operate on all the variables minus the outcome factor
    FUN = function(nm) {
      # permute the column, calling it by name
      new_data = test_df %>% mutate_at(nm, sample)
      # create a list with the new predicted probabilities
      permForest_preds = array_branch(
        predict(forest.ranger, data=new_data, predict.all = T, seed = .seed)$predictions, margin = 3 
      )
      # calculate the new AUCs
      permAUCs = map_dbl(permForest_preds, .f = function(Y) {
        new_prob = Y[, target.i]
        perm.auc <- MLmetrics::AUC(y_pred = new_prob, 
                                   y_true = ifelse(test_df[, factor.i] == target.class, 1, 0))
        # return them; map_dbl will convert them to a vector
        perm.auc
      })
      
      # Subtract the real AUCs by the permuted values to calculate importance
      treewise_importances <- as.vector(treewise_auc) - as.vector(permAUCs)
      # Average these results (compute AUC_VIM for this variable)
      list(
        "AUC_VIM" = mean(treewise_importances),
        # Also return the AUC-based importance for this variable within each tree of the forest
        "treewise_importances" = data.frame(tree = paste0("tree_", 1:length(treewise_importances)),
                                            "importance" = treewise_importances, 
                                            row.names="tree")
      )
    } # end the function(nm)
  ) # end mclapply()
  
  # Importances is now a list, containing one item for each variable; so label it as such:
  names(Importances) <- names(test_data)[-factor.i]
  # Each element of the list has the importance for the variable (for the whole forest);
  # and also the raw importances to each tree in the forest that were used to compute the VIM
  
  return(Importances)
}

# # Example. NOT RUN
# newIris <- iris %>% 
#   # recode to predict if "Species" is setosa, or not
#   mutate_at("Species", ~factor(ifelse(.=="setosa", "setosa", "not_setosa"), 
#                                levels = c("setosa","not_setosa")))
# train.i <- sample(1:nrow(newIris), size = round(nrow(newIris)*.5))
# train_iris <- newIris[train.i, ]
# test_iris <- newIris[-train.i, ]
# # Fit ranger model
# train_irisRanger = ranger(Species ~., data = train_iris, probability = T, importance="permutation")
# vi = testSetAUC_VIM(train_irisRanger, test.data = test_iris, target.class = "setosa")
# data.frame(
#   "testSetAUC_VI" = sapply(vi, function(x) x$AUC_VIM),
#   "OOBAccuracy_VI" = importance(train_irisRanger),
# )
# # END NOT RUN