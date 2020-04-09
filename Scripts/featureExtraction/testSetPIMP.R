##########################################################
# Function to compute the permutation importance p-value
# using the PIMP algorithm (Alltman et al 2014); 
# adapted for test-set variable importance,
# implemented with ranger::ranger()
# Ian J. Douglas - April 2020
# version 0.0.1                     
#
# Note: currently implemented with AUC-gain-based importance (not error loss)
#

testSetPIMP <- function(
  forest.ranger, test.data = NULL, xtest=NULL, ytest=NULL, target.class, 
  nPerms.null = 1000, nPerms.imp = 1, random.seed = NULL, nCores=NULL
) {
  # require packages
  require(dplyr); require(purrr); require(MLmetrics); require(ranger); require(parallel)
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
  
  # 2. Compute the AUCs on the predictions to the test set
  ## 2a. Extract the predited probabilities (for each test subject & both classes) for each tree
  ### This is a subject x class x tree 3-D array, so now convert it to a list (length == num.trees)
  treewise_result = purrr::array_branch(array = prediction_obj$predictions, margin = 3)
  ## 2b. Compute the actual teset-set AUC for each individual tree; add this to each tree's results inplace
  testSetAUC_vector <- map_dbl(.x = treewise_result, .f = function(x) {
    # extract the predicted probability of being in the target class:
    target_proba = x[, target.i]
    # compute the AUC:
    MLmetrics::AUC(y_pred = target_proba,
                   y_true = ifelse(test_df[,factor.i] == target.class, 1, 0))
  })
  ###### Prior to iteratively permuting each X variable; compute the null distribution of Y
  newY <- vector(mode = "list", length = nPerms.null)
  for (i in 1:nPerms.null) {newY[[i]] <- sample(test_df[, factor.i])}
  newY_df <- purrr::reduce(newY, data.frame)
  # Now using these permuted Y vectors, compute the AUC from real the model, but a random Y
  nullAUCs_fromRealX.list = lapply(newY_df, function(newYcol) {
    # create the vector of null AUCs for each tree, at each random premutation of Y:
    map_dbl(.x = treewise_result, .f = function(x) {
      # extract the predicted probability of being in the target class:
      target_proba = x[, target.i]
      # compute the AUC:
      MLmetrics::AUC(y_pred = target_proba,
                     y_true = ifelse(newYcol == target.class, 1, 0))
    }) # return it from map_dbl()
  })
  # Now, for each random permutation of Y, we have the corresponding null AUC from the real model 
  ###### Thus, each p-value is generated against the same null distr.
  
  # 3. Compute the AUC-based permutation importances as follows (adapted from Janitza et al., 2013)
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
      # permute the column whose name is stored in "nm", nPerms times (and aggregate the random results)
      newX <- vector(mode = "list", length = nPerms.imp)
      for (i in 1:nPerms.imp) {newX[[i]] <- sample(test_df[,nm])}
      new_data = test_df %>% mutate_at(nm, ~rowMeans(reduce(newX, cbind))) 
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
      treewise_importances <- as.vector(testSetAUC_vector) - as.vector(permAUCs)
      # Average these results (compute AUC_VIM for this variable)
      auc_vim <- mean(treewise_importances)
      
      # 4. Permutation p-values:
      ### i. Generate a matrix of permuted clones of the Y column
      ## ii. Using the real model, comute a null AUC for each of these random Y vectors
      ###### (i) and (ii) completed above; now:
      # iii. Compute the AUCs on each of these random Y vectors, using the permuted X data
      ## iv. Subtract the new AUCs from the those predicting random Y from a real X matrix
      ### v. Average each resulting vector, for the null distribution of bAgg-ed importances
      
      ### 4iii:
      nullImps <- sapply(1:ncol(newY_df), function(n) {
        # for each tree from the permutated X data, compute AUC at each permuted version of Y
        nullTree_AUC <- map_dbl(.x = permForest_preds, .f = function(Y) {
          new_prob <- Y[, target.i]
          # compute the null AUC:
          MLmetrics::AUC(y_pred = new_prob,
                         y_true = ifelse(newY_df[,n] == target.class, 1, 0))
        })
        # Subtract the null AUC from this tree, at this null permutation of Y
        # (4iv):
        impsPerTree <- as.vector(nullAUCs_fromRealX.list[[n]]) - as.vector(nullTree_AUC)
        # Finally, bAgg the importances at each tree for the variable's importance at this random Y
        mean(impsPerTree) # return this (step 4v).
      })
      # Now nullImps has one variable importance at each random Y
      # Note, we are inside a single iteration for one variable. So this is variable nm's null distr.
      list(
        "AUC_VIM" = auc_vim, # this variable's AUC_VIM
        "null.AUC_VIM.distr" = nullImps
      )
    } # end the function(nm)
  ) # end mclapply()
  
  # Importances is now a list, containing one item for each variable; so label it as such:
  nulls <- lapply(Importances, function(x) {
    vec <- x$null.AUC_VIM.distr
    names(vec) <- NULL
    vec
  })
  names(nulls) <- names(test_df)[-factor.i]
  # compute p.values
  pvals = sapply(Importances, function(x) {
    (sum(x$AUC_VIM < x$null.AUC_VIM.distr) + 1) / (nPerms.null + 1)
  })
  impFrame = data.frame("variable" = names(test_df)[-factor.i],
                        "importance" = sapply(Importances, function(x) x$AUC_VIM),
                        "p.value" = pvals,
                        stringsAsFactors = FALSE)

  out = list("crossValScore" = mean(testSetAUC_vector),
             "testSetImportance" = list(varImps = impFrame, NullAUCs = nulls))
  return(out)
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
# vi = testSetPIMP(forest.ranger = train_irisRanger, 
#                  test.data = test_iris, 
#                  target.class = "setosa", 
#                  random.seed = 1,
#                  nPerms.null = 100,
#                  nPerms.imp = 100,
#                  nCores = 3)
# data.frame(
#   "testSetAUC_VI" = sapply(vi$testSetImportance, function(x) x$AUC_VIM),
#   "OOBAccuracy_VI" = importance(train_irisRanger)
# )
#              testSetAUC_VI OOBAccuracy_VI
# Sepal.Length    0.01163579    0.010103367
# Sepal.Width     0.00433281    0.005097003
# Petal.Length    0.26533830    0.196612073
# Petal.Width     0.28656593    0.206918675
# # END NOT RUN