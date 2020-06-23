t0 <- Sys.time()
# load requirements and data
library(MLmetrics)
library(tidyverse)
library(gbm)
library(parallel)
library(vip)
# read in the data
DATA <- readRDS("../data/noCC-master-StrDataLong_GAM-Adjusted_2020-05-20.rds") %>%
  # Change the GROUP variable to a binary indicator: 1 if PI, 0 if COMP
  mutate_at("GROUP", ~as.numeric(. == "PI")) %>%
  # drop the covariates (already used to adjust predictors anyway)
  dplyr::select(-GENDER_FEMALE, -brain_age_yrs, -WAVE, -EstimatedTotalIntraCranialVol)
# DATA <- readRDS("~/DANL/SB/ianMasters/data/master/noCC-master-StrDataLong_GAM-Adjusted_2020-05-20.rds") %>%
#   # Change the GROUP variable to a binary indicator: 1 if PI, 0 if COMP
#   mutate_at("GROUP", ~as.numeric(. == "PI")) %>%
#   # drop the covariates (already used to adjust predictors anyway)
#   dplyr::select(-GENDER_FEMALE, -brain_age_yrs, -WAVE, -EstimatedTotalIntraCranialVol)
##### Note-retained colums: subjWeight, IDENT_SUBID

# PART 1: Stratified train-test splits
# Define the function to make n train-test splits stratified by group.
# Set the randomization seed one time prior to splitting.
# Do so in a for-loop so that splits are made sequentially.
# Thus, if necessary it will be possible to reproduce the splits themeselves.
# -----------------------
makeSplits = function(n, data)
{ 
  dataPartitions = vector(mode = 'list', length = 0)
  pi.index = which(as.logical(data$GROUP)) # converts 1 to TRUE, passes along to which()
  comp.index = which(!as.logical(data$GROUP)) # converts 1 to F, reverses, sends to which()
  pi.subid = data$IDENT_SUBID[pi.index] # subject ids for the pi group
  comp.subid = data$IDENT_SUBID[comp.index] # same for comps
  pi.weights = (1/n_distinct(pi.subid)) * data$subjWeight[pi.index]
  comp.weights = (1/n_distinct(comp.subid)) * data$subjWeight[comp.index]
  # define the train set to be balanced with respect to the groups
  for (i in 1:n) {
    # sample into a training set:
    train.index = c( 
      # create a sample with 70% of the number of people (subject ids)
      sample(pi.index, size = round(n_distinct(pi.subid)*.7), prob = pi.weights), # 70 / 30 split
      sample(comp.index, size = round(n_distinct(pi.subid)*.7), prob = comp.weights) # downsample
    )
    traindata = data[sample(train.index), ] # shuffle the order
    # test data is now those SUBJECTS (not rows) 
    testdata = data[!data$IDENT_SUBID %in% traindata$IDENT_SUBID, ] 
    # Now make a modification to the train data so that the first 70% of
    # the rows contain the true train subset that will be heldout for an internal tuning loop
    # by the gbm() function.
    ids <- unique(sample(traindata$IDENT_SUBID)) # shuffle the order again
    indexer <- 1
    tunedata <- traindata[0, ] # create an empty data frame with the same column names
    while(nrow(tunedata)/nrow(traindata) < .33) {
      tunedata <- rbind(tunedata, 
                        traindata %>% dplyr::filter(IDENT_SUBID == ids[indexer]))
      traindata <- traindata %>% dplyr::filter(IDENT_SUBID != ids[indexer])
      indexer <- indexer + 1
    }
    traindata <- rbind(traindata, tunedata)
    dataPartitions[[paste0('Resample_', i)]] <- list('train' = traindata,
                                                     'tunedata' = tunedata,
                                                     'trainfrac' = (nrow(traindata)-nrow(tunedata))/nrow(traindata),
                                                     'test' = testdata, 
                                                     'train.indices' = train.index)
  }
  return(dataPartitions)
}
# Split the data 1000 times
# SET ONE SEED NOW, THEN CREATE ALL RESAMPLES SEQUENTIALLY.
set.seed(2020)
ttSplits <- makeSplits(100, DATA)

# PART 2
# Function to fit each gbm, internally cross-validate the hyperparameters of depth and shrinkage, and save best mod.
PARAMS <- expand.grid(lambda = c(.0005, .0010, .0015, .0020),
                      depth = c(1, 2, 3, 4, 5),
                      nodemin = seq(1,31, by=2))
gbmCV <- function(ttSplit, param_grid)
{
  grid_search <- apply(param_grid, 1, function(params) {
    model <- try(gbm(
      GROUP ~.-IDENT_SUBID-subjWeight, 
      data = ttSplit$train, # supply only the train data
      distribution = "bernoulli", # model the binary outcome with a logit link fn
      n.trees = 750, # a large number to ensure the minimum criterion is reached
      shrinkage = params["lambda"],
      interaction.depth = params["depth"],
      n.minobsinnode = params["nodemin"],
      train.fraction = ttSplit$trainfrac, # construct the gbm on the first train.fraction*nrow(data) rows
      bag.fraction = 1 # do not resample from the train set, we manually created a test set.
    ))
    if (class(model) == "try-error") {
      return(list(Tree = NA_integer_, TestAUC = -Inf))
    } # this will also end the anonymous function inide the apply call.
    # Otherwise, if the model fit successfully:
    # now for each tree in this one gbm model, get the AUC to our test set.
    tree.num <- gbm.perf(model, method = "test", plot.it = F)
    testAUC <- AUC(predict(model, newdata = ttSplit$test, type = "response", n.trees = tree.num),
                   y_true = ttSplit$test$GROUP)
    best = list(Tree = tree.num, TestAUC = testAUC)
    # if the best model is the last tree, then assume it hadn't yet converged
    # Add 16 trees at a time, and stop the model when the best model is at least 15 trees in the past
    # If the number of trees reaches 2000, then abort, and return the results of 2000th tree
    while((model$n.trees - best$Tree) < 15) {
      model <- gbm.more(model, n.new.trees = 16)
      tree.num <- gbm.perf(model, method = "test", plot.it = F)
      testAUC <- AUC(predict(model, newdata = ttSplit$test, type = "response", n.trees = tree.num),
                     y_true = ttSplit$test$GROUP)
      best = list(Tree = tree.num, TestAUC = testAUC)
      if (model$n.trees >= 2000) {break}
    } # end of while loop
    
    # Now best is either the normally converging results, or the extended results from "while"
    # We need to return the best model, the AUC, as well as # of the best tree (not in that order)
    list(TheModel = model, BestTree = best$Tree, TestAUC = best$TestAUC)
  }) # ends the anonymous function inside the apply() and writes into "grid_search"
  # Now we have one model for each hyperparameter combination in "grid_search"
  # Grid Search CV is done. Now we simply delete all the worse models, and output the final model.
  # Identify the best one:
  bestModNum <- which.max(sapply(grid_search, function(x) {x$TestAUC}))
  THEBESTMODEL <- grid_search[[bestModNum]]$TheModel
  # Now that we know bestModNum is where the best model occurred, output it and related results
  # We also want to compute the variable importances and output them along with this model.
  # First we need to define a helper function
  predict.gbm_bestTree <- function(object, newdata) 
  {
    predict(object, newdata, type = "response", n.trees = grid_search[[bestModNum]]$BestTree)
  }
  pvi = vi_permute(
    object = THEBESTMODEL, # supply the model
    feature_names = names(DATA %>% dplyr::select(-subjWeight, -IDENT_SUBID, -GROUP)), # just predictors
    train = ttSplit$test, # supply the TEST data instead
    target = ttSplit$test$GROUP, # test set response
    metric = "auc", # AUC-based importance
    reference_class = 1, # PI are coded 1 (since using "auc")
    type = "difference", # PVI = auc - auc*
    nsim = 100, # permute each column 1000 times to compute its importance
    keep = TRUE, # KEEP ALL PERMUTATIONS
    sample_frac = 1, # use the whole test set
    pred_wrapper = predict.gbm_bestTree # instruct each permutation to predict from this fn
  )
  bestModelPkg <- list(
    "GBM"= list('BestModelTestSetPreds' = predict(THEBESTMODEL, newdata = ttSplit$test, type='response',n.trees = grid_search[[bestModNum]]$BestTree),
                'BestModel_BestNumTrees' = grid_search[[bestModNum]]$BestTree,
                'BestModel_TestAUC' = grid_search[[bestModNum]]$TestAUC,
                # For ease of access, we also want to know what the best parameters were
                'BestParams' = as.data.frame(param_grid) %>% slice(bestModNum)),
    "PVI" = attr(pvi, "raw_scores") # just return this matrix with all scores, we can summarize later
  )
  return(bestModelPkg) # one will be returned for each ttSplit
} # end of gbmCV function

# Run it in parallel on each element of ttSplits
gbmModelList = mclapply(
  mc.cores = 20,
  X = ttSplits,
  FUN = function(x) {gbmCV(ttSplit = x, param_grid = PARAMS)}
)
# just in case the next part hits an error, save these models immediately.
saveRDS(gbmModelList, "../output/noCC_classification_gbmCV-resultsList_2020-31-05.rds")
print("models fit")
# gbmModelList = lapply(
#   X = ttSplits,
#   FUN = function(x) gbmCV(ttSplit = x, param_grid = PARAMS)
# )
# Now just loop through and attach the train indices and test subj id for each iteration
for (i in 1:length(gbmModelList)) {
  gbmModelList[[i]]$data = list(
     "train.indices"= ttSplits[[i]]$train.indices,
     "test.subid" = ttSplits[[i]]$test$IDENT_SUBID
  )
}

# Save the result
saveRDS(gbmModelList, "../output/noCC_classification_gbmCV-resultsList_2020-31-05.rds")
print('Done')
print('Elapsed:')
Sys.time() - t0
