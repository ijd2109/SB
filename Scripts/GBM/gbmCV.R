set.seed(111)
require(xgboost)
#require(lime)
require(tidyverse)
require(doParallel)

StrData = readRDS("../../data/processed/structuralLabelled.rds")
StrData_noWBV = select(StrData,-EstimatedTotalIntraCranialVol)

hyper_grid <- expand.grid(
  eta = c(.01, .05, .1, .3),
  max_depth = c(1, 3, 5, 7),
  min_child_weight = c(1, 3, 5, 7),
  subsample = c(.65, .8, 1), 
  colsample_bytree = c(.8, .9, 1),
  optimal_trees = 0,               # a place to dump results
  max_AUC = 0,
  min_logloss = 0# as place to dump results
)




y_str = as.numeric(ifelse(StrData_noWBV$GROUP =="PI",1,0))
X_str = select(StrData_noWBV, -SUBJECTID_long, -age, -GROUP) %>%
  select_if(is.numeric) %>%
  as.matrix()



# just run on the structural data for now
for(i in 1:nrow(hyper_grid)) {
  
  # create parameter list
  params <- list(
    eta = hyper_grid$eta[i],
    max_depth = hyper_grid$max_depth[i],
    min_child_weight = hyper_grid$min_child_weight[i],
    subsample = hyper_grid$subsample[i],
    colsample_bytree = hyper_grid$colsample_bytree[i]
  )
  
  # reproducibility
  set.seed(111)
  
  # train model
  xgb.tune <- xgb.cv(
    params = params,
    data = X_str,
    label = y_str,
    nrounds = 5000,
    nfold = 5,
    stratified = TRUE,
    metrics = list("auc", "logloss"),
    objective = "binary:logistic",
    verbose = 0,               # silent,
    early_stopping_rounds = 10 # stop if no improvement for 10 consecutive trees
  )
  
  # add min training error and trees to grid
  hyper_grid$optimal_trees[i] <- which.max(xgb.tune$evaluation_log$test_auc_mean)
  hyper_grid$max_AUC[i] <- max(xgb.tune$evaluation_log$test_auc_mean)
  hyper_grid$min_logloss[i] <- min(xgb.tune$evaluation_log$test_logloss_mean)
}

saveRDS(hyper_grid,"hyperGrid_gbmResults.rds")
