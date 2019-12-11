# set seed for reproudicibility.
set.seed(111)
t1 = Sys.time()
# Load (or install) required packages
## xgboost
if ("xgboost" %in% installed.packages()) {
  library(xgboost)
} else {
  install.packages("xgboost")
  library(xgboost, quietly = TRUE)
}
## Skip lime for now
# if ("lime" %in% installed.packages()) {
#   library(lime)
# } else {
#   install.packages("lime")
#   library(lime)
# }
## tidyverse
if ("tidyverse" %in% installed.packages()) {
  library(tidyverse)
} else {
  install.packages("tidyverse")
  library(tidyverse, quietly = TRUE)
}
## MLmetrics
if ("MLmetrics" %in% installed.packages()) {
  library(MLmetrics)
} else {
  install.packages("MLmetrics")
  library(MLmetrics, quietly = TRUE)
}

### Read in the structural data (has been more reliable in this sample)
StrData = readRDS("../../data/processed/structuralLabelled.rds")
StrData_noWBV = select(StrData,-EstimatedTotalIntraCranialVol)

### Prepare the data for compatibility with `xgb.cv()`
# Convert the GROUP variable to a dummy variable for the PI group
y_str = as.numeric(ifelse(StrData_noWBV$GROUP =="PI",1,0))
X_str = StrData_noWBV %>%
  select(-SUBJECTID_long, -age, -GROUP, -cbcl_totprob_t, -wave_to_pull) %>%
  select_if(is.numeric) %>% # just a failsafe here.
  # Predictor variables are required to be in matrix form for xgboost package
  as.matrix()


# Make the param grid
hyper_grid <- expand.grid(
  eta = c(.01, .05, .1, .3),
  max_depth = c(1, 3, 5, 7),
  min_child_weight = c(1, 3, 5, 7),
  subsample = c(.65, .8, 1), 
  colsample_bytree = c(.8, .9, 1),
  optimal_trees = 0, # a place to capture results
  max_AUC = 0,
  min_logloss = 0 # a place to capture results
)


# Run the loop (Grid Search)
# (just run on the structural data for now)
for(i in 1:nrow(hyper_grid)) {
  
  # create the unique parameter list for the i-th iteration
  params <- list(
    eta = hyper_grid$eta[i],
    max_depth = hyper_grid$max_depth[i],
    min_child_weight = hyper_grid$min_child_weight[i],
    subsample = hyper_grid$subsample[i],
    colsample_bytree = hyper_grid$colsample_bytree[i]
  )
  
  # train model
  xgb.tune <- xgb.cv(
    params = params, # put the params here.
    data = X_str,
    label = y_str,
    nrounds = 5000, # grow out each machine at MOST to 5000 trees.
    nfold = 5, # k-fold CV
    stratified = TRUE, # stratified sampling by the predicted factor.
    metrics = list("auc", "logloss"), #
    objective = "binary:logistic", # objective and link functions
    verbose = 0,
    early_stopping_rounds = 10 # stop if no improvement for 10 consecutive trees
  )
  
  # add min training error and trees to grid
  hyper_grid$optimal_trees[i] <- which.max(xgb.tune$evaluation_log$test_auc_mean)
  hyper_grid$max_AUC[i] <- max(xgb.tune$evaluation_log$test_auc_mean)
  hyper_grid$min_logloss[i] <- min(xgb.tune$evaluation_log$test_logloss_mean)
}


# now has the results plugged in:
saveRDS(hyper_grid, "../../output/GBM/hyperGrid_gbmResults.rds") 
gbm_res = readRDS("../../output/GBM/hyperGrid_gbmResults.rds")


# Fit the best XGB model for futher evaluation using LIME
best_params = gbm_res[which.max(gbm_res$max_AUC), 1:5] %>% # select the 5 tuned params over
  do.call("list", .) # conver to list.
# add the objective and link to the params
best_params[["objective"]] <- "binary:logistic"

# fit to the full sample:
bestGBM = xgboost(
  params = best_params,
  data = X_str,
  label = y_str,
  nrounds = 5000,
  early_stopping_rounds = 50,
  verbose = 0,
  stratified = TRUE
)

#ANOTHER fit to the full sample: 
### use xgb.train() for compatibility (later) with lime()
bestXGB_4LIME = xgb.train(
  best_params,
  xgb.DMatrix(X_str, label = y_str),
  nrounds = 5000,
  verbose = 0,
  stratified = TRUE
)

# train-test cross-val:
PARAMS = best_params
PARAMS[["eval_metric"]] <- "error"
R = 1000
crossValScore = NULL
for (i in 1:R) {
  set.seed(i)
  iTrain = sample(1:length(y_str), size = round(length(y_str)*.7))
  test = xgb.DMatrix(X_str[-iTrain,], label = y_str[-iTrain])
  mod = xgb.train(
    PARAMS,
    xgb.DMatrix(X_str[iTrain,], label = y_str[iTrain]),
    early_stopping_rounds = 50,
    stratified = TRUE,
    verbose = 0,
    nrounds = 5000,
    watchlist = list(validation = test)
  )
  crossValScore[i] = mod$best_score
  if (i == R) {set.seed(111)}
}

#bootstrap and aggregate the model errors for a boot.ci
GBM = function(X, y) {
  mod = xgboost(
    params = best_params,
    data = X,
    label = y,
    nrounds = 5000,
    early_stopping_rounds = 50,
    verbose = 0,
    stratified = TRUE
  )
  return(mod)
}

B = 1000
bootOut = data.frame(inBag_error = rep(NA, times = B), 
                     OOB_error = rep(NA, times = B))
for (i in 1:B) {
  # stratified bootstrap sampling (test set is imbalanced):
  n_min = min(sum(y_str == 1), sum(y_str == 0))
  inds = c(sample(which(y_str==1), size = n_min, replace = TRUE),
           sample(which(y_str==0), size = n_min, replace = TRUE))
  fit = GBM(X = X_str[inds,], y = y_str[inds])
  bootOut$inBag_error[i] = fit$best_score
  pred = ifelse(predict(fit, newdata = X_str[-inds, ]) >= .5, 1, 0)
  cm = MLmetrics::ConfusionMatrix(pred, y_str[-inds])
  bootOut$OOB_error[i] = 1 - sum(diag(cm))/sum(cm)
}


saveRDS(bestXGB_4LIME,"../../output/GBM/bestXG_trainFullSamp.rds")
saveRDS(bestGBM,"../../output/GBM/bestGBM.rds")
saveRDS(bootOut,"../../output/GBM/bootOutput.rds")
saveRDS(crossValScore, "../../output/GBM/GBM_crossValScores.rds")

print("Success")
Sys.time()
duration = Sys.time() - t1
print(paste0("duration: ", duration))