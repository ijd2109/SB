.libPaths()
if (!require(vip)) {install.packages("vip"); library(vip)}
library(randomForest)
library(tidyverse)
library(parallel)
library(MLmetrics)
source('min_max.R')
source('descending_rank.R')
startTime <- Sys.time()
print(paste0("Packages read in, start time: ", startTime))
# Read in the data that contains all waves for all subjects (subcortical volume data)
adjustedData <- readRDS("../data/masterAdjustedStrDataLong_2020-04-28.rds")
# create "weighted" data by duplicating rows for subjects with only one observation
w.rf_data <- adjustedData %>%
  rbind(., filter(adjustedData, subjWeight == 1)) %>%
  select(-WAVE,-brain_age_yrs,-GENDER_FEMALE, -EstimatedTotalIntraCranialVol,
         # DELETE THE CORPUS CALLOSUM WHITE MATTER VARIABLES
         -starts_with("CC_"))

# Set up a function to fit a single tree with identical parameters, to a bootstrap sample
# define a function to produce the trees
rfClassifier <- function(seed)
{
  set.seed(seed)
  randomForest(
    GROUP ~.-IDENT_SUBID-subjWeight, data = w.rf_data,
    mtry = sqrt(ncol(w.rf_data) - 2),
    ntree = 1, # one tree,
    nodesize = 3, # terminal nodes must contain at least 3 subjects (with the same prediction)
    strata = w.rf_data$GROUP,
    sampsize = rep(n_distinct(adjustedData$IDENT_SUBID[adjustedData$GROUP=="PI"]), 
                   times = n_distinct(adjustedData$GROUP)),
    importance = TRUE,
    keep.inbag = TRUE,
    keep.forest = TRUE,
    replace = TRUE
  )
}

print(paste0("Fitting trees. Start time: ", Sys.time()))
# Fit a forest of these single bootstrapped trees
w.forest <- mclapply(X = 1:901, FUN = function(seed) {rfClassifier(seed = seed)}, mc.cores = 12)
#w.forest <- lapply(X = 1:901, FUN=function(seed) {rfClassifier(seed = seed)})

print(paste0("Summarizing random forest results. Start time: ", Sys.time()))
# now add to each result the IDENT_SUBID of all subjects in bag and oob
# Note, this is done manually b/c inbag does not mean a subject is not ALSO out of bag
for (i in 1:length(w.forest)) {
  the_model <- w.forest[[i]]
  inbag.index <- row(the_model$inbag)[, 1][the_model$inbag[, 1] != 0]
  w.forest[[i]]$inbag.id <- w.rf_data$IDENT_SUBID[inbag.index]
  w.forest[[i]]$oob.id <- w.rf_data$IDENT_SUBID[!w.rf_data$IDENT_SUBID %in% w.forest[[i]]$inbag.id]
  w.forest[[i]]$the_tree <- getTree(the_model, k = 1, labelVar = TRUE) # also add the tree
  # Now get the out of bag predictions to the people in "oob.id"
  w.forest[[i]]$oob.predictions <- predict(
    the_model, 
    newdata = w.rf_data[w.rf_data$IDENT_SUBID %in% w.forest[[i]]$oob.id, ],
    type = "response" # this will be either 1 or 0, as an integer rather than character
  )
  w.forest[[i]]$oob.confusion <- ConfusionMatrix(
    y_pred = w.forest[[i]]$oob.predictions,
    y_true = w.rf_data$GROUP[w.rf_data$IDENT_SUBID %in% w.forest[[i]]$oob.id]
  )
  w.forest[[i]]$oob.f1_score <- F1_Score(
    w.forest[[i]]$oob.predictions, 
    y_true = w.rf_data$GROUP[w.rf_data$IDENT_SUBID %in% w.forest[[i]]$oob.id], 
    positive = "PI") 
  w.forest[[i]]$oob.Accuracy <- with(w.forest[[i]], sum(diag(oob.confusion))/sum(oob.confusion))
}
# Forest score
w.rf_results <- list()
for (i in c("oob.Accuracy", "oob.confusion", "oob.f1_score")) {
  if (i == "oob.confusion") {
    w.rf_results$Trees[[i]] <- lapply(w.forest, function(x) x[[i]])
  } else w.rf_results$Trees[[i]] <- sapply(w.forest, function(x) x[[i]])
}
w.rf_results$Forest$oob.f1_score <- mean(w.rf_results$Trees$oob.f1_score)
w.rf_results$Forest$oob.confusion <- matrix(rowMeans(map_dfc(
  w.rf_results$Trees$oob.confusion, function(m) {data.frame(x = as.vector(m))}
)), ncol = 2, dimnames = list(c("PI", "COMP"), c("PI", "COMP")))
w.rf_results$Forest$oob.Accuracy <- mean(w.rf_results$Trees$oob.Accuracy)

# Save elements of the results in a list
saveRDS(list("model" = w.forest, "results" = w.rf_results),
        "../output/ManualRandomForestWeighted_object+Results.rds")
# Clear up the environment of un-needed objects
rm(list = c("w.rf_results", "w.forest", "adjustedData"))
# Read the list back in:
w.rf_resultsList <- readRDS("../output/ManualRandomForestWeighted_object+Results.rds")

print(paste0("Computing variable importances per tree. Start time: ", Sys.time()))
# Compute the variable importances for each tree to its corresponding OOB set
forest_of_imps <- mclapply(
  mc.cores = 12, # set the number of parallel cores
  X = w.rf_resultsList$model, # for each (1 tree random forest) model; do:
  FUN = function(rf) {
    vi_permute(
      object = rf, 
      feature_names = names(select(w.rf_data, -IDENT_SUBID, -GROUP, -subjWeight)),
      train = w.rf_data %>% filter(IDENT_SUBID %in% rf$oob.id), # select OOB subjects
      target = w.rf_data$GROUP[w.rf_data$IDENT_SUBID %in% rf$oob.id], # same
      metric = metric_accuracy, # accuracy-based permutation importance
      smaller_is_better = F, # larger accuracy is better
      type = "difference", # VI = accuracy_true - accuracy_permuted
      # function to get the predicted OOB accuracy:
      pred_wrapper = function(object, newdata) {predict(object, newdata, type = "response")},
      nsim = 100 # number of times to permute each predictor to calculate its importances
    ) %>%
      # the function outputs importance, importance SD, and the variable name; calculate rank:
      mutate(Rank = descending_rank(Importance),
             absoluteRank = descending_rank(abs(Importance))) %>%
      arrange(Rank)  # arrange each df by Rank
  }
)

print(paste0("Calculating variable importances. Start time: ", Sys.time()))
# Save the entire forest of raw variable importances, and the summarized (bagged) version:
### First the summary:
aggImps = reduce(.x = forest_of_imps, .f = rbind) %>%
  group_by(Variable) %>%
  summarize_all(mean) %>%
  ungroup() %>%
  arrange(desc(Importance)) %>%
  mutate_at(vars(Variable), ~factor(., levels = rev(.)))
# And the raw values, with some summary statistics added and gathered into long format.
allImps = reduce(.x = forest_of_imps, .f = rbind) %>%
  mutate(tree = rep(1:(nrow(.)/n_distinct(Variable)), each = n_distinct(Variable))) %>%
  mutate_at("Variable", ~factor(., levels = levels(aggImps$Variable))) %>%
  group_by(Variable) %>% # now get point estimates for each variables's mean and SD
  mutate(VI = mean(Importance), impSD = sd(Importance), 
         avgRank = mean(Rank), rankSD = sd(Rank), 
         avgAbsRank = mean(absoluteRank), aRankSD = sd(absoluteRank)) %>% ungroup()
### Save them
saveRDS(aggImps, "../output/aggImps.rds"); saveRDS(allImps, "../output/allImps.rds")
print(paste0("Done. Elapsed time: ", Sys.time() - startTime))




