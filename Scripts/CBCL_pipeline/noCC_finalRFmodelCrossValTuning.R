# # # # # # # # # # # # # # # # # # # # # #
# Grid Search Cross Validation for hyper-
# parameter tuning of final models
# Ian Douglas - May 2020
# ----
set.seed(111) # SET SEED FOR REPRODUCIBILITY
t0 <- Sys.time()
# Load required packages
library(randomForest)
library(tidyverse)
library(parallel)
# ----
# Read in the data.
# The data read in is a named list containing three data frames
# Each data frame has one unique CBCL subscale (the response variable)
# Each data frame has the same structural brain volume predictors
df.list <- readRDS("../data/finalStrAdjDF-LIST_2020-05-07.rds")
# ----
# Preprocess each data frame identically for each set
print(paste0(Sys.time(), "- Preprocessing data..."))
df.list <- lapply(df.list, function(df) {
  df %>% 
    rename_at(vars(contains("CBCL")), ~replace(., TRUE, "y")) %>%
    select(-GENDER_FEMALE, -brain_age_yrs, -WAVE, -subjWeight, -EstimatedTotalIntraCranialVol) %>%
    # Now delete all the corpus callosum variables!
    select(-starts_with('CC_')) %>%
    column_to_rownames("IDENT_SUBID")
})
# ----
# Set up the param grid, and also attach its corresponding data frame
print(paste0(Sys.time(), "- Generating param grid..."))
params_and_data.list = lapply(df.list, function(df) {
  list(
    grid = expand.grid(
      .mtry = 2:(ncol(df) - 3), # minus the response and GROUP; the maximum is p - 1
      .ntree = seq(500, 1550, by = 75),
      .nodesize = seq(1, 20, by = 1)),
    dataframe = df
  )
})
# ----
# Set up a function to iteratively fit with each combination of data and parameters
fit_rForest <- function(.data, ...)
{
  randomForest(
    y ~ .-GROUP,
    data = .data,
    strata = .data$GROUP,
    sampsize = sum(.data$GROUP=="PI") * n_distinct(.data$GROUP),
    # save computational cost by limiting the computation of certain outputs
    importance = FALSE, proximity = FALSE, keep.forest = FALSE, oob.prox = FALSE,
    # allow user to supply further arguments (these will be hyper parameters)
    ...)
}
# ----
# Grid search.
# On each element of the params_and_data.list, search the param grid for the best
# hyperparameters of the random forest model for the corresponding data.
print(paste0(Sys.time(), "- Conducting grid search..."))
gridSearchResults <- mclapply(params_and_data.list, function(list) {
  # For each param grid and data frame list, do:
  mclapply(1:nrow(list$grid), function(i) {
    fit_rForest(.data = list$dataframe, 
                mtry = list$grid$.mtry[i],
                ntree = list$grid$.ntree[i],
                nodesize = list$grid$.nodesize[i])
  }, mc.cores = 5)
}, mc.cores = 3)
# ----
# save the result
saveRDS(gridSearchResults, "../output/noCC_finalCBCL_ModelsGridCV_2020-05-13.rds")
print(paste0("Elapsed time: ", Sys.time() - t0))
  
  
  
