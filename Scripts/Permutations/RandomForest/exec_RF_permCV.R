setwd("/home/idouglas/ianMasters/")
  # Cross validation of random forest models, and permutation testing for model accuracy metrics
  ## Load packages
if(isFALSE(require(tidyverse))){install.packages("tidyverse")}
if(isFALSE(require(randomForest))){install.packages("randomForest")}
if(isFALSE(require(caret))){install.packages("caret")}
if(isFALSE(require(doParallel))){install.packages("doParallelerse")}
if(isFALSE(require(pROC))){install.packages("pROC")}
if(isFALSE(require(plotROC))){install.packages("plotROC")}
if(isFALSE(require(MLmetrics))){install.packages("MLmetrics")}
library(tidyverse)
library(randomForest)
library(caret)
library(doParallel)
library(pROC)
library(plotROC)



## Read in the data frames

# functional conn. and dissim. data
fdisBeta1 = readRDS('/home/idouglas/ianMasters/data/labelledFDissBeta1.rds')
fcon = readRDS('/home/idouglas/ianMasters/data/labelledFCor.rds')
# read in PCA data
pc.fdBeta1 = readRDS('/home/idouglas/ianMasters/data/fdBeta1PCAScoresLbl.rds')
pc.fc = readRDS('/home/idouglas/ianMasters/data/fcPCAScoresLbl.rds')
# note, each PCA required 35 dimensions to reach 80% variance explained.
# Filter by that so that k is smaller than n (necessary because p was not smaller than n).
topPC.fdBeta1 = pc.fdBeta1[ , -c(grep("^PC36$", names(pc.fdBeta1)):ncol(pc.fdBeta1))]
topPC.fc = pc.fc[ , -c(grep("^PC36$", names(pc.fc)):ncol(pc.fc))]
# structural data:
StrData = readRDS("/home/idouglas/ianMasters/data/structuralLabelled.rds")
StrData_noWBV = select(StrData,-EstimatedTotalIntraCranialVol)
# Now PCA
StrPCA = readRDS("/home/idouglas/ianMasters/data/strPCAscoresLabelled.rds")
StrPCA_noWBV = readRDS("/home/idouglas/ianMasters/data/strPCAscoresLabelled_noWBV.rds")
# Put all the data into a named list:
data_list = list(
  "FD" = fdisBeta1,
  "FC" = fcon,
  "FDPCA" = topPC.fdBeta1, # note, using the numberof PCs required to explain 80% variance
  "FCPCA" = topPC.fc, # note, using the numberof PCs required to explain 80% variance
  "Str" = StrData,
  "Str_noWBV" = StrData_noWBV,
  "StrPCA" = StrPCA,
  "StrPCA_noWBV" = StrPCA_noWBV
)
# and remove the data from the environment:
rm(list=c("fdisBeta1", "fcon","topPC.fdBeta1","topPC.fc", "pc.fdBeta1","pc.fc",
          "StrData","StrData_noWBV", "StrPCA","StrPCA_noWBV"))
# quick pre-process: make the IDENT_SUBID column the rownmaes, and then delete it
data_list = lapply(data_list, function(x) {
  rownames(x) = x$IDENT_SUBID
  x = select(x, -IDENT_SUBID)
  return(x)
})


data_list = vector(mode = "list", length = 8)
names(data_list) = c("FD","FC","FDPCA","FCPCA","Str",
                     "Str_noWBV","StrPCA","StrPCA_noWBV")


## Define the list of parameters for each model, based on results of 'rfGridSearchCV.rmd' for each dataset:

param_list = list(
  "FD" = c("mtry"=2350,"ntree"=1501,"nodesize"= 7),
  "FC" = c("mtry"=2350,"ntree"=1501,"nodesize"= 7),
  "FDPCA" = c("mtry"=30,"ntree"=801,"nodesize"= 7),
  "FCPCA" = c("mtry"=30,"ntree"=801,"nodesize"= 7),
  "Str" = c("mtry"=3,"ntree"=851,"nodesize"= 5),
  "Str_noWBV" = c("mtry"=5,"ntree"=851,"nodesize"= 5),
  "StrPCA" = c("mtry"=19,"ntree"=2001,"nodesize"= 12),
  "StrPCA_noWBV" = c("mtry"=17,"ntree"=851,"nodesize"= 5)
)
# as.data.frame(param_list) # do not print


# Fit the "baseline" models to generate OOB scores and variable importances.
# This differs from the later models in that it conducts bagging, 
# but not cross validation.

set.seed(111) # for reproducibility.
bestForest.list = mclapply(X = names(data_list), FUN = function(x) {
  dat = data_list[[x]]; params = param_list[[x]]
  # pre-proc
  dat = na.omit(
    select(dat, -one_of("age","IDENT_SUBID","SUBJECTID_long",
                        "wave_to_pull", "cbcl_totprob_t")))
  dat$GROUP = factor(dat$GROUP)
  model = randomForest(GROUP ~., data = dat,
                       mtry = params[1],
                       ntree = params[2],
                       nodesize = params[3],
                       strata = dat$GROUP,
                       sampsize = rep(sum(dat$GROUP=="PI"), times = 2),
                       importance = TRUE)
}, mc.cores = (detectCores() - 1))
names(bestForest.list) = names(data_list)
# compute the OOB_AUCs for all above models
OOB_AUCs = sapply(rf.list, function(x) {
  AUC(ifelse(rf.list$FD$predicted=="PI",1,0), 
      ifelse(rf.list$FD$y=="PI",1,0))
})


# Run the cross-validation algorithm/permutation testing algorithm on all datasets

permutationResults = mclapply(X = names(data_list), mc.cores = detectCores() - 1,
  FUN = function(x) {
    dat = data_list[[x]]; params = param_list[[x]]
    # pre-proc
    dat = na.omit(
      select(dat, -one_of("age","IDENT_SUBID","SUBJECTID_long",
                          "wave_to_pull", "cbcl_totprob_t")))
    #train test split; stratified by group (and undersample majority class)
    dat$GROUP = factor(dat$GROUP)
    index_pi = which(dat$GROUP == "PI"); index_comp = which(dat$GROUP !="PI")
    train_i = replicate(n = 1000, simplify = FALSE, 
                        expr=c(sample(index_pi,size=round(length(index_pi)*.75)),
                               sample(index_comp,size=round(length(index_pi)*.75))))
    crossVal.res = mclapply(X = train_i, mc.cores = detectCores() - 1,
      FUN = function(y, dat = dat, params = params) {
        training = dat[y, ]; test = dat[-y, ]
        fit = randomForest(GROUP ~ ., data = training,
                           mtry = params[1],
                           ntree = params[2],
                           nodesize = params[3],
                           strata = training$GROUP,
                           sampsize = c(round(length(index_pi)*.75),
                                        round(length(index_pi)*.75))
                           )
        # get the predicted, actual, AUC, and null dist of 1000 permuted AUC
        pred = factor(predict(fit, newdata = test), levels = c("COMP", "PI"))
        actual = factor(test$GROUP, levels = c("COMP","PI"))
        #note: collecting the probabilities is only necessary for plotting ROC curves.
        # prob1 = predict(fit, newdata = test, type="prob")[,"PI"]
        # predActual.dat = as.data.frame(
        #   list("IDENT_SUBID"=rownames(test), "pred"=pred, "actual"=actual, "proba.pi"=prob1))
        crossVal_AUC = AUC(y_pred = ifelse(pred == "PI", 1, 0), 
                           y_true = ifelse(actual =="PI", 1, 0))
        nullDistr = data.frame(
          "null_AUC" = replicate(simplify = TRUE, n = 1000,
                                 expr = AUC(y_pred = ifelse(pred == "PI", 1, 0),
                                            y_true = sample(ifelse(actual =="PI", 1, 0)))
        ))
        results = list(#"predActual.dat" = predActual.dat, # don't need the probailities
                       "CV_AUC"= crossVal_AUC,
                       "nullDistr_AUC.dat" = nullDistr)
        return(results)
      })
    return(crossVal.res)
  }
)

## Compile results:

# extract the AUCs
AUCs = NULL
for (i in 1:length(permutationResults)) {
  tmp.AUC = vector(length = 1000, mode = "double")
  for (j in 1:1000) {
    tmp.AUC[j] = permutationResults[[i]][[j]]$CV_AUC
  }
  AUCs[[i]] = data.frame(estimate = mean(tmp.AUC), SD = sd(tmp.AUC))
  rm(list="tmp.AUC")
}
names(AUCs) = names(data_list)

# extract the null distributions
nullDistr = lapply(permutationResults, function(x) {
  tmp = lapply(x, function(y) {
    sort(y[["nullDistr_AUC.dat"]]$null_AUC) # extract the null predictions
  })
  return(rowMeans(Reduce("cbind", tmp)))
})
names(nullDistr) = names(data_list)

#create the average null distribution from all models and associated p vals
masterNull = rowMeans(Reduce("cbind",nullDistr))

# calculate permutation p-values
# p-value: (100% - percent of permuted values closer to chance than the observed)/100
perm.pval = lapply(names(data_list), function(x) {
  n = length(nullDistr[[x]])
  # comparing the mean of all 1000 test-set accuracies to the mean (sorted) null distribution
  (1 + sum(nullDistr[[x]] > mean(AUCs[[x]]$estimate)))/(1 + n)
})
names(perm.pval) = names(data_list)

#compared to common null
pval = lapply(names(data_list), function(x) {
  n = length(masterNull)
  # comparing the mean of all 1000 test-set accuracies to the mean (sorted) null distribution
  (1 + sum(masterNull > mean(AUCs[[x]]$estimate)))/(1 + n)
})
names(pval) = names(data_list)


## Results:
res = as.data.frame(list("OOB_AUC" = round(OOB_AUCs,4),
                         "CV_AUC" = round(sapply(AUCs, function(x) x$estimate),4),
                         "CV_AUC_sd" = round(sapply(AUCs, function(x) x$SD),4),
                         "Null_AUC" = round(sapply(nullDistr, mean),4),
                         "Null_sd" = round(sapply(nullDistr, sd),4),
                         "p" = round(unlist(perm.pval),3),
                         "global.p" = round(unlist(pval),3)), 
                    row.names = names(data_list))
saveRDS(res, "/home/idouglas/ianMasters/output/resTable.rds")



# Visualization (hashed out for now)

# Visualize the distribution of the accuracies
## Compile results into dataframes for plotting

## Add some labels and convert from wide to long format to plot distributions
# dataNames = names(data_list)
# dataType = c(rep(c("connectivity", "dissimilarity"),2), rep("structural",4))
# rawAcc = lapply(permutationResults, function(x) {
#   sapply(x, function(y) {
#     y[[2]]
#   })
# })
# perm_plt_data = Reduce("rbind", lapply(1:8, function(x) {
#   n = length(rawAcc[[x]])
#   data.frame("model" = rep(dataNames[x], times= (n+1000)),
#              "dataType" = rep(dataType[x], times = (n+1000)),
#              "Distribution" = c(rep("Test.Set.Repetitions",times=n), 
#                                 rep("Permuted.Null", times = 1000)),
#              "Accuracy" = c(rawAcc[[x]], masterNull),
#              stringsAsFactors = FALSE)
# }))
# 
# # seprate functional and structural data for plotting
# fMRI_plt_data = perm_plt_data %>% 
#   filter(dataType != "structural")
# StrMRI_plt_data = perm_plt_data %>% 
#   filter(dataType == "structural")
# 
# 
# {r, echo=FALSE, eval=FALSE}
# saveRDS(perm_plt_data, "../../../output/permutations/FINAL_permResults4plotting.rds")
# saveRDS(fMRI_plt_data, "../../../output/permutations/FINAL_FMRIpermResults4plotting.rds")
# saveRDS(StrMRI_plt_data, "../../../output/permutations/FINAL_STRMRIpermResults4plotting.rds")
# 
# 
# {r, eval=TRUE, echo=FALSE}
# perm_plt_data = readRDS("../../../output/permutations/FINAL_permResults4plotting.rds")
# fMRI_plt_data = readRDS("../../../output/permutations/FINAL_FMRIpermResults4plotting.rds")
# StrMRI_plt_data = readRDS("../../../output/permutations/FINAL_STRMRIpermResults4plotting.rds")
# 
# 
# ## Generate plots
# 
# fMRI_plt = ggplot(fMRI_plt_data, aes(Accuracy, fill = Distribution)) +
#   geom_density(alpha = .3) +
#   geom_vline( # calculate the means
#     data = (
#       data.frame("model"=dataNames, 
#                  "avg" = sapply(rawAcc, mean)) %>%
#         filter(grepl("^F", model))
#     ),
#     aes(xintercept = avg)) +
#   facet_grid(~model) +
#   ggtitle("Functional Data Model Accuracies and Permutation Test Results") +
#   theme(panel.background = element_rect(fill="white"),
#         plot.title = element_text(hjust = .5))
# 
# StrMRI_plt = ggplot(StrMRI_plt_data, aes(Accuracy, fill = Distribution)) +
#   geom_density(alpha = .3) +
#   geom_vline( # calculate the means
#     data = (
#       data.frame("model"=dataNames,
#                  "avg" = sapply(rawAcc, mean)) %>%
#         filter(!grepl("^F", model))
#     ),
#     aes(xintercept = avg)) +
#   facet_grid(~model) +
#   ggtitle("Structural Data Model Accuracies and Permutation Test Results") +
#   theme(panel.background = element_rect(fill="white"),
#         plot.title = element_text(hjust = .5))
# 
# 
# {r, eval=FALSE, echo=FALSE}
# ggsave("../../../results/permutations/plots/fMRI_AccGridCV_FINALdensities.pdf",
#        plot = fMRI_plt,
#        height = 3, width = 8, units = "in", device = "pdf")
# ggsave("../../../results/permutations/plots/StrMRI_AccGridCV_FINALdensities.pdf", 
#        plot = StrMRI_plt,
#        height = 3, width = 8, units = "in", device = "pdf")
# 
# ## Plot each against the global null distribution for all models
# {r, fig.width=10,fig.height=3}
# fMRI_plt
# 
# {r, fig.width=10,fig.height=3}
# StrMRI_plt



