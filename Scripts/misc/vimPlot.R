####################################################
# Function for producing a variable importance plot
# from a randomForest object produced via
# randomForest::randomForest()
# Ian Douglas - March 2020
# 0.0.1 (first version)
# 

vimPlot <- function(model, num.vars = 30, fillcolor = 'lightgreen', return.imps = FALSE, ...) {
  # `...` arguments are passed to ggplot() + theme(...)
  
  # Use randomForest::importance to extract the variable importances
  impFrame = as.data.frame(randomForest::importance(model))
  
  # Based upon the model type, extract different types of importance values
  if (model$type == "classification") {
    if (is.null(model$importanceSD)) {
      nm <- "MeanDecreaseGini"; y.lab <- "Gini_Importance"
      imp.SD <- NULL
    } else {
      nm <- "MeanDecreaseAccuracy"; y.lab <- "Accuracy_Importance"
      imp.SD <- as.numeric(model$importanceSD[, nm])
    }
  } else {
    if (is.null(model$importanceSD)) {
      nm <- "IncNodePurity"; y.lab <- "NodePurity_Importance"
      imp.SD <- NULL
    } else {
      nm <- "%IncMSE"; y.lab <- "MSE_Importance"
      imp.SD <- as.numeric(model$importanceSD)
    }
  }
  
  # the first column is MSE increase (for reg; analogous for classification)
  impFrame = impFrame[nm] %>%
    mutate(variable = rownames(.), impSD = imp.SD) %>%
    # by default importances are "scaled" by their SD
    # Unscale them to units of % change in MSE (regression), or % Accuracy (classification)
    # or % change in Gini if classification AND importance was not set to TRUE in function call
    rowwise() %>%
    mutate_at(nm, ~ifelse(!is.null(model$importanceSD), . * impSD, .)) %>%
    ungroup() %>%
    arrange_at(nm, ~plyr::desc(.)) %>% # importance is (necessarily) the first column
    # select the desired number of variables
    slice(1:ifelse(num.vars=="all", nrow(.), num.vars))
  
  # set the variable labels for plotting aesthetics
  impFrame = impFrame %>%
    mutate(variable = factor(impFrame$variable, levels = rev(impFrame$variable)),
           importance = unlist(select(., all_of(nm))))
  
  # generate plot:
  plt = ggplot(impFrame) + 
    geom_bar(stat='identity', aes(x = variable, y = importance),fill=fillcolor) +
    coord_flip() +
    ylab(y.lab) +
    theme(plot.title = element_text(hjust = .5)) +
    theme(...)
  
  # Print the plot, adding titles depending on the type of importance (and errorbars if SD exists)
  if (exists("impSD", impFrame)) {
    print(plt +
            ggtitle("Variable Importances within 95% Bootstrap confidence intervals") +
            geom_errorbar(aes(x = variable, y = importance,
                              ymin = importance - 2*impSD, ymax = importance + 2*impSD)))
  } else print(plt + ggtitle("Variable Importances"))
  
  # Print some information / warnings
  warning("Top 30 variables displayed by default. Set num.vars to 'all' to plot all variables, or enter the number of variables you wish to plot explicitly")
  if (is.null(model$importanceSD)) {
    warning("Importances reflect the average (in-bag) node purity index. Set 'importance=TRUE' when running randomForest() to obtain robust out-of-bag permutation importances.")
  }
  if (return.imps) {
    return(pltFrame %>% select(variable, importance, impSD))
  }
}