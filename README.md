# RCADS predictive modeling
### Highlights
#### Random Forest distributions
![](./rcadsMountainPlot.png)
Distributions of RCADS total *t*-score predictions generated from individual trees built on bootstrap resamples of the data. The prediction from the random forest is outlined in white and masked in green, while the dashed black line represents the true distribution of RCADS scores.

#### Variable importances
`source(Scripts/misc/vimPlot)`
![](./FC_vimplot.jpeg)
These are the most explanatory variables in predicting RCADS from functional connections between (Harvard-Oxfordr) parcels in the brain. Overall, 20% of the variance is explained by the random forest model. *Generate this plot using `vimPlot.R`*