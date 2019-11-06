# SB
SB data analysis scripts and output.

TO-DO now: 
- decide one outcome for all different models.
- DISCUSS: random forest with structural PCs
- DISCUSS: cross validation on hold-out set
- DISCUSS: discuss with mvt and pab potentially redundant structural variables
-To try later: clustering the 69 regions first into features, and then using those to predit CBCL and group classification.
-- ROC curve (pROC package)
-- compare the
-- create a confint for Accuracy from each model.
- potential (very last thing): predict PACCT outcomes using behaviral measures, and compare to the validation of the models built with SB.
Notes about the data files:
* Data used in PCA and random forest are the wave 1 data, unless a participant had no wave 1 data, but wave 3 was always omitted (different scanner); the youngest age collected was used if possible.

