#################################
# Rank the values in a vector in
# descending order, such that the largest
# value gets the rank of 1, and the smallest
# value gets the "highest" rank.
# Ian Douglas - March 2020

descending_rank = function(data) { # maximum is ranked #1 (minimum rank) and so forth
  require(data.table)
  reverseScaled = 1 - min_max(data)
  data.table::frank(reverseScaled)
}