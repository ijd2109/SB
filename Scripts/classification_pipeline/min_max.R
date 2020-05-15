################################################
# Min-max scale a vector, and allow for a small
# value (smaller than the second smallest value)
# to be added to the minimum of zero if desired
# by the user (for the purposes of visualization)
# Ian Douglas - March 2020
#

min_max = function(data, repel.zero = FALSE) {
  out <- (data - min(data)) / (max(data) - min(data))
  if (repel.zero) {
    second.lowest <- min(replace(out, out == 0, NA), na.rm = T)
    out <- replace(out, out == 0, seq(0, second.lowest, length.out = 3)[2])
  }
  return(out)
}