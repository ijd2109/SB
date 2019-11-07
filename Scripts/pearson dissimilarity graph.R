# Pearson dissimilarity plot
d = expand.grid("rho"=seq(-.9,.9,by=.1), "beta" = c(-4:-1,1:4))
twoParamDiss.fn = function(rho, beta) {
  sqrt( ( (1 - rho)/(1 + rho))^beta)
}

# 2-parameter dissimilarity (rho, and beta)
for (i in unique(d$beta)) {
  rho = d[d$beta == i, "rho"]
  beta = d[d$beta == i, "beta"]
  dissimilarity = twoParamDiss.fn(rho, beta)
  plot(rho, dissimilarity, 
       # assign the colors incrementally from 1 to 8
       col = c(1:8)[which(unique(d$beta)==i)],
       # line graph, 
       type = "l", lwd = 2,
       xlim = c(-.9,.9), ylim=c(0,8))
  if (!tail(unique(d$beta), 1) == i) {
    par(new= TRUE)
  }
}

# beta parameter
# -4 -3 -2 -1  1  2  3  4
# Corresponding color: 
# 1   2  3  4  5  6  7  8  
# black, red, green, blue, cyan, magenta, yellow, grey