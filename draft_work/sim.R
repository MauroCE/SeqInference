library(SMC)
n <- 500
dat <- stochastic_volatility(n, c(0.9, 1, 1))
test <- pmmh2(1000, dat$y, 400, c(0.5, 0.5), 0.3)
colMeans(test)
plot(test[,1], type = "l")
plot(test[,2], type = "l")
