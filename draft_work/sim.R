library(SMC)
n <- 500
dat <- stochastic_volatility(n, c(0.9, 1, 1))
test <- APF(dat$y, 400, c(0.9, 1, 1))$states
plot(test, type = "l", col = "red")
lines(dat$x)
sum((dat$x - test[-1])^2)
