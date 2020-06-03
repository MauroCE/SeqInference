stochastic_volatility <- function(nobs, param) {
  x <- rep(0, nobs)
  y <- rep(0, nobs)
  x[1] <- rnorm(1, sd = sqrt(param[3]^2 / (1 - param[1]^2)))
  y[1] <- rnorm(1, sd = param[2] * exp(x[1] / 2))
  for (i in 2:nobs) {
    x[i] <- param[1] * x[i - 1] + rnorm(1, sd = param[3])
    y[i] <- rnorm(1, sd = param[2] * exp(x[i] / 2))  
  }
  return(data.frame(x=x, y=y))
}

n <- 1000
dat <- stochastic_volatility(n, c(0.9, 0.5, 1))
test <- APF(dat$y, 800, c(0.5, 0.1, 2))
