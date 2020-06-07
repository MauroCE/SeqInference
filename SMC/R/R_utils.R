#' Synthetic Stochastic Volatility Data
#'
#'@param nobs The number of observations to generate.
#'@param param A vector of model parameters used to generate the data.
#'
#'@return A data frame containing (x,y) pairs of observations. The states are 
#'given by x and the observations are given by y.
stochastic_volatility <- function(nobs, param) {
  x <- rep(0, nobs)
  y <- rep(0, nobs)
  x[1] <- stats::rnorm(1, sd = sqrt(param[3]^2 / (1 - param[1]^2)))
  y[1] <- stats::rnorm(1, sd = param[2] * exp(x[1] / 2))
  for (i in 2:nobs) {
    x[i] <- param[1] * x[i - 1] + stats::rnorm(1, sd = param[3])
    y[i] <- stats::rnorm(1, sd = param[2] * exp(x[i] / 2))  
  }
  return(data.frame(x=x, y=y))
}