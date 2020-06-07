#' Synthetic Stochastic Volatility Data
#'
#'@param nobs The number of observations to generate.
#'@param param A vector of model parameters used to generate the data.
#'
#'@return A data frame containing (x,y) pairs of observations. The states are 
#'given by x and the observations are given by y.
#'
#'@export
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

#' Auxiliary Particle Filter
#' 
#' This function is currently limited to the stochastic volatility model.
#' 
#' @param obs A vector of observations to be filtered.
#' @param num_particles A positive integer number of particles to be used
#' in the simulation.
#' @param param A vector of model parameters (alpha, beta, sigma).
#' 
#' @return A list containing a sample from the empirical distribution; the
#' approximated marginal log-likelihood of the data; the filtered states.
APF <- function(obs, N, param) {
  results <- APF_Cpp(obs, N, param)
  weights <- results$weights
  particles <- results$particles
  states <- rep(0, nrow(particles))
  for (i in 1:ncol(particles)) {
    states <- states + weights[i] * particles[,i]
  }
  return(list("states" = states,
              "sample" = results["sample"],
              "log_marginal" = results["log_marginal"]))
}