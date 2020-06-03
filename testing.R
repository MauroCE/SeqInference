sim_data <- function(nobs, param) {
  states <- rep(0, nobs)
  obs <- rep(0, nobs)
  x0 <- rnorm(1, 0, param[3])
  states[1] <- param[1] * x0 + rnorm(1, sd = param[3])
  obs[1] <- rnorm(1, sd = param[2] * exp(states[1] / 2))
  for (i in 2:nobs) {
    states[i] = param[1] * states[i - 1] + rnorm(1, sd = param[3])
    obs[i] = rnorm(1, sd = param[2] * exp(states[i] / 2))
  }
  return(data.frame(states = states, obs = obs))
}

dat <- sim_data(500, c(0.91, 0.5, 1))
plot(dat$states, type = "l")

filter <- APF(dat$obs, 1000, c(0.91, 0.5, 1))
lines(filter, col = "blue")
