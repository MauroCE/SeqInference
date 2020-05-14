library(ggplot2)

## MODEL SPECIFIC FUNCTIONS:
# Model - Binary data - Thought this would be interesting?

logistic <- function(x, phi, alpha) {
  1 / (1 + exp(- alpha - t(phi) %*% x))
}

qsim <- function(x, theta) {
  if (length(x) == 1) Sigma = matrix(theta, nrow = 1, ncol = 1)
  else Sigma = diag(theta)
  MASS::mvrnorm(1, x, Sigma)
}

fsim <- function(x, phi, alpha) {
  p <- logistic(x, phi, alpha)
  rbinom(1, 1, p)
}

## GENERIC FUNCTIONS FOR STATE SPACE MODEL:

# Function looks at the current latent state (at time t) and produces a 
# new obervation at time t + 1.
step <- function(x, theta, phi, alpha) {
  x1 <- qsim(x, theta)
  y1 <- fsim(x1, phi, alpha)
  return(c(x1, y1))
}

state_model_simulation <- function(x0, qsim, fsim, theta=rep(0.01, length(x0)), 
                                   phi=rep(1, length(x0)), alpha=0, 
                                   Tmax=1000) {
  
  obs0 <- step(x0, theta, phi, alpha)
  trajectory <- data.frame(t(obs0))
  x <- obs0[-length(obs0)]
  for (t in 2:Tmax) {
    obs <- step(x, theta, phi, alpha)
    trajectory <- rbind(trajectory, obs)
    x <- obs[-length(obs)] # Update x
  }
  names <- c(paste("x", 1:length(x0), sep=""), "y")
  colnames(trajectory) <- names
  return(trajectory)
}

pl_sms_1D <- function(trajectory) {
  pl <- ggplot(data = trajectory) + 
        geom_point(mapping = aes(x=1:nrow(trajectory), y=x1, 
                                 col=as.character(y))) +
        labs(color="y")
  pl
}

# Note that qsim and fsim can be changed to whatever we want.
# data does not have to be binary.
sim1 <- state_model_simulation(0, qsim, fsim, Tmax = 1000)
print(head(sim1))
pl_sms_1D(sim1)

# Looking at the plots I don't know if this model is a good idea. Random walk
# tends to drit off and produce a lot of 1's or a lot of 0's.