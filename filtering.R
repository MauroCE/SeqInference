library(methods)

## OOP CLASS/GENERIC/METHOD DEFINITIONS:

setClass("BootstrapFilter",
  slots = c(
    obs = "vector",
    chains = "matrix",
    num_particles = "numeric",
    initial = "function",
    prior_sample = "function",
    likelihood_density = "function"
  )         
)

BootstrapFilter <- function(inital, prior_sample, likelihood_density, num_particles = 100) {
  new("BootstrapFilter",
    obs = vector(),
    chains = matrix(initial(num_particles), ncol = num_particles),
    num_particles = num_particles,
    initial = initial,
    prior_sample = prior_sample,
    likelihood_density = likelihood_density
  )
}

setGeneric("chains", function(x) standardGeneric("chains"))
setMethod("chains", "BootstrapFilter",
  function(x) return(x@chains)
)

setGeneric("update<-", function(x, t, value) standardGeneric("update<-"))
setMethod("update<-", "BootstrapFilter", 
  function(x, t, value) {
    x@obs <- c(x@obs, value)
    state_samples <- prior_sample(x@chains[t - 1,])
    x@chains <- rbind(x@chains, state_samples)
    weights <- likelihood_density(x@obs[t - 1], state_samples) # Index t - 1 to account 
    weights <- weights / sum(weights)                  # for inital sample at time t = 0
    resample <- sample(1:length(weights), replace = TRUE, prob = weights)
    x@chains <- x@chains[,resample]
    return(x)
  }        
)

## DATA GENERATING FUNCTION:

synthetic_state_space <- function(tmax, initial, prior_sample, robs) {
  x <- rep(0, tmax) ; y <- rep(0, tmax)
  x[1] <- initial(1) ; y[1] <- (x[1])
  for (t in 1:(tmax - 1)) {
    x[t + 1]  <- prior_sample(x[t])
    y[t + 1] <- robs(x[t + 1])
  }
  return(data.frame("x" = x, "y" = y))
}

robs <- function(x, b = 0.5) {
  rnorm(1, 0, b * exp(x / 2))
}

initial <- function(num_particles, a = 0.91, sd = 1) {
  rnorm(num_particles, 0, sqrt(sd^2 / (1 - a^2)))
}

prior_sample <- function(x, a = 0.91, sd = 1) {
  num_particles <- length(x)
  samples <- rep(0, num_particles)
  for(i in 1:num_particles) {
    samples[i] <- rnorm(1, a * x[i], sd)
  }
  return(samples)
}

likelihood_density <- function(y, x, b = 0.5) {
  num_particles <- length(x)
  densities <- rep(0, num_particles)
  for (i in 1:num_particles) {
    densities[i] <- dnorm(y, 0, b * exp(x[i] / 2))
  }
  return(densities)
}

## TEST IF MODEL IS PERFORMING AS EXPECTED:

main <- function(tmax = 100) {
  data <- synthetic_state_space(tmax, initial, prior_sample, robs)
  model <- BootstrapFilter(initial, prior_sample, likelihood_density)
  for (t in 2:tmax) {
    update(model, t) <- data$y[t]
  }
  chains <- chains(model)
  means <- rowMeans(chains)
  plot(1:tmax, data$x, type = "l")
  lines(1:tmax, means, type = "l", col = "red")
}

main()
