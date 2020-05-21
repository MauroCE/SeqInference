library(methods)

## OOP CLASS/GENERIC/METHOD DEFINITIONS:

setClass("BootstrapFilter",
  slots = c(
    obs = "vector",
    chains = "matrix",
    num_particles = "numeric",
    initial = "function",
    transition = "function",
    likelihood = "function"
  )
)

BootstrapFilter <- function(inital, transition, likelihood, num_particles = 100) {
  new("BootstrapFilter",
    obs = vector(),
    chains = matrix(initial(num_particles), ncol = num_particles),
    num_particles = num_particles,
    initial = initial,
    transition = transition,
    likelihood = likelihood
  )
}

setGeneric("chains", function(model) standardGeneric("chains"))
setMethod(
  "chains", "BootstrapFilter",
  function(model) {
    return(model@chains)
  }
)

setGeneric("update<-", function(model, t, value) standardGeneric("update<-"))
setMethod(
  "update<-", "BootstrapFilter",
  function(model, t, value) {
    model@obs <- c(model@obs, value) # value is essentially y_t
    # IMPORTANCE SAMPLING STEP
    state_samples <- transition(model@chains[t - 1, ])
    model@chains <- rbind(model@chains, state_samples)
    weights <- likelihood(model@obs[t - 1], state_samples) # Index t - 1 to account
    weights <- weights / sum(weights) # for inital sample at time t = 0
    # RESAMPLING
    resample <- sample(1:length(weights), replace = TRUE, prob = weights)
    model@chains <- model@chains[, resample]
    return(model)
  }
)

## DATA GENERATING FUNCTION:

synthetic_state_space <- function(tmax, initial, transition, emission) {
  x <- rep(0, tmax)
  y <- rep(0, tmax)
  x[1] <- initial(1)
  y[1] <- emission(x[1])
  for (t in 1:(tmax - 1)) {
    x[t + 1] <- transition(x[t])
    y[t + 1] <- emission(x[t + 1])
  }
  return(data.frame("x" = x, "y" = y))
}

emission <- function(x, b = 0.5) {
  rnorm(1, 0, b * exp(x / 2))
}

initial <- function(num_particles, a = 0.91, sd = 1) {
  rnorm(num_particles, 0, sqrt(sd^2 / (1 - a^2)))
}

transition <- function(x, a = 0.91, sd = 1) rnorm(length(x), a * x, sd)

likelihood <- function(y, x, b = 0.5) dnorm(y, 0, b * exp(x / 2))

## TEST IF MODEL IS PERFORMING AS EXPECTED:

main <- function(tmax = 100) {
  data <- synthetic_state_space(tmax, initial, transition, emission)
  model <- BootstrapFilter(initial, transition, likelihood)
  for (t in 2:tmax) {
    update(model, t) <- data$y[t]
  }
  chains <- chains(model)
  means <- rowMeans(chains)
  plot(1:tmax, data$x, type = "l")
  lines(1:tmax, means, type = "l", col = "red")
}

main()
