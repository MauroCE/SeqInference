synthetic_state_space <- function(tmax, rstate, robs, x0 = 1) {
    x <- rep(0, tmax) ; y <- rep(0, tmax)
    x[1] <- x0 ; y[1] <- robs(x[1])
    for (t in 1:(tmax - 1)) {
        x[t + 1]  <- rstate(x[t])
        y[t + 1] <- robs(x[t + 1])
    }
    return(data.frame("x" = x, "y" = y))
}

# Function accepts rstate() which must generate state samples, and 
# dobs which must compute the observation density for a given state (this
# function must be vectorized with respect to the state variable).
bootstrap_filter <- function(y, x0, rstate, dobs, m = 100) {
    tmax <- length(y)
    chains <- matrix(nrow = tmax, ncol = m)
    chains[1,] <- x0
    for (t in 2:tmax) {
        sampled_x <- rstate(chains[t - 1,])
        weights <- dobs(y[t], sampled_x)
        weights <- weights / sum(weights)
        chains[t,] <- sample(sampled_x, m,replace = TRUE, prob = weights)
    }
    return(chains)
}

# Model Proposal: Stochasitc volatility model.

rstate <- function(x, a = 0.91, sd = 1) {
    m <- length(x)
    samples <- rep(0, m)
    for(i in 1:m) {
        samples[i] <- rnorm(1, a * x[i], sd)
    }
    return(samples)
}

dobs <- function(y, x, b = 0.5) {
    m <- length(x)
    densities <- rep(0, m)
    for (i in 1:m) {
        densities[i] <- dnorm(y, 0, b * exp(x[i] / 2))
    }
    return(densities)
}

robs <- function(x, b = 0.5) {
    rnorm(1, 0, b * exp(x / 2))
}

main <- function(tmax = 5000, m = 1000, a = 0.91, sd = 1) {
    x0 <- rnorm(1, 0, sqrt(sd^2 / (1 - a^2)))
    data <- synthetic_state_space(tmax, rstate, robs, x0)
    plot(1:tmax, data$y, col = "red")
    lines(1:tmax, data$x, type = "l")
    # This assumes we start from the correct state
    chains <- bootstrap_filter(data$y, x0, rstate, dobs, m)
    means <- rowMeans((chains))
    plot(1:tmax, data$x, type = "l", col = "red") # Hopefully diagonal straight line.
    lines(1:tmax, means, type = "l")
}

main()


