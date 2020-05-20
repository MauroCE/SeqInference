import numpy as np

from math import exp, sqrt
from numpy import array, reshape
from numpy.random import normal, multivariate_normal
from scipy.stats import norm
from pandas import DataFrame

## DATA GENERATION:

def robs(x, b=0.5):
    SD = b * exp(x / 2)
    return normal(loc=0, scale=SD, size=1)

def initial_state(num_particles, a=0.91, SD=1.1):
    #scale = sqrt(SD ** 2 / (1 - a ** 2))
    scale = SD # Fix paramter constraint at a = 1
    return normal(loc=0, scale=scale, size=num_particles)

def transition(x, a=0.91, SD=1):
    num_particles = len(x) # x must be supplied as a 1D array.
    samples = np.empty(num_particles)
    for i in range(num_particles):
        loc = a * x[i]
        samples[i] = normal(loc=loc, scale=SD, size=1)
    return samples

def synthetic_state_space(tmax, initial_state, transition, robs):
    x = np.empty(tmax) ; y = np.empty(tmax)
    x[0] = initial_state(1) ; y[0] = robs(x[0])
    for t in range(tmax - 1):
        x[t + 1] = transition(x[t:(t + 1)]) # Must input a 1D array by
        y[t + 1] = robs(x[t + 1])           # taking a slice.
    return DataFrame({"x": x, "y": y})

## SIMULATION FUNCTIONS:

def state_mean(x, theta):
    return theta[0] * x

def delta_to_h(delta):
    h_sq = 1 - ((3 * delta - 1) / 2 * delta) ** 2
    return sqrt(h_sq)

def theta_mean(theta, theta_sample, delta = 0.98):
    h = delta_to_h(delta)
    l = sqrt(1 - h ** 2)
    vec_sum = np.empty((1, theta_sample.shape[1]))
    for i in range(theta_sample.shape[0]):
        vec_sum = vec_sum + theta_sample[i,:]
    return l * theta + (1 - l) * vec_sum / vec_sum.shape[1]

def initial_theta(num_particles, mean = 0):
    mean_vec = np.repeat(mean, 2)
    cov = np.identity(2)
    return multivariate_normal(mean_vec, cov, size=num_particles)


def theta_transition(num_particles, theta, theta_sample, delta = 0.98):
    mean = theta_mean(theta, theta_sample, delta)
    h = delta_to_h(delta)
    V = np.cov(theta_sample)
    cov = (h ** 2) * V
    sample = multivariate_normal(mean, cov, size=num_particles)
    return sample

def likelihood(y, x, theta):
    SD = theta[1] * exp(x / 2)
    return norm.pdf(y, 0, abs(SD))

def weights(y, state_sample, theta_sample, likelihood, delta = 0.98):
    num_particles = theta_sample.shape[0]
    weights = np.empty(num_particles)
    for i in range(num_particles):
        m = theta_mean(theta_sample[i,:], theta_sample, delta).flatten()
        smean = state_mean(state_sample[i], theta_sample[i,:])
        lik = likelihood(y, state_sample[i], theta_sample[i,:])
        lik_aux = likelihood(y, smean, m)
        if (lik_aux == 0): print("y:", y, " smean:", smean, "m:", m)
        weights[i] = lik / lik_aux
    return weights

def Filter(tmax, y, initial_theta, initial_state, num_particles = 100, 
           delta = 0.98):
    theta_sample = initial_theta(num_particles)
    chains = np.empty((tmax + 1, num_particles))
    for i in range(num_particles):
        chains[0,i] = initial_state(1, theta_sample[i,0])
    w = weights(y[0], chains[0,:], theta_sample, likelihood, delta)
    # main iteration
    #for t in range(tmax):

    return w

#data = synthetic_state_space(100, initial_state, transition, robs)
#print(data.head())
#weights = Filter(5, array(data["y"]), initial_theta, initial_state)
#print(weights)

# PROBLEM: Auxilliary Likelihood function returns 0 and cannot compute weights.

#print(likelihood(-0.4, -0.8, array([-0.6,0.01])))