import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


class BootstrapFilter:
    """
    Base class implementing a Bootstrap Filter (see Sequential Monte Carlo in Practice - A. Doucet, page 11).
    """
    def __init__(self, p_x0, transition, likelihood, n, y):
        """
        Constructor of Bootstrap Filter. It takes the marginal prior probability p(x_0), the transition probability
        p(x_t | x_{t-1}) and the likelihood density function p(y_t | x_t). Notice that since this SMC algorithm is a
        Bootstrap Filter weights are found by evaluating the likelihood.

        :param p_x0: Marginal prior probability p(x_0) used to sample the initial hidden states.
        :type p_x0: function
        :param transition: Transition probability used by the hidden Markov chain to move forward in time.
        :type transition: function
        :param likelihood: Likelihood function given x_t (since y are conditionally independent).
        :type likelihood: function
        :param n: Number of particles. This is the number of samples used at each step.
        :type n: int
        :param y: Observed data. This is generated further down in the script.
        :type y: np.array
        """
        # Store values into fields and functions into methods
        self._initial_sampler = p_x0
        self._transition = transition
        self._likelihood = likelihood
        self.n = n
        self.y = y
        self.t_max = np.size(self.y, 0)
        self.x = np.zeros((self.t_max, self.n))

    def initial_sampler(self):
        """
        Method used to sample particles at time t=0 of the algorithm.

        :return: Samples from p(x_0).
        :rtype: np.array
        """
        return self._initial_sampler(n=self.n)

    def transition(self, x):
        """
        Method implementing the transition probability p(x_t | x_{t-1}). Set by the constructor __init__().

        :param x: Current value of the hidden Markov chain, denoted x_{t-1}. Will be used to sample x_t.
        :type x: np.array
        :return: A sample x_t from p(x_t | x_{t-1}).
        :rtype: np.array
        """
        return self._transition(x)

    def likelihood(self, y_t, x_t):
        """
        Likelihood function p(y_t | x_t). This is used to compute the weights.

        :param y_t: Observed value y at time t, for which we want to evaluate the likelihood.
        :type y_t: np.array
        :return: New weight p(y_t | x_t).
        :rtype: np.array
        """
        return self._likelihood(y_t=y_t, x_t=x_t)

    def run(self):
        """
        Runs the Bootstrap Filter algorithm.
        :return: Nothing to return.
        :rtype: None
        """
        # Sample from the initial distribution
        self.x[0, :] = np.ravel(self.initial_sampler())
        for t in range(1, self.t_max):
            # Go forward in time in the hidden Markov chain using transition probability
            self.x[t, ] = self.transition(self.x[t-1, ])
            # Compute importance weights using likelihood function
            weights = self.likelihood(y_t=self.y[t, ], x_t=self.x[t, ])
            weights = weights / np.sum(weights)
            # Resample the particles according to the weights
            resampled_indexes = np.random.choice(np.arange(self.n), size=self.n, replace=True, p=weights)
            self.x = self.x[:, resampled_indexes]


def initial_sampler(n, a=0.91, b=1.0, dim=1):
    """
    Function used to sample the hidden state at the beginning of the algorithm, p(x_0). In this case it is a MVN where
    the mean is zero and the variance-covariance matrix is diagonal with variances all equal to b**2 / (1 - a**2).

    :param a: Scalar value used to scale the variance-covariance matrix according to b**2 / (1 - a**2).
    :type a: float
    :param b: Scalar value used to scale the variance-covariance matrix according to b**2 / (1 - a**2).
    :type b: float
    :param n: Number of samples drawn from the MVN.
    :param dim: Number of dimensions. For illustrative purposes will be 1
    :type dim: int
    :return: n samples from the MVN.
    :rtype: np.array
    """
    return np.random.multivariate_normal(mean=np.zeros(dim), cov=(b**2) * np.eye(dim) / (1 - a**2), size=n)


def transition(x, a=0.91, b=1.0):
    """
    Transition probability used to move forward in time by the hidden Markov chain. Parameters a and b are used to scale
    the mean and the variance-covariance matrix of a MVN respectively. It simply samples from a MVN.

    :param x: Current value of the hidden Markov chain x_{t-1}. Will be the (unscaled) mean of the MVN.
    :type x: np.array
    :param a: Scalar value used to scale the mean of the MVN. The mean is given by a*x.
    :type a: float
    :param b: Scalar value used to scale variance-covariance matrix. The variance-covariance is just an identity matrix
              scaled by b**2.
    :type b: float
    :return: Sample from the MVN representing x_t.
    :rtype: np.array
    """
    return np.random.multivariate_normal(mean=a*x, cov=(b**2)*np.eye(np.size(x)))


def likelihood(y_t, x_t, b=0.5):
    """
    Likelihood function p(y_t | x_t) used to compute weights. It returns the value of a multivariate normal distribution
    at y_t, with mean [0, 0] and diagonal variance-covariance matrix scaled by (b * np.exp(x / 2))**2

    :param y_t: Current observed state at which we want to compute the likelihood.
    :type y_t: np.array
    :param x_t: Current sampled hidden state, corresponding to y_t.
    :type x_t: np.array
    :param b: Used to scale the variance-covariance matrix. See documentation above.
    :type b: float
    :return: Value of the multivariate normal density at y_t.
    :rtype: float
    """
    return stats.norm.pdf(x=y_t, loc=np.zeros_like(x_t), scale=b*np.exp(x_t/2)**2)


def emission(x_t, b=0.5):
    """
    Emission probability used to generate the data.

    :param x_t: Current hidden state used to sample the corresponding y_t.
    :type x_t: np.array
    :param b: Scalar used to scale the variance-covariance matrix.
    :type b: float
    :return: Emission sample y_t ~ p(y_t | x_t).
    :rtype: np.array
    """
    return np.random.multivariate_normal(mean=np.zeros_like(x_t), cov=np.diag((b*np.exp(x_t/2))**2))


# GENERATE SYNTHETIC DATA FOR PLOTTING
t_max = 100                       # Maximum time t up to which we generate data
n = 100                           # Number of particles
dim = 1                           # Number of dimensions. For illustrative purposes will be 1
x = np.zeros((t_max+1, dim))      # Container for x hidden values
y = np.zeros((t_max+1, dim))      # Container for y observed values
x[0, :] = initial_sampler(n=1)    # Sample x_0 from the initial distribution
y[0, :] = np.nan                  # Set y_0 to be NA since observed process starts at y_1
# Alternate between transition and emission sampling
for t in range(1, t_max+1):
    x[t, :] = transition(x[t-1, :])   # Sample x_t from the transition distribution p(x_t | x_{t-1})
    y[t, :] = emission(x[t, :])       # Sample y_t given x_t from the emission distribution p(y_t | x_t)

# RUN THE MODEL
model = BootstrapFilter(p_x0=initial_sampler, transition=transition, likelihood=likelihood, n=n, y=y)
model.run()

# TAKE THE MEAN OF THE CHAINS
means = np.mean(model.x, axis=1)
fig, ax = plt.subplots()
ax.plot(np.arange(101), means)
ax.plot(np.arange(101), x)
plt.show()




