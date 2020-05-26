import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def transition2(x_t, theta_t, sample_size=(1, )):
    # Return a sample from a multivariate normal distribution.
    # x_t: (N samples, 1) contains different samples of x_t.
    # theta_t: Contains
    if len(np.shape(theta_t)) == 1:
        return np.random.normal(loc=theta_t[0]*x_t, scale=theta_t[1], size=sample_size)
    else:
        return np.random.normal(loc=theta_t[:, 0]*x_t, scale=theta_t[:, 1], size=sample_size)


def mu(x_samples, samples_per_samples):
    # samples_per_samples: number of samples that we take for each x_t^{(j)}
    # Can make use of Numpy broadcasting features by simply passing in both x and theta samples
    size_of_x = np.size(x_samples, 0)
    samples = transition(x=x_samples, n=(size_of_x, samples_per_samples))
    return np.mean(samples, axis=1).reshape(-1, 1)


def sample_auxiliary(g_vec):
    n = np.size(g_vec, 0)
    return np.random.choice(np.arange(n), size=n, replace=True, p=np.ravel(g_vec))

def resample(x, m, indeces):
    return x[indeces], m[indeces]

def compute_v_and_theta_bar(theta_samples, wvec):
    theta_bar = np.sum(wvec*theta_samples) / np.sum(wvec)
    V_t = np.sum(wvec*(theta_samples - theta_bar)**2) / np.sum(wvec)
    return theta_bar, V_t


def get_a(delta = 0.95):
    a = (3 * delta - 1) / (2 * delta)
    return a


def m_t(theta_samples, a):
    m_t = a * theta_samples + (1 - a) * np.mean(theta_samples, axis = 0)
    return m_t


def likelihood(y_t, x_t, beta):
    return stats.norm.pdf(y_t, loc=0.0, scale=beta*np.exp(x_t/2))


def g_current(y_current, mu_current, m_previous, weights_previous):
    g = weights_previous * likelihood(y_t=y_current, x_t=mu_current, beta=m_previous)
    return g / np.sum(weights_previous)


def initial_sampler(n, alpha=0.91, sigma=1.0):
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
    return np.random.normal(loc=0.0, scale=sigma/np.sqrt(1 - alpha**2), size=n)


def emission(x_t, beta=0.5):
    """
    Emission probability used to generate the data.

    :param x_t: Current hidden state used to sample the corresponding y_t.
    :type x_t: np.array
    :param b: Scalar used to scale the variance-covariance matrix.
    :type b: float
    :return: Emission sample y_t ~ p(y_t | x_t).
    :rtype: np.array
    """
    return np.random.normal(loc=0.0, scale=np.exp(x_t)*(beta**2))


def transition(x, alpha=0.91, sigma=1.0, n=(1, )):
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
    return np.random.normal(loc=alpha*x, scale=np.array([sigma]), size=n)


# GENERATE SYNTHETIC DATA FOR PLOTTING
def generate_data(t_max=100):
    x = np.zeros((t_max + 1, 1))  # Container for x hidden values
    y = np.zeros((t_max + 1, 1))  # Container for y observed values
    x[0, :] = initial_sampler(n=1)  # Sample x_0 from the initial
    y[0, :] = np.nan  # Set y_0 to be NA since observed process starts at y_1
    # Alternate between transition and emission sampling
    for t in range(1, t_max + 1):
        x[t, :] = transition(x[t - 1, :]) # Sample x_t from p(x_t | x_{t-1})
        y[t, :] = emission(x[t, :])  # Sample y_t given x_t from p(y_t | x_t)
    return x, y


x, y = generate_data()


