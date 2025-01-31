import numpy as np
from scipy.special import gamma, beta, factorial, gammainc, gammaincc, erf
from scipy.stats import norm, lognorm, uniform, expon, poisson, beta as beta_dist, gamma as gamma_dist, bernoulli

# Collection of Cumulative Distribution Functions (CDFs)

# Normal CDF
def normal_cdf(x, mu, sigma):
    return 0.5 * (1 + erf((x - mu) / (sigma * np.sqrt(2))))

# Lognormal CDF
def lognormal_cdf(x, mu, sigma):
    assert x > 0, "x must be greater than 0"
    return 0.5 * (1 + erf((np.log(x) - mu) / (sigma * np.sqrt(2))))

# Gaussian CDF (same as Normal)
def gaussian_cdf(x, mu, sigma):
    return normal_cdf(x, mu, sigma)

# Uniform CDF
def uniform_cdf(x, a, b):
    assert a < b, "a must be less than b"
    if x < a:
        return 0
    elif x > b:
        return 1
    else:
        return (x - a) / (b - a)

# Exponential CDF
def exponential_cdf(x, lambd):
    assert x >= 0, "x must be greater than or equal to 0"
    assert lambd > 0, "lambda must be greater than 0"
    return 1 - np.exp(-lambd * x)

# Poisson CDF
def poisson_cdf(x, lambd):
    assert x >= 0 and isinstance(x, int), "x must be a non-negative integer"
    assert lambd > 0, "lambda must be greater than 0"
    return poisson.cdf(x, lambd)

# Beta CDF
def beta_cdf(x, alpha, beta_val):
    assert 0 <= x <= 1, "x must be between 0 and 1"
    assert alpha > 0, "alpha must be greater than 0"
    assert beta_val > 0, "beta must be greater than 0"
    return beta_dist.cdf(x, alpha, beta_val)

# Gamma CDF
def gamma_cdf(x, alpha, beta_val):
    assert x > 0, "x must be greater than 0"
    assert alpha > 0, "alpha must be greater than 0"
    assert beta_val > 0, "beta must be greater than 0"
    return gammainc(alpha, beta_val * x)

# Bernoulli CDF
def bernoulli_cdf(x, p):
    assert 0 <= p <= 1, "p must be between 0 and 1"
    if x < 0:
        return 0
    elif x < 1:
        return 1 - p
    else:
        return 1

# Binomial CDF
def binomial_cdf(x, n, p):
    assert 0 <= x <= n and isinstance(x, int), "x must be an integer between 0 and n"
    assert 0 <= p <= 1, "p must be between 0 and 1"
    return binom.cdf(x, n, p)

# Chi-Square CDF
def chi_square_cdf(x, k):
    assert x >= 0, "x must be greater than or equal to 0"
    assert k > 0, "degrees of freedom must be positive"
    return chi2.cdf(x, k)

# Cauchy (Lorentzian) CDF
def cauchy_cdf(x, x0, gamma_val):
    return 0.5 + (1/np.pi) * np.arctan((x - x0) / gamma_val)
