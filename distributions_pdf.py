import numpy as np
from scipy.special import gamma, beta, factorial, gammainc, gammaincc, erf
from scipy.stats import norm, lognorm, uniform, expon, poisson, beta as beta_dist, gamma as gamma_dist, bernoulli, binom, chi2, cauchy

# x is a numpy array of values
# Collection of Probability Density Functions (PDFs)

# Normal PDF
def normal_pdf(x, mu, sigma):
    return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-0.5*((x-mu)/sigma)**2)

# Lognormal PDF
def lognormal_pdf(x, mu, sigma):
    # find all values equal to 0 and replace with e-12
    x = np.where(x == 0, 1e-12, x)
    assert x.all() > 0, "x must be greater than 0"
    return 1/(x*sigma*np.sqrt(2*np.pi)) * np.exp(-0.5*((np.log(x)-mu)/sigma)**2)

def lognormal_offset_scaled_pdf(x, y0, A, w, c):
    """
    Custom Lognormal PDF with offset and scaling.

    Parameters:
        x (array): Input values.
        y0 (float): Constant offset.
        A (float): Amplitude scaling factor.
        w (float): Shape parameter (related to standard deviation of ln(x)).
        c (float): Location parameter (related to mean of ln(x)).

    Returns:
        array: Computed PDF values.
    """
    # Ensure x > 0 to avoid log(0)
    x = np.maximum(x, 1e-10)
    return y0 + A / (np.sqrt(2 * np.pi) * w * x) * np.exp(-(np.log(x)/(x/c))**2 / (2 * w**2))

# Gaussian PDF (same as Normal)
def gaussian_pdf(x, mu, sigma):
    return normal_pdf(x, mu, sigma)

# Uniform PDF
def uniform_pdf(x, a, b):
    assert a < b, "a must be less than b"
    return (1/(b-a)) * (a <= x <= b)

# Exponential PDF
def exponential_pdf(x, lambd):
    assert x >= 0, "x must be greater than or equal to 0"
    assert lambd > 0, "lambda must be greater than 0"
    return lambd * np.exp(-lambd*x)

# Poisson PDF
def poisson_pdf(x, lambd):
    assert x.all() >= 0 and isinstance(x, int), "x must be a non-negative integer"
    assert lambd > 0, "lambda must be greater than 0"
    return (lambd**x) * np.exp(-lambd) / factorial(x)

# Beta PDF
def beta_pdf(x, alpha, beta_val):
    assert x.all() >= 0 and x.all() <= 1, "x must be between 0 and 1"
    assert alpha > 0, "alpha must be greater than 0"
    assert beta_val > 0, "beta must be greater than 0"
    return (x**(alpha-1) * (1-x)**(beta_val-1)) / beta(alpha, beta_val)

# Gamma PDF
def gamma_pdf(x, alpha, beta_val):
    assert x > 0, "x must be greater than 0"
    assert alpha > 0, "alpha must be greater than 0"
    assert beta_val > 0, "beta must be greater than 0"
    return (beta_val**alpha * x**(alpha-1) * np.exp(-beta_val*x)) / gamma(alpha)

# Bernoulli PDF
def bernoulli_pdf(x, p):
    assert x in [0, 1], "x must be 0 or 1"
    assert 0 <= p <= 1, "p must be between 0 and 1"
    return p**x * (1-p)**(1-x)

# Binomial PDF
def binomial_pdf(x, n, p):
    assert 0 <= x <= n and isinstance(x, int), "x must be an integer between 0 and n"
    assert 0 <= p <= 1, "p must be between 0 and 1"
    return binom.pmf(x, n, p)

# Chi-Square PDF
def chi_square_pdf(x, k):
    assert x >= 0, "x must be greater than or equal to 0"
    assert k > 0, "degrees of freedom must be positive"
    return (x**(k/2 - 1) * np.exp(-x/2)) / (2**(k/2) * gamma(k/2))

# Cauchy (Lorentzian) PDF
def cauchy_pdf(x, x0, gamma_val):
    return (1/np.pi) * (gamma_val / ((x - x0)**2 + gamma_val**2))



