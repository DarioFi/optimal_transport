import numpy as np
import sympy as sp
from scipy.special import gamma, betainc
from scipy.integrate import quad

# Define alpha as a symbolic variable
alpha = sp.symbols('alpha')


def N_sphere_cap(angle, n):
    # Case (i): 1/2 * pi < alpha <= pi
    if (np.pi / 2) < angle <= np.pi:
        return 1

    # Case (ii): 1/4 * pi + 1/2 * sin^-1(1/n) <= alpha <= 1/2 * pi
    elif (np.pi / 4 + 0.5 * np.arcsin(1 / n)) <= angle <= (np.pi / 2):
        result = np.floor((2 * np.sin(angle) ** 2) / (2 * np.sin(angle) ** 2 - 1))
        return result

    # Case (iii): 1/4 * pi < alpha <= 1/4 * pi + 1/2 * sin^-1(1/n)
    elif (np.pi / 4) < angle <= (np.pi / 4 + 0.5 * np.arcsin(1 / n)):
        return n + 1

    # Case (iv): alpha = 1/4 * pi
    elif np.isclose(angle, np.pi / 4):  # To account for floating-point precision
        return 2 * n

    elif 0 < angle < np.pi / 4:
        return N_star(angle, n)


    # If alpha does not fall into any of the above categories, return None or raise an error
    else:
        raise ValueError("Alpha value out of range for defined N(alpha) function.")


def N_star(angle, n):
    # Calculate beta
    beta = np.arcsin(np.sqrt(2) * np.sin(angle) / 2)

    # Define the integrand for the integral
    def integrand(theta, n, beta):
        return (np.sin(theta)) ** (n - 2) * (np.cos(theta) - np.cos(beta))

    # Perform the integral
    integral, _ = quad(integrand, 0, beta, args=(n, beta))

    # Calculate the function value
    numerator = np.pi * gamma((n - 1) / 2) * np.sin(beta) * np.tan(beta)
    denominator = 2 * gamma(n / 2) * integral

    N_star = numerator / denominator

    return N_star


def theta_bound(alpha):
    return np.arccos(max(2 ** (2 * alpha - 1) - 1, 0))
