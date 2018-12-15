"""Main ad server logic. Handles solution of
system of differential equations. Contains
criterions used to compare functions
"""
import numpy as np

import interpolation
import integral
import equation


def correction_function(traffic, plan, beta, x):
    """Computes correction for y(t)

    Args:
        traffic (float): traffic function at point t
        plan (float): plan function value at point t
        beta (float): parameter
        x (float): x(t) value
    """

    diff_plan = (x - plan)

    return beta * diff_plan


def solve(users_pdf, plan, traffic, x0, y0, beta, tau):
    """Finds threshold and shows.

    Computes threshold y(t) and shows x(t) functions by
    solving a system of differential equations:
    dx / dt = traffic'(t) integral_y^1(users_pdf(w)) dw
    dy / dt = correction_function(traffic, plan, beta, x)

    Args:
        users_pdf (ndarray): grid function representing distribution of the audience for
            the probability of hitting the target
        plan (ndarray): grid function representing increasing number of shows
        traffic (ndarray): grid function representing increasing total traffic
        x0 (float): initial parameter for cauchy problem x(0) = x0
        y0 (float): initial parameter for cauchy problem y(0) = y0
    """

    r = interpolation.Spline(users_pdf)
    s = interpolation.Spline(plan)
    z = interpolation.Spline(traffic)

    fs = [
        lambda t, v: z(t, df=True) * integral.definite_integral(r, v[1], 1),
        lambda t, v: correction_function(z(t), s(t), beta, v[0])
    ]

    grid, solutions = equation.runge_kutta4(
        funcs=fs,
        t0=0,
        tN=tau,
        x0=np.array([x0, y0]),
    )

    shows = np.dstack([grid, solutions[:, 0]]).reshape(-1, 2)
    threshold = np.dstack([grid, solutions[:, 1]]).reshape(-1, 2)

    return shows, threshold


def crit1(real_shows, threshold, users_pdf, x0, tau):
    """Computes accuracy of hitting target.

    This criterion measures the proportion of ads shown to a non-target audience
    relative to the total number of impressions.

    Args:
        real_shows (ndarray): Grid function x(t), which is a found solution
        threshold (ndarray): Grid function y(t), which is a found solution
        users_pdf (ndarray): Grid function representing probability density
        x0 (float): Initial condition
        tau (float): Time period

    Returns:
        Value in [0, 1], lower is better
    """
    x = interpolation.Spline(real_shows)
    y = interpolation.Spline(threshold)
    r = interpolation.Spline(users_pdf)

    @np.vectorize
    def under_integral(t):
        return x(t, df=True) * integral.definite_integral(lambda w: w * r(w), y(t), 1)

    return 1 - (integral.definite_integral(under_integral, 0, tau)) / (x(tau) - x0)


def crit2(plan, real_shows):
    """Computes second critrion.

    This criterion measures the accuracy of the condition by the number of shows.

    Args:
        plan (ndarray): Grid function S(t)
        real_shows (ndarray): Grid function x(t)

    Returns:
        Value of a criterion, lower is better
    """
    total_plan = plan[-1, 1]
    total_shows = real_shows[-1, 1]

    return np.abs(total_plan - total_shows) / total_plan
