"""Main ad server logic"""
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
    return beta * (x - traffic)


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
