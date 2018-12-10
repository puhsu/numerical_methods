"""
Differential equations solver module.
Used for computing threshold and
shows plan
"""
import numpy as np


def compute(*, u, z, s, x0, y0, tau, beta):
    x = np.linspace(0, 10, s.shape[0])

    real_shows = np.dstack([x, np.sqrt(x) * 10]).reshape(-1, 2)
    threshold = np.dstack([x, np.sin(x)]).reshape(-1, 2)
    return real_shows, threshold


def runge_kutta4(funcs, t0, tN, x0, grid_size=100):
    """Solves Cauchy problem.

    Finds solution to cauchy problem given vector of functions boundaries for
    arguments and initial condition using Runge–Kutta 4th order method.

    Args:
        func: vector of callable objects representing functions from system of
            equations in form of f(t, x)
        t0: lower limit of interval
        tN: upper limit of interval
        x0: initial conditions, vector of x(t0)
        grid_size: number of grid nodes

    Returns:
        Something
    """

    step_size = (tN - t0) / grid_size
    grid = np.linspace(t0, tN, grid_size + 1)
    solutions = np.zeros(shape=(grid_size + 1, len(funcs)))

    solutions[0] = x0
    for i, (t, x) in enumerate(zip(grid[:-1], solutions[:-1]), 1):
        k1 = step_size * np.array([f(t, x) for f in funcs])
        k2 = step_size * np.array([f(t + step_size / 2, x + k1 / 2) for f in funcs])
        k3 = step_size * np.array([f(t + step_size / 2, x + k2 / 2) for f in funcs])
        k4 = step_size * np.array([f(t + step_size, x + k3) for f in funcs])
        solutions[i] = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return grid, solutions
