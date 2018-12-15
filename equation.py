"""Differential equations solver module. Used for computing threshold and shows
plan. Also has some auxilary functions used in the process of solving system of
differential equaitons.
"""
import numpy as np


def runge_kutta4(funcs, t0, tN, x0, grid_size=49):
    """Solves Cauchy problem.

    Finds solution to cauchy problem given vector of functions boundaries for
    arguments and initial condition using Runge–Kutta 4th order method.

    Args:
        func (ndarray): vector of callable objects representing functions from system of
            equations in form of f(t, x)
        t0 (float): lower limit of interval
        tN (float): upper limit of interval
        x0 (ndarray): initial conditions, vector of x(t0)
        grid_size (int): number of grid nodes

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

        # problem specifics
        if solutions[i, 1] < 0:
            solutions[i, 1] = 0
        if solutions[i, 1] > 1:
            solutions[i, 1] = 1

    return grid, solutions
