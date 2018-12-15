"""Integration module.
"""
import numpy as np


def integrate(grid_function):
    """Computes integral of a function with trapezodial rule.

    Finds integral of a given grid function numerically with accumulated error
    of O(h^2), where h is a step size in grid.

    Args:
        func: numpy.ndarray storing grid function with x, y in columns

    Returns:
        integral value as a single float number

    """
    x = grid_function[:, 0]
    y = grid_function[:, 1]

    dx = x[1:] - x[:-1]
    sy = y[1:] + y[:-1]
    return np.sum(dx * sy) / 2


def definite_integral(f, low, high, grid_size=50):
    """Finds definite integral of a function.

    Given a callable compute it's definite integral on the segment
    set by low and high.

    Args:
        f (callable): any callable object representing a function
        low (float): lower limit for integral
        high (float): upper limit for integral
    """

    if low < 0:
        low = 0
    x = np.linspace(low, high, grid_size)
    grid = np.dstack([x, f(x)]).reshape(-1, 2)
    return integrate(grid)
