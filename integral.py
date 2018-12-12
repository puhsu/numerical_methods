"""Integration module.
"""
import numpy as np


def integrate(func):
    """Computes integral of a function with trapezodial rule.

    Finds integral of a given grid function numerically with accumulated error
    of O(h^2), where h is a step size in grid.

    Args:
        func: numpy.ndarray storing grid function with x, y in columns

    Returns:
        integral value as a single float number

    """
    x = func[:, 0]
    y = func[:, 1]

    dx = x[1:] - x[:-1]
    sy = y[1:] + y[:-1]
    return np.sum(dx * sy) / 2


def compute(func):
    """Finds integral of a function.
    Return: g(t) = ∫_0^t func(x) dx (tabulated)
    """
    x = func[:, 0]

    res = []
    for i in range(func.shape[0]):
        res.append((x[i], integrate(func[i:])))
    return np.array(res)
