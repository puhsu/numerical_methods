"""
Integration module.
"""
import numpy as np


def integrate(func):
    """
    Compute integral of a function using
    trapezoidal rule.
    Args:
      func (numpy.ndarray) grid function with x, y in columns
    Return:
      integral (float)
    """
    x = func[:, 0]
    y = func[:, 1]

    dx = x[1:] - x[:-1]
    sy = y[1:] + y[:-1]
    return np.sum(dx * sy) / 2


def compute(func):
    """
    Find integral of a function.
    Return: g(t) = ∫_0^t func(x) dx (tabulated)
    """
    x = func[:, 0]

    res = []
    for i in range(func.shape[0]):
        res.append((x[i], integrate(func[i:])))
    return np.array(res)
