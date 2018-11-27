"""
Integration module.
"""
import numpy as np


def integrate(func):
    x = func[:, 0]
    y = func[:, 1]
    area = np.sum(y) * (x[-1] - x[0]) / x.size
    return area


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
