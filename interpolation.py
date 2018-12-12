"""Interpolation module"""
import numpy as np

import util


def solve_tridiagonal_system(diag, below, above, f):
    """Solves system of equations with tridiagonal matrix.

    Returns solution to linear system of equations of form Ax = f, where A is
    tridiagonal -- only has main diagonal and two diagonals above and below it.

    Args:
        diag (ndarray): vector of shape N with elements from main diagonal
        below (ndarray): vector of shape N-1 with elements below main diagonal
        above (ndarray): vector of shape N-1 with elements above main diagonal
        f (ndarray): vector of shape N with values from the right

    Returns:
        Vector of x values -- solution to initial system.
    """

    p = [above[0] / -diag[0]]
    q = [f[0] / diag[0]]

    below = np.r_[0, below]
    above = np.r_[above, 0]
    n = len(f)

    for i in range(1, n):
        denominator = -diag[i] - below[i] * p[-1]
        p.append(above[i] / denominator)
        q.append((below[i] * q[-1] - f[i]) / denominator)
    p.pop()

    x = [(below[n-1] * q[n-2] - f[n-1]) / (-diag[n-1] - below[n-1] * p[n-2])]
    for i in reversed(range(n - 1)):
        x.append(p[i] * x[-1] + q[i])

    return np.array(list(reversed(x)))


class Spline(object):
    """Cubic spline interpolation.

    Class used to interpolate grid functions with cubic splines.

    Example:
        grid = np.array([[0, 1], [1, 1.5], [3, 5]])
        f = Spline(grid)
        f(0) == 1    # True
        f(1) == 1.5  # True

    Attributes:
        ends (ndarray): Saved grid nodes, used to determine which coefficients to
            use at each step
    """
    def __init__(self, grid_function):
        self.beg = grid_function[:-1, 0]
        self.end = grid_function[1:, 0]
        self._interpolate(grid_function)

    @util.vectorize
    def __call__(self, x, df=False):
        for i, (left, right) in enumerate(zip(self.beg, self.end)):
            if left <= x and x <= right:
                if df:
                    return (
                        self.b[i] +
                        2 * self.c[i] * (x - right) +
                        3 * self.d[i] * (x - right)**2
                    )
                return (
                    self.a[i] +
                    self.b[i] * (x - right) +
                    self.c[i] * (x - right)**2 +
                    self.d[i] * (x - right)**3
                )

    def _interpolate(self, grid_function):
        """Finds coefficients for spline interpolation.

        Sets (a, b, c, d) attributes with vectors for polinoms of form:
        a + b(x - x_0) + c(x - x_0)^2 + d(x - x_0)^3
        """
        x = grid_function[:, 0]
        y = grid_function[:, 1]

        h = x[1:] - x[:-1]
        a = y[1:]

        c = solve_tridiagonal_system(
            diag=2 * (h[:-1] + h[1:]),
            below=h[:-1],
            above=h[1:],
            f=3 * ((a[1:] - a[:-1]) / h[1:] - (a[:-1] - y[:-2]) / h[:-1])
        )

        padded_c = np.r_[0, c, 0]
        self.a = a
        self.b = (a - y[:-1]) / h + h * (2 * padded_c[1:] + padded_c[:-1]) / 3
        self.c = padded_c[1:]
        self.d = (padded_c[1:] - padded_c[:-1]) / (3 * h)
