import numpy as np


def solve_tridiagonal(a, b, c, f):
    """
    Returns solution to A*x = f

    Where Matrix A is tridiagonal
    a -- below main diagonal
    b -- main diagonal
    c -- above main diagonal

    f -- values vector
    """
    p = [c[0] / -b[0]]
    q = [f[0] / b[0]]
    n = len(f)
    a = np.r_[0, a]
    c = np.r_[c, 0]

    # first step
    for i in range(1, n):
        denominator = -b[i] - a[i] * p[-1]
        p.append(
            c[i] / denominator
        )
        q.append(
            (a[i] * q[-1] - f[i]) / denominator
        )

    p.pop()

    # second step
    x = [(a[n-1] * q[n-2] - f[n-1]) / (-b[n-1] - a[n-1] * p[n-2])]
    for i in reversed(range(n - 1)):
        x.append(p[i] * x[-1] + q[i])

    return(np.array(list(reversed(x))))


def interpolate(func, points=100):
    x = func[:, 0]
    y = func[:, 1]

    # prepare coefficients for linear system of equations
    h = x[1:] - x[:-1]
    a = y[1:]

    # find c coefficients
    c = solve_tridiagonal(
        a=h[:-1],
        b=2 * (h[:-1] + h[1:]),
        c=h[1:],
        f=3 * ((a[1:] - a[:-1]) / h[1:] - (a[:-1] - y[:-2]) / h[:-1])  # please work
    )

    # infer other coefficients from c's
    full_c = np.r_[0, c, 0]
    d = (full_c[1:] - full_c[:-1]) / (3 * h)
    b = (a - y[:-1]) / h + h * (2 * full_c[1:] + full_c[:-1]) / 3
    c = full_c[1:]

    return list(zip(a, b, c, d))


def get_spline_хy(coefs, xs):
    beg = xs[:-1]
    end = xs[1:]

    points_x = np.linspace(beg[0], end[-1], 400)
    points_y = np.array([])
    
    
    for i, (coef, l, r) in enumerate(zip(coefs, beg, end)):
        a, b, c, d = coef
        x = points_x[(l <= points_x) & (points_x < r)]
        if i == len(coefs) - 1:
            x = points_x[(l <= points_x) & (points_x <= r)]
        y = a + b * (x - r) + c * (x - r)**2 + d * (x - r)**3
        points_y = np.concatenate([points_y, y])

    return points_x, points_y


if __name__ == "__main__":
    x = np.array([0, 1, 2, 3])
    y = np.array([0, 0.5, 2, 1.5])

    print(interpolate(
        np.dstack([x,y]).reshape(-1, 2)
    ))
