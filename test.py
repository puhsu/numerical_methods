import argparse

import numpy as np
import scipy.integrate as si
import matplotlib.pyplot as plt
from matplotlib import rcParams

import integral
import interpolation
import equation

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14


def smooth(x):
    return np.arctan(x + 2) + 1


def oscilating(x):
    return np.sin(100 * x)


def non_continuous(x):
    return (x - 1) * (x < 0) + (x**3 + 2 * x + 2) * (x >= 0)


def plot_integral_еrror_scipy():
    """
    Plot error with respect to
    scipy built-in integration method
    for three different functions
    """
    plt.figure(figsize=(15, 10))
    for func in smooth, oscilating, non_continuous:
        print(f'Run computations for {func.__name__}')
        sp_result = si.quad(func, -2, 2)[0]
        errors = []
        for n in 10**np.arange(1, 7):
            x = np.linspace(-2, 2, n)
            y = func(x)
            error = np.abs(sp_result - integral.integrate(np.dstack([x, y]).reshape(-1, 2)))
            errors.append(error)
        plt.loglog(10**np.arange(1, 7), errors, label=func.__name__)
        plt.xlabel('N')
        plt.ylabel('Error')
    plt.title('E(N) compared to scipy.integrate.quad')
    plt.grid(True)
    plt.legend()
    plt.savefig('report/sp_compare.png', bbox_inches='tight')


def plot_integral_error_runge():
    plt.figure(figsize=(15, 10))
    for func in smooth, oscilating, non_continuous:
        print(f'Run computations for {func.__name__}')
        errors = []
        for n in 10**np.arange(1, 7):
            x = np.linspace(-2, 2, n // 2)
            y = func(x)

            x2 = np.linspace(-2, 2, n)
            y2 = func(x2)

            area = integral.integrate((np.dstack([x, y]).reshape(-1, 2)))
            area2 = integral.integrate((np.dstack([x2, y2]).reshape(-1, 2)))
            error = np.abs(area2 - area) / 3
            errors.append(error)
        plt.loglog(10**np.arange(1, 7), errors, label=func.__name__)
        plt.xlabel('N')
        plt.ylabel('Error')
    plt.title('E(N) upper bound')
    plt.grid(True)
    plt.legend()
    plt.savefig('report/runge.png', bbox_inches='tight')


def plot_interpolation():
    for func in smooth, oscilating, non_continuous:
        plt.figure(figsize=(20, 10))
        plt.subplot(121)
        x_func = np.linspace(-2, 2, 400)
        y_func = func(x_func)

        plt.plot(x_func, y_func, color='green', label='f(x)', zorder=1)

        # plot splines
        for n, c, z in ((8, 'orange', 3), (20, 'blue', 2)):
            x = np.linspace(-2, 2, n)
            y = func(x)
            plt.scatter(x, y, color=c, marker='o', zorder=z)
            f = interpolation.Spline(np.dstack([x, y]).reshape(-1, 2))
            plt.plot(x_func, f(x_func), color=c, linestyle='--', label=f'N={n}', zorder=z)

        plt.title('Cubic spline interpolation of f(x)')
        plt.grid(True)
        plt.legend()

        # plot E(x) = f(x) - S(x)
        plt.subplot(122)
        plt.plot(x_func, y_func - f(x_func))
        plt.grid(True)
        plt.title('Error E(x) = f(x) - S(x)')

        plt.savefig('report/' + func.__name__ + '_cubic.png', bbox_inches='tight')


def plot_equation_vector_field():
    def true_x(t):
        return np.exp(-2 * t) * 5 * np.cos(3 * t) + 1

    def true_y(t):
        return np.exp(-2 * t) * (4 * np.cos(3 * t) + 3 * np.sin(3 * t)) + 1

    funcs = [
        lambda t, x: 2 * x[0] - 5 * x[1] + 3,
        lambda t, x: 5 * x[0] - 6 * x[1] + 1,
    ]
    x0 = np.array([6, 5])

    grid, solution = equation.runge_kutta4(funcs, 0, 10, x0)
    x = solution[:, 0]
    y = solution[:, 1]

    # firstly plot solutions with original functions on two subplots
    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    plt.plot(grid, x, label='Runge–Kutta', color='red')
    plt.plot(grid, true_x(grid), label='True $x(t)$', color='blue')
    plt.grid(True)
    plt.legend()
    plt.title('$x(t)$ with numerical solution')

    plt.subplot(122)
    plt.plot(grid, y, label='Runge-Kutta', color='red')
    plt.plot(grid, true_y(grid), label='True $y(t)$', color='blue')
    plt.grid(True)
    plt.legend()
    plt.title('$y(t)$ with numerical solution')

    plt.savefig('report/diffeq_functions.png', bbox_inches='tight')

    # plot vector field and trajectories for solutions
    plt.figure(figsize=(10, 10))
    X, Y = np.meshgrid(np.linspace(-2.5, 7.5, 15), np.linspace(-2.5, 7.5, 15))
    u, v = np.zeros(X.shape), np.zeros(Y.shape)
    NI, NJ = X.shape
    for i in range(NI):
        for j in range(NJ):
            x_v = X[i, j]
            y_v = Y[i, j]
            u[i, j] = 2 * x_v - 5 * y_v + 3
            v[i, j] = 5 * x_v - 6 * y_v + 1

    plt.quiver(X, Y, u, v, color='black', width=0.002)
    plt.xlabel('$x(t)$')
    plt.ylabel('$y(t)$')
    plt.title('Vector field of $f(t, x)$')
    plt.grid(False)

    plt.plot(true_x(grid), true_y(grid), color='blue', label='analytical solution')
    plt.plot(x, y, color='red', label='numerical solution')
    plt.legend()
    plt.savefig('report/diffeq_vector_field.png', bbox_inches='tight')


def plot_equation_cauchy_error():
    def true_x(t):
        return np.exp(-2 * t) * 5 * np.cos(3 * t) + 1

    def true_y(t):
        return np.exp(-2 * t) * (4 * np.cos(3 * t) + 3 * np.sin(3 * t)) + 1

    funcs = [
        lambda t, x: 2 * x[0] - 5 * x[1] + 3,
        lambda t, x: 5 * x[0] - 6 * x[1] + 1,
    ]
    x0 = np.array([6, 5])

    grid, solution = equation.runge_kutta4(funcs, 0, 10, x0)

    steps = []
    x_errs = []
    y_errs = []
    for n in 10**np.arange(2, 6):
        grid, solution = equation.runge_kutta4(funcs, 0, 10, x0, grid_size=n)
        x = solution[:, 0]
        y = solution[:, 1]

        x_errs.append(np.abs(x - true_x(grid)).max())
        y_errs.append(np.abs(y - true_y(grid)).max())
        steps.append(10 / n)

    plt.figure(figsize=(10, 10))
    plt.loglog(steps, x_errs, color='red', label='$x(t)$ error')
    plt.loglog(steps, y_errs, color='blue', label='$y(t)$ error')
    plt.grid(True)
    plt.legend()
    plt.xlabel('h')
    plt.ylabel('error')
    plt.title('Runge–Kutta method error')
    plt.savefig('report/diffeq_error.png', bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Tests for all numerical methods modules. Mostly building plots.'
    )

    parser.add_argument(
        '--integral',
        action='store_true',
        help='Test integration module',
    )

    parser.add_argument(
        '--interpolation',
        action='store_true',
        help='Test interpolation module',
    )

    parser.add_argument(
        '--cauchy',
        action='store_true',
        help='Test differential equations solver',
    )

    args = parser.parse_args()

    if args.integral:
        plot_integral_error_runge()
        plot_integral_еrror_scipy()

    if args.interpolation:
        plot_interpolation()
        plot_equation_vector_field()

    if args.cauchy:
        plot_equation_cauchy_error()
