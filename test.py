import numpy as np
import scipy.integrate as si
import matplotlib as mpl
from matplotlib import rcParams

import integral

mpl.use('Agg')
import matplotlib.pyplot as plt
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14


def smooth(x):
    return np.arctan(x + 2) + 1


def oscilating(x):
    return np.sin(100 * x)


def non_continuous(x):
    return (x - 1) * (x < 0) + (x**3 + 2 * x - 1) * (x >= 0)


def plot_еrror_scipy():
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
    plt.savefig('sp_compare.png', bbox_inches='tight')


def plot_error_runge():
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
    plt.savefig('runge.png', bbox_inches='tight')


if __name__ == "__main__":
    plot_еrror_scipy()
    plot_error_runge()
    
