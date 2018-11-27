"""
Differential equations solver module.
Used for computing threshold and
shows plan
"""
import numpy as np


def compute(*, u, z, s, x0, y0, tau, beta):
    # TODO implement
    x = np.linspace(0, 10, s.shape[0])

    real_shows = np.dstack([x, np.sqrt(x) * 10]).reshape(-1, 2)
    threshold = np.dstack([x, np.sin(x)]).reshape(-1, 2)
    return real_shows, threshold
