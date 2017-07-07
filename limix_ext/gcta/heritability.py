import numpy as np
from numpy import asarray

from .core import estimate_h2


def estimate(G, y, prevalence):
    y = asarray(y, float).copy()
    return _bernoulli_estimator(G, y, prevalence)


def _bernoulli_estimator(G, y, prevalence):
    h2 = estimate_h2(G, y, prevalence)
    h2 = np.asscalar(np.asarray(h2, float))
    return min(max(h2, 0.), 1.)

