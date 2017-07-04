from numpy import asarray

from .core import estimate_h2
from ..util import gower_normalization


def estimate(y, K, prevalence, timeout=None):
    K = gower_normalization(asarray(K, float))
    y = asarray(y, float).copy()
    if timeout is None:
        return estimate_h2(K, y, prevalence)
    return estimate_h2(K, y, prevalence, timeout=timeout)
