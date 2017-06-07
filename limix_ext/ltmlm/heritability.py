from numpy import asarray

from core import estimate_h2
from ..util import gower_normalization


def estimate(y, K, prevalence):
    K = gower_normalization(asarray(K, float))
    y = asarray(y, float).copy()
    return _bernoulli_estimator(y, K, prevalence)

def _bernoulli_estimator(y, K, prevalence):
    return estimate_h2(K, y, prevalence)
