import numpy as np
from numpy import asarray

from ..util import gower_normalization, maf
from .core import test_ltmlm


def scan(y, X, K, prevalence):
    K = gower_normalization(asarray(K, float))
    X = asarray(X, int)
    y = asarray(y, float)

    (_, pvals, stats) = test_ltmlm(X, K, y, prevalence)

    pvals = np.asarray(pvals, float).ravel()
    stats = np.asarray(stats, float).ravel()

    if not np.all(np.isfinite(pvals)):
        raise ValueError("There is some non-finite stuff in this p-values.")

    return (stats, pvals)
