import numpy as np
from numpy import asarray

from core import test_ltmlm
from limix_tool.genotype import maf
from limix_tool.kinship import gower_kinship_normalization


def scan(y, X, K, prevalence):
    K = gower_kinship_normalization(asarray(K, float))
    X = asarray(X, int)
    y = asarray(y, float)

    mafs = maf(X)

    ok = np.full(X.shape[1], True, dtype=bool)
    ok[mafs <= 0.05] = False

    (_, pvals_, stats_) = test_ltmlm(X[:,ok], K, y, prevalence)

    pvals = np.ones(X.shape[1])
    pvals[ok] = np.asarray(pvals_, float).ravel()

    stats = np.zeros(X.shape[1])
    stats[ok] = np.asarray(stats_, float).ravel()

    if not np.all(np.isfinite(pvals)):
        raise ValueError("There is some non-finite stuff in this p-values.")

    return (stats, pvals)
