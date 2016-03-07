from numpy import asarray
import numpy as np
from limix_tool.kinship import gower_kinship_normalization
from core import test_ltmlm

def scan(y, X, K, prevalence):
    K = gower_kinship_normalization(asarray(K, float))
    X = asarray(X, float).copy()
    y = asarray(y, float).copy()
    (_, pvals, stats) = test_ltmlm(X, K, y, prevalence)
    pvals = np.asarray(pvals, float).ravel()
    nok = np.logical_not(np.isfinite(pvals))
    pvals[nok] = 1.
    stats = np.asarray(stats, float).ravel()
    stats[nok] = 0.
    return (stats, pvals)
