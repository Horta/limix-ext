from numpy import asarray
import numpy as np
from limix_util.data_ import gower_kinship_normalization
from core import test_ltmlm

def scan(y, X, K, prevalence):
    K = gower_kinship_normalization(asarray(K, float))
    X = asarray(X, float).copy()
    y = asarray(y, float).copy()
    (_, pvals, _) = test_ltmlm(X, K, y, prevalence)
    pvals = np.asarray(pvals, float).ravel()
    pvals[np.logical_not(np.isfinite(pvals))] = 1.
    return pvals
