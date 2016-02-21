from numpy import asarray
import numpy as np
from core import leap_scan
from limix_util.data_ import gower_kinship_normalization

def scan(y, covariate, X, K, K_nsnps, prevalence):
    K = gower_kinship_normalization(asarray(K, float))
    X = asarray(X, float).copy()
    y = asarray(y, float).copy()

    (_, pvals, _) = leap_scan(X, K, y, prevalence, K_nsnps,
                                cutoff=np.inf, covariates=covariate)
    pvals = np.asarray(pvals, float).ravel()
    pvals[np.logical_not(np.isfinite(pvals))] = 1.

    return pvals
