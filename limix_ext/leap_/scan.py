from numpy import asarray
import numpy as np
from core import leap_scan
from limix_util.kinship import gower_kinship_normalization

def scan(y, covariate, X, K, K_nsnps, prevalence):
    K = gower_kinship_normalization(asarray(K, float))
    X = asarray(X, float).copy()
    y = asarray(y, float).copy()
    covariate = asarray(covariate, float).copy()

    ok = X.std(0) > 0
    pvals = np.ones(X.shape[1])


    (_, pvals_, _) = leap_scan(X[:,ok], K, y, prevalence, K_nsnps,
                                cutoff=np.inf, covariates=covariate)
    pvals[ok] = np.asarray(pvals_, float).ravel()
    pvals[np.logical_not(np.isfinite(pvals))] = 1.

    return pvals
