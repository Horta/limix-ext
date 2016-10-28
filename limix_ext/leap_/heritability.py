import numpy as np
from numpy import asarray

from core import leap_scan
from limix_tool.kinship import gower_kinship_normalization


def estimate(y, covariate, K, K_nsnps, prevalence):
    K = gower_kinship_normalization(asarray(K, float))
    y = asarray(y, float).copy()
    return _bernoulli_estimator(y, covariate, K, K_nsnps, prevalence)

def _bernoulli_estimator(y, covariate, K, K_nsnps, prevalence):
    n = K.shape[0]
    G = np.random.randint(0, 3, size=(n, 1))
    G = asarray(G, float)
    (_, _, h2) = leap_scan(G, K, y, prevalence, K_nsnps,
                                    cutoff=np.inf, covariates=covariate)

    h2 = np.asscalar(np.asarray(h2, float))
    return h2
