from numpy import asarray
from limix_tool.kinship import gower_kinship_normalization
from core import estimate_h2

def estimate(y, K, prevalence):
    K = gower_kinship_normalization(asarray(K, float))
    y = asarray(y, float).copy()
    return _bernoulli_estimator(y, K, prevalence)

def _bernoulli_estimator(y, K, prevalence):
    return estimate_h2(K, y, prevalence)
