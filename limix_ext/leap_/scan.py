from __future__ import absolute_import

import logging

import numpy as np
from numpy import asarray

from limix_tool.kinship import gower_kinship_normalization

from .core import leap_scan


def scan(y, covariate, X, K, K_nsnps, prevalence):
    logger = logging.getLogger(__name__)
    logger.info('Gower normalizing')
    K = gower_kinship_normalization(asarray(K, float))
    X = asarray(X, float).copy()
    y = asarray(y, float).copy()
    covariate = asarray(covariate, float).copy()

    ok = X.std(0) > 0
    pvals = np.ones(X.shape[1])
    stats = np.zeros(X.shape[1])
    logger.info('leap_scan started')
    (stats_, pvals_, _) = leap_scan(X[:,ok], K, y, prevalence, K_nsnps,
                                cutoff=np.inf, covariates=covariate)
    logger.info('leap_scan finished')
    pvals[ok] = np.asarray(pvals_, float).ravel()
    stats[ok] = np.asarray(stats_, float).ravel()

    nok = np.logical_not(np.isfinite(pvals))
    pvals[nok] = 1.
    stats[nok] = 0.

    return (stats, pvals)
