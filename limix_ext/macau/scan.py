from __future__ import absolute_import
from numpy import asarray
import numpy as np
import logging
from .core import run_scan
from limix_tool.kinship import gower_kinship_normalization

def scan(y, ntrials, covariate, X, K):
    logger = logging.getLogger(__name__)
    logger.info('Gower normalizing')
    K = gower_kinship_normalization(asarray(K, float))
    X = asarray(X, float).copy()
    y = asarray(y, float).copy()
    covariate = asarray(covariate, float).copy()
    ntrials = asarray(ntrials, float).copy()

    ok = X.std(0) > 0
    pvals = np.ones(X.shape[1])
    stats = np.zeros(X.shape[1])
    logger.info('macau scan started')


    # (stats_, pvals_, _) = runscan(y, ntrials, K, X[:,0])
    import ipdb; ipdb.set_trace()
    run_scan(y, ntrials, K, X[:,0])
    logger.info('macau scan finished')
    # pvals[ok] = np.asarray(pvals_, float).ravel()
    # stats[ok] = np.asarray(stats_, float).ravel()
    #
    # nok = np.logical_not(np.isfinite(pvals))
    # pvals[nok] = 1.
    # stats[nok] = 0.
    #
    # return (stats, pvals)
