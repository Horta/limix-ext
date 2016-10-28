from __future__ import absolute_import

import logging

import numpy as np
from numpy import asarray

from ..util import gower_normalization
from ._core import run_scan


def binomial_scan(nsuccesses, ntrials, covariate, X, K):
    logger = logging.getLogger(__name__)
    logger.info('Gower normalizing')
    K = gower_normalization(asarray(K, float))
    X = asarray(X, float).copy()
    nsuccesses = asarray(nsuccesses, float).copy()
    covariate = asarray(covariate, float).copy()
    ntrials = asarray(ntrials, float).copy()

    ok = X.std(0) > 0
    pvals = np.ones(X.shape[1])
    stats = np.zeros(X.shape[1])
    logger.info('macau scan started')


    pvals_ = run_scan(nsuccesses, ntrials, K, X[:,ok])

    logger.info('macau scan finished')
    pvals[ok] = np.asarray(pvals_, float).ravel()

    return (stats, pvals)
