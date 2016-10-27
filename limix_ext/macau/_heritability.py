from __future__ import absolute_import

from numpy import asarray

import numpy as np

import logging

from .core import run_scan

from ..util import gower_normalization

def heritability(y, ntrials, covariate, K):
    logger = logging.getLogger(__name__)
    logger.info('Gower normalizing')
    K = gower_normalization(asarray(K, float))

    random = np.random.RandomState(3984)
    X = random.randn(len(y), 1)
    y = asarray(y, float).copy()
    covariate = asarray(covariate, float).copy()
    ntrials = asarray(ntrials, float).copy()

    ok = X.std(0) > 0

    pvals = np.ones(X.shape[1])
    stats = np.zeros(X.shape[1])
    logger.info('macau heritability started')

    pvals_ = run_scan(y, ntrials, K, X[:,ok])
    logger.info('macau heritability finished')
    pvals[ok] = np.asarray(pvals_, float).ravel()

    return 0.5
