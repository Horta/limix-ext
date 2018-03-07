from __future__ import absolute_import

import logging

import numpy as np
from numpy import asarray

from ..util import gower_normalization
from .core import run_heritability


def binomial_estimate(nsuccesses, ntrials, covariate, K, NP=10):
    logger = logging.getLogger(__name__)
    logger.info('Gower normalizing')
    K = gower_normalization(asarray(K, float))

    random = np.random.RandomState(3984)
    nsuccesses = asarray(nsuccesses, float).copy()
    covariate = asarray(covariate, float).copy()
    ntrials = asarray(ntrials, float).copy()

    logger.info('macau heritability started')

    h2 = run_heritability(nsuccesses, ntrials, K, NP)
    logger.info('macau heritability finished')

    return h2
