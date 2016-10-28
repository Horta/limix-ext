from __future__ import absolute_import

import logging

import numpy as np
from numpy import asarray

from limix_tool.kinship import gower_kinship_normalization

from ._core import train_associations


def normal_scan(y, covariate, X, K, ntrials=None):
    logger = logging.getLogger(__name__)
    logger.info('Gower normalizing')
    K = gower_kinship_normalization(asarray(K, float))
    X = asarray(X, float).copy()
    y = asarray(y, float).copy()
    covariate = asarray(covariate, float).copy()

    if ntrials is not None:
        y /= ntrials

    y -= y.mean()
    std = y.std()
    if std > 0.:
        y /= std

    y = y[:, np.newaxis]


    logger.info('train_association started')
    (stats, pvals, _, _, _) = train_associations(X, y, K, C=covariate,
                                             addBiasTerm=False)
    logger.info('train_association finished')
    pvals = np.asarray(pvals, float).ravel()
    nok = np.logical_not(np.isfinite(pvals))
    pvals[nok] = 1.

    stats = np.asarray(stats, float).ravel()
    stats[nok] = 0.

    return (stats, pvals)
