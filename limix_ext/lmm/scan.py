from __future__ import absolute_import
from numpy import asarray
import numpy as np
from .core import train_associations
from limix_util.kinship import gower_kinship_normalization

def scan(y, covariate, X, K):
    K = gower_kinship_normalization(asarray(K, float))
    X = asarray(X, float).copy()
    y = asarray(y, float).copy()

    y -= y.mean()
    std = y.std()
    if std > 0.:
        y /= std

    y = y[:, np.newaxis]

    (_, pvals, _, _, _) = train_associations(X, y, K, C=covariate,
                                             addBiasTerm=False)

    pvals = np.asarray(pvals, float).ravel()
    pvals[np.logical_not(np.isfinite(pvals))] = 1.


    return pvals
