from __future__ import print_function

import logging

import numpy as np

import fastlmm.util.VertexCut as vc
from calc_h2 import calc_h2
from eigd import eigenDecompose
from leap_gwas import gwas
from probit import probit


def apply_this_kinship(G, K, y, prevalence, nsnps_back, cutoff,
                       covariates=None):

    logger = logging.getLogger(__name__)
    nK = K.copy()

    #These are the indexes of the IIDs to remove
    remove_set = set(np.sort(vc.VertexCut().work(K, cutoff)))
    logger.info('Marking %d individuals to be removed due to high relatedness.',
                len(remove_set))

    vinds = set(range(K.shape[0])) - remove_set
    vinds = np.array(list(vinds), int)

    S, U = eigenDecompose(K[np.ix_(vinds, vinds)])
    eigen = dict(XXT=K, arr_1=S, arr_0=U)
    # S, U = leapUtils.eigenDecompose(K[vinds, vinds])
    n = len(vinds)

    logger.info('Heritability estimation')
    h2 = calc_h2(dict(vals=y), prevalence, eigen, keepArr=vinds,
                 h2coeff=1.0, numRemovePCs=10, lowtail=False)
    logger.info('Heritability estimation finished')
    if h2 >= 1.:
        logger.warn("LEAP found h2 greater than or equal to 1: %f. %s", h2,
                    "Clipping it to 0.9.")
        h2 = 0.9
    if h2 <= 0.:
        logger.warn("LEAP found h2 smaller than or equal to 0: %f. %s", h2,
                    "Clipping it to 0.1.")
        h2 = 0.1

    logger.info('Probit fitting')
    liabs = probit(nsnps_back, n, dict(vals=y), h2, prevalence, U, S,
                   covar=covariates)

    logger.info('gwas method started')
    (stats, pvals) = gwas(nK, G, liabs['vals'], h2, covariate=covariates)
    logger.info('gwas method finished')
    return (stats, pvals, h2)
