from __future__ import print_function
import logging
import numpy as np
from leap_gwas import gwas
import scipy.linalg as la
from calc_h2 import calc_h2
import fastlmm.util.VertexCut as vc
from probit import probit
import time

def eigenDecompose(XXT):
    logger = logging.getLogger(__file__)
    t0 = time.time()
    logger.debug('Computing eigendecomposition...')
    s,U = la.eigh(XXT)
    if (np.min(s) < -1e-4):
        raise Exception('Negative eigenvalues found')
    s[s<0]=0
    ind = np.argsort(s)
    ind = ind[s>1e-12]
    U = U[:, ind]
    s = s[ind]
    logger.debug('Done in %0.2f seconds', time.time()-t0)
    return s,U


def apply_this_kinship(G, K, y, prevalence, nsnps_back, cutoff,
                       covariates=None):

    logger = logging.getLogger(__file__)
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

    h2 = calc_h2(dict(vals=y), prevalence, eigen, keepArr=vinds,
                 h2coeff=1.0, numRemovePCs=10, lowtail=False)

    if h2 >= 1.:
        logger.warn("LEAP found h2 greater than or equal to 1: %f. %s", h2,
                    "Clipping it to 0.9.")
        h2 = 0.9
    if h2 <= 0.:
        logger.warn("LEAP found h2 smaller than or equal to 0: %f. %s", h2,
                    "Clipping it to 0.1.")
        h2 = 0.1

    liabs = probit(nsnps_back, n, dict(vals=y), h2, prevalence, U, S,
                   covar=covariates)

    (stats, pvals) = gwas(nK, G, liabs['vals'], h2, covariate=covariates)
    return (stats, pvals, h2)
