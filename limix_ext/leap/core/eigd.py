import logging
import time

import numpy as np
import scipy.linalg as la


def eigenDecompose(XXT):
    logger = logging.getLogger(__name__)
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
