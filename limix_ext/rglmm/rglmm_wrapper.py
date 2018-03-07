from __future__ import division

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import h5py

import os
import warnings
import numpy as np
from rpy2.rinterface import RRuntimeWarning
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri

numpy2ri.activate()

from numpy import abs as npy_abs
from numpy import asarray, clip, inf, sqrt
from numpy.testing import assert_
from scipy.stats import chi2

from numpy_sugar import epsilon


def rglmm_binomial(nsuc, ntri, X, K, G=None, npoints=10 * 6):
    import limix

    ntri = np.atleast_2d(ntri.T).T
    nsuc = np.atleast_2d(nsuc.T).T

    warnings.filterwarnings("ignore", category=RRuntimeWarning)

    r_source = ro.r['source']
    folder = os.path.dirname(os.path.realpath(__file__))
    r_source(os.path.join(folder, "rglmm.R"))
    glmm_binomial = ro.r['rglmm_binomial']

    if G is None:
        r = glmm_binomial(nsuc, ntri, X, K, npoints)
    else:
        r = glmm_binomial(nsuc, ntri, X, K, npoints, G=G)

    null_lml = r[r.names.index('lml')][0]
    intercept = r[r.names.index('intercept')][0]
    effsizes = np.asarray(r[r.names.index('effsizes')])
    v_g = r[r.names.index('v_g')][0]
    v_e = r[r.names.index('v_e')][0]

    d = dict(
        null_lml=null_lml,
        intercept=intercept,
        beta=effsizes,
        v_g=v_g,
        v_e=v_e)

    if G is not None:
        d['alt_lmls'] = np.asarray(r[r.names.index('alt_lmls')])
        pv = limix.stats.lrt_pvalues(null_lml, d['alt_lmls'])
        d['pv'] = pv

    return d


if __name__ == '__main__':
    import numpy as np

    nsuc = np.load("nsuc.npy")
    ntri = np.load("ntri.npy")
    K = np.load("K.npy")
    X0 = np.load("X.npy")

    X = X0[:, :2].copy()
    G = X0[:, 2:].copy()

    r = rglmm_binomial(nsuc, ntri, X, K, G)
    assert_(r['pv'][0] <= r['pv'][3])
