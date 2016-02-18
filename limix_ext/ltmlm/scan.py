from time import time
from limix.utils.preprocess import covar_rescale
import numpy as np

def bernoulli_scan(y, K, prevalence, X):
    from lib import test_ltmlm

    y = y.copy()
    X = X.copy()
    K = K.copy()

    print("-- LTMLM began --")
    RV = dict()
    RV['pv'] = np.full(X.shape[1], np.nan, float)

    start = time()
    try:
        K = covar_rescale(K)
        (h2, pvals, _) = test_ltmlm(X, K, y, prevalence)
        pvals = np.asarray(pvals, float).ravel()
        pvals[np.logical_not(np.isfinite(pvals))] = 1.
        RV['pv'][:] = pvals
        RV['error_status'] = 0
        RV['error_msg'] = None

    except Exception as e:
        RV['error_status'] = 1
        RV['error_msg'] = str(e)
        print RV['error_msg']
    RV['elapsed'] = time()-start

    print("-- LTMLM finished (%.3f s) --" % RV['elapsed'])

    return RV
