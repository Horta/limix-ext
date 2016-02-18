import numpy as np
from time import time
from lib import estimate_h2
from limix.utils.preprocess import covar_rescale

def bernoulli_h2(y, covariate, K, prevalence):
    K = covar_rescale(K)
    y = y.copy()
    covariate = covariate.copy()

    print("-- LTMLM began --")
    RV = dict()
    RV['h2'] = np.nan

    start = time()
    try:

        K = covar_rescale(K)

        h2 = estimate_h2(K, y, prevalence)

        RV['h2'] = h2
        RV['error_status'] = 0
        RV['error_msg'] = None

    except Exception as e:
        RV['error_status'] = 1
        RV['error_msg'] = str(e)
        print RV['error_msg']
    RV['elapsed'] = time()-start

    print("-- LTMLM finished (%.3f s) --" % RV['elapsed'])

    return RV
