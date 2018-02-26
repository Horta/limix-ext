from __future__ import absolute_import
from __future__ import division

import logging
import pystan
import limix
import os
import numpy as np
import pickle as pkl

from ..util import gower_normalization
from ..util import clone


def binomial_scan(nsuc, ntri, X, G, K):
    logger = logging.getLogger(__name__)

    logger.info('Gower normalizing')

    nsuc = np.asarray(clone(nsuc), int)
    ntri = np.asarray(clone(ntri), int)
    X = clone(X)
    G = clone(G)
    K = clone(K)
    N = K.shape[0]
    P = X.shape[1]

    gower_normalization(K, out=K)

    sm = load_model()
    data = dict(N=N, P=P, nsuc=nsuc, ntri=ntri, X=X, K=K)

    fit = sm.sampling(data=data)
    params = extract_params(N, P, fit)
    null_lml = params['lp']

    alt_lmls = []
    for i in range(G.shape[1]):
        data['X'] = np.concatenate((X, G[:, i:i + 1]), axis=1)
        data['P'] = data['X'].shape[1]
        fit = sm.sampling(data=data)
        params = extract_params(N, P, fit)
        alt_lmls.append(params['lp'])

    pvals = limix.stats.lrt_pvalues(null_lml, alt_lmls)

    return pvals


def load_model():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    fp_pickle = os.path.join(dir_path, 'glmm.pickle')
    fp_model = os.path.join(dir_path, 'glmm.stan')
    if not os.path.exists(fp_pickle):
        stan_code = open(fp_model).read()
        sm = pystan.StanModel(model_code=stan_code)
        with open(fp_pickle, 'wb') as f:
            pkl.dump(sm, f, pkl.HIGHEST_PROTOCOL)
    else:
        with open(fp_pickle, 'rb') as f:
            sm = pkl.load(f)
    return sm


def extract_params(N, P, fit):
    nitems = dict(effsiz=P, u_effsiz=N, e=N, u=N)
    pars = fit.model_pars
    for k in pars:
        if k not in nitems:
            nitems[k] = 1
    params = fit.get_posterior_mean()
    i = 0
    r = dict()
    for k in pars:
        r[k] = np.mean(params[i:i + nitems[k], :], 1)
        i += nitems[k]
    r['lp'] = np.mean(params[-1, :])
    for k in ['lp', 'sigma_g', 'sigma_e']:
        r[k] = r[k].item()
    for k in ['u', 'u_effsiz', 'e']:
        if k in r:
            del r[k]
    return r
