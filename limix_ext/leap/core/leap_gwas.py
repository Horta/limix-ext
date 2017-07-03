import numpy as np
import scipy as sp

from fastlmm.inference.lmm_cov import LMM as fastLMM


def gwas(K, G, liabs, h2, covariate=None):

    G -= np.mean(G, 0)
    G /= np.std(G, 0)
    if covariate is None:
        covariate = np.ones((K.shape[0], 1))
    lmm = fastLMM(X=covariate, Y=liabs[:,np.newaxis], K=K, inplace=True)
    res = lmm.nLLeval(h2=h2, dof=None, scale=1.0, penalty=0.0, snps=G)
    beta = res['beta']

    chi2stats = beta*beta/res['variance_beta']
    p_values = sp.stats.f.sf(chi2stats,1,lmm.U.shape[0]-3)[:,0]

    return (chi2stats, p_values)
