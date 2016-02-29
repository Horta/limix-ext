import numpy as np
from numpy import asarray
from limix_util.heritability import h2_observed_space_correct
from limix_util.kinship import gower_kinship_normalization
import core

def estimate(y, covariate, K, prevalence):
    y = asarray(y, float).copy()
    return _bernoulli_estimator(y, covariate, K, prevalence)

def _bernoulli_estimator(y, covariate, K, prevalence):
    y = y.copy()
    covariate = covariate.copy()
    ascertainment = float(sum(y==1.)) / len(y)
    y -= y.mean()
    std = y.std()
    if std > 0.:
        y /= std
    K = gower_kinship_normalization(asarray(K, float))

    n = K.shape[0]
    G = np.random.randint(0, 3, size=(n, 1))
    G = np.asarray(G, float)
    y_ = y[:, np.newaxis]

    (_, _, ldelta, sigg2, beta0) = core.train_associations(G, y_,
                                                  K, C=covariate,
                                                  addBiasTerm=False)

    m = np.dot(covariate, beta0.T)
    varc = np.var(m)

    delta = np.exp(ldelta)
    sige2 = delta * sigg2
    h2 = sigg2 / (sigg2 + sige2 + varc)
    h2 = np.asscalar(np.asarray(h2, float))

    h2 = h2_observed_space_correct(h2, prevalence, ascertainment)

    return h2