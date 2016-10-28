import numpy as np
from numpy.testing import assert_allclose

from limix_ext.lmm.heritability import binomial_estimate


# def test_bernoulli():
#     random = np.random.RandomState(981)
#     n = 500
#     p = n+4
#
#     M = np.ones((n, 1)) * 0.4
#     G = random.randint(3, size=(n, p))
#     G = np.asarray(G, dtype=float)
#     G -= G.mean(axis=0)
#     G /= G.std(axis=0)
#     G /= np.sqrt(p)
#
#     K = np.dot(G, G.T)
#     Kg = K / K.diagonal().mean()
#     K = 0.5*Kg + 0.5*np.eye(n)
#     K = K / K.diagonal().mean()
#
#     z = random.multivariate_normal(M.ravel(), K)
#     y = np.zeros_like(z)
#     y[z>0] = 1.
#     prevalence = 0.5
#     covariate = np.ones((n, 1))
#
#     h2 = estimate(y, covariate, K, prevalence)
#     assert_allclose(h2, 0.81995708512)

def test_binomial():
    random = np.random.RandomState(981)
    n = 500
    p = n+4

    M = np.ones((n, 1)) * 0.4
    G = random.randint(3, size=(n, p))
    G = np.asarray(G, dtype=float)
    G -= G.mean(axis=0)
    G /= G.std(axis=0)
    G /= np.sqrt(p)

    K = np.dot(G, G.T)
    Kg = K / K.diagonal().mean()
    K = 0.5*Kg + 0.5*np.eye(n)
    K = K / K.diagonal().mean()

    z = random.multivariate_normal(M.ravel(), K)
    y = np.zeros_like(z)
    y[z>0] = 1.
    prevalence = 0.5
    covariate = np.ones((n, 1))
    ntrials = np.ones(n)

    h2 = binomial_estimate(y, ntrials, covariate, K)
    assert_allclose(h2, 0.4750208125210601)
