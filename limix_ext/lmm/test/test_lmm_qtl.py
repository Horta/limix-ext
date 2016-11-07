from numpy.random import RandomState
from numpy import (sqrt, ones, asarray, zeros_like, dot, eye)
from numpy.testing import assert_allclose

from limix_ext.lmm.qtl import binomial_scan
from limix_ext.lmm.qtl import poisson_scan

def test_binomial():
    random = RandomState(981)
    n = 50
    p = n+4

    M = ones((n, 1)) * 0.4
    G = random.randint(3, size=(n, p))
    G = asarray(G, dtype=float)
    G -= G.mean(axis=0)
    G /= G.std(axis=0)
    G /= sqrt(p)

    K = dot(G, G.T)
    Kg = K / K.diagonal().mean()
    K = 0.5*Kg + 0.5*eye(n)
    K = K / K.diagonal().mean()

    z = random.multivariate_normal(M.ravel(), K)
    nsuccesses = zeros_like(z)
    nsuccesses[z>0] = 1.
    prevalence = 0.5
    covariates = ones((n, 1))
    ntrials = ones(n)

    pvalues = binomial_scan(nsuccesses, ntrials, G, K, covariates)
    assert_allclose(pvalues[:5], [0.454928853761, 0.520126645015,
                                  0.800757778798, 0.350573357244,
                                  0.499519169745])

def test_poisson():
    random = RandomState(981)
    n = 50
    p = n+4

    M = ones((n, 1)) * 0.4
    G = random.randint(3, size=(n, p))
    G = asarray(G, dtype=float)
    G -= G.mean(axis=0)
    G /= G.std(axis=0)
    G /= sqrt(p)

    K = dot(G, G.T)
    Kg = K / K.diagonal().mean()
    K = 0.5*Kg + 0.5*eye(n)
    K = K / K.diagonal().mean()

    z = random.multivariate_normal(M.ravel(), K)
    noccurrences = zeros_like(z)
    noccurrences[z>0] = 1.
    prevalence = 0.5
    covariates = ones((n, 1))

    pvalues = poisson_scan(noccurrences, G, K, covariates)
    assert_allclose(pvalues[:5], [0.454928853761, 0.520126645015,
                                  0.800757778798, 0.350573357244,
                                  0.499519169745])

if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
