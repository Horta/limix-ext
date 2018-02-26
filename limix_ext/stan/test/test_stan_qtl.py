from numpy.random import RandomState
from numpy import (sqrt, ones, asarray, zeros_like, dot, eye)
from numpy.testing import assert_

from limix_ext.stan import binomial_scan


def test_binomial():
    random = RandomState(981)
    n = 10
    p = n + 4

    M = ones((n, 1)) * 0.4
    G = random.randint(3, size=(n, p))
    G = asarray(G, dtype=float)
    G -= G.mean(axis=0)
    G /= G.std(axis=0)
    G /= sqrt(p)

    K = dot(G, G.T)
    Kg = K / K.diagonal().mean()
    K = 0.5 * Kg + 0.5 * eye(n)
    K = K / K.diagonal().mean()

    z = random.multivariate_normal(M.ravel(), K)
    nsuccesses = zeros_like(z)
    nsuccesses[z > 0] = 1.
    prevalence = 0.5
    covariates = ones((n, 1))
    ntrials = ones(n)

    pvalues = binomial_scan(nsuccesses, ntrials, covariates, G[:, :2], K)
    assert_(pvalues[0] <= pvalues[1])


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
