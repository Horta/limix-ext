from __future__ import division

import pytest
import sys

import limix_ext as lxt

from numpy import array, dot, empty, hstack, ones, pi, sqrt, zeros
from numpy import asarray
from numpy.random import RandomState
from numpy.testing import assert_almost_equal

@pytest.mark.skipif('linux' not in sys.platform,
                    reason="requires Linux")
def test_macau():


    random = RandomState(139)
    nsamples = 1000
    nfeatures = 1500

    G = random.randn(nsamples, nfeatures) / sqrt(nfeatures)

    u = random.randn(nfeatures)

    z = 0.1 + 2 * dot(G, u) + random.randn(nsamples)

    ntrials = random.randint(10, 500, size=nsamples)

    y = zeros(nsamples)
    for i in range(len(ntrials)):
        y[i] = sum(z[i] + random.logistic(scale=pi / sqrt(3),
                                          size=ntrials[i]) > 0)
    M = ones((nsamples, 1))

    K = G.dot(G.T)
    (stats, pvals) = lxt.macau.scan(y, ntrials, M, G[:,0:5], K)

    pvals_expected = array([0.02004, 0.4066, 0.5314, 0.9621, 0.8318])
    stats_expected = array([0, 0, 0, 0, 0])

    pvals = asarray(pvals)
    stats = asarray(stats)

    assert_almost_equal(pvals/10, pvals_expected/10, decimal=1)
    assert_almost_equal(stats/10, stats_expected/10, decimal=1)
