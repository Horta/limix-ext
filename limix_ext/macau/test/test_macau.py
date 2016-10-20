from __future__ import division

import pytest
import sys

import limix_ext as lxt

from numpy import array, dot, empty, hstack, ones, pi, sqrt, zeros
from numpy.random import RandomState
from numpy.testing import assert_almost_equal

@pytest.mark.skipif('linux' not in sys.platform,
                    reason="requires Linux")
def test_macau():


    random = RandomState(139)
    nsamples = 300
    nfeatures = 1000

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
    lxt.macau.scan(y, ntrials, M, G, K)
