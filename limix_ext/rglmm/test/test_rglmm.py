import limix_ext as lext
from numpy.testing import assert_
import numpy as np
import os
from os.path import join


def test_rglmm_binomial():
    folder = os.path.dirname(os.path.realpath(__file__))

    nsuc = np.load(join(folder, "nsuc.npy"))
    ntri = np.load(join(folder, "ntri.npy"))
    K = np.load(join(folder, "K.npy"))
    X0 = np.load(join(folder, "X.npy"))

    X = X0[:, :2].copy()
    G = X0[:, 2:].copy()

    r = lext.rglmm.rglmm_binomial(nsuc, ntri, X, K, G)
    pv = r['pv']
    assert_(pv[0] <= 1)
