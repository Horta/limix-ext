from __future__ import absolute_import

from sys import platform as _platform

from numpy import copyto, empty_like


def gower_normalization(K, out=None):
    """Perform Gower normalizion on covariance matrix K.

    The rescaled covariance matrix has sample variance of 1.
    """
    c = (K.shape[0] - 1) / (K.trace() - K.mean(0).sum())
    if out is None:
        return c * K

    copyto(out, K)
    out *= c


def clone(X):
    if X is None:
        return None
    Y = empty_like(X, dtype=float, order='C')
    copyto(Y, X)
    return Y


def maf(X):
    ok = _check_encoding(X)
    if not ok:
        raise ValueError("It assumes that X encodes 0, 1, and 2 only.")
    s0 = X.sum(0)
    s0 = s0 / (2 * X.shape[0])
    s1 = 1 - s0
    return minimum(s0, s1)


def _check_encoding(X):
    u = unique(X)
    u = u[isfinite(u)]
    if len(u) > 3:
        return False
    return all([i in set([0, 1, 2]) for i in u])


def platform():
    """Returns whether it is running on `linux`, `osx`, or `win`."""
    if _platform == "linux" or _platform == "linux2":
        return 'linux'
    elif _platform == "darwin":
        return 'osx'
    elif _platform == "win32":
        return 'win'
    return 'unknown'
