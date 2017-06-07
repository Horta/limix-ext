from numpy import empty_like
from numpy import copyto

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
    r"""Compute minor allele frequencies.

    It assumes that `X` encodes 0, 1, and 2 representing the number
    of alleles.

    Args:
        X (array_like): Genotype matrix.

    Returns:
        array_like: minor allele frequencies.

    Examples
    --------

        .. doctest::

            >>> from numpy.random import RandomState
            >>> from limix.stats import maf
            >>>
            >>> random = RandomState(0)
            >>> X = random.randint(0, 3, size=(100, 10))
            >>>
            >>> print(maf(X))
            [ 0.49   0.49   0.445  0.495  0.5    0.45   0.48   0.48   0.47   0.435]
    """
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
