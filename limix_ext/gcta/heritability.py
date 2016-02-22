from numpy import asarray
import numpy as np
from core import estimate_h2

def estimate(G, y, prevalence):
    y = asarray(y, float).copy()
    return _bernoulli_estimator(G, y, prevalence)

def _bernoulli_estimator(G, y, prevalence):
    h2 = estimate_h2(G, y, prevalence)
    return np.asscalar(np.asarray(h2, float))

if __name__ == '__main__':
    random = np.random.RandomState(842)
    G = random.randint(0, 3, (5, 10))
    y = random.randint(0, 2, 5)
    print _bernoulli_estimator(G, y, 0.5)
