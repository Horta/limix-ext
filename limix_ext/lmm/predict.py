from __future__ import division
import numpy as np
from sklearn import gaussian_process
from numpy import asarray
import core

def predict(y, covariate_train, G_train, covariate_test, G_test):

    G_train = np.hstack([covariate_train, G_train])
    G_test = np.hstack([covariate_test, G_test])

    gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
    gp.fit(G_train, y)
    y_pred, sigma2_pred = gp.predict(G_test, eval_MSE=True)

    return y_pred

if __name__ == '__main__':

    np.random.seed(32908742)
    G_train = np.random.randn(500, 600)
    G_test = np.random.randn(50, 600)

    covariate_train = np.ones((500, 1))
    covariate_test = np.ones((50, 1))

    y = np.random.randint(0, 2, size=500)

    y_pred, sigma2_pred = predict(y, covariate_train, G_train, covariate_test, G_test)
