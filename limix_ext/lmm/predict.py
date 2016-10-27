# from __future__ import division
# import numpy as np
# import GPy
# from numpy import asarray
# from . import core
#
# def predict(y, covariate_train, G_train, covariate_test, G_test):
#
#     G_train = np.hstack([covariate_train, G_train])
#     G_test = np.hstack([covariate_test, G_test])
#
#     kernel = GPy.kern.Linear(input_dim=G_train.shape[1])
#     m = GPy.models.GPRegression(G_train, y[:,np.newaxis], kernel)
#     m.optimize(messages=True)
#     p = m.predict(G_test)
#     return -p[0].ravel()
#
# if __name__ == '__main__':
#
#     np.random.seed(32908742)
#     G_train = np.random.randn(500, 600)
#     G_test = np.random.randn(50, 600)
#
#     covariate_train = np.ones((500, 1))
#     covariate_test = np.ones((50, 1))
#
#     y = np.random.randint(0, 2, size=500)
#
#     y_pred, sigma2_pred = predict(y, covariate_train, G_train, covariate_test, G_test)
