import numpy as np
import pandas
import scipy as sp

import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
import rpy2.robjects.pandas2ri
from rpy2.robjects.packages import importr

rpy2.robjects.numpy2ri.activate()
rpy2.robjects.pandas2ri.activate()

def _create_r_env(y, ntrials, G):
    yy = np.concatenate([ntrials-y, y])
    # yy = np.concatenate([y, ntrials-y])
    ryy = ro.r['matrix'](ro.FloatVector(yy), ncol=2)
    return (ryy, G)

def logpdf(y_train, ntrials_train, y_test, ntrials_test, G_train, G_test):
    glmnet = importr("glmnet")
    (ryy, G) = _create_r_env(y_train, ntrials_train, G_train)
    model = glmnet.glmnet(G, ryy, family="binomial")

    l = glmnet.predict_glmnet(model, newx=G_test, type="link", s=0.01)
    l = np.asarray(l).ravel()
    psuc = 1. / (1. + np.exp(-l))

    lpdf = []
    for i in range(len(psuc)):
        lpdf += [sp.stats.binom(ntrials_test[i], psuc[i]).logpmf(y_test[i])]

    return lpdf

def logpdf_outcome(y_train, ntrials_train, y_test, ntrials_test, G_train, G_test, parallel=False):
    glmnet = importr("glmnet")
    nfolds = 5
    if parallel:
        doMC = importr("doMC")
        doMC.registerDoMC(nfolds)
    (ryy, G) = _create_r_env(y_train, ntrials_train, G_train)
    model = glmnet.cv_glmnet(G, ryy, family="binomial", alpha=0, nfolds=nfolds, parallel=parallel)


    l = glmnet.predict_cv_glmnet(model, newx=G_test, type="link", s="lambda.min")
    l = np.asarray(l).ravel()
    psuc = 1. / (1. + np.exp(-l))

    lpdf = []
    outcome = []
    for i in range(len(psuc)):
        binom = sp.stats.binom(ntrials_test[i], psuc[i])
        logpdfs = [binom.logpmf(nc) for nc in range(ntrials_test[i]+1)]

        lpdf += [logpdfs[int(y_test[i])]]
        outcome += [np.argmax(logpdfs)]

    return (lpdf, outcome)


def logpdf_outcome_mean(y_train, ntrials_train, y_test, ntrials_test, G_train, G_test, parallel=False):
    glmnet = importr("glmnet")
    nfolds = 5
    if parallel:
        doMC = importr("doMC")
        doMC.registerDoMC(nfolds)
    (ryy, G) = _create_r_env(y_train, ntrials_train, G_train)
    model = glmnet.cv_glmnet(G, ryy, family="binomial", alpha=0, nfolds=nfolds, parallel=parallel)


    l = glmnet.predict_cv_glmnet(model, newx=G_test, type="link", s="lambda.min")
    l = np.asarray(l).ravel()
    psuc = 1. / (1. + np.exp(-l))

    lpdf = []
    outcome = []
    mean = []
    for i in range(len(psuc)):
        binom = sp.stats.binom(ntrials_test[i], psuc[i])
        logpdfs = [binom.logpmf(nc) for nc in range(ntrials_test[i]+1)]

        lpdf += [logpdfs[int(y_test[i])]]
        outcome += [np.argmax(logpdfs)]
        mean += [psuc[i] * ntrials_test[i]]

    return (lpdf, outcome, mean)
