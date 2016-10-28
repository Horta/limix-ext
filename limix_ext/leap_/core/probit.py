import logging
import time

import numpy as np
import scipy.optimize as opt
import scipy.stats as stats


def _create_iid(nsamples):
    return [(str(i), str(i)) for i in xrange(nsamples)]

def probit(nsnps, nsamples, pheno, h2, prev, U, S, outFile=None, covar=None,
           thresholds=None, nofail=False, numSkipTopPCs=0, mineig=1e-3,
           hess=False, recenter=True, maxFixedIters=100, epsilon=1e-3):

    logger = logging.getLogger(__name__)
    # bed, pheno = leapUtils._fixupBedAndPheno(bed, pheno)
    biid = _create_iid(nsamples)

    if isinstance(pheno, dict):
        phe = pheno['vals']
    else:
        phe = pheno
    if (len(phe.shape)==2):
        if (phe.shape[1]==1):
            phe=phe[:,0]
        else:
            raise Exception('More than one phenotype found')


    S = S * nsnps
    U = U
    # S = eigen['arr_1'] * nsnps
    # U = eigen['arr_0']
    S = np.sqrt(S)
    goodS = (S>mineig)
    if (numSkipTopPCs > 0): goodS[-numSkipTopPCs:] = False
    if (np.sum(~goodS) > 0):
        logger.info('Removing %d PCs with low variance.', np.sum(~goodS))
    G = U[:, goodS]*S[goodS]

    #Set binary vector
    pheUnique = np.unique(phe)
    if (pheUnique.shape[0] != 2):
        raise Exception('phenotype file has more than two values')
    pheMean = phe.mean()
    cases = (phe>pheMean)
    phe[~cases] = 0
    phe[cases] = 1

    #run probit regression
    t = stats.norm(0,1).isf(prev)
    if (thresholds is not None): t = thresholds

    #Recenter G    to only consider the unrelated individuals
    if recenter: G -= np.mean(G[:, :], axis=0)
    else: G -= np.mean(G, axis=0)

    numFixedFeatures = 0
    if (covar is not None):
        covar *= np.mean(np.std(G, axis=0))
        G = np.concatenate((covar, G), axis=1)
        numFixedFeatures += covar.shape[1]

    #Run Probit regression
    probitThresh = (t if thresholds is None else t[:])
    beta = probitRegression(G[:, :], phe[:], probitThresh, nsnps, numFixedFeatures, h2, hess, maxFixedIters, epsilon, nofail)

    #Predict liabilities for all individuals
    meanLiab = G.dot(beta)
    liab = meanLiab.copy()
    indsToFlip = ((liab <= t) & (phe>0.5)) | ((liab > t) & (phe<0.5))
    liab[indsToFlip] = stats.norm(0,1).isf(prev)

    if (outFile is not None):
        #save liabilities
        f = open(outFile+'.liabs', 'w')
        for ind_i,[fid,iid] in enumerate(biid): f.write(' '.join([fid, iid, '%0.3f'%liab[ind_i]]) + '\n')
        f.close()

        #save liabilities after regressing out the fixed effects
        if (numFixedFeatures > 0):
            liab_nofixed = liab - G[:, :numFixedFeatures].dot(beta[:numFixedFeatures])
            f = open(outFile+'.liab_nofixed', 'w')
            for ind_i,[fid,iid] in enumerate(biid): f.write(' '.join([fid, iid, '%0.3f'%liab_nofixed[ind_i]]) + '\n')
            f.close()

            liab_nofixed2 = meanLiab - G[:, :numFixedFeatures].dot(beta[:numFixedFeatures])
            indsToFlip = ((liab_nofixed2 <= t) & (phe>0.5)) | ((liab_nofixed2 > t) & (phe<0.5))
            liab_nofixed2[indsToFlip] = stats.norm(0,1).isf(prev)
            f = open(outFile+'.liab_nofixed2', 'w')
            for ind_i,[fid,iid] in enumerate(biid): f.write(' '.join([fid, iid, '%0.3f'%liab_nofixed2[ind_i]]) + '\n')
            f.close()

    #Return phenotype struct with liabilities
    liabsStruct = {
        'header':[None],
        'vals':liab,
        'iid':biid
    }
    return liabsStruct

def evalProbitReg(beta, X, cases, controls, thresholds, invRegParam, normPDF,
                  _):
    XBeta = np.ravel(X.dot(beta)) - thresholds
    phiXBeta = normPDF.pdf(XBeta)
    PhiXBeta = normPDF.cdf(XBeta)

    logLik = np.sum(np.log(PhiXBeta[cases])) + np.sum(np.log(1-PhiXBeta[controls]))
    w = np.zeros(X.shape[0])
    w[cases] = -phiXBeta[cases] / PhiXBeta[cases]
    w[controls] = phiXBeta[controls] / (1-PhiXBeta[controls])
    grad = X.T.dot(w)

    #regularize
    logLik -= 0.5*invRegParam * beta.dot(beta)    #regularization
    grad += invRegParam * beta
    return (-logLik, grad)


def probitRegression(X, y, thresholds, numSNPs, numFixedFeatures, h2, useHess, maxFixedIters, epsilon, nofail):

    logger = logging.getLogger(__name__)

    import sklearn.linear_model
    regParam = h2 /  float(numSNPs)
    Linreg = sklearn.linear_model.Ridge(alpha=1.0/(2*regParam), fit_intercept=False, normalize=False, solver='lsqr')
    Linreg.fit(X, y)
    initBeta = Linreg.coef_
    np.random.seed(1234)

    normPDF = stats.norm(0, np.sqrt(1-h2))
    invRegParam = 1.0/regParam
    controls = (y==0)
    cases = (y==1)
    funcToSolve = evalProbitReg

    assert useHess == False
    # hess =(probitRegHessian if useHess else None)
    hess = None
    jac= True
    method = 'Newton-CG'
    args = (X, cases, controls, thresholds, invRegParam, normPDF, h2)
    logger.debug('Beginning Probit regression.')
    t0 = time.time()
    optObj = opt.minimize(funcToSolve, x0=initBeta, args=args, jac=jac, method=method, hess=hess)
    logger.debug('Done in %0.2f seconds.', time.time()-t0)
    if (not optObj.success):
        logger.debug('Optimization status: %s.', optObj.status)
        logger.debug(optObj.message)
        if (nofail == 0):
            raise Exception('Probit regression failed with message: '
                            + optObj.message)
    beta = optObj.x

    #Fit fixed effects
    if (numFixedFeatures > 0):
        thresholdsEM = np.zeros(X.shape[0]) + thresholds

        for i in xrange(maxFixedIters):
            logger.debug('Beginning fixed effects iteration %d.', i+1)
            t0 = time.time()
            prevBeta = beta.copy()

            #Learn fixed effects
            thresholdsTemp = thresholdsEM - X[:, numFixedFeatures:].dot(beta[numFixedFeatures:])
            args = (X[:, :numFixedFeatures], cases, controls, thresholdsTemp, 0, normPDF, h2)

            optObj = opt.minimize(funcToSolve, x0=beta[:numFixedFeatures], args=args, jac=True, method=method, hess=hess)
            if (not optObj.success):
                logger.warn(optObj.message)
                #raise Exception('Learning failed with message: ' + optObj.message)
            beta[:numFixedFeatures] = optObj.x

            #Learn random effects
            thresholdsTemp = thresholdsEM - X[:, :numFixedFeatures].dot(beta[:numFixedFeatures])
            args = (X[:, numFixedFeatures:], cases, controls, thresholdsTemp, invRegParam, normPDF, h2)
            optObj = opt.minimize(funcToSolve, x0=beta[numFixedFeatures:], args=args, jac=True, method=method, hess=hess)
            if (not optObj.success):
                logger.warn(optObj.message)
                #raise Exception('Learning failed with message: ' + optObj.message)
            beta[numFixedFeatures:] = optObj.x

            diff = np.sqrt(np.mean(beta[:numFixedFeatures]**2 - prevBeta[:numFixedFeatures]**2))
            logger.debug('Done in %0.2f seconds.', time.time()-t0)
            logger.debug('Diff: %0.4e.', diff)
            if (diff < epsilon): break
    return beta
