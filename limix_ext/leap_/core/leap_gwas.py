import numpy as np
import scipy as sp
from fastlmm.inference.lmm_cov import LMM as fastLMM
# from pysnptools.standardizer import DiagKtoN

def gwas(K, G, liabs, h2, covariate=None):

    G -= np.mean(G, 0)
    G /= np.std(G, 0)
    if covariate is None:
        covariate = np.ones((K.shape[0], 1))
    lmm = fastLMM(X=covariate, Y=liabs[:,np.newaxis], K=K, inplace=True)
    res = lmm.nLLeval(h2=h2, dof=None, scale=1.0, penalty=0.0, snps=G)
    beta = res['beta']

    chi2stats = beta*beta/res['variance_beta']
    p_values = sp.stats.f.sf(chi2stats,1,lmm.U.shape[0]-3)[:,0]

    return (chi2stats, p_values)

    # (test_snps,pheno,
    #                  G0=None, G1=None, mixing=None,
    #                  covar=None, output_file_name=None, h2=None, log_delta=None,
    #                  cache_file = None):

# def gwas(bedSim, bedTest, pheno, h2, outFile, eigenFile, covar):

    # bedSim, pheno = leapUtils._fixupBedAndPheno(bedSim, pheno)
    # bedTest, pheno = leapUtils._fixupBedAndPheno(bedTest, pheno)

    #Run GWAS
    # logdelta = np.log(1.0/h2 - 1)
    # G0 = (bedSim if eigenFile is None else None)
    # print 'Performing LEAP GWAS...'
    # results_df = fastlmm.association.single_snp(bedTest, pheno, G0=G0,
    #         covar=covar, output_file_name=outFile, log_delta=logdelta,
    #         cache_file=eigenFile)
    # return results_df
