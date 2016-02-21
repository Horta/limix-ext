import numpy as np
import rpy2.robjects as robjects
#@profile
def save_grm(K, filename):
    print "Saving grm file %s..." % filename
    robjects.r('''
        f <- function(grm, filename) {
            BinFile = file(filename, "wb")
            writeBin(grm, BinFile, size=4)
            close(BinFile)
        }
    ''')
    fun = robjects.globalenv['f']
    fun(robjects.FloatVector(K[np.tril_indices_from(K)]), filename)
#@profile
def save_id(K, filename):
    print "Saving grm id file %s..." % filename
    with open(filename, 'w') as f:
        for i in xrange(K.shape[0]):
            f.write("FAM%d\tIND%d\n" % (i, i))
#@profile
def save_nsnps(nsnps, K, filename):
    print "Saving snps file %s..." % filename
    robjects.r('''
        f <- function(nsnps, filename) {
            BinFile = file(filename, "wb")
            writeBin(nsnps, BinFile, size=4)
            close(BinFile)
        }
    ''')
    fun = robjects.globalenv['f']
    nrep = len(np.tril_indices_from(K)[0])
    fun(robjects.FloatVector([nsnps] * nrep), filename)
#@profile
def save_pheno(filename, y):
    print "Saving phenotype file %s..." % filename
    if np.all([yi in [0.0, 1.0] for yi in y]):
        with open(filename, 'w') as f:
            for i in xrange(y.size):
                f.write("FAM%d IND%d %d\n" % (i, i, y[i]))
    else:
        with open(filename, 'w') as f:
            for i in xrange(y.size):
                f.write("FAM%d IND%d %.16f\n" % (i, i, y[i]))
