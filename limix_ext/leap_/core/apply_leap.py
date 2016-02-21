from __future__ import print_function
import numpy as np
import leap.leapMain as leapMain
import leap.leapUtils as leapUtils
from pysnptools.snpreader.bed import Bed
from pandas import DataFrame, Series
from collections import OrderedDict
import os
from subprocess import call
import tempfile
import shutil
from limix_util.plink_ import create_ped, create_map
from calc_h2 import calc_h2

# def create_bed_files(nfrX, nbgX, y):
#     save_pheno('/Users/horta/tmp/dataset/prefix.phen', y)
#     # save_nsnps(nbgX, K, '/Users/horta/tmp/dataset/prefix.bin')
# def create_ped(filepath, frX, bgX, y):
#     # chromosome 1 = frX
#     # chromosome 2 = bgX
#     n = y.shape[0]
#     p1 = frX.shape[1]
#     p2 = bgX.shape[1]
#
#     l = [('Family ID', Series([0]*n)),
#          ('Individual ID', Series(np.arange(n))),
#          ('Paternal ID', Series([0]*n)),
#          ('Maternal ID', Series([0]*n)),
#          ('Sex', Series([0]*n)),
#          ('Phenotype', Series(np.asarray(y, dtype=int)))]
#
#     ii = 0
#     for X in [frX, bgX]:
#         for i in xrange(X.shape[1]):
#             left = []
#             right = []
#             for j in xrange(X.shape[0]):
#                 if X[j,i] == 0.:
#                     left.append('A')
#                     right.append('A')
#                 elif X[j,i] == 1.:
#                     left.append('A')
#                     right.append('B')
#                 elif X[j,i] == 2.:
#                     left.append('B')
#                     right.append('B')
#                 else:
#                     assert False
#             s1 = Series(left)
#             s2 = Series(right)
#
#             l.append(('SNP_left%d' % ii, s1))
#             l.append(('SNP_right%d' % ii, s2))
#             ii += 1
#
#     d = DataFrame(OrderedDict(l))
#     d.to_csv(filepath + '.ped', sep=' ', header=False, index=False)
#
#     chroms = np.concatenate((np.ones(frX.shape[1], dtype=int),\
#                             2*np.ones(bgX.shape[1], dtype=int))
#                            )
#     l = [('chromosome', chroms),
#          ('SNP ID', np.arange(p1+p2, dtype=int)),
#          ('Genetic distance', np.arange(p1+p2)),
#          ('Base-pair position', np.arange(p1+p2))]
#
#     d = DataFrame(OrderedDict(l))
#     d.to_csv(filepath + '.map', sep=' ', header=False, index=False)

# def create_ped_low_memory(filepath, frX, bgX, y):
#
#     # chromosome 1 = frX
#     # chromosome 2 = bgX
#     n = y.shape[0]
#     p1 = frX.shape[1]
#     p2 = bgX.shape[1]
#
#     with open(filepath + '.ped', 'w') as f:
#
#         for j in xrange(n):
#             genos = []
#             for X in [frX, bgX]:
#                 for i in xrange(X.shape[1]):
#                     if X[j,i] == 0.:
#                         genos.append('A')
#                         genos.append('A')
#                     elif X[j,i] == 1.:
#                         genos.append('A')
#                         genos.append('B')
#                     elif X[j,i] == 2.:
#                         genos.append('B')
#                         genos.append('B')
#                     else:
#                         assert False
#
#             f.write('0 %d 0 0 0 %d ' % (j, y[j]))
#             f.write(' '.join(genos))
#             f.write('\n')
#
#     with open(filepath + '.map', 'w') as f:
#         ii = 0
#         for (c, X) in enumerate([frX, bgX]):
#             for i in xrange(X.shape[1]):
#                 f.write('%d %d %d %d\n' % ((c+1), ii, ii, ii))
#                 ii += 1

def create_phen(filepath, y):
    n = y.shape[0]
    l = [('Family ID', Series([0]*n)),
         ('Individual ID', Series(np.arange(n))),
         ('Phenotype', Series(np.asarray(y, dtype=int)))]
    d = DataFrame(OrderedDict(l))
    d.to_csv(filepath + '.phe', sep=' ', header=False, index=False)

def create_bed(filepath):
    call(["plink", "--file", filepath, "--out", filepath, "--make-bed",
          "--noweb", "--1"])

def apply_this_kinship(G, K, y, prevalence, nsnps_back, cutoff, covariates=None):
    import fastlmm.util.VertexCut as vc
    from probit import probit

    nK = K.copy()

    #These are the indexes of the IIDs to remove
    remove_set = set(np.sort(vc.VertexCut().work(K, cutoff)))
    print('Marking', len(remove_set),
        'individuals to be removed due to high relatedness')

    vinds = set(range(K.shape[0])) - remove_set
    vinds = np.array(list(vinds), int)

    S, U = leapUtils.eigenDecompose(K[np.ix_(vinds, vinds)])
    eigen = dict(XXT=K, arr_1=S, arr_0=U)
    # S, U = leapUtils.eigenDecompose(K[vinds, vinds])
    n = len(vinds)

    h2 = calc_h2(dict(vals=y), prevalence, eigen, keepArr=vinds,
                 h2coeff=1.0, numRemovePCs=10, lowtail=False)

    if h2 >= 1.:
        print("Warning: LEAP found h2",
            "greater than or equal to 1.: %f." % h2)
        h2 = 0.9
    if h2 <= 0.:
        print("Warning: LEAP found h2",
            "smaller than or equal to 0.: %f." % h2)
        h2 = 0.1

    liabs = probit(nsnps_back, n, dict(vals=y), h2, prevalence, U, S,
                   covar=covariates)

    from leap_gwas import gwas
    (stats, pvals) = gwas(nK, G, liabs['vals'], h2, covariate=covariates)
    return (stats, pvals, h2)


def apply_this(frX, bgX, y, prevalence, cutoff, covariates=None):
    # cutoff = 0.05

    outfolder = tempfile.mkdtemp()

    try:
        filepath = os.path.join(outfolder, 'data')
        # create_ped(filepath, frX, bgX, y)
        create_ped_low_memory(filepath, frX, bgX, y)
        create_bed(filepath)
        create_phen(filepath, y)

        bfile = filepath
        phenoFile = bfile+'.phe'
        fg_chrom = 1
        bg_chrom = 2

        bed = Bed(bfile).read().standardize()

        # #Create a bed object excluding SNPs from the current chromosome
        bedBackground = leapUtils.getExcludedChromosome(bfile, fg_chrom)
        # #Create a bed object including only SNPs from the current chromosome
        bedForeground = leapUtils.getExcludedChromosome(bfile, bg_chrom)

        # #1. Find individuals to exclude to eliminate relatedness
        #     (kinship coeff > 0.05)
        indsToKeep = leapUtils.findRelated(bed, cutoff=cutoff)

        # #2. Compute eigendecomposition for the data
        eigenFile = os.path.join(outfolder, 'eigen.npz')
        eigen = leapMain.eigenDecompose(bedBackground, outFile=eigenFile)
        # #3. compute heritability explained by this data

        h2 = leapMain.calcH2(phenoFile, prevalence, eigen, keepArr=indsToKeep,
                             h2coeff=1.0)
        # #4. Compute liabilities explained by this data
        liabs = leapMain.probit(bedBackground, phenoFile, h2, prevalence, eigen, keepArr=indsToKeep, covar=covariates)

        # #5. perform GWAS, using the liabilities as the observed phenotypes
        results_df = leapMain.leapGwas(bedBackground, bedForeground, liabs, h2)

        # frame_list.append(results_df)

        pvals = results_df['PValue'].as_matrix()

        indices = np.asarray(results_df['SNP'].as_matrix(), dtype=long)

        pvals[indices] = pvals
        chi2stats = results_df['chi2stats'].as_matrix()
        chi2stats[indices] = chi2stats

    except Exception as e:
        shutil.rmtree(outfolder)
        raise

    shutil.rmtree(outfolder)
    return (pvals, h2, chi2stats)

def estimate_h2(bgX, y, prevalence, cutoff):
    # cutoff = 0.05

    outfolder = tempfile.mkdtemp()
    chroms = np.ones(bgX.shape[0])

    try:
        filepath = os.path.join(outfolder, 'data')
        # create_ped2(filepath, [bgX], y)
        # create_ped2_low_memory(filepath, [bgX], y)
        create_ped(filepath, y, bgX)
        create_map(filepath, chroms)
        create_bed(filepath)
        create_phen(filepath, y)

        bfile = filepath
        phenoFile = bfile+'.phe'

        bed = Bed(bfile).read().standardize()

        # # #Create a bed object excluding SNPs from the current chromosome
        # bedBackground = leapUtils.getExcludedChromosome(bfile, fg_chrom)
        # # #Create a bed object including only SNPs from the current chromosome
        # bedForeground = leapUtils.getExcludedChromosome(bfile, bg_chrom)

        # #1. Find individuals to exclude to eliminate relatedness
        #     (kinship coeff > 0.05)
        indsToKeep = leapUtils.findRelated(bed, cutoff=cutoff)

        # #2. Compute eigendecomposition for the data
        eigenFile = os.path.join(outfolder, 'eigen.npz')
        eigen = leapMain.eigenDecompose(bed, outFile=eigenFile)
        # #3. compute heritability explained by this data

        h2 = leapMain.calcH2(phenoFile, prevalence, eigen, keepArr=indsToKeep,
                             h2coeff=1.0)

    except Exception as e:
        shutil.rmtree(outfolder)
        h2 = 0.
        raise

    shutil.rmtree(outfolder)
    return (h2, indsToKeep)
