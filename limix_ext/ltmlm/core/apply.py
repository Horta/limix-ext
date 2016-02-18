import numpy as np
from os.path import join
from gwarped.util.file import open_temp_folder
from gwarped.util.linalg import economic_QS, ddot
import os
import subprocess
from subprocess import check_output
from subprocess import CalledProcessError

_letras = np.array([['A', 'C'],
 ['A', 'G'],
 ['A', 'T'],
 ['C', 'A'],
 ['G', 'A'],
 ['T', 'A'],
 ['G', 'T']])

def _write_geno(bgX, folder, prefix):
    fname = "%s.geno" % prefix
    # np.savetxt(join(folder, fname), bgX.T, '%.16f', ' ')
    np.savetxt(join(folder, fname), bgX.T, '%d', '')
    with open(join(folder, fname), 'rb+') as filehandle:
        filehandle.seek(-1, os.SEEK_END)
        filehandle.truncate()
    return fname

def _write_snp(bgX, folder, prefix):

    nsnps = bgX.shape[1]
    fname = "%s.snp" % prefix
    nletras = _letras.shape[0]
    with open(join(folder,  fname), "w") as f:
        for i in xrange(nsnps):

            ii = i % nletras
            fina = "%s %s" % (_letras[ii, 0], _letras[ii, 1])

            f.write("rs%d 11 0.0 %d %s" % (i, i*100000, fina))
            if i < nsnps - 1:
                f.write("\n")
    return fname

def _write_cov(folder, K, prefix):
    fname = "%s.cov" % prefix
    np.savetxt(join(folder,  fname), K, '%.8f', ',')
    return fname

def _write_ind(y, folder, prefix):

    n = y.shape[0]
    fname = "%s.ind" % prefix
    with open(join(folder,  fname), "w") as f:
        for i in xrange(n):
            status = 'Case' if y[i] == 1 else 'Control'
            sex = 'F' if i % 2 == 0 else 'M'
            f.write("SAMPLE%d %s %s" % (i, sex, status))
            # if i < n - 1:
            f.write("\n")
    return fname

#create a parameter file to be used by convertf that looks like this:
##genotypename:    exampleForGRM.geno
##snpname:         exampleForGRM.snp
##indivname:       example.ind
##outputformat:    PACKEDPED
##genotypeoutname: example.bed
##snpoutname:      example.bim
##indivoutname:    example.fam
def _write_eig2bed(folder, genoFileName, snpFileName, indFileName,
                  prefixName):
    fname = "eigToBed."+prefixName+".par"
    bedFileName = join(folder, prefixName+".bed")
    bimFileName = join(folder, prefixName+".bim")
    famFileName = join(folder, prefixName+".fam")
    with open(join(folder, fname), "w") as f:
        f.write("genotypename:    "+join(folder, genoFileName)+"\r\n")
        f.write("snpname:         "+join(folder, snpFileName)+"\r\n")
        f.write("indivname:       "+join(folder, indFileName)+"\r\n")
        f.write("outputformat:    PACKEDPED\r\n")
        f.write("genotypeoutname: "+bedFileName+"\r\n")
        f.write("snpoutname:      "+bimFileName+"\r\n")
        f.write("indivoutname:    "+famFileName+"\r\n")
    return fname

#create a parameter file to be used by LTMLM that looks like this:
##genotypename:  example.geno
##snpname:       example.snp
##indivname:     example.ind
##covname:     exampleGCTA.cov
##outputname:    example.chisq
##heritname:       example.herit
##pmlname:       example.pml
def _write_chi2file(folder, genof, snpf, indf, covf, prefix):

    fname = "chisq."+prefix+".par"

    genof = join(folder, genof)
    snpf = join(folder, snpf)
    indf = join(folder, indf)
    covf = join(folder, covf)

    outf = join(folder, prefix + ".chisq")
    heritf = join(folder, prefix + ".herit")
    pmlf = join(folder, prefix + ".pml")

    conf_str = ""
    conf_str += "genotypename:  "+genof+"\n"
    conf_str += "snpname:       "+snpf+"\n"
    conf_str += "indivname:     "+indf+"\n"
    conf_str += "covname:     "+covf+"\n"
    conf_str += "outputname:    "+outf+"\n"
    conf_str += "heritname:       "+heritf+"\n"
    conf_str += "pmlname:       "+pmlf+"\n"

    with open(join(folder, fname), "w") as f:
        f.write(conf_str)

    print "LTMLM configuration file:"
    print conf_str
    print "-------------------"
    return fname

def _convert2bed(folder, eig2bedf):
    import os
    cfolder = os.path.dirname(os.path.realpath(__file__))
    conv = join(cfolder, "convertf")

    from subprocess import call
    fname = join(folder, eig2bedf)
    return_code = call([conv + " -p %s" % fname], shell=True)
    if return_code != 0:
        raise Exception("Error while running convertf for file %s." % eig2bedf)

def _apply_gcta(folder, prefix):
    fname = join(folder, prefix)
    from subprocess import call
    cmd = "gcta64 --bfile "+fname+" --make-grm --out "+fname
    return_code = call(cmd, shell=True)
    if return_code != 0:
        raise Exception("Error while running gcta64 for file %s." % fname)

def _convert_gcta2cov(folder, indf, prefix):
    indf = join(folder, indf)
    binf = join(folder, prefix + ".grm.bin")
    covf = prefix + "GCTA.cov"

    import os
    cfolder = os.path.dirname(os.path.realpath(__file__))
    rscript = join(cfolder, "convertGctaGrmToCov.R")

    cmd = "R CMD BATCH '--args "+binf+" "+indf+" "+\
           join(folder, covf)+"' %s" % rscript

    try:
        print check_output(cmd, shell=True)
    except CalledProcessError:
        print("Error while running R/convertGctaGrmToCov.R "+
                        "for files %s and %s." % (indf, binf))
        raise

    return covf

def _run_ltmlm(folder, threshold, chi2f, heritMax):
    from subprocess import call
    chi2f = join(folder, chi2f)

    import os
    cfolder = os.path.dirname(os.path.realpath(__file__))

    cmd = join(cfolder, "LTMLM") + " -t "+threshold+"   -p "\
           +chi2f+"  -H "+heritMax

    print "Shell command:", cmd
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         shell=True)
    output, output_err = p.communicate()
    return_code = 0 if output_err is None else 1

    import re
    output = re.sub(r"Diagonal component of Tinv.*\n[^\n]*\n", "", output)
    output = re.sub(r"The PML multivatriate\n[^\n]*\n", "", output)
    output = re.sub(r"About to print: psuedoFam\n[^\n]*\n", "", output)
    print output

    # return_code = call(cmd, shell=True, stdout=out, stderr=err)
    if return_code != 0:
        raise Exception("Error while running LTMLM for file %s." % chi2f)

def _read_heritability(herif):
    with open(herif, "r") as f:
        f.readline()
        line = f.readline()
        return float(line.split(":")[1])

def _read_chi2(chi2f):
    chi2vals = []

    with open(chi2f, "r") as f:
        f.readline()
        while True:
            line = f.readline()
            if not line: break
            vals = line.split(",")
            if len(vals) == 2:
                chi2vals.append(float(vals[1]))

    return np.array(chi2vals)

def estimate_h2(K, y, prevalence):
    X = np.random.randint(0, 3, (K.shape[0], 2))
    (h2, _, _) = test_ltmlm(X, K, y, prevalence)
    return h2

def old_test_ltmlm(X, K, y, prevalence):
    (Q, S) = economic_QS(K=K)
    L = ddot(Q, np.sqrt(S), left=False)
    return test_ltmlm_geno_bg(X, L, y, prevalence)

def test_ltmlm_geno_bg(fgX, bgX, y, prevalence):
    import scipy as sp
    import scipy.stats

    threshold = str(-sp.stats.norm.ppf(prevalence))
    heritMax = '1.0'

    bgX = np.asarray(bgX, dtype=float)
    fgX = np.asarray(fgX, dtype=float)
    y = np.asarray(y, dtype=int)
    # def or_(x1, x2, x3):
        # return np.logical_or(np.logical_or(x1, x2), x3)
    # assert np.all(or_(0 == bgX, 1 == bgX, 2 == bgX))

    prefix = "example"
    with open_temp_folder() as folder:
        indf = _write_ind(y, folder, prefix)

        genof_bg = _write_geno(bgX, folder, prefix + "_for_grm")
        snpf_bg = _write_snp(bgX, folder, prefix + "_for_grm")

        genof_fg = _write_geno(fgX, folder, prefix)
        snpf_fg = _write_snp(fgX, folder, prefix)

        eig2bedf = _write_eig2bed(folder, genof_bg, snpf_bg, indf, prefix)
        _convert2bed(folder, eig2bedf)
        _apply_gcta(folder, prefix)
        covf = _convert_gcta2cov(folder, indf, prefix)

        chi2f = _write_chi2file(folder, genof_bg, snpf_bg, indf,
                                covf, prefix)

        _run_ltmlm(folder, threshold, chi2f, heritMax)

        herif = prefix + ".herit"
        h2 = _read_heritability(join(folder, herif))

        chi2vals = _read_chi2(join(folder, prefix + ".chisq"))

    h2 = max(0., h2)
    pvals = sp.stats.chi2(df=1).sf(chi2vals)
    stats = chi2vals

    return (h2, pvals, stats)



def test_ltmlm(X, K, y, prevalence):
    import scipy as sp
    import scipy.stats

    threshold = str(-sp.stats.norm.ppf(prevalence))
    heritMax = '1.0'

    y = np.asarray(y, dtype=int)

    prefix = "example"
    with open_temp_folder() as folder:
        indf = _write_ind(y, folder, prefix)

        genof_fg = _write_geno(X, folder, prefix)
        snpf_fg = _write_snp(X, folder, prefix)

        covf = _write_cov(folder, K, prefix)

        chi2f = _write_chi2file(folder, genof_fg, snpf_fg, indf,
                                covf, prefix)

        _run_ltmlm(folder, threshold, chi2f, heritMax)

        herif = prefix + ".herit"
        h2 = _read_heritability(join(folder, herif))

        chi2vals = _read_chi2(join(folder, prefix + ".chisq"))

    h2 = max(0., h2)
    pvals = sp.stats.chi2(df=1).sf(chi2vals)
    stats = chi2vals

    return (h2, pvals, stats)


if __name__ == '__main__':
    np.random.seed(0)
    X = np.random.randint(0, 3, (10, 4))
    G = np.random.randint(0, 3, (10, 300))
    K = np.dot(G, G.T) / float(G.shape[1])
    y = np.random.randint(0, 2, 10)
    new_test_ltmlm(X, K, y, 0.1)
