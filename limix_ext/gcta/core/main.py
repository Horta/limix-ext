import os
import numpy as np
import subprocess
import shutil
from result import Result
import tempfile
from limix_util.plink_ import create_ped
from limix_util.plink_ import create_map
from limix_util.plink_ import create_phen
from limix_util.plink_ import create_bed
from limix_util.system_ import platform
from limix_util.array_ import isint_alike

def _create_their_kinship(bed_folder, prefix):
    if platform() != 'linux':
        raise OSError('The %s platform is currently not supported.'
                      % platform())

    print "Creating their kinship matrix..."
    cwd = os.path.abspath(os.path.dirname(__file__))

    shell_list = ['./gcta64', '--bfile', os.path.join(bed_folder, prefix),
                  '--autosome', '--make-grm',
                  '--out', os.path.join(bed_folder, prefix)]
    print "Running shell list %s..." % str(shell_list)
    p = subprocess.Popen(shell_list, cwd=cwd)
    p.wait()

def prepare_for_their_kinship(folder, prefix, G, y, chromosome):
    print "Preparing for their kinship..."
    G = G.astype(int)
    chromosome = chromosome.astype(int)

    filepath = os.path.join(folder, prefix)
    create_ped(filepath + '.ped', y, G)
    create_map(filepath + '.map', chromosome)
    create_phen(filepath + '.phe', y)
    create_bed(filepath)

    _create_their_kinship(folder, prefix)

def _run_gcta(prefix, phen_filename, preva, diag_one=False, nthreads = 1):
    outfolder = tempfile.mkdtemp()
    shell_list = ['gcta64', '--grm', prefix, '--pheno', phen_filename,
                  '--prevalence', '%.7f' % preva,
                  '--reml', '--out', os.path.join(outfolder, 'result'), '--thread-num', str(nthreads)]
    if diag_one:
        shell_list.append('--reml-diag-one')

    print "Running shell list %s..." % str(shell_list)
    p = subprocess.Popen(shell_list)
    p.wait()
    r = Result(os.path.join(outfolder, 'result.hsq'))
    shutil.rmtree(outfolder)
    return r

def run_gcta(bed_folder, prefix, prevalence, diag_one=False):
    phen_filename = os.path.join(prefix + '.phe')

    result = _run_gcta(os.path.join(bed_folder, prefix), os.path.join(bed_folder, phen_filename),
                       prevalence,
                       diag_one = diag_one, nthreads = 1)
    return result

def estimate_h2_gcta(G, y, prevalence):

    if not isint_alike(G):
        raise ValueError('The genetic markers matrix must contain only integer'+
                         'values.')

    outdir = tempfile.mkdtemp()

    try:
        chromosome = np.ones(G.shape[1])
        prepare_for_their_kinship(outdir, 'data', G, y, chromosome)
        result = run_gcta(outdir, 'data', prevalence)
    except Exception as e:
        shutil.rmtree(outdir)
        print str(e)
        raise

    shutil.rmtree(outdir)
    return result.heritability_liability_scale
