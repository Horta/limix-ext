import logging
import os
import shutil
import subprocess
import tempfile

import numpy as np

from limix_math import isint_alike
from limix_tool import plink_
from limix_util.path import temp_folder
from limix_util.sys import platform
from result import Result


def _create_their_kinship(bed_folder, prefix):
    logger = logging.getLogger(__name__)

    if platform() != 'linux':
        raise OSError('The %s platform is currently not supported.'
                      % platform())

    logger.debug("Creating GCTA kinship matrix.")
    cwd = os.path.abspath(os.path.dirname(__file__))

    cmd = ['./gcta64', '--bfile', os.path.join(bed_folder, prefix),
           '--autosome', '--make-grm',
           '--out', os.path.join(bed_folder, prefix)]

    logger.debug("Running shell list %s.", str(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=cwd)
    out, err = p.communicate()
    logger.debug(out)
    if len(err) > 0:
        logger.warn(err)


def prepare_for_their_kinship(folder, prefix, G, y, chromosome):
    logger = logging.getLogger(__name__)
    logger.debug("Preparing for their kinship.")
    G = G.astype(int)
    chromosome = chromosome.astype(int)

    filepath = os.path.join(folder, prefix)
    plink_.create_ped(filepath + '.ped', y, G)
    plink_.create_map(filepath + '.map', chromosome)
    plink_.create_phen(filepath + '.phe', y)
    plink_.create_bed(filepath)

    _create_their_kinship(folder, prefix)


def _run_gcta(prefix, phen_filename, preva, diag_one=False, nthreads=1):
    outfolder = tempfile.mkdtemp()
    cmd = ['./gcta64', '--grm', prefix, '--pheno', phen_filename,
           '--prevalence', '%.7f' % preva,
           '--reml', '--out', os.path.join(outfolder, 'result'), '--thread-num', str(nthreads)]
    if diag_one:
        cmd.append('--reml-diag-one')

    logger = logging.getLogger(__name__)
    logger.debug("Running shell list %s.", str(cmd))

    cwd = os.path.abspath(os.path.dirname(__file__))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=cwd)

    out, err = p.communicate()
    logger.debug(out)
    if len(err) > 0:
        logger.warn(err)

    r = Result(os.path.join(outfolder, 'result.hsq'))
    shutil.rmtree(outfolder)

    return r


def kinship_estimation(G, y, prevalence):
    chromosome = np.ones(G.shape[1])
    with temp_folder() as outdir:
        prepare_for_their_kinship(outdir, 'data', G, y, chromosome)
        g = np.fromfile(os.path.join(outdir, 'data.grm.bin'), dtype='f4')
        k = g.shape[0]
        n = (-1 + np.sqrt(1 + 8 * k)) / 2.
        n = int(n)
        K = np.empty((n, n))
        K[np.triu_indices_from(K)] = g
        K.T[np.triu_indices_from(K)] = g
    return K


def run_gcta(bed_folder, prefix, prevalence, diag_one=False):
    phen_filename = os.path.join(prefix + '.phe')

    result = _run_gcta(os.path.join(bed_folder, prefix), os.path.join(bed_folder, phen_filename),
                       prevalence,
                       diag_one=diag_one, nthreads=1)
    return result


def estimate_h2_gcta(G, y, prevalence):

    if not isint_alike(G):
        raise ValueError('The genetic markers matrix must contain only integer' +
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
