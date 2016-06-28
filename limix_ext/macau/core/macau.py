import os
import logging
import numpy as np
import subprocess
from pandas import DataFrame
import tempfile

def _run_scan(y_fp, ntrials_fp, K_fp, predictor_fp):
    outfolder = tempfile.mkdtemp()
    cmd = ['./macau', '-g', y_fp, '-t', ntrials_fp,
                  '-p', predictor_fp,
                  '-k', K_fp, '-o', outfolder]

    logger = logging.getLogger(__name__)
    logger.debug("Running shell list %s.", str(cmd))
    cwd = os.path.abspath(os.path.dirname(__file__))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=cwd)
    out, err = p.communicate()

    logger.debug(out)
    if len(err) > 0:
        logger.warn(err)

    return None

def run_scan(y, ntrials, K, predictor):
    outfolder = tempfile.mkdtemp()

    np.savetxt(os.path.join(outfolder, 'K.txt'), K)

    y_df = DataFrame()
    y_df.index = ['site1']
    y_df.index.name = 'site'

    ntrials_df = DataFrame()
    ntrials_df.index = ['site1']
    ntrials_df.index.name = 'site'

    for i in range(K.shape[0]):
        y_df['sample_%d' % i] = int(y[i])
        ntrials_df['sample_%d' % i] = int(ntrials[i])

    y_df.to_csv(os.path.join(outfolder, 'y.txt'), sep=" ")
    ntrials_df.to_csv(os.path.join(outfolder, 'ntrials.txt', sep=" "))

    predictor = np.atleast_2d(predictor)
    if predictor.shape[1] > 1:
        predictor = predictor.T

    np.savetxt(os.path.join(outfolder, 'predictor.txt'),
               np.atleast_2d(predictor))

    return _run_scan(os.path.join(outfolder, 'y.txt'),
                     os.path.join(outfolder, 'ntrials.txt'),
                     os.path.join(outfolder, 'K.txt'),
                     os.path.join(outfolder, 'predictor.txt'))
