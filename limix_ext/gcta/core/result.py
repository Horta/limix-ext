import re

import numpy as np


class Result(object):
    def __init__(self, filename):
        with open(filename, 'r') as f:
            f.readline()

            line = f.readline().split('\t')
            self.var_g = float(line[1])
            self.var_g_se = float(line[2])

            line = f.readline().split('\t')
            self.var_n = float(line[1])
            self.var_n_se = float(line[2])

            line = f.readline().split('\t')
            self.var_total = float(line[1])
            self.var_total_se = float(line[2])

            f.readline()
            f.readline()
            line = f.readline()

            match = re.match( r'.* in the sample = (.*); User-specified disease prevalence = (.*)\).*', line)
            self.prevalence_in_sample = float(match.group(1))
            self.prevalence_specified = float(match.group(2))

            line = f.readline()

            self._heritability_liability_scale = float(line.split('\t')[1])

            if np.abs(self.var_g + self.var_n - self.var_total) > 1e-5:
                raise Exception("Total variance differ from var_g + var_n: %.6f." % np.abs(self.var_g + self.var_n - self.var_total))

    @property
    def heritability_liability_scale(self):
        return self._heritability_liability_scale

    @property
    def heritability_observed_scale(self):
        return self.var_g / self.var_total

    # def __str__(self):
    #     return tabulate([['genetic var', self.var_g],
    #                      ['noise var', self.var_n],
    #                      ['total var', self.var_total],
    #                      ['heritability', self.heritability]])

class ResultContinuous(object):
    def __init__(self, filename):
        with open(filename, 'r') as f:
            f.readline()

            line = f.readline().split('\t')
            self.var_g = float(line[1])
            self.var_g_se = float(line[2])

            line = f.readline().split('\t')
            self.var_n = float(line[1])
            self.var_n_se = float(line[2])

            line = f.readline().split('\t')
            self.var_total = float(line[1])
            self.var_total_se = float(line[2])

            line = f.readline().split('\t')
            self._heritability_liability_scale = float(line[1])

    @property
    def heritability_liability_scale(self):
        return self._heritability_liability_scale

    @property
    def heritability_observed_scale(self):
        return self.var_g / self.var_total

    # def __str__(self):
    #     return tabulate([['genetic var', self.var_g],
    #                      ['noise var', self.var_n],
    #                      ['total var', self.var_total],
    #                      ['heritability', self.heritability]])
