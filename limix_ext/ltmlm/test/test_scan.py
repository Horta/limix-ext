import numpy as np
import unittest
from limix_ext.ltmlm.scan import scan
from limix_util.system_ import platform

class TestScan(unittest.TestCase):
    def setUp(self):
        pass

    @unittest.skipUnless(platform() == "linux", "requires Linux")
    def test_bernoulli(self):
        random = np.random.RandomState(981)
        n = 500
        p = n+4

        M = np.ones((n, 1)) * 0.4
        Ginit = random.randint(3, size=(n, p))
        G = np.asarray(Ginit, dtype=float)
        G -= G.mean(axis=0)
        G /= G.std(axis=0)
        G /= np.sqrt(p)

        K = np.dot(G, G.T)
        Kg = K / K.diagonal().mean()
        K = 0.5*Kg + 0.5*np.eye(n)
        K = K / K.diagonal().mean()

        z = random.multivariate_normal(M.ravel(), K)
        y = np.zeros_like(z)
        y[z>0] = 1.
        prevalence = 0.5

        X = Ginit[:,:int(Ginit.shape[1]/2)]

        pvals = scan(y, X, Kg, prevalence)

        opvals = [ 0.57518908,  0.83497672,  0.62043047,  0.76377594,
                   0.71293504,  0.1431161,
                   0.39383587,  0.52709649,  0.72885669,  0.66417552,
                   0.71505313,  0.9812729,
                   0.24055988,  0.50900454,  0.64838761,  0.82242069,
                   0.77315745,  0.6944050,
                   0.11073432,  0.44805439,  0.07482495,  0.32267304,
                   0.16660047,  0.4097000,
                   0.95352217,  0.12892675,  0.10550449,  0.68860379,
                   0.3554866 ,  0.9512225,
                   0.66509699,  0.13170771,  0.67827865,  0.53538683,
                   0.25081831,  0.0206677,
                   0.30126867,  0.65014529,  0.04948836,  0.98376624,
                   0.88349989,  0.4374463,
                   0.13149378,  0.13775762,  0.66999669,  0.24313754,
                   0.94113062,  0.0010821,
                   0.40968798,  0.69420956,  0.20172656,  0.37929058,
                   0.37289124,  0.770916,
                   0.06767916,  0.68490462,  0.12618592,  0.92237459,
                   0.61407948,  0.4025677,
                   0.21725364,  0.95533512,  0.6291097 ,  0.61760014,
                   0.0049632 ,  0.1650596,
                   0.02395179,  0.76917066,  0.33135194,  0.37535127,
                   0.1172497 ,  0.2354701,
                   0.61055124,  0.42689132,  0.71413582,  0.35384225,
                   0.86029216,  0.073270,
                   0.70998493,  0.32893502,  0.13940271,  0.4786691 ,
                   0.46586828,  0.0251495,
                   0.08884761,  0.79282758,  0.32343288,  0.50816151,
                   0.90544395,  0.120631,
                   0.35118935,  0.01142883,  0.98334827,  0.13460803,
                   0.78451449,  0.6593936,
                   0.29067551,  0.3274034 ,  0.23526401,  0.89300236,
                   0.72948503,  0.061508,
                   0.18526693,  0.81805714,  0.04247237,  0.28308491,
                   0.10709459,  0.8232988,
                   0.28072432,  0.71701166,  0.61937215,  0.65955385,
                   0.08082718,  0.3551155,
                   0.44933531,  0.80040357,  0.60175704,  0.41396375,
                   0.95789398,  0.3430285,
                   0.4158375 ,  0.28816602,  0.39397479,  0.73244553,
                   0.84643018,  0.3468320,
                   0.00356866,  0.345269  ,  0.33306011,  0.68740317,
                   0.17449219,  0.3565933,
                   0.39321531,  0.73052187,  0.36055736,  0.18937156,
                   0.3141852 ,  0.3071989,
                   0.01466462,  0.09510853,  0.88036242,  0.18093469,
                   0.67852762,  0.0277931,
                   0.62367648,  0.74207185,  0.39499693,  0.1849754 ,
                   0.05217495,  0.5957602,
                   0.63869725,  0.061089  ,  0.67825482,  0.71501122,
                   0.39958095,  0.2919928,
                   0.91212937,  0.20046822,  0.44780567,  0.50704138,
                   0.5558943 ,  0.3719591,
                   0.49329372,  0.33439615,  0.334865  ,  0.09119907,
                   0.08477206,  0.7650928,
                   0.37106856,  0.0580772 ,  0.99853383,  0.13609813,
                   0.06702645,  0.8980513,
                   0.41177153,  0.69332164,  0.73058737,  0.88503287,
                   0.31317934,  0.3545231,
                   0.62033572,  0.19770732,  0.73168292,  0.04451209,
                   0.16132308,  0.3489798,
                   0.34768646,  0.73765324,  0.91787672,  0.36165013,
                   0.87766786,  0.2353238,
                   0.86170319,  0.1461864 ,  0.93376031,  0.86085241,
                   0.16658496,  0.8863172,
                   0.25111137,  0.23810136,  0.74202822,  0.46269477,
                   0.58071759,  0.1282747,
                   0.21306902,  0.90983468,  0.55904583,  0.82136945,
                   0.43697128,  0.3826599,
                   0.34888321,  0.55892381,  0.87828777,  0.51172949,
                   0.12664212,  0.2637454,
                   0.63754978,  0.80239226,  0.35582434,  0.84377908,
                   0.43339584,  0.8061133,
                   0.791743  ,  0.208912  ,  0.56928578,  0.4791418 ,
                   0.85789012,  0.1664708,
                   0.9207093 ,  0.338848  ,  0.31386535,  0.66498625,
                   0.17551105,  0.9199471,
                   0.42464333,  0.19876713,  0.76214933,  0.0372173 ,
                   0.5064398 ,  0.5691594,
                   0.43700277,  0.41239101,  0.35353029,  0.4896756 ,
                   0.11415688,  0.046315,
                   0.37763583,  0.40212086,  0.95477202,  0.50713066,
                   0.90777116,  0.9084403 ]
        opvals = np.asarray(opvals, float)
        np.testing.assert_almost_equal(pvals, opvals, decimal=4)

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()