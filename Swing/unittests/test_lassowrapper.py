__author__ = 'Justin Finkle'
__email__ = 'jfinkle@u.northwestern.edu'

import unittest
from Swing.util.linear_wrapper import LassoWrapper
from Swing.util import linear_wrapper
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class TestLassoWrapper(unittest.TestCase):
    def setUp(self):
        # Load data
        file_path = "../../data/dream4/insilico_size10_1_timeseries.tsv"
        df = pd.DataFrame.from_csv(file_path, sep='\t')
        times = df.index.values[~np.isnan(df.index.values)]
        times_set = set(times)
        data = df.values

        # Remove NaNs from TSV
        data = data[~np.isnan(data).all(axis=1)]
        self.lassowrapper = LassoWrapper(data)

    def test_get_coef(self):
        #todo: Find a functioning test case
        coef = np.array([[0,1,1],[1,0,1], [1,1,0]])
        n_features = len(coef)
        x_step = np.ones(n_features)
        data = x_step.copy()
        n_steps = 100
        for ii in range(n_steps):
            x_step = np.dot(coef, x_step)
            data = np.vstack((data, x_step))
        n_samples = 50
        start = np.random.randint(0, len(data)/2)
        sample = data[start:start+n_samples]

        lasso = LassoWrapper(sample)

    def test_get_max_alpha(self):
        alpha_precision = 1e-9
        max_alpha = self.lassowrapper.get_max_alpha()
        num_coef_at_max_alpha = np.count_nonzero(self.lassowrapper.get_coeffs(max_alpha))
        num_coef_less_max_alpha = np.count_nonzero(self.lassowrapper.get_coeffs(max_alpha-alpha_precision))
        self.assertTrue(num_coef_at_max_alpha == 0)
        self.assertTrue(num_coef_less_max_alpha > 0)

    def test_cross_validate_alpha(self):
        data = self.lassowrapper.data.copy()
        alpha_range = np.linspace(0, self.lassowrapper.get_max_alpha())
        q_list = [self.lassowrapper.cross_validate_alpha(alpha) for alpha in alpha_range]

    def test_sum_of_squares(self):
        data = np.reshape(np.arange(6), (3,2))
        expected_ss = 16
        calc_ss = linear_wrapper.sum_of_squares(data)
        self.assertEqual(calc_ss, expected_ss)

if __name__ == '__main__':
    unittest.main()
"""
Old test code

if __name__ == '__main__':
    # Load data
    file_path = "../../data/dream4/insilico_size10_1_timeseries.tsv"
    df = pd.DataFrame.from_csv(file_path, sep='\t')
    times = df.index.values[~np.isnan(df.index.values)]
    times_set = set(times)
    genes = df.columns.values
    replicates = len(times)/float(len(times_set))
    data = df.values

    # Remove NaNs from TSV
    data = data[~np.isnan(data).all(axis=1)]

    # Initialize lassowrapper
    lasso_wrapper = LassoWrapper(data)
    alpha = 0.0
    coef = lasso_wrapper.get_coeffs(alpha)
    m = lasso_wrapper.get_max_alpha()
    print m

"""