__author__ = 'Justin Finkle'
__email__ = 'jfinkle@u.northwestern.edu'

import unittest
import Swing
import pandas as pd
import numpy as np
import numpy.testing as npt
import pdb

class TestRFRWindow(unittest.TestCase):
    def setUp(self):
        file_path = "../../data/dream4/insilico_size10_1_timeseries.tsv"
        gene_start_column = 1
        time_label = "Time"
        separator = "\t"
        gene_end = None

        self.roller = Swing.Swing(file_path, gene_start_column, gene_end, time_label, separator, window_type = "RandomForest")
        index = 0
        window_info = {"time_label": self.roller.time_label, "gene_start": self.roller.gene_start,
                       "gene_end": self.roller.gene_end, 'nth_window': index}
        self.test_rfr = Swing.RandomForestRegressionWindow(self.roller.get_window_raw(index),window_info)
        self.jobs = 1
        self.trees = 10
        self.permutes = 10

    def test_make_edge_table(self):
        pass

    def test_permute_coeffs(self):
        pass

    def test_calc_p_value(self):
        pass

    def test_initialize_params(self):
        pass

    def test_fit_window(self):
        pass

    def test_fitstack_coeffs(self):
        pass

    def test_get_coeffs(self):
        pass

    def test_get_coeffs(self):
        # With alpha at 0 everything should be nonzero except the diagonal values
        expected_non_zero = len(self.test_rfr.genes)**2-len(self.test_rfr.genes)
        calc_coeffs = self.test_rfr.get_coeffs(self.trees, n_jobs=self.jobs)
        calc_non_zero = np.count_nonzero(calc_coeffs)
        self.assertTrue(expected_non_zero == calc_non_zero)

    def test_run_permutation_test(self):
        # The model must first be initialized
        self.test_rfr.initialize_params(self.trees)
        self.test_rfr.fit_window(self.jobs)
        self.test_rfr.run_permutation_test(self.permutes, self.jobs)
        n_genes = len(self.test_rfr.genes)
        self.assertTrue(self.test_rfr.permutation_means.shape == (n_genes, n_genes))
        self.assertTrue(self.test_rfr.permutation_sd.shape == (n_genes, n_genes))

    def test_make_edge_table(self):
        self.test_rfr.initialize_params(self.trees)
        self.test_rfr.fit_window(self.jobs)
        self.test_rfr.run_permutation_test(self.permutes, self.jobs)
        pdb.set_trace()
        self.test_rfr.make_results_table()
        old_order = self.test_rfr.results_table['regulator-target'].values
        self.test_rfr.sort_edges()
        new_order = self.test_rfr.results_table['regulator-target'].values
        self.assertFalse(np.array_equal(old_order, new_order))

if __name__ == '__main__':
    unittest.main()
