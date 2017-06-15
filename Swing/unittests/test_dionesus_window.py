__author__ = 'Justin Finkle'
__email__ = 'jfinkle@u.northwestern.edu'

import unittest
import Swing
import pandas as pd
import numpy as np
import numpy.testing as npt
import pdb

class TestDionesusWindow(unittest.TestCase):
    def setUp(self):
        file_path = "../../data/dream4/insilico_size10_1_timeseries.tsv"
        gene_start_column = 1
        time_label = "Time"
        separator = "\t"
        gene_end = None

        self.roller = Swing.Swing(file_path, gene_start_column, gene_end, time_label, separator,
                                    window_type="Dionesus")

        self.roller.create_windows()
        self.test_dionesus = self.roller.window_list[0]
        self.permutes = 10

    def test_make_edge_table(self):
        pass

    def test_sort_edges(self):
        pass

    def test_generate_results_table(self):
        pass

    def test_rank_results(self):
        pass

    def test_run_permutation_test(self):
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
        # All coefficients and vip scores should be nonzero except along the diagonal
        expected_non_zero = len(self.test_dionesus.genes)**2-len(self.test_dionesus.genes)
        calc_coeffs, calc_vip = self.test_dionesus.get_coeffs()
        calc_non_zero = np.count_nonzero(calc_coeffs)
        calc_vip_non_zero = np.count_nonzero(calc_vip)
        self.assertTrue(expected_non_zero == calc_non_zero)
        self.assertTrue(expected_non_zero == calc_vip_non_zero)

    def test_run_permutation_test(self):
        # The model must first be initialized
        self.test_dionesus.initialize_params()
        self.test_dionesus.fit_window()
        self.test_dionesus.run_permutation_test(self.permutes)
        n_genes = len(self.test_dionesus.genes)
        self.assertTrue(self.test_dionesus.permutation_means.shape == (n_genes, n_genes))
        self.assertTrue(self.test_dionesus.permutation_sd.shape == (n_genes, n_genes))

    def test_make_edge_table(self):
        self.test_dionesus.initialize_params()
        self.test_dionesus.fit_window()
        self.test_dionesus.run_permutation_test(self.permutes)
        #self.test_dionesus.generate_results_table()
        self.test_dionesus.make_edge_table()
        old_order = self.test_dionesus.results_table['regulator-target'].values
        self.test_dionesus.sort_edges()
        new_order = self.test_dionesus.results_table['regulator-target'].values
        
        self.assertFalse(np.array_equal(old_order, new_order))

if __name__ == '__main__':
    unittest.main()
