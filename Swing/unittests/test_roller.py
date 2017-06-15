import unittest
import Swing
import numpy as np
import pandas as pd
import numpy.testing as npt
import matplotlib.pyplot as plt

class TestRoller(unittest.TestCase):
    def setUp(self):
        # Setup a different roller
        file_path = "../../data/dream4/insilico_size10_1_timeseries.tsv"
        gene_start_column = 1
        time_label = "Time"
        separator = "\t"
        gene_end = None

        self.dream_roller = Swing.Swing(file_path, gene_start_column, gene_end, time_label, separator, window_type = "Lasso")

        # Only make 2 windows, so that that testing doesn't take forever
        self.dream_roller.set_window(self.dream_roller.overall_width-1)

    def test_get_n_windows(self):
        pass

    def test_get_window_raw(self):
        pass

    def test_remove_blank_rows(self):
        pass

    def test_get_n_genes(self):
        pass

    def test_create_windows(self):
        "Change this to test if the lags, min_lag, etc are valid" 

        self.dream_roller.create_windows()
        correct_n_windows = self.dream_roller.get_n_windows()
        n_windows = len(self.dream_roller.window_list)
        self.assertTrue(correct_n_windows == n_windows)
    
    def test_check_lags(self):
        pass

    def test_strip_dataframe(self, dataframe):
        pass

    def test_get_window_data(self, index, explanatory_indices):
        pass

    def test_get_window_object(self):
        print self.dream_roller.window_width
        print self.dream_roller.overall_width
        print self.dream_roller.get_n_windows()
        print self.dream_roller.get_window_raw(self.dream_roller.current_step)
        print self.dream_roller.get_window(self.dream_roller.current_step).head()
    
    def test_initialize_windows(self):
        self.dream_roller.create_windows()
        self.dream_roller.initialize_windows()
        for window in self.dream_roller.window_list:
            self.assertTrue(window.alpha is not None)
            self.assertTrue(window.beta_coefficients is not None)

    def test_rank_windows(self):
        self.dream_roller.create_windows()
        self.dream_roller.initialize_windows()
        self.dream_roller.rank_windows(10, 10)
        for window in self.dream_roller.window_list:
            self.assertTrue(window.edge_table is not None)

    def test_optimize_params(self):
        pass

    def test_fit_windows(self):
        pass

    def test_rank_edges(self):
        pass

    def test_average_rank(self):
        pass

    def test_zscore_all_data(self):
        pass

    def test_get_window_stats(self):
        pass

    def test_compile_roller_edges(self):
        pass

    def test_make_static_edge_dict(self):
        pass

    def test_make_sort_df(self):
        pass

    def test_calc_edge_importance_cutoff(self):
        pass

    def test_get_samples(self):
        pass

    def test_score(self):
        pass

    def test_get_only_genes(self):
        only_genes = self.dream_roller.get_window(self.dream_roller.current_step)
        header = only_genes.columns.values
        correct_header = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10']
        self.assertTrue(np.array_equal(correct_header, header))




if __name__ == '__main__':
    unittest.main()
