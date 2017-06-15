import unittest
import numpy as np
import Swing
import random
from random import randint
import numpy.testing as npt
import pdb

class TestWindow(unittest.TestCase):
    def setUp(self):
        file_path = "../../data/dream4/insilico_size10_1_timeseries.tsv"
        gene_start_column = 1
        time_label = "Time"
        separator = "\t"
        gene_end = None

        self.roller = Swing.Swing(file_path, gene_start_column, gene_end, time_label, separator, window_type = "Lasso")
        self.roller.set_window(width=20)
        self.roller.create_windows()
        self.test_window = self.roller.window_list[0]


    def test_possible_edge_list_self(self):
        expected_edges = self.test_window.n_genes**2
        num_edges = len(self.test_window.edge_list)
        self.assertTrue(expected_edges == num_edges)

    def test_possible_edge_list_no_self(self):
        genes = self.test_window.genes.copy()
        expected_edges = self.test_window.n_genes**2 - self.test_window.n_genes
        self.test_window.edge_list = self.test_window.possible_edge_list(genes, genes, False)
        num_edges = len(self.test_window.edge_list)
        self.assertTrue(expected_edges == num_edges)

    def test_resample_window(self):

        resampled_values = self.test_window.resample_window()

        # Confirm shapes are true
        self.assertTrue(self.test_window.window_values.shape == resampled_values.shape)
        num_rows, num_columns = self.test_window.window_values.shape

        # Verify that the resampled matrix doesn't equal the original matrix
        self.assertFalse(np.array_equal(resampled_values, self.test_window.window_values))

        # Verify that values in each column of resampled matrix are values in the same column of the original window
        truth_table = np.array(
            [[value in self.test_window.window_values[:, column] for value in resampled_values[:, column]] for column in
             range(num_columns)]).T
        self.assertTrue(np.all(truth_table))

    def test_add_noise_to_window(self):
        # Generate test data frame
        original_values = self.test_window.window_values
        max_random = 0.3

        # Get noisy values
        noise_values = self.test_window.add_noise_to_values(original_values, max_random=max_random)

        # Make sure noise is within set range
        noise_magnitude = np.abs((noise_values-original_values)/original_values)
        self.assertTrue(np.all(noise_magnitude <= max_random))

    def test_mean_calculation(self):
        # initialize result dict, which is a dict with mean, n, ss

        result = {'n': 0, 'mean': 0, 'ss': 0}
        new_samples = [randint(0, 100) for r in xrange(5)]
        new_result = self.test_window.update_variance_1D(result, new_samples)

        correct_mean = np.mean(new_samples)
        self.assertEqual(new_result["mean"], correct_mean)

    def test_large_mean_calculation(self):
        # initialize result dict, which is a dict with mean, n, ss

        result = {'n': 0, 'mean': 0, 'ss': 0}
        new_samples = [randint(0, 100) for r in xrange(10000)]
        new_result = self.test_window.update_variance_1D(result, new_samples)

        correct_mean = np.mean(new_samples)
        #get within 6 decimals?
        self.assertEqual("%.6f" % new_result["mean"], "%.6f" % correct_mean)

    def test_iterative_mean_calculation(self):
        new_result = {'n': 0, 'mean': 0, 'ss': 0}
        gold_samples = []

        for i in xrange(1000):
            new_samples = [random.uniform(0.001, 0.009) for r in xrange(10)]
            gold_samples = gold_samples + new_samples
            new_result = self.test_window.update_variance_1D(new_result, new_samples)
        correct_mean = np.mean(gold_samples)
        print 'Correct Mean: ', (correct_mean)
        print 'Calculated Mean: ', (new_result['mean'])
        self.assertEqual("%.12f" % new_result["mean"], "%.12f" % correct_mean)

    def test_iterative_variance_calculation(self):
        new_result = {'n': 0, 'mean': 0, 'ss': 0}
        gold_samples = []

        for i in xrange(1000):
            new_samples = [random.uniform(0.001, 0.009) for r in xrange(10)]
            gold_samples = gold_samples + new_samples
            new_result = self.test_window.update_variance_1D(new_result, new_samples)
        correct_variance = np.var(gold_samples, ddof=1)
        print 'Correct Var: ', (correct_variance)
        print 'Calculated Var: ', (new_result['variance'])
        self.assertEqual("%.12f" % new_result["variance"], "%.12f" % correct_variance)


    def test_variance_calculation(self):
        # initialize result dict, which is a dict with mean, n, ss

        result = {'n': 0, 'mean': 0, 'ss': 0}
        new_samples = [randint(0, 100) for r in xrange(100)]
        new_result = self.test_window.update_variance_1D(result, new_samples)

        correct_var = np.var(new_samples, ddof=1)
        self.assertEqual("%.6f" % new_result["variance"], "%.6f" % correct_var)

    def test_incremental_variance_calculation(self):
        # initialize result dict, which is a dict with mean, n, ss

        result = {'n': 0, 'mean': 0, 'ss': 0}
        new_samples_A = [randint(0, 100) for r in xrange(2)]
        new_result = self.test_window.update_variance_1D(result, new_samples_A)

        new_samples_B = [randint(0, 100) for r in xrange(2)]
        new_result = self.test_window.update_variance_1D(new_result, new_samples_B)
        combined_samples = new_samples_A + new_samples_B

        correct_var = np.var(combined_samples, ddof=1)
        self.assertEqual("%.6f" % new_result["variance"], "%.6f" % correct_var)

    def test_incremental_mean_calculation(self):
        # initialize result dict, which is a dict with mean, n, ss

        result = {'n': 0, 'mean': 0, 'ss': 0}
        new_samples_A = [randint(0, 10000) for r in xrange(2)]
        new_result = self.test_window.update_variance_1D(result, new_samples_A)

        new_samples_B = [randint(0, 10000) for r in xrange(2)]
        new_result = self.test_window.update_variance_1D(new_result, new_samples_B)
        combined_samples = new_samples_A + new_samples_B

        correct_var = np.mean(combined_samples)
        self.assertEqual("%.6f" % new_result["mean"], "%.6f" % correct_var)

    def test_incremental_2D_array_mean(self):
        zeros = np.zeros((10, 10))
        result = {'n': zeros.copy(), 'mean': zeros.copy(), 'ss': zeros.copy()}

        new_samples_A = np.random.random((10, 10))
        A_list = []
        A_list.append(new_samples_A)
        new_result = self.test_window.update_variance_2D(result, A_list)

        new_samples_B = np.random.random((10, 10))
        B_list = []
        B_list.append(new_samples_B)
        new_result = self.test_window.update_variance_2D(new_result, B_list)
        combined_samples = np.dstack((new_samples_A, new_samples_B))
        # dstack coordinates are formatted as follows:
        # (y coord, x coord, z coord)
        # or alternatively, (row index, col index, sample index)

        correct_var = np.mean(combined_samples, axis=2)

        npt.assert_array_almost_equal(new_result["mean"], correct_var, decimal=12)

    def test_iterative_2D_array_mean(self):
        zeros = np.zeros((10, 10))
        new_result = {'n': zeros.copy(), 'mean': zeros.copy(), 'ss': zeros.copy()}

        A_list = []

        combined_samples = np.empty([10, 10])
        for i in xrange(1000):
            new_samples_A = np.random.random((10, 10))
            A_list.append(new_samples_A)
            temp_list = []
            temp_list.append(new_samples_A)
            new_result = self.test_window.update_variance_2D(new_result, temp_list)

        combined_samples = np.dstack(A_list)
        # dstack coordinates are formatted as follows:
        # (y coord, x coord, z coord)
        # or alternatively, (row index, col index, sample index)

        correct_mean = np.mean(combined_samples, axis=2)

        npt.assert_array_almost_equal(new_result["mean"], correct_mean, decimal=12)

    def test_iterative_2D_array_variance(self):
        zeros = np.zeros((10, 10))
        new_result = {'n': zeros.copy(), 'mean': zeros.copy(), 'ss': zeros.copy()}

        A_list = []

        combined_samples = np.empty([10, 10])
        for i in xrange(1000):
            new_samples_A = np.random.random((10, 10))
            A_list.append(new_samples_A)
            temp_list = []
            temp_list.append(new_samples_A)
            new_result = self.test_window.update_variance_2D(new_result, temp_list)

        combined_samples = np.dstack(A_list)
        # dstack coordinates are formatted as follows:
        # (y coord, x coord, z coord)
        # or alternatively, (row index, col index, sample index)

        correct_var = np.var(combined_samples, axis=2, ddof=1)

        npt.assert_array_almost_equal(new_result["variance"], correct_var, decimal=12)

    def test_incremental_2D_array_variance(self):
        zeros = np.zeros((10, 10))
        result = {'n': zeros.copy(), 'mean': zeros.copy(), 'ss': zeros.copy()}

        new_samples_A = np.random.random((10, 10))
        A_list = []
        A_list.append(new_samples_A)
        new_result = self.test_window.update_variance_2D(result, A_list)

        new_samples_B = np.random.random((10, 10))
        B_list = []
        B_list.append(new_samples_B)

        new_result = self.test_window.update_variance_2D(new_result, B_list)
        combined_samples = np.dstack((new_samples_A, new_samples_B))
        # dstack coordinates are formatted as follows:
        # (y coord, x coord, z coord)
        # or alternatively, (row index, col index, sample index)

        correct_var = np.var(combined_samples, axis=2, ddof=1)

        npt.assert_array_almost_equal(new_result["variance"], correct_var, decimal=12)

if __name__ == '__main__':
    unittest.main()
