import unittest
from Swing.util.permutation_test import Permuter
import numpy as np
from random import randint
import numpy.testing as npt
import pdb
import random
"""I'm performing a running mean and variance calculation. It seems to be a close enough approximation, and this unit test will check if the answer is within 6 decimal places of the desired answer"""

#todo: organize these methods so that they are in order. also, there is alot of repetition in this code. should I write methods? should unittest methods call/be dependent on eachother?


class TestPermutations(unittest.TestCase):
    def setUp(self):
        self.permuter = Permuter()

    def test_mean_calculation(self):
        #initialize result dict, which is a dict with mean, n, ss

        result = {'n':0, 'mean':0, 'ss':0}
        new_samples = [randint(0,100) for r in xrange(5)]
        new_result=self.permuter.update_variance_1D(result,new_samples)

        correct_mean = np.mean(new_samples)
        self.assertEqual(new_result["mean"], correct_mean)

    def test_large_mean_calculation(self):
        #initialize result dict, which is a dict with mean, n, ss

        result = {'n':0, 'mean':0, 'ss':0}
        new_samples = [randint(0,100) for r in xrange(10000)]
        new_result=self.permuter.update_variance_1D(result,new_samples)

        correct_mean = np.mean(new_samples)
        #get within 6 decimals?
        self.assertEqual("%.6f" % new_result["mean"],"%.6f" % correct_mean)

    def test_iterative_mean_calculation(self):
        new_result = {'n':0, 'mean':0, 'ss':0}
        gold_samples = []

        for i in xrange(1000):
            new_samples = [random.uniform(0.001,0.009) for r in xrange(10)]
            gold_samples = gold_samples + new_samples
            new_result = self.permuter.update_variance_1D(new_result, new_samples)
        correct_mean = np.mean(gold_samples)
        print(correct_mean)
        print(new_result['mean'])
        self.assertEqual("%.12f" % new_result["mean"],"%.12f" % correct_mean)

    def test_iterative_variance_calculation(self):
        new_result = {'n':0, 'mean':0, 'ss':0}
        gold_samples = []

        for i in xrange(1000):
            new_samples = [random.uniform(0.001,0.009) for r in xrange(10)]
            gold_samples = gold_samples + new_samples
            new_result = self.permuter.update_variance_1D(new_result, new_samples)
        correct_variance = np.var(gold_samples, ddof=1)
        print(correct_variance)
        print(new_result['variance'])
        self.assertEqual("%.12f" % new_result["variance"],"%.12f" % correct_variance)


    def test_variance_calculation(self):
        #initialize result dict, which is a dict with mean, n, ss

        result = {'n':0, 'mean':0, 'ss':0}
        new_samples = [randint(0,100) for r in xrange(100)]
        new_result=self.permuter.update_variance_1D(result,new_samples)

        correct_var = np.var(new_samples, ddof=1)
        self.assertEqual("%.6f" % new_result["variance"], "%.6f" % correct_var)

    def test_incremental_variance_calculation(self):
        #initialize result dict, which is a dict with mean, n, ss

        result = {'n':0, 'mean':0, 'ss':0}
        new_samples_A = [randint(0,100) for r in xrange(2)]
        new_result=self.permuter.update_variance_1D(result,new_samples_A)

        new_samples_B = [randint(0,100) for r in xrange(2)]
        new_result = self.permuter.update_variance_1D(new_result,new_samples_B)
        combined_samples = new_samples_A + new_samples_B

        correct_var = np.var(combined_samples, ddof=1)
        self.assertEqual("%.6f" % new_result["variance"], "%.6f" % correct_var)

    def test_incremental_mean_calculation(self):
        #initialize result dict, which is a dict with mean, n, ss

        result = {'n':0, 'mean':0, 'ss':0}
        new_samples_A = [randint(0,10000) for r in xrange(2)]
        new_result=self.permuter.update_variance_1D(result,new_samples_A)

        new_samples_B = [randint(0,10000) for r in xrange(2)]
        new_result = self.permuter.update_variance_1D(new_result,new_samples_B)
        combined_samples = new_samples_A + new_samples_B

        correct_var = np.mean(combined_samples)
        self.assertEqual("%.6f" % new_result["mean"], "%.6f" % correct_var)

    def test_incremental_2D_array_mean(self):
        zeros = np.zeros((10,10))
        result = {'n':zeros.copy(), 'mean': zeros.copy(), 'ss': zeros.copy()}

        new_samples_A = np.random.random((10,10))
        A_list = []
        A_list.append(new_samples_A)
        new_result=self.permuter.update_variance_2D(result,A_list)

        new_samples_B = np.random.random((10,10))
        B_list = []
        B_list.append(new_samples_B)
        new_result = self.permuter.update_variance_2D(new_result, B_list)
        combined_samples = np.dstack((new_samples_A,new_samples_B))
        #dstack coordinates are formatted as follows:
        # (y coord, x coord, z coord)
        # or alternatively, (row index, col index, sample index)

        correct_var = np.mean(combined_samples, axis=2)

        npt.assert_array_almost_equal(new_result["mean"], correct_var, decimal = 12)

    def test_iterative_2D_array_mean(self):
        zeros = np.zeros((10,10))
        new_result = {'n':zeros.copy(), 'mean': zeros.copy(), 'ss': zeros.copy()}

        A_list = []

        combined_samples = np.empty([10,10])
        for i in xrange(1000):
            new_samples_A = np.random.random((10,10))
            A_list.append(new_samples_A)
            temp_list = []
            temp_list.append(new_samples_A)
            new_result=self.permuter.update_variance_2D(new_result,temp_list)

        combined_samples = np.dstack(A_list)
        #dstack coordinates are formatted as follows:
        # (y coord, x coord, z coord)
        # or alternatively, (row index, col index, sample index)

        correct_mean = np.mean(combined_samples, axis=2)

        npt.assert_array_almost_equal(new_result["mean"], correct_mean, decimal = 12)

    def test_iterative_2D_array_variance(self):
        zeros = np.zeros((10,10))
        new_result = {'n':zeros.copy(), 'mean': zeros.copy(), 'ss': zeros.copy()}

        A_list = []

        combined_samples = np.empty([10,10])
        for i in xrange(1000):
            new_samples_A = np.random.random((10,10))
            A_list.append(new_samples_A)
            temp_list = []
            temp_list.append(new_samples_A)
            new_result=self.permuter.update_variance_2D(new_result,temp_list)

        combined_samples = np.dstack(A_list)
        #dstack coordinates are formatted as follows:
        # (y coord, x coord, z coord)
        # or alternatively, (row index, col index, sample index)

        correct_var = np.var(combined_samples, axis=2, ddof=1)

        npt.assert_array_almost_equal(new_result["variance"], correct_var, decimal = 12)

    def test_incremental_2D_array_variance(self):
        zeros = np.zeros((10,10))
        result = {'n':zeros.copy(), 'mean': zeros.copy(), 'ss': zeros.copy()}

        new_samples_A = np.random.random((10,10))
        A_list = []
        A_list.append(new_samples_A)
        new_result=self.permuter.update_variance_2D(result,A_list)

        new_samples_B = np.random.random((10,10))
        B_list = []
        B_list.append(new_samples_B)

        new_result = self.permuter.update_variance_2D(new_result, B_list)
        combined_samples = np.dstack((new_samples_A,new_samples_B))
        #dstack coordinates are formatted as follows:
        # (y coord, x coord, z coord)
        # or alternatively, (row index, col index, sample index)

        correct_var = np.var(combined_samples, axis=2, ddof=1)

        npt.assert_array_almost_equal(new_result["variance"], correct_var, decimal = 12)



if __name__ == '__main__':
    unittest.main()
