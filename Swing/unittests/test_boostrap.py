__author__ = 'Justin Finkle'
__email__ = 'jfinkle@u.northwestern.edu'

import Swing
import numpy as np
import sys
from Swing.util.linear_wrapper import LassoWrapper
from Swing.util import Ranker
from Swing.util import Grapher
from scipy import stats
import matplotlib.pyplot as plt
import time
import unittest
import numpy.testing as npt

class TestPermutations(unittest.TestCase):
    def setUp(self):
        self.bootstrapper = Ranker.LassoBootstrapper()

    def test_auc(self):
        x = np.arange(11)
        y = np.ones(11)
        expected_area = 10.0
        self.bootstrapper.auc(y, x)
        self.assertEquals(expected_area, self.bootstrapper.edge_stability_auc)

    def test_get_nth_window_auc(self):
        first_window = np.random.random([5,5])
        second_window = np.random.random([5,5])
        window = np.dstack((first_window, second_window))
        self.bootstrapper.edge_stability_auc = window.copy()
        retrieved_first_window = self.bootstrapper.get_nth_window_auc(0)
        npt.assert_array_equal(retrieved_first_window, first_window)

if __name__ == '__main__':
    unittest.main()