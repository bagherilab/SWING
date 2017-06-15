import unittest
import numpy as np
import Swing
import random
from random import randint
import numpy.testing as npt
import pdb

class TestAggrStats(unittest.TestCase):
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

    def test_timeseries_sep(self):
        window = self.test_window
        data_list = window.pack_values(window.raw_data)
        time_n = 20
        series_n = 5
        success = []
        for series in data_list:
            if len(series) == 20:
                success.append(1)
            else:
                success.append(0)
        if len(data_list) == 5:
            success.append(1)
        else:
            success.append(0)
        self.assertTrue( 0 not in success )

    def test_get_rates(self):
        rates_list = self.test_window.get_rates(1)
        #should get 5 lists
        #shape should be 19, 10
        test_data = rates_list[0]
        pdb.set_trace()
        self.assertTrue(test_data.shape == (19,10))

    def test_rate_analysis(self):
        rate_dict = self.test_window.get_rate_analysis(1)
        self.assertTrue(rate_dict['all_rates'].shape == (95,10))

    def test_get_average(self):
        averages = self.test_window.get_average()
        pdb.set_trace()
if __name__ == '__main__':
    unittest.main()
 
