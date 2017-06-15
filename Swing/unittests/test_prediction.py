import unittest
import numpy as np
import Swing
import random
from random import randint
import numpy.testing as npt
import pdb
import sklearn.metrics as skmet
import Swing.util.utility_module as Rutil

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

    def test_model_is_saved(self):
        model_list = self.test_window.model
        n_genes = self.test_window.n_genes
        self.assertTrue(len(model_list),n_genes)

    def test_prediction(self):
        model_list = self.test_window.model
        model = model_list[0]['model']
        response_train = model_list[0]['response']
        predictor_train = model_list[0]['predictor']

        #get training scores
        training_scores = Rutil.get_cragging_scores(model, predictor_train, response_train)

        #get test set from the roller model
        test_data = Rutil.get_test_set(self.test_window.raw_data, self.roller.raw_data)
        response_col = 0
        response_test = test_data.ix[:,response_col].values
        predictor_test = test_data.drop(test_data.columns[response_col],1).values

        #get prediction scores
        test_scores = Rutil.get_cragging_scores(model, predictor_test, response_test)

    def test_fit(self):
        self.roller.optimize_params()
        self.roller.fit_windows()
        self.test_window.fit_window()
        pdb.set_trace()
if __name__ == '__main__':
    unittest.main()

