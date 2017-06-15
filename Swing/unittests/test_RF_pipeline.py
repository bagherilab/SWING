import unittest
import numpy as np
import Swing
import pdb

class TestRFPipeline(unittest.TestCase):
    def setUp(self):
        file_path = "../../data/dream4/insilico_size10_1_timeseries.tsv"
        gene_start_column = 1
        time_label = "Time"
        separator = "\t"
        gene_end = None

        self.roller = Swing.Swing(file_path, gene_start_column, gene_end, time_label, separator,window_type = "RandomForest")

    def test_crap(self):
        self.roller.set_window(20)
        self.roller.create_windows()
        self.roller.optimize_params()
        self.roller.fit_windows()
        self.roller.rank_edges(permutation_n=10)
        #pdb.set_trace()
        self.roller.average_rank(rank_by='p_value', ascending=False)
        #score some edge lists
        #first score the sorted average edge list
        gold_standard = "../../data/dream4/insilico_size10_1_goldstandard.tsv"
        averaged_aupr = self.roller.score(self.roller.averaged_ranks, gold_standard)
        #next score each individual edge list
        aupr_list = []
        for window in self.roller.window_list:
            aupr = self.roller.score(window.results_table,gold_standard)
            aupr_list.append(aupr)
        #pdb.set_trace()
if __name__ == '__main__':
    unittest.main()
