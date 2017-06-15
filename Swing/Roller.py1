import pandas as pd
import numpy as np
from Window import Window
from LassoWindow import LassoWindow
from RFRWindow import RandomForestRegressionWindow, tdRFRWindow
from DionesusWindow import DionesusWindow
from util import utility_module as utility
from util.Evaluator import Evaluator
import pdb
import random


class Roller(object):
    """
    A thing that grabs different timepoints of data, can set window and step size.

    """

    def __init__(self, file_path, gene_start=None, gene_end=None, time_label="Time", separator="\t",
                 window_type="RandomForest"):
        """
        Initialize the roller object. Read the file and put it into a pandas dataframe
        :param file_path: file-like object or string
                        The file to read
        :param gene_start: int
        :param gene_end: int
        """
        # Read the raw data into a pandas dataframe object
        self.raw_data = pd.read_csv(file_path, sep=separator)
        self.raw_data = self.raw_data.dropna(axis=0, how='all')

        self.file_path = file_path
        self.window_type = window_type

        # Set roller defaults
        self.current_step = 0
        self.window_width = 3
        self.step_size = 1
        self.time_label = time_label

        # Get overall width of the time-course
        self.time_vec = self.raw_data[self.time_label].unique()
        self.overall_width = len(self.time_vec)

        if gene_end is not None:
            self.gene_end = gene_end
        else:
            self.gene_end = len(self.raw_data.columns)
        if gene_start is not None:
            self.gene_start = gene_start
        else:
            self.gene_start = 0

        self.gene_list = self.raw_data.columns.values[self.gene_start:self.gene_end]
        self.window_list = []

    def make_possible_edge_list(self, parents, children, self_edges=True):
        """
        Create a list of all the possible edges between parents and children

        :param parents: array
            labels for parents
        :param children: array
            labels for children
        :param self_edges:
        :return: array, length = parents * children
            array of parent, child combinations for all possible edges
        """
        parent_index = range(len(parents))
        child_index = range(len(children))
        a, b = np.meshgrid(parent_index, child_index)
        parent_list = parents[a.flatten()]
        child_list = children[b.flatten()]
        possible_edge_list = None
        if self_edges:
            possible_edge_list = zip(parent_list, child_list)

        elif not self_edges:
            possible_edge_list = zip(parent_list[parent_list != child_list], child_list[parent_list != child_list])

        return possible_edge_list

    def get_n_windows(self):
        """
        Calculate the number of windows

        Called by:
            create_windows
            get_window_stats

        :return: int
        """
        total_windows = (self.overall_width - self.window_width + 1) / (self.step_size)
        return total_windows

    def get_window(self, start_index):
        # todo: start_index should be used
        """
        Select a window from the full data set, only keeping the data corresponding to genes

        Called by:
            __init__
            set_max_alpha in Ranker.py (necessary)
            permutation_test.py (deprecated)


        :param start_index: int
            The start of the window

        :return: data-frame
        """
        raw_window = self.get_window_raw(0)
        only_genes = raw_window.iloc[:, self.gene_start:self.gene_end]
        return only_genes

    def get_window_raw(self, start_index, random_time=False):
        """
        Select a window from the full data set. This is fancy data-frame slicing

        Called by:
            create_windows
            get_window_stats
            get_window

        :param start_index: int
            The start of the window
        :param random_time: bool, optional
        :return: data-frame
        """
        if random_time:
            # select three random timepoints
            time_window = self.time_vec[start_index]
            choices = self.time_vec
            choices = np.delete(choices, start_index)
            for x in range(0, self.window_width - 1):
                chosen_time = random.choice(choices)
                time_window = np.append(time_window, chosen_time)
                chosen_index = np.where(choices == chosen_time)
                choices = np.delete(choices, chosen_index)
        else:
            end_index = start_index + self.window_width
            time_window = self.time_vec[start_index:end_index]
        data = self.raw_data[self.raw_data[self.time_label].isin(time_window)]
        return data

    def set_window(self, width):
        """
        Set the window width

        Called by:
            pipeline

        :param width: int

        :return:
        """
        self.window_width = width

    def set_step(self, step):
        """
        Set the window step size

        Called by:


        :param step:
        :return:
        """

        self.step_size = step

    def reset(self):
        """
        Reset the window to the beggining of time

        Called by:

        :return:
        """
        self.current_step = 0

    # need to do something about this method. keep for now, but currently need a "preprocess" method.
    def remove_blank_rows(self):
        """
        Removes a row if the sum of that row is NaN

        Called by:


        :return:
        """
        """calculates sum of rows. if sum is NAN, then remove row"""
        coln = len(self.raw_data.columns)
        sums = [self.raw_data.iloc[:, x].sum() for x in range(0, coln)]
        ind = np.where(np.isnan(sums))[0]
        self.raw_data.iloc[:, ind] = 0

    def get_n_genes(self):
        """
        Calculate the number of genes in the data set

        Called by:


        :return:
        """

        return len(self.raw_data.columns) - 1

    def create_windows(self, random_time=False):
        """
        Create window objects for the roller to use

        Called by:
            pipeline

        :return:
        """
        window_list = [self.get_window_object(self.get_window_raw(index, random_time),
                                              {"time_label": self.time_label,
                                               "gene_start": self.gene_start,
                                               "gene_end": self.gene_end,
                                               "nth_window": index}) if (
        index + self.window_width <= self.overall_width) else '' for index in range(self.get_n_windows())]
        self.window_list = window_list

    def get_window_object(self, dataframe, window_info_dict):
        """
        Return a window object from a data-frame

        Called by:
            create_windows

        :param dataframe: data-frame
        :param window_info_dict: dict
            Dictionary containing information needed for window initialization
        :return:
        """
        if self.window_type == "Lasso":
            window_obj = LassoWindow(dataframe, window_info_dict, self.raw_data)
        elif self.window_type == "RandomForest":
            window_obj = RandomForestRegressionWindow(dataframe, window_info_dict, self.raw_data)
        elif self.window_type == "Dionesus":
            window_obj = DionesusWindow(dataframe, window_info_dict, self.raw_data)

        return window_obj

    def initialize_windows(self):
        """
        deprecated - Initialize window parameters and do a preliminary fit

        Called by:
        Currently only called by unittest Roller/unittests/test_roller.py

        todo: delete
        :return:
        """
        for window in self.window_list:
            window.initialize_params()
            window.fit_window()

    def rank_windows(self, n_permutes=1000, n_bootstraps=1000, n_alphas=20, noise=0.2):
        """
        Run tests to score and rank windows

        Called by:


        :param n_permutes: int, optional
            Number of permutes to run. Default is 1,000
        :param n_bootstraps: int, optional
            Number of bootstraps to run. Default is 1,000
        :param n_alphas: int, optional
            Number of alpha values to test if using Lasso. Default is 20
        :param noise: float ([0,1]), optional
            The amount of noise to add to bootstrapped windows. Default is 0.2

        :return:
        """
        for window in self.window_list:
            window.run_permutation_test(n_permutes)
            window.run_bootstrap(n_bootstraps, n_alphas, noise)
            window.make_edge_table()

    def optimize_params(self):
        """
        Optimize window fit parameters

        Called by:
            pipeline


        :return:
        """

        for window in self.window_list:
            window.initialize_params()
        return self.window_list

    def fit_windows(self, alpha=None, n_trees=None, show_progress=True):
        #todo: need a better way to pass parameters to fit functions
        """
        Fit each window in the list

        Called by:
            pipeline

        :param alpha:
        :param n_trees:
        :return:
        """

        for window in self.window_list:
            if self.window_type == "Lasso":
                if alpha != None:
                    window.alpha = alpha
            if self.window_type == "RandomForest":
                if n_trees != None:
                    window.n_trees = n_trees
            if show_progress:
                print "Fitting window %i of %i" %((window.nth_window+1), len(self.window_list))
            window.fit_window()

        return self.window_list

    def rank_edges(self, n_bootstraps=1000, permutation_n=1000):
        """
        Run tests to rank edges in windows

        Called by:
            pipeline

        :param n_bootstraps:
        :param permutation_n:
        :return:
        """

        if self.window_type == "Lasso":
            for window in self.window_list:
                window.run_permutation_test(n_permutations=permutation_n)
                print("Running bootstrap...")
                window.run_bootstrap(n_bootstraps=n_bootstraps)
                window.generate_results_table()
        if self.window_type == "RandomForest":
            for window in self.window_list:
                if window.include_window == True:
                    print("Running permutation on window %i...")%window.nth_window
                    window.run_permutation_test(n_permutations=permutation_n)
                    window.make_edge_table()
        return self.window_list

    def average_rank(self, rank_by, ascending):
        """
        Average window edge ranks

        Called by:
            pipeline


        :param rank_by: string
            The parameter to rank edges by
        :param ascending: Bool
        :return:
        """
        if self.window_type == "Lasso":
            ranked_result_list = []
            for window in self.window_list:
                ranked_result = window.rank_results(rank_by, ascending)
                ranked_result_list.append(ranked_result)
        if self.window_type == "RandomForest":
            ranked_result_list = []
            for window in self.window_list:
                ranked_result = window.sort_edges(rank_by)
                ranked_result_list.append(ranked_result)

        aggr_ranks = utility.average_rank(ranked_result_list, rank_by + "-rank")
        # sort tables by mean rank in ascending order
        mean_sorted_edge_list = aggr_ranks.sort(columns="mean-rank", axis=0)
        self.averaged_ranks = mean_sorted_edge_list
        return self.averaged_ranks

    # todo: this method sucks. sorry.
    def score(self, sorted_edge_list, gold_standard_file):
        """
        Scores some stuff, I think...
        Called by:
            pipeline
        :param sorted_edge_list:
        :param gold_standard_file:
        :return:
        """
        if len(sorted_edge_list) < 15:
            pdb.set_trace()
        evaluator = Evaluator(gold_standard_file, sep='\t')
        edge_cutoff = len(evaluator.gs_flat)
        precision, recall, aupr = evaluator.calc_pr(sorted_edge_list[0:edge_cutoff + 1])
        score_dict = {"precision": precision, "recall": recall, "aupr": aupr}
        return score_dict

    def zscore_all_data(self):
        #todo: this should not replace raw_data, a new feature should be made
        #todo: scipy.stats.zscore can be used with the correct parameters for 1 line
        """
        Zscore the data in a data-frame

        Called by:
            pipeline

        :return:
        """
        # zscores all the data
        dataframe = self.raw_data

        # for each column, calculate the zscore
        # zscore is calculated as X - meanX / std(ddof = 1)
        for item in dataframe.columns:
            if item != self.time_label:
                dataframe[item] = (dataframe[item] - dataframe[item].mean()) / dataframe[item].std(ddof=1)
        self.raw_data = dataframe

    def get_window_stats(self):
        """
        Generate a dictionary of relevant information from a window
            N : the number of data points in this window,
            time_labels: the names of the time points in a roller model
            step_size: the step-size of the current model
            window_size: the size of the window of the current model
            total_windows: the number of windows total
            window_index: the index of the window. counts start at 0. ie if the window index is 0 it is the 1st window.
            If the window index is 12, it is the 12th window in the series.

        Called by:


        :return: dict
        """
        """for each window, get a dict:
            N : the number of datapoints in this window,
            time_labels: the names of the timepoints in a roller model
            step_size: the step-size of the current model
            window_size: the size of the window of the current model
            total_windows: the number of windows total
            window_index: the index of the window. counts start at 0. ie if the window index is 0 it is the 1st window. if the window index is 12, it is the 12th window in the series."""
        current_window = self.get_window_raw()

        """calculate the window index. todo: move into own function later"""
        min_time = np.amin(current_window[self.time_label])
        window_index = np.where(self.time_vec == min_time) / self.step_size
        # to calculate the nth window, time vector
        # index of the time-vector, step size of 2? window 4, step size 2
        #
        # total windows = total width (10) - window_width (2) +1 / step size
        # 10 time points 0 1 2 3 4 5 6 7 8 9
        # width is 2: 0 and 1
        # step size is 2
        # 01, 12, 23, 34, 45, 56, 67, 78, 89

        # todo: so the issue is that total windows (get n windows) is the true number of windows, and window index is the nth -1 window... it would be great to consolidate these concepts but no big deal if they can't be.

        window_stats = {'N': len(current_window.index),
                        'time_labels': current_window[self.time_label].unique(),
                        'step_size': self.step_size,
                        'window_size': self.window_width,
                        'total_windows': self.get_n_windows(),
                        'window_index': window_index}
        return window_stats
