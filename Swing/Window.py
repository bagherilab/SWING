__author__ = 'Justin Finkle'
__email__ = 'jfinkle@u.northwestern.edu'

from .util import utility_module as utility
import sys
import numpy as np
import pandas as pd
from scipy import stats
import pdb
import statsmodels.stats.stattools as st
import statsmodels.stats.diagnostic as std

class Window(object):
    """
    A window object is a snapshot created from the full data. Windows operate independently from each other. Windows
    should be sub-classed to have network inference specific features and methods.
    """

    def __init__(self, raw_dataframe, window_info, roller_data, td_window, explanatory_dict, response_dict):
        #todo: unclear if roller_data is necessary
        """
        Initialize a window object. Extract information from the passed data-frame. Generate edge list.

        :param raw_dataframe: pandas data-frame
        :param window_info: dict
            Dictionary that provides data that can be used to uniquely identify a window
        :return:
        """
        self.td_window = td_window

        # Unpack data dictionaries
        self.explanatory_data = explanatory_dict['explanatory_data']
        self.explanatory_window = explanatory_dict['explanatory_window']
        self.explanatory_times = explanatory_dict['explanatory_times']
        self.explanatory_labels = explanatory_dict['explanatory_labels']
        self.response_data = response_dict['response_data']
        self.response_window = response_dict['response_window']
        self.response_times = response_dict['response_times']
        self.response_labels = response_dict['response_labels']
        self.earlier_windows = list(set(self.explanatory_window))

        # Unpack window information
        self.time_label = window_info['time_label']
        self.gene_start = window_info['gene_start']
        self.gene_end = window_info['gene_end']
        self.nth_window = window_info['nth_window']

        # Calculate additional attributes for the window
        self.data = raw_dataframe
        self.df = raw_dataframe.iloc[:, self.gene_start:self.gene_end].copy()
        self.n_samples = len(self.response_data)
        self.genes = self.df.columns.values
        self.n_genes = len(self.genes)
        self.results_table = pd.DataFrame()
        self.edge_list = utility.make_possible_edge_list(self.explanatory_labels, self.response_labels)

        # Add edge list to edge table
        self.results_table['regulator-target'] = self.edge_list
        self.roller_data = roller_data

        # Initialize attributes used for ranking edges
        self.permutation_means = None
        self.permutation_sd = None
        self.permutation_p_values = None
        self.edge_mse_diff = None

        """
        The training score is a list of descriptors for how well the model fit the training data for each response
        variable for the current window the test score is a list of descriptors for how well the model fit the test data
        for each response variable for the current window.
        """
        self.training_scores = []
        self.test_scores = []

        # Additional options for window slicing

        self.remove_stationary_ts = False
        self.var_threshold = 0.10

    def create_linked_list(self, numpy_array_2D, value_label):
        """labels and array should be in row-major order"""
        linked_list = pd.DataFrame({'regulator-target': self.edge_list, value_label: numpy_array_2D.flatten()})
        return linked_list

    def get_window_stats(self):
        """for each window, get a dict:
            N :             the number of datapoints in this window,
            time_labels:    the names of the timepoints in a roller model
            step_size:      the step-size of the current model
            window_size:    the size of the window of the current model
            total_windows:  the number of windows total
            window_index:   the index of the window. counts start at 0. ie if the window index is 0 it is the 1st window
                            if the window index is 12, it is the 12th window in the series."""

        window_stats = {'n_samples': self.n_samples,
                        'n_genes': self.n_genes}
        return window_stats

    def resample_window(self):
        """
        Resample window values, along a specific axis
        :param window_values: array

        :return: array
        """
        n, p = self.explanatory_data.shape

        # For each column randomly choose samples
        resample_values = np.array([np.random.choice(self.explanatory_data[:, ii], size=n) for ii in range(p)]).T

        # resample_window = pd.DataFrame(resample_values, columns=self.df.columns.values.copy(),
        #                               index=self.df.index.values.copy())
        return resample_values

    def add_noise_to_values(self, window_values, max_random=0.2):
        """
        Add uniform noise to each value
        :param window: dataframe

        :param max_random: float
            Amount of noise to add to each value, plus or minus
        :return: array

        """

        noise = np.random.uniform(low=1 - max_random, high=1 + max_random, size=window_values.shape)
        noisy_values = np.multiply(window_values, noise)
        return noisy_values

    def initialize_params(self):
        """
        Initialize a model for the window and calculate the necessary parameters
        :return:
        """
        pass

    def permute_data(self, array):
        """Warning: Modifies data in place. also remember the """
        new_array = array.copy()
        _ = [np.random.shuffle(i) for i in new_array]
        return new_array

    def update_variance_1D(self, prev_result, new_samples):
        """
        incremental calculation of means: accepts new_samples, which is a list of samples. then calculates a new mean.
        this is a useful function for calculating the means of large arrays
        """

        n = float(prev_result["n"])
        mean = float(prev_result["mean"])
        sum_squares = float(prev_result["ss"])

        for x in new_samples:
            n = n + 1
            # delta = float(x) - mean
            old_mean = mean
            mean = old_mean + (float(x) - old_mean) / n
            sum_squares = sum_squares + (float(x) - mean) * (float(x) - old_mean)

        if (n < 2):
            return 0

        variance = sum_squares / (n - 1)
        result = {"mean": mean,
                  "ss": sum_squares,
                  "variance": variance,
                  "n": n}
        return result

    def update_variance_2D(self, prev_result, new_samples):
        """incremental calculation of means: accepts new_samples, which is a list of samples.
        then calculates a new mean. this is a useful function for calculating the means of large arrays"""
        n = prev_result["n"]  # 2D numpy array with all zeros or watev
        mean = prev_result["mean"]  # 2D numpy array
        sum_squares = prev_result["ss"]  # 2D numpy array

        # new_samples is a list of arrays
        #x is a 2D array
        for x in new_samples:
            n += 1
            #delta = float(x) - mean
            old_mean = mean.copy()
            mean = old_mean + np.divide((x - old_mean), n)
            sum_squares = sum_squares + np.multiply((x - mean), (x - old_mean))

        if n[0, 0] < 2:
            result = {"mean": mean,
                      "ss": sum_squares,
                      "n": n}
            return result

        variance = np.divide(sum_squares, (n - 1))
        result = {"mean": mean,
                  "ss": sum_squares,
                  "variance": variance,
                  "n": n}
        return result

    def identify_stationary_experiments(self,timeseries):
        """Test if the experiments are equivalent to flat lines. Flat lines have a variance of 0. Arbitrary threshold set by self.var_threshold"""
        time_vec = self.response_times
        # find unique number of timepoints, and calc number of experiments
        time_n = len(set(self.response_times))
        split_n = len(self.response_times)/time_n
        # generate a list of separate experiments
        print("%s %s"%(split_n,timeseries.shape ))
        ts_list = np.split(timeseries, split_n)

        var_list = []
        for ts in ts_list:
            residuals = ts - np.mean(ts)
            var = ((residuals**2).sum())/len(ts)
            var_list.append(var)

        remove_exp_idx = [i for i,v in enumerate(var_list) if v < self.var_threshold]

        return((remove_exp_idx,ts_list))



    def _initialize_coeffs(self, data, y_data, x_labels, y_labels, x_window, nth_window, s_edges = False):
        """ Returns a copy of the vector, an empty array with a defined shape, an empty list, and the maximum number of
        nodes
        """

        coeff_matrix = np.array([], dtype=np.float64).reshape(0, data.shape[1])

        model_list = []

        model_inputs = []

        # Construct a list of tuples:
        # Tuple = (Response, Explanatory, Index)


        for col_index, target_y in enumerate(y_data.T):
            if self.remove_stationary_ts is True:
                exclude_list,ts_list = self.identify_stationary_experiments(target_y)
                if exclude_list:
                    
                    # find time-series that was not excluded and use in target matrix
                    ts_list2 = [idx for i, idx in enumerate(ts_list) if i not in exclude_list]
                    # if there are no useable time-series, use all of the timeseries available
                    if len(ts_list2) is 0:
                        ts_list2 = ts_list
                    target_y = np.hstack(ts_list2)
                    time_n = len(set(self.response_times))
                    split_n = len(self.response_times)/time_n
                    x_matrix = data.astype(np.float64, copy=True)
                    x_series = np.split(x_matrix, split_n)
                    x_series2 = [idx for i, idx in enumerate(x_series) if i not in exclude_list]
                    if len(x_series2) is 0:
                        x_series2 = x_series
                    x_matrix = np.vstack(x_series2)
                else:
                    x_matrix = data.astype(np.float64, copy=True)
            else:
                x_matrix = data.astype(np.float64,copy=True)

            # Identify experiments that are stationary

            insert_index = col_index

            if nth_window in x_window:
                keep_columns = ~((x_window == self.nth_window) & (x_labels == y_labels[col_index]))
                if False in keep_columns:
                    insert_index = list(keep_columns).index(False)
                x_matrix = x_matrix[:, keep_columns].astype(np.float64, copy=True)

            input_tuple = (target_y, x_matrix, insert_index)
            model_inputs.append(input_tuple)

        return(coeff_matrix, model_list, model_inputs)

    def pack_values(self, df):
        #pack the values into separate time series. It outputs a list of pandasdataframes such that each series can be analyzed separately.
        #this is important because otherwise the algorithm will combine calculations from separate time series
        time = self.data[self.time_label].unique()
        time_n = len(time)
        series_n = len(self.data[self.time_label])/len(time)
        time_series_list = []
        for i in range(0, series_n):
            #get data[first time point : last timepoint]
            series = df[i*time_n:(i*time_n)+time_n]
            time_series_list.append(series)
        return time_series_list

    def get_rates(self, n=1):
        series_list = self.pack_values(self.df)
        rates_list = []
        for series in series_list:
            rise = np.diff(series, n, axis = 0)
            time = self.data[self.time_label].unique()
            rrun = np.array([j-i for i,j in list(zip(time[:-1], time[1:]))])
            #the vector represents the scalars used to divide each row
            rates = rise/rrun[:,None]
            rates_list.append(rates)

        return rates_list

    def get_rate_analysis(self, n=1):
        # get max rates
        rates = self.get_rates(n)
        all_rates = np.vstack(rates)

        rate_dict = {}
        rate_dict['max'] = all_rates.max(axis=0)
        rate_dict['min'] = all_rates.min(axis=0)
        rate_dict['mean'] = all_rates.mean(axis=0)
        rate_dict['median'] = np.median(all_rates,axis=0)
        rate_dict['std'] = all_rates.std(axis=0, ddof=1)
        rate_dict['all_rates'] = all_rates
        return rate_dict

    def get_linearity(self):
        n_genes = self.response_data.shape[1]
        linearity = []
        for gene_index in range(0,n_genes):
            xi = self.data[self.time_label].unique()
            y = self.response_data[gene_index,:]
            slope, intercept, r_value, p_value, std_er = stats.linregress(xi, y)
        linearity.append(r_value)
        return linearity

    def get_average(self):
        averages = self.response_data.mean(axis=0)
        return averages

    def crag_window(self, model_params):
        model = model_params['model']
        response_train = model_params['response']
        predictor_train = model_params['predictor']
        response_col = model_params['col_index']
        training_scores = utility.get_cragging_scores(model, predictor_train, response_train)
        test_data = utility.get_test_set(self.data, self.roller_data)

        response_test = test_data.ix[:, response_col].values
        predictor_test = test_data.drop(test_data.columns[response_col],1).values

        test_scores = utility.get_cragging_scores(model,predictor_test, response_test)
        return((training_scores, test_scores))

    ###################################################################################################################
    # Abstract methods listed below
    ###################################################################################################################

    def run_permutation_test(self):
        """
        Run the permutation test and update the edge_table with p values. It is expected that this method will vary
        depending on the type of method used by the window
        :return:
        """
        pass

    def fit_window(self):
        """
        Fit the window with the specified window

        """
        pass

    def run_bootstrap(self, n_bootstraps):
        """
        Run bootstrapping and update the edge_table with stability values. It is expected that this method will vary
        depending on the type of method used by the window
        :return:
        """
        pass

    def sort_edges(self, method):
        """
        Rank the edges in the edge table. This may eventually be window type specific.

        :param method: The method to use for ranking the edges in edge_table
        :return: list of tuples
            Sorted list [(regulator1, target1), (regulator2, target2)...] that can be scored with aupr or auroc
        """
        pass

    def get_coeffs(self, *args):
        """
        Get the beta coefficients
        :param args:
        :return:
        """
        pass
