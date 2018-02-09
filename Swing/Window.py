import numpy as np
import pandas as pd
from .util import utility_module as util


class Window(object):
    """
    A window object is a snapshot created from the full data. Windows operate independently from each other. Windows
    should be sub-classed to have network inference specific features and methods.
    """

    def __init__(self, raw_dataframe, window_info, td_window, explanatory_dict, response_dict):
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

        self.edge_list = util.make_possible_edge_list(self.explanatory_labels, self.response_labels)
        # Add edge list to edge table
        self.results_table['regulator-target'] = self.edge_list

        # Initialize attributes used for ranking edges
        self.permutation_means = None
        self.permutation_sd = None
        self.permutation_p_values = None

        """
        The training score is a list of descriptors for how well the model fit the training data for each response
        variable for the current window the test score is a list of descriptors for how well the model fit the test data
        for each response variable for the current window.
        """
        self.training_scores = []
        self.test_scores = []

    def create_linked_list(self, numpy_array_2D, value_label):
        """labels and array should be in row-major order"""
        linked_list = pd.DataFrame({'regulator-target': self.edge_list, value_label: numpy_array_2D.flatten()})
        return linked_list

    def resample_window(self):
        """
        Resample window values, along a specific axis
        :param window_values: array

        :return: array
        """
        n, p = self.explanatory_data.shape

        # For each column randomly choose samples
        resample_values = np.array([np.random.choice(self.explanatory_data[:, ii], size=n) for ii in range(p)]).T

        return resample_values

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

            old_mean = mean
            mean = old_mean + (float(x) - old_mean) / n
            sum_squares = sum_squares + (float(x) - mean) * (float(x) - old_mean)

        if n < 2:
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

        return coeff_matrix, model_list, model_inputs

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

    def get_coeffs(self, *args):
        """
        Get the beta coefficients
        :param args:
        :return:
        """
        pass

    def make_edge_table(self):
        """
        Make an edge table that can be combined.
        """
        pass