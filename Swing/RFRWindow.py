import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy import stats

from .Window import Window

import pdb

class RandomForestRegressionWindow(Window):
    def __init__(self, dataframe, window_info, roller_data, td_window, explanatory_dict, response_dict):
        super(RandomForestRegressionWindow, self).__init__(dataframe, window_info, roller_data, td_window,
                                                           explanatory_dict, response_dict)
        self.edge_importance = None
        self.n_trees = None
        self.n_jobs = None

    def make_edge_table(self, calc_mse=False):
        """
        Make the edge table
        :return:
        """

        # Build indexing method for all possible edges. Length = number of parents * number of children
        parent_index = range(self.edge_importance.shape[1])
        child_index = range(self.edge_importance.shape[0])
        a, b = np.meshgrid(parent_index, child_index)

        # Flatten arrays to be used in link list creation
        df = pd.DataFrame()
        df['Parent'] = self.edge_importance.columns.values[a.flatten()]
        df['Child'] = self.edge_importance.index.values[b.flatten()]
        df['Importance'] = self.edge_importance.values.flatten()
        df['P_window'] = self.explanatory_window[a.flatten()]

        # Calculate the window of the child node, which is equivalent to the current window index
        child_values = np.array([self.nth_window] * self.edge_importance.shape[0])
        df['C_window'] = child_values[b.flatten()]

        if self.permutation_p_values is not None:
            df["p_value"] = self.permutation_p_values.flatten()

        # Remove any self edges
        df = df[~((df['Parent'] == df['Child']) & (df['P_window'] == df['C_window']))]

        if calc_mse:
            df['MSE_diff'] = self.edge_mse_diff.flatten()
        return df

    def _permute_coeffs(self, zeros, crag, n_permutations, n_jobs):
        """

        :param zeros:
        :param crag:
        :param n_permutations:
        :param n_jobs:
        :return:
        """
        # initialize running calculation
        result = {'n': zeros.copy(), 'mean': zeros.copy(), 'ss': zeros.copy()}
        # inner loop: permute the window N number of times
        for nth_perm in range(0, n_permutations):
            # if (nth_perm % 200 == 0):
            # print 'Perm Run: ' +str(nth_perm)

            # permute data
            permuted_data = self.permute_data(self.explanatory_data)

            # fit the data and get coefficients
            permuted_coeffs, _ = self.get_coeffs(self.n_trees, crag=crag, x_data=permuted_data, n_jobs=n_jobs)
            dummy_list = [permuted_coeffs]
            result = self.update_variance_2D(result, dummy_list)

        self.permutation_means = result['mean'].copy()
        self.permutation_sd = np.sqrt(result['variance'].copy())
        self.permutation_p_values = self.calc_p_value()

    def run_permutation_test(self, crag=False, n_permutations=1000, n_jobs=1):
        """

        :param crag:
        :param n_permutations:
        :param n_jobs:
        :return:
        """
        # initialize permutation results array
        self.permutation_means = np.empty(self.edge_importance.shape)
        self.permutation_sd = np.empty(self.edge_importance.shape)
        zeros = np.zeros(self.edge_importance.shape)

        self._permute_coeffs(zeros, crag=crag, n_permutations=n_permutations, n_jobs=n_jobs)

    def calc_p_value(self, value=None, mean=None, sd=None):
        """

        :param value:
        :param mean:
        :param sd:
        :return:
        """
        if value is None:
            value = self.edge_importance.copy()
        if mean is None:
            mean = self.permutation_means.copy()
        if sd is None:
            sd = self.permutation_sd.copy()

        z_scores = (value - mean) / sd
        cdf = stats.norm.cdf((-1 * abs(z_scores)))
        p_values = 2 * cdf
        return p_values

    def initialize_params(self, n_trees=None):
        """
        Choose the value of alpha to use for fitting
        :param n_trees: float, optional
            The alpha value to use for the window. If none is entered the alpha will be chosen by cross validation
        :return:
        """
        if n_trees is None:
            # Select number of trees with default parameters
            self.n_trees = 500
        elif n_trees >= 0 and type(n_trees) == int:
            self.n_trees = n_trees
        else:
            raise ValueError("Number of trees must be int (>=0) or None")
        return

    def fit_window(self, crag=False, n_jobs=1, calc_mse=False):
        """
        Set the attributes of the window using expected pipeline procedure and calculate beta values
        :return:
        """
        self.edge_importance, self.edge_mse_diff = self.get_coeffs(self.n_trees, crag=crag, n_jobs=self.n_jobs,
                                                                   calc_mse=calc_mse)

    def _fitstack_coeffs(self, coeff_matrix, model_list, x_matrix, target_y, col_index, n_trees, n_jobs, crag):

        # Initialize the random forest object
        rfr = RandomForestRegressor(n_estimators=n_trees, n_jobs=n_jobs, max_features="sqrt")

        # Fit the model
        rfr.fit(x_matrix, target_y)

        # Save model parameters
        model_params = {'col_index': col_index,
                        'response': target_y,
                        'predictor': x_matrix,
                        'model': rfr}
        model_list.append(model_params)
        importance_vector = rfr.feature_importances_
        # artificially add a 0 to where the col_index is
        # to prevent self-edges
        if coeff_matrix.shape[1] - len(importance_vector) == 1:
            importance_vector = np.insert(importance_vector, col_index, 0)

        coeff_matrix = np.vstack((coeff_matrix, importance_vector))
        # there's some scoping issues here. cragging needs the roller's raw data but the window does not know what
        # roller contains (outside scope). have to pass in the roller's raw data and save it somehow :/

        if crag:
            training_scores, test_scores = self.crag_window(model_params)
            self.training_scores.append(training_scores)
            self.test_scores.append(test_scores)

        return coeff_matrix, model_list

    def get_coeffs(self, n_trees, crag=False, x_data=None, n_jobs=1, calc_mse=False):
        """
        :param x_data:
        :param n_trees:
        :return: array-like
            An array in which the rows are children and the columns are the parents
        """
        # initialize items
        y_data = self.response_data
        if x_data is None:
            x_data = self.explanatory_data

        coeff_matrix, model_list, model_inputs = self._initialize_coeffs(data = x_data, y_data = y_data, x_labels = self.explanatory_labels, y_labels = self.response_labels, x_window = self.explanatory_window, nth_window = self.nth_window)

        mse_matrix = None

        for target_y, x_matrix, insert_index in model_inputs:
            coeff_matrix, model_list = self._fitstack_coeffs(coeff_matrix, model_list, x_matrix, target_y, insert_index,
                                                             n_trees, n_jobs, crag)


            if calc_mse:
                base_mse = mean_squared_error(model_list[insert_index]['model'].predict(x_matrix), target_y)

                f_coeff_matrix, f_model_list, _ = self._initialize_coeffs(data=x_matrix, y_data=y_data, x_labels = self.explanatory_labels, y_labels = self.response_labels, x_window = self.explanatory_window, nth_window = self.nth_window)
                mse_list = []
                for idx in range(x_matrix.shape[1]):
                    adj_x_matrix = np.delete(x_matrix, idx, axis=1)
                    f_coeff_matrix, f_model_list = self._fitstack_coeffs(f_coeff_matrix, f_model_list, adj_x_matrix,
                                                                         target_y, idx, n_trees, n_jobs, crag)
                    mse_diff = base_mse - mean_squared_error(f_model_list[idx]['model'].predict(adj_x_matrix), target_y)
                    mse_list.append(mse_diff)
                if mse_matrix is None:
                    mse_matrix = np.array(mse_list)
                else:
                    mse_matrix = np.vstack((mse_matrix, np.array(mse_list)))

        importance_dataframe = pd.DataFrame(coeff_matrix, index=self.response_labels, columns=self.explanatory_labels)
        importance_dataframe.index.name = 'Child'
        importance_dataframe.columns.name = 'Parent'
        return importance_dataframe, mse_matrix
