from scipy import stats
from sklearn.cross_decomposition import PLSRegression
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from .Window import Window


class DionesusWindow(Window):
    """
    A window that runs Dionesus as the network inference algorithm. The PLSR function is from sci-kit learn for
    implementation consistency between window types

    For more information about Dionesus see:

    Ciaccio, Mark F., et al. "The DIONESUS algorithm provides scalable and accurate reconstruction of dynamic
    phosphoproteomic networks to reveal new drug targets." Integrative Biology (2015).

    """

    def __init__(self, dataframe, window_info, td_window, explanatory_dict, response_dict):
        super(DionesusWindow, self).__init__(dataframe, window_info, td_window, explanatory_dict,
                                             response_dict)
        self.num_pcs = None

        self.beta_coefficients = None
        self.vip = None
        self.cv_table = None
        self.bootstrap_matrix = None
        self.freq_matrix = None
        self.edge_stability_auc = None

    def make_edge_table(self):
        """

        :return:

        Called by:
            Swing.rank_edges()
        """
        # Build indexing method for all possible edges. Length = number of parents * number of children
        parent_index = range(self.beta_coefficients.shape[1])
        child_index = range(self.beta_coefficients.shape[0])
        a, b = np.meshgrid(parent_index, child_index)

        # Flatten arrays to be used in link list creation
        df = pd.DataFrame()
        df['Parent'] = self.beta_coefficients.columns.values[a.flatten()]
        df['Child'] = self.beta_coefficients.index.values[b.flatten()]
        df['Importance'] = self.vip.values.flatten()
        df['Beta'] = self.beta_coefficients.values.flatten()
        df['P_window'] = self.explanatory_window[a.flatten()]

        # Calculate the window of the child node, which is equivalent to the current window index
        child_values = np.array([self.nth_window] * self.beta_coefficients.shape[0])
        df['C_window'] = child_values[b.flatten()]

        if self.permutation_p_values is not None:
            df["p_value"] = self.permutation_p_values.flatten()

        # Remove any self edges
        df = df[~((df['Parent'] == df['Child']) & (df['P_window'] == df['C_window']))]

        return df

    def generate_results_table(self):

        # generate edges for initial model
        initial_edges = self.create_linked_list(self.beta_coefficients, 'B')

        # permutation edges
        permutation_mean_edges = self.create_linked_list(self.permutation_means, 'p-means')
        permutation_sd_edges = self.create_linked_list(self.permutation_sd, 'p-sd')
        stability_edges = self.create_linked_list(self.edge_stability_auc, 'stability')

        aggregated_edges = initial_edges.merge(permutation_mean_edges, on='regulator-target').merge(
            permutation_sd_edges, on='regulator-target').merge(stability_edges, on='regulator-target')

        # p-value calculation of permutation test
        valid_window = aggregated_edges
        initial_B = valid_window['B']
        sd = valid_window['p-sd']
        mean = valid_window['p-means']
        valid_window['final-z-scores-perm'] = (initial_B - mean) / sd
        valid_window['cdf-perm'] = (-1 * abs(valid_window['final-z-scores-perm'])).apply(stats.norm.cdf)
        # calculate t-tailed pvalue
        valid_window['p-value-perm'] = (2 * valid_window['cdf-perm'])
        self.results_table = valid_window
        return self.results_table

    def run_permutation_test(self, n_permutations=1000):

        # initialize permutation results array
        self.permutation_means = np.empty((self.n_genes, self.n_genes))
        self.permutation_sd = np.empty((self.n_genes, self.n_genes))
        zeros = np.zeros(self.beta_coefficients.shape)

        # initialize running calculation
        result = {'n': zeros.copy(), 'mean': zeros.copy(), 'ss': zeros.copy()}

        # inner loop: permute the window N number of times
        for nth_perm in range(0, n_permutations):
            # permute data
            permuted_data = self.permute_data(self.explanatory_data)

            # fit the data and get coefficients

            result_tuple = self.get_coeffs(x_data=permuted_data)
            permuted_coeffs = result_tuple[0]
            permuted_vip = result_tuple[1]

            dummy_list = [permuted_coeffs]
            result = self.update_variance_2D(result, dummy_list)

        self.permutation_means = result['mean'].copy()
        self.permutation_sd = np.sqrt(result['variance'].copy())
        self.permutation_p_values = self._calc_p_value()

    def _calc_p_value(self, value=None, mean=None, sd=None):
        if value is None:
            value = self.beta_coefficients.copy()
        if mean is None:
            mean = self.permutation_means.copy()
        if sd is None:
            sd = self.permutation_sd.copy()

        z_scores = (value - mean) / sd
        cdf = stats.norm.cdf((-1 * abs(z_scores)))
        p_values = 2 * cdf
        return p_values

    def initialize_params(self):
        """
        Optimize the number of PCs to use.

        :return:
        """

        # calculate the Q2 score using PC=1,2,3,4,5

        # pick the PCs that maximizes the Q2 score-PCs tradeoff, using the elbow rule,
        # maximizing the second derivative or maximum curvature.
        result_tuple = self.get_coeffs()
        model_list = result_tuple[2]
        model_inputs = result_tuple[3]

        explained_variances = None
        size_test = []
        for response, explanatory, index in model_inputs:
            size_test.append(explanatory.shape)

        min_dim = sorted(size_test, key=lambda x: x[1], reverse=False)[0][1]

        for response, explanatory, index in model_inputs:
            pca = PCA()
            pca.fit(explanatory)
            if explained_variances is None:
                explained_variances = pca.explained_variance_ratio_
            else:
                try:
                    explained_variances = np.vstack((explained_variances, pca.explained_variance_ratio_))
                except ValueError:
                    try:
                        explained_variances = np.vstack((explained_variances[:, :min_dim],
                                                         pca.explained_variance_ratio_[:min_dim]))
                    except IndexError:
                        truncated_index = min_dim
                        explained_variances = np.vstack((explained_variances[:truncated_index],
                                                         pca.explained_variance_ratio_[:truncated_index]))

        explained_variances_mean = np.mean(explained_variances, axis=0)
        test_pcs = [x for x in range(1, len(explained_variances_mean) + 1)]
        elbow_x, elbow_y = self._elbow_criteria(test_pcs, explained_variances_mean)
        self.num_pcs = elbow_x

    def fit_window(self, pcs=3):
        """
        Set the attributes of the window using expected pipeline procedure and calculate beta values

        :return:
        """
        if self.num_pcs is not None:
            pcs = self.num_pcs
        result_tuple = self.get_coeffs(pcs)

        self.beta_coefficients = result_tuple[0]
        self.vip = result_tuple[1]
        self.model_list = result_tuple[2]

    def _fitstack_coeffs(self, n_pcs, coeff_matrix, vip_matrix, model_list, x_matrix, target_y, col_index):
        """
        :param n_pcs:
        :param coeff_matrix:
        :param vip_matrix:
        :param model_list:
        :param x_matrix:
        :param target_y:
        :param col_index:
        :return:
        """
        pls = PLSRegression(n_pcs, False)

        # Fit the model
        pls.fit(x_matrix, target_y)

        model_params = {'col_index': col_index,
                        'response': target_y,
                        'predictor': x_matrix,
                        'model': pls}

        model_list.append(model_params)

        # artificially add a 0 to where the col_index is to prevent self-edges
        coeffs = pls.coef_
        coeffs = np.reshape(coeffs, (len(coeffs),))
        vips = self._vipp(x_matrix, target_y, pls.x_scores_, pls.x_weights_)
        vips = np.reshape(vips, (len(vips),))
        if coeff_matrix.shape[1] - len(coeffs) == 1:
            coeffs = np.insert(coeffs, col_index, 0)
            vips = np.insert(vips, col_index, 0)

        coeff_matrix = np.vstack((coeff_matrix, coeffs))
        vip_matrix = np.vstack((vip_matrix, vips))

        return coeff_matrix, vip_matrix, model_list

    def _elbow_criteria(self, x, y):
        x = np.array(x)
        y = np.array(y)
        # Slope between elbow endpoints
        m1 = self._point_slope(x[0], y[0], x[-1], y[-1])
        # Intercept
        b1 = y[0] - m1 * x[0]

        # Slope for perpendicular lines
        m2 = -1 / m1

        # Calculate intercepts for perpendicular lines that go through data point
        b_array = y - m2 * x
        x_perp = (b_array - b1) / (m1 - m2)
        y_perp = m1 * x_perp + b1

        # Calculate where the maximum distance to a line connecting endpoints is
        distances = np.sqrt((x_perp - x) ** 2 + (y_perp - y) ** 2)
        index_max = np.where(distances == np.max(distances))[0][0]
        elbow_x = x[index_max]
        elbow_y = y[index_max]
        return elbow_x, elbow_y

    @staticmethod
    def _point_slope(x1, y1, x2, y2):
        slope = (y2 - y1) / float(x2 - x1)
        return slope

    def get_coeffs(self, num_pcs=2, x_data=None, y_data=None):
        """
        :param x_data:
        :param n_trees:
        :return: array-like
            An array in which the rows are children and the columns are the parents
        """
        # initialize items
        if y_data is None:
            y_data = self.response_data
        if x_data is None:
            x_data = self.explanatory_data

        coeff_matrix, model_list, model_inputs = self._initialize_coeffs(data=x_data, y_data=y_data,
                                                                         x_labels=self.explanatory_labels,
                                                                         y_labels=self.response_labels,
                                                                         x_window=self.explanatory_window,
                                                                         nth_window=self.nth_window)
        vip_matrix = coeff_matrix.copy()

        # Calculate a model for each target column
        for target_y, x_matrix, insert_index in model_inputs:
            coeff_matrix, vip_matrix, model_list = self._fitstack_coeffs(num_pcs, coeff_matrix, vip_matrix, model_list,
                                                                         x_matrix, target_y, insert_index)

        coeff_dataframe = pd.DataFrame(coeff_matrix, index=self.response_labels, columns=self.explanatory_labels)
        coeff_dataframe.index.name = 'Child'
        coeff_dataframe.columns.name = 'Parent'

        importance_dataframe = pd.DataFrame(vip_matrix, index=self.response_labels, columns=self.explanatory_labels)
        importance_dataframe.index.name = 'Child'
        importance_dataframe.columns.name = 'Parent'
        return coeff_dataframe, importance_dataframe, model_list, model_inputs

    @staticmethod
    def _vipp(x, y, t, w):
        # initializing
        [p, h] = w.shape
        co = np.matrix(np.zeros([1, h]))

        # Calculate s		
        for ii in range(h):
            corr = np.corrcoef(y, t[:, ii], rowvar=0)
            co[0, ii] = corr[0, 1] ** 2
            s = np.sum(co)
        # Calculate q		
        # This has been linearized to replace the original nested for loop		
        w_power = np.power(w, 2)
        d = np.multiply(w_power, co)
        q = np.sum(d, 1)
        vip = np.sqrt(p * q / s)
        return vip
