__author__ = 'Justin Finkle'
__email__ = 'jfinkle@u.northwestern.edu'
from linear_wrapper import LassoWrapper
import numpy as np
from scipy import integrate

#todo: Make a general object

class LassoBootstrapper(object):
    # Method specific
    # Generate n models
    # Keep beta value ranges for each model
    # Count how many times a beta value appears
    def __init__(self):
        self.max_alpha = None
        self.bootstrap_matrix = None
        self.freq_matrix = None
        self.edge_stability_auc = None
        self.roller_object = None


    def run_bootstrap(self, roller_object, window_size, n_bootstraps, n_alphas, noise=0.2):
        # Assign roller object and set maximum alpha
        self.roller_object = roller_object
        self.set_max_alpha()

        alpha_range = np.linspace(0, self.max_alpha, n_alphas)
        self.roller_object.set_window(window_size)
        for ii, alpha in enumerate(alpha_range):
            current_coef = self.roller_object.fit_model(window_size, alpha=alpha, resamples=n_bootstraps, noise=noise)
            if ii is 0:
                empty_shape = list(current_coef.shape) + [len(alpha_range)]
                self.bootstrap_matrix = np.empty(tuple(empty_shape))
            self.bootstrap_matrix[:, :, :, :, ii] = current_coef
        self.freq_matrix = self.calc_edge_freq()/float(n_bootstraps)
        self.auc(self.freq_matrix, alpha_range)
        return alpha_range

    def set_max_alpha(self):
        # Get maximum possible alpha for the whole data set
        self.roller_object.set_window(self.roller_object.overall_width)
        current_window = self.roller_object.get_window()
        lasso = LassoWrapper(current_window.values)
        self.max_alpha = lasso.get_max_alpha()

    def calc_edge_freq(self):
        "This is agnostic to the edge sign, only whether it exists"
        edge_exists = self.bootstrap_matrix != 0
        freq_matrix = np.sum(edge_exists, axis=3)
        return freq_matrix

    def auc(self, matrix, xvalues, axis=-1):
        """
        Calculate area under the curve
        :return:
        """
        self.edge_stability_auc = integrate.trapz(matrix, xvalues, axis=axis)

    def get_nth_window_auc(self, nth):
        auc = self.edge_stability_auc[:,:, nth]
        return auc
