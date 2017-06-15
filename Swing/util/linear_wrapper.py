from sklearn import linear_model
import sys
from sklearn.linear_model import Lasso
import numpy as np
import warnings
from sklearn.cross_validation import KFold

#Note: if the optimal alpha value is very small, then another method other than LASSO should be used

class LassoWrapper:
    #I want this to spit out a matrix of coefficients, where row is the gene target and columns are the regulators


    def __init__(self, X):
        """
        :param X: array
            Numpy array of that data that Lasso will run
        :return:
        """
        self.data = X

    def get_coeffs(self, alpha=0.2):
        """returns a 2D array with target as rows and regulators as columns"""
        clf = linear_model.Lasso(alpha)
        #loop that iterates through the target genes
        all_data = self.data
        coeff_matrix = np.array([],dtype=np.float_).reshape(0, all_data.shape[1])

        for col_index,column in enumerate(all_data.T):
            #delete the column that is currently being tested
            X_matrix = np.delete(all_data, col_index, axis=1)
            #take out the column so that the gene does not regress on itself
            target_TF = all_data[:,col_index]
            clf.fit(X_matrix, target_TF)
            coeffs = clf.coef_
            #artificially add a 0 to where the col_index is
            #to prevent self-edges
            coeffs = np.insert(coeffs,col_index,0)
            coeff_matrix=np.vstack((coeff_matrix,coeffs))
        return coeff_matrix

    def get_max_alpha(self, max_expected_alpha=1e4, min_step_size=1e-9):
        """
        Get the smallest value of alpha that returns a lasso coefficient matrix of all zeros

        :param max_expected_alpha: float, optional

            Largest expected value of alpha that will return all zeros. This is a guess and is dependent on the data.
            The function will step from the minimum to this value with a step size one power of 10 less than this value

        :param min_step_size: float, optional

            The smallest step size that will be taken. This also defines the precision of the alpha value returned

        :return: float

            The smallest alpha value that will return all zero beta values, subject to the precision of min_step_size

        """
        warnings.simplefilter("ignore")
        # Get maximum edges, assuming all explanors are also response variables and no self edges
        [n, p] = self.data.shape
        max_edges = p * (p-1)

        # Raise exception if Lasso doesn't converge with alpha == 0
        if np.count_nonzero(self.get_coeffs(0)) != max_edges:
            raise ValueError('Lasso does not converge with alpha = 0')

        # Raise exception if max_expected_alpha does not return all zero betas
        if np.count_nonzero(self.get_coeffs(max_expected_alpha)) != 0:
            raise ValueError('max_expected_alpha not high enough, coefficients still exist. Guess higher')

        # Set ranges of step sizes, assumed to be powers of 10
        powers = int(np.log10(max_expected_alpha/min_step_size))
        step_sizes = [max_expected_alpha/(10**ii) for ii in range(powers+1)]

        # Intialize loop values
        cur_min = 0
        alpha_max = step_sizes[0]

        # Start stepping with forward like selection
        for ii, cur_max in enumerate(step_sizes[:-1]):

            # Set the maximum for the range to scan
            if alpha_max > cur_max:
                cur_max = alpha_max

            # Set the current step size and new range to look through
            cur_step = step_sizes[ii+1]
            cur_range = np.linspace(cur_min, cur_max, (cur_max-cur_min)/cur_step+1)

            # In the current range, check when coefficients start popping up
            for cur_alpha in cur_range:
                num_coef = np.count_nonzero(self.get_coeffs(cur_alpha))
                if num_coef > 0:
                    cur_min = cur_alpha
                elif num_coef == 0:
                    # Found a new maximum that eliminates all betas, no need to keep stepping
                    alpha_max = cur_alpha
                    break
        return alpha_max

    def cross_validate_alpha(self, alpha, n_folds=3):
        '''
        Get a Q^2 value for the alpha value
        :param alpha:
        :param n_folds: int
            when number of folds is the same as number of samples this is equivalent to leave-one-out
        :return:
        '''
        data = self.data.copy()
        n_elements = len(data)
        kf = KFold(n_elements, n_folds)

        press = 0.0
        ss = 0.0

        for train_index, test_index in kf:
            x_train = data[train_index]
            x_test = data[test_index]
            y_test = x_test.copy()

            # Run Lasso
            lasso = LassoWrapper(x_train)
            current_coef = lasso.get_coeffs(alpha)

            y_predicted = np.dot(x_test, current_coef)

            # Calculate PRESS and SS
            current_press = np.sum(np.power(y_predicted-y_test, 2))
            current_ss = sum_of_squares(y_test)

            press += current_press
            ss += current_ss
        q_squared = 1-press/ss
        return q_squared

def sum_of_squares(X):
    column_mean = np.mean(X, axis=0)
    ss = np.sum(np.power(X-column_mean,2))
    return ss

