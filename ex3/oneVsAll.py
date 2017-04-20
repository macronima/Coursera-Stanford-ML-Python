import numpy as np
#import lrCostFunction as lrcf
from scipy.optimize import minimize

from lrCostFunction import lrCostFunction
from ex2.gradientFunctionReg import gradientFunctionReg


def oneVsAll(X, y, num_labels, Lambda):
    """trains multiple logistic regression classifiers and returns all
        the classifiers in a matrix all_theta, where the i-th row of all_theta
        corresponds to the classifier for label i
    """

# Some useful variables
    m, n = X.shape

# You need to return the following variables correctly 
    all_theta = np.zeros((num_labels, n + 1))

# Add ones to the X data matrix
    X = np.column_stack((np.ones((m, 1)), X))

# ====================== YOUR CODE HERE ======================
# Instructions: You should complete the following code to train num_labels
#               logistic regression classifiers with regularization
#               parameter lambda. 
#
# Hint: theta(:) will return a column vector.
#
# Hint: You can use y == c to obtain a vector of0 1's and 0's that tell use
#       whether the ground truth is true/false for this class.
#
# Note: For this assignment, we recommend using fmincg to optimize the cost
#       function. It is okay to use a for-loop (for c = 1:num_labels) to
#       loop over the different classes.

    for c in xrange(num_labels):
        # initial theta for c/class
        initial_theta = np.zeros((n + 1, 1))

        print("Training {:d} out of {:d} categories...".format(c + 1, num_labels))
        myargs = (X, (y % 10 == c).astype(int), Lambda, True)
         #minimize(lrcf.costFunctionReg, x0=initial_theta, args=myargs, options={'disp': True, 'maxiter': 13},method="Newton-CG", jac=True)
        theta = minimize(lrCostFunction, initial_theta, args=myargs, method=None, jac=True, options={'maxiter': 50})
        all_theta[c, :] = theta["x"]

    # =========================================================================

    return all_theta

