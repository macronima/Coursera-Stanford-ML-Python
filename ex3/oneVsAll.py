import numpy as np
import lrCostFunction as lrcf
from scipy.optimize import minimize
from lrCostFunction import lrgradientReg
from lrCostFunction import lrCostFunction

from scipy import optimize

def optimizeTheta(mytheta,myX,myy,mylambda=0.):
    result = optimize.fmin_cg(lrgradientReg, fprime=lrCostFunction, x0=mytheta, \
                              args=(myX, myy, mylambda), maxiter=13, disp=False,\
                              full_output=True)
    return result[0], result[1]


def oneVsAll(X, y, num_labels, lambda_reg):
    # ONEVSALL trains multiple logistic regression classifiers and returns all
    # the classifiers in a matrix all_theta, where the i-th row of all_theta
    # corresponds to the classifier for label i
    #   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
    #   logisitc regression classifiers and returns each of these classifiers
    #   in a matrix all_theta, where the i-th row of all_theta corresponds
    #   to the classifier for label i


    # Some useful variables
#    m, n = X.shape

    # You need to return the following variables correctly
   # all_theta = np.zeros((num_labels, n + 1))

    # Add ones to the X data matrix
   # X = np.column_stack((np.ones((m, 1)), X))

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should complete the following code to train num_labels
    #               logistic regression classifiers with regularization
    #               parameter lambda.
    #
    mylambda = lambda_reg
    initial_theta = np.zeros((X.shape[1], 1)).reshape(-1)
    Theta = np.zeros((num_labels, X.shape[1]))
    for i in xrange(num_labels):
        iclass = i if i else num_labels  # class "10" corresponds to handwritten zero
    #    print "Optimizing for handwritten number %d..." % i
        logic_Y = np.array([1 if x == iclass else 0 for x in y])  # .reshape((X.shape[0],1))
        itheta, imincost = optimizeTheta(initial_theta, X, logic_Y, mylambda)
        Theta[i-1] = itheta
    #print "Done!"
    return Theta
