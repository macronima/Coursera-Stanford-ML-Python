from numpy import log
from sigmoid import sigmoid
import numpy as np

def costFunction(theta, X,y, return_grad=False):
    """ computes the cost of using theta as the
    parameter for logistic regression and the
    gradient of the cost w.r.t. to the parameters."""

    # Initialize some useful values
    m = len(y)  # number of training examples

    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)



# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta
#
# Note: grad should have the same dimensions as theta

    one = y * np.transpose(np.log(sigmoid(np.dot(X, theta))))
    two = (1 - y) * np.transpose(np.log(1 - sigmoid(np.dot(X, theta))))
    J = -(1. / m) * (one + two).sum()
    grad = (1. / m) * np.dot(sigmoid(np.dot(X, theta)).T - y, X).T

    if return_grad == True:
        return J, np.transpose(grad)
    elif return_grad == False:
        return J  # for use in fmin/fmin_bfgs optimization function
