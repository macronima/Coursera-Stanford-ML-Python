import numpy as np
from ex2.sigmoid import sigmoid
import sys


def lrCostFunction(theta, X, y, lambda_reg, return_grad=False):
    # LRCOSTFUNCTION Compute cost and gradient for logistic regression with
    # regularization
    #   J = LRCOSTFUNCTION(theta, X, y, lambda_reg) computes the cost of using
    #   theta as the parameter for regularized logistic regression and the
    #   gradient of the cost w.r.t. to the parameters.



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

    # taken from costFunctionReg.py

    one = y * (np.log(sigmoid(X.dot(theta)))).T.sum()
    two = (1.0 - y) * (np.log(1 - sigmoid(X.dot(theta)))).T.sum()
    reg = lambda_reg / 2.0 / m * (np.power(theta[1:theta.shape[0]],2)).sum()
    J = -(1.0 / m) * (one + two) + reg

    beta = ((sigmoid(X.dot(theta)).T - y).dot(X)).T

    grad = (1.0 / m) * beta + (lambda_reg / m) * (theta)

    # the case of j = 0 (recall that grad is a n+1 vector)
    grad_no_regularization = (1.0 / m) * beta

    # and then assign only the first element of grad_no_regularization to grad
    grad[0] = grad_no_regularization[0]

    # display cost at each iteration
    sys.stdout.write("Cost: %f   \r" % (J))
    sys.stdout.flush()

    if return_grad:
        return J, grad
    else:
        return J
