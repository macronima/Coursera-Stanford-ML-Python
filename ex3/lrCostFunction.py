import numpy as np
from ex2.sigmoid import sigmoid
import sys



def lrCostFunction(theta, X, y, lambda_reg):
    # LRCOSTFUNCTION Compute cost and gradient for logistic regression with
    # regularization
    #   J = LRCOSTFUNCTION(theta, X, y, lambda_reg) computes the cost of using
    #   theta as the parameter for regularized logistic regression and the
    #   gradient of the cost w.r.t. to the parameters.



    # Initialize some useful values
    m = len(y)  # number of training examples

    # You need to return the following variables correctly 
    J = 0

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    #

    # taken from costFunctionReg.py

    h = sigmoid(X.dot(theta))

    one =  (np.log(h)).T.dot(y)
    two = (np.log(1 - h)).T.dot(1.0-y)
    reg = lambda_reg / 2.0 / m * ((theta[1:])**2).sum()
    J = -(1.0 / m) * (one + two) + reg
    return J



def lrgradientReg(theta,X,y,reg):
    m=y.size
    h = sigmoid(X.dot(theta))

    grad = (1 / m) * X.T.dot(h - y) + (reg / m) * np.r_[[[0]], theta[1:].reshape(-1, 1)]

    return (grad.flatten())
