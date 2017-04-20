from numpy import asfortranarray, squeeze, asarray
from sigmoid import sigmoid
import numpy as np
from gradientFunction import gradientFunction


def gradientFunctionReg(theta, X, y, Lambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    m = len(y)   # number of training examples

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the gradient of a particular choice of theta.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta


# =============================================================

    h = sigmoid(X.dot(theta))

    grad = (1.0 / m) * X.T.dot(h - y) + (Lambda / m) * np.r_[[0], theta[1:]]

    return (grad.flatten())