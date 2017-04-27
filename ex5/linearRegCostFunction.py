import numpy as np
def linearRegCostFunction(X, y, theta, reg):
    """computes the
    cost of using theta as the parameter for linear regression to fit the
    data points in X and y. Returns the cost in J and the gradient in grad
    """
# Initialize some useful values
    return_grad = True
    m = len(y)  # number of training examples

    # force to be 2D vector
   # theta = np.reshape(theta, (-1, y.shape[1]))

    # You need to return the following variables correctly


    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost and gradient of regularized linear
    #               regression for a particular choice of theta.
    #
    #               You should set J to the cost and grad to the gradient.
    #

    # cost function
    J = (1. / (2 * m)) * np.power((np.dot(X, theta) - y), 2).sum() + (float(reg) / (2 * m)) * np.power(
        theta[1:theta.shape[0]], 2).sum()

    # regularized gradient
    grad = (1. / m) * np.dot(X.T, np.dot(X, theta) - y) + (float(reg) / m) * theta

    # unregularize first gradient
    grad_no_regularization = (1. / m) * np.dot(X.T, np.dot(X, theta) - y)
    grad[0] = grad_no_regularization[0]

    if return_grad == True:
        return J, grad.flatten()
    elif return_grad == False:
        return J