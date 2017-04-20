from computeCostMulti import computeCostMulti
import numpy as np

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    """
     Performs gradient descent to learn theta
       theta = gradientDescent(x, y, theta, alpha, num_iters) updates theta by
       taking num_iters gradient steps with learning rate alpha
    """

    # Initialize some useful values
    J_history = np.zeros((num_iters, 1))
    m = y.size  # number of training examples

    for i in xrange(num_iters):
        #   ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta.
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.
        #



        # ============================================================
        theta = theta - alpha * (1.0 / m) * X.T.dot(X.dot(theta) - y.T)
        # Save the cost J in every iteration
        J_history[i] = computeCostMulti(X, y, theta)

    return theta, J_history