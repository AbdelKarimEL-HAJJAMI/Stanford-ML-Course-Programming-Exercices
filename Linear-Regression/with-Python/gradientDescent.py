import numpy as np
import computeCost as costModule

def gradientDescent(X, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn theta
    theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
    taking num_iters gradient steps with learning rate alpha
    """
    m = y.size # number of training examples
    J_history = np.zeros((num_iters, 1))
    
    for iter in range(num_iters):
         prediction = np.dot(X, theta)
         errors = np.subtract(prediction, y)
         theta = theta - (alpha/m)*np.dot(X.T, errors)
         #save the cost J in every iteration
         J_history[iter] = costModule.computeCost(X, y, theta)
     
    return J_history, theta
