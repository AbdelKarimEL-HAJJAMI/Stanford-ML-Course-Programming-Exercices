import numpy as np
from sigmoid import sigmoid

def costFunction(theta, X, y, reg_lambda = None, flattenResult=False):
    """
    Compute cost for logistic regression
    J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the  
    parameter for logistic regression.
    """
    
    m,n = X.shape
    theta = theta.reshape((n,1))
    predictions = sigmoid(np.dot(X, theta))

    cost_y_1 = np.multiply(np.multiply(-1, y), np.log(predictions))
    cost_y_0 = np.multiply(np.subtract(1,y), np.log(1-predictions))
    
    J = (1.0/m)*np.sum(np.subtract(cost_y_1, cost_y_0))
    
    if reg_lambda:
        J =  J + (reg_lambda/(2*m))*np.sum(np.power(theta[1:], 2))
   
    return J
