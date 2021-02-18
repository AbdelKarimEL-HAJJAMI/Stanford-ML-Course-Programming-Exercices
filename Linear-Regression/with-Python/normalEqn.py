import numpy as np

def normalEqn(X, y):
    """
    Computes the closed-form solution to linear regression 
    using the normal equations
    """
    f = X.shape[1] # count features
    theta = np.zeros((f, 1))
    print("shape theta", theta.shape)
    
    
    theta = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, y))
    
    return theta
