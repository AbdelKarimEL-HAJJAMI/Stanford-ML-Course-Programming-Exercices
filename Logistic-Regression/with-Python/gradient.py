import numpy as np
from sigmoid import sigmoid

def gradient(theta,X,y, reg_lambda=None, flattenResult=False):
     m,n = X.shape
     theta = theta.reshape((n,1))
     predictions = sigmoid(np.dot(X, theta))
     errors = np.subtract(predictions, y)
     grad = (1.0/m)*np.dot(X.T, errors)
         
     if reg_lambda:
         grad0 = grad[0, :].reshape((1,1))
         grad = grad[1:] + (reg_lambda/m)*theta[1:]
         grad = np.r_[grad0, grad]
     
        
     if  flattenResult:    
         return grad.flatten()
     
     return grad
