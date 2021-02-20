import matplotlib.pyplot as plt
import numpy as np

def plotData(X, y, xlabel, ylabel, posLineLabel, negLineLabel):
    """
    PLOTDATA Plots the data points X and y into a new figure 
    PLOTDATA(x,y) plots the data points with + for the positive examples
    and o for the negative examples. X is assumed to be a Mx2 matrix.
    """
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    plt.plot(X[pos, 0], X[pos, 1], marker='+', color='black', label=posLineLabel , linestyle='None')[0]
    plt.plot(X[neg, 0], X[neg, 1], marker='o', color='yellow', label=negLineLabel, linestyle='None')[0]
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
