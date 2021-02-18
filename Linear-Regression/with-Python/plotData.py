import matplotlib.pyplot as plt

def plotData(x, y):
    """Plots the data points x and y into a new figure"""
  
    datasetPlot = plt.plot(x, y,  linestyle='None', color='red', marker='x', markersize=10, label="Training data")
    plt.xlabel('Profit in $10,000')
    plt.ylabel('Population of city in 10,000s')
    
    return datasetPlot
