import numpy as np

def readMNISTData():
    dataPath = '/Users/subhasis/myWorks/resources/mnist-in-csv/mnist_train.csv'

    print("Reading data...")
    data = np.genfromtxt(dataPath, delimiter=',')
    print("Data reading done...")
    data = data[1:, :]
    opData = data[:, 0]
    ipData = data[:, 1:]
    return ipData, opData