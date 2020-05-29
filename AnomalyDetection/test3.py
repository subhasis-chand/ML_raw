import numpy as np
import matplotlib.pyplot as plt
from anomalyDetection import detectAnomaly, separateAnomalyData, generateMeanSig, gaussian

data = np.genfromtxt('../resources/breastCancerData/dataMean.csv', delimiter=',', dtype='float')
opData = data[:, 0]
ipData = data[:, 1:]

meanSigArr = generateMeanSig(ipData)

for j in range(ipData.shape[1]):
    d = ipData[:, j]
    x = np.linspace(d.min(), d.max(), 50)
    y = gaussian(x, meanSigArr[j][0], meanSigArr[j][1])

    histMax, binMax, _ = plt.hist(d, bins=30)
    print("feature no: ", j+1)
    plt.plot(x, y*histMax.max())
    plt.show()

sortedProb, prob = detectAnomaly(ipData)
print("Sorted probabilities of data being anomalous. Check for a threshold value: \n", sortedProb)

threshold = 1.5e-06 #Selected from above print
properData, anomalousData = separateAnomalyData(data, threshold)


