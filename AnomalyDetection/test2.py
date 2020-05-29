import numpy as np
import matplotlib.pyplot as plt
from anomalyDetection import detectAnomaly, separateAnomalyData, gaussian, generateMeanSig

data = np.genfromtxt('../resources/breastCancerData/breast-cancer-wisconsin.data', delimiter=',', dtype='float')

data = data[:, 1:] #Removing first column, which is for patient id
data = data[~np.isnan(data).any(axis=1)] #Removing all NaN rows

ipData = data[:, 0:-1]
opData = data[:, -1]

meanSigArr = generateMeanSig(ipData)

# Visualise how the gaussians fit the features
for j in range(ipData.shape[1]):
    d = ipData[:, j]
    x = np.linspace(d.min(), d.max(), 50)
    y = gaussian(x, meanSigArr[j][0], meanSigArr[j][1])

    histMax, binMax, _ = plt.hist(d, bins=10)
    print("feature no: ", j+1)
    plt.plot(x, y*histMax.max())
    plt.show()

#We assume all malignant cases to be anomalies, and try to detect them
# This is not perfect but we can see most of the results are promising 
sortedProb, probList = detectAnomaly(ipData)
malignant, benign =[], []
for i  in range(probList.shape[0]):
    if opData[i] == 2:
        benign.append(probList[i])
    else:
        malignant.append(probList[i])

plt.plot(malignant, 'r.', label='Malignant or Anomaly')
plt.plot(benign, 'g.', label='Benign')
plt.legend()
plt.show()