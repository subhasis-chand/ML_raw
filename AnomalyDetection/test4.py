import numpy as np
import matplotlib.pyplot as plt
from anomalyDetection import multivariateGauusianAnomaly

data = np.genfromtxt('../resources/breastCancerData/breast-cancer-wisconsin.data', delimiter=',', dtype='float')

data = data[:, 1:] #Removing first column, which is for patient id
data = data[~np.isnan(data).any(axis=1)] #Removing all NaN rows

ipData = data[:, 0:-1]
opData = data[:, -1]

sortedProbList, probList = multivariateGauusianAnomaly(ipData)
print(sortedProbList)

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