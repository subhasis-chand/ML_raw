import numpy as np
import matplotlib.pyplot as plt
from anomalyDetection import detectAnomaly, separateAnomalyData

fig, ax = plt.subplots(2)
plt.subplots_adjust(hspace = 0.8)

data = np.genfromtxt('../resources/serverData/tr_server_data.csv', delimiter=',', dtype='float')
data = data[1:, :]

sortedProb, prob = detectAnomaly(data)
print("Sorted probabilities of data being anomalous. Check for a threshold value: \n", sortedProb)

threshold = 0.028 #Selected from above print
properData, anomalousData = separateAnomalyData(data, threshold, prob)

ax[0].title.set_text("All the data points")
ax[1].title.set_text("Anomaly in the data")
ax[0].set_xlabel('Latency(ms)')
ax[1].set_xlabel('Latency(ms)')
ax[0].set_ylabel('Throughput(mb/s)')
ax[1].set_ylabel('Throughput(mb/s)')

ax[0].plot(data[:, 0], data[:, 1], '.b')
ax[1].plot(properData[:, 0], properData[:, 1], '.g', label='Proper Data')
ax[1].plot(anomalousData[:, 0], anomalousData[:, 1], '.r', label='Anomaly Data')
plt.legend()
plt.show()




