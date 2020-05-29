import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))#/np.sqrt(2 * np.pi * sig ** 2)

def generateMeanSig(data):
    meanSigArr = np.zeros((data.shape[1], 2))
    for i in range(data.shape[1]):
        meanSigArr[i][0] = np.mean(data[:, i])
        meanSigArr[i][1] = np.std(data[:, i])
    return meanSigArr

def detectAnomalyProbability(dataPoint, meanSigArr):
    anomProb = 1
    for i in range(meanSigArr.shape[0]):
        asdf = gaussian(dataPoint[i], meanSigArr[i][0], meanSigArr[i][1])
        anomProb = anomProb * asdf
    return anomProb

def detectAnomaly(data):
    meanSigArr = generateMeanSig(data)
    probList = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        probList[i] = detectAnomalyProbability(data[i], meanSigArr)
    return np.sort(probList), probList


def separateAnomalyData(data, th):
    sortedProblist, probList = detectAnomaly(data)
    properData = data[probList > th]
    anomalousData = data[probList <= th]
    return properData, anomalousData

# data = np.genfromtxt('../resources/breastCancerData/dataMean.csv', delimiter=',', dtype='float')
data = np.genfromtxt('../resources/serverData/tr_server_data.csv', delimiter=',', dtype='float')
# data = np.genfromtxt('/Users/subhasis/Downloads/breast-cancer-wisconsin.data', delimiter=',', dtype='float')
# positiveData = data[data[:, 0] == 1]
# negativeData = data[data[:, 0] == 0]
# positiveData = positiveData[:, 1:]
# negativeData = negativeData[:, 1:]

# op = data[:, -1]

data = data[1:, :]
# data[:, 3] = np.log10(data[:, 3])
# data[:, 6] = np.log10(data[:, 6])
# data[:, 7] = np.log10(data[:, 7])
# data = np.delete(data, 5, axis=1) # for breast-cancer-wisconsin.data

meanSigArr = generateMeanSig(data)
probList = np.zeros(data.shape[0])
for i in range(data.shape[0]):
    probList[i] = detectAnomalyProbability(data[i], meanSigArr)
print(np.sort(probList))

probList = np.array(probList)
properData = data[probList > 2.8e-02]
anomalousData = data[probList <= 2.8e-02]
print("proper data: ", anomalousData, anomalousData.shape, data.shape)
exit()

# for j in range(data.shape[1]):
#     d = data[:, j]
#     x = np.linspace(d.min(), d.max(), 50)
#     y = gaussian(x, meanSigArr[j][0], meanSigArr[j][1])

#     histMax, binMax, _ = plt.hist(d, bins=30)
#     print(j)
#     plt.plot(x, y*histMax.max())
#     plt.show()

print(probList.min(), probList.max())
histMax, binMax, _ = plt.hist(probList, bins=100)
# plt.plot(np.sort(probList), '.')
plt.show()

# for breast-cancer-wisconsin.data
# malignant, benign =[], []
# for i  in range(probList.shape[0]):
#     if op[i] == 2:
#         benign.append(probList[i])
#     else:
#         malignant.append(probList[i])

# plt.plot(malignant, 'r.')
# plt.plot(benign, 'g.')
# plt.show()
exit()

# col_1 = data[:, 3]

# # print(col_1)

# # histMax, binMax, _ = plt.hist(col_1, bins=int(np.sqrt(data.shape[0]) ))
# # plt.show()

# c1mean = np.mean(col_1)
# print(c1mean)
# c1sig = np.std(col_1)
# print(c1sig)
# colMin = col_1.min()
# colMax = col_1.max()

# x = np.linspace(colMin, colMax, 50)
# normCol1 = gaussian(91, c1mean, c1sig)
# print("norm col: ", normCol1, 0.5**10)
# exit()
# plt.plot(x, normCol1)
# plt.show()
