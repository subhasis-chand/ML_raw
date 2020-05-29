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
