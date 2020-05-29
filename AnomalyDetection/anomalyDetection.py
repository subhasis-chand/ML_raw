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


def separateAnomalyData(data, th, probList):
    properData = data[probList > th]
    anomalousData = data[probList <= th]
    return properData, anomalousData

def multivariateGauusian(x, mu, cov, det):
    expTerm = -0.5 * (x - mu).T * np.linalg.inv(cov) * (x - mu)
    # The next line is actual formula and the probability distribution is a prob density function
    # which means the area under the curve is 1.
    # Howevr it gives the prob values to be very very small
    # So we use the scaled version. except for the area under the curve it does not affect anything else
    # as we are just removing the constant term 
    # prob = (1.0/(np.power(2*np.pi, x.shape[0]/2) * np.sqrt(det))) * np.power(np.e, expTerm) #Actual formula
    prob = np.power(np.e, expTerm) # Scaled version
    return prob[0][0]

def multivariateGauusianAnomaly(data):
    mu = np.mean(data, axis=0)
    mu = np.reshape(np.matrix(mu), (data.shape[1], 1))
    cov = np.matrix(np.cov(data.T))
    det = np.linalg.det(cov)

    probList = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        x = np.matrix(data[i]).T
        probList[i] = multivariateGauusian(x, mu, cov, det)

    return np.sort(probList), probList

    