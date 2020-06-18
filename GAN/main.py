import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
from LoadData import readMNISTData
import matplotlib.pyplot as plt

noiseArrayLen = 100

class Discriminator(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        n_features = 784
        n_out = 1
        
        self.hidden0 = nn.Sequential( 
            nn.Linear(n_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(256, n_out),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

class Generator(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self):
        super(Generator, self).__init__()
        n_features = 100
        n_out = 784
        
        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )
        
        self.out = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

def randomNoise(batchSize):
    arr = np.zeros((batchSize, noiseArrayLen))
    for i in range(batchSize):
        arr[i] = np.random.normal(0, 0.4, noiseArrayLen)
    return torch.from_numpy(arr).float()
    # return torch.randn(batchSize, noiseArrayLen)

dNet = Discriminator()
gNet = Generator()

print(dNet, gNet)

optimizer_g = torch.optim.Adam(gNet.parameters(), lr=0.0002)
optimizer_d = torch.optim.Adam(dNet.parameters(), lr=0.0002)

loss = nn.BCELoss()

print("Loading data...")
ipData, opData = readMNISTData()
print("Done Loading Data...")

batchSize = 100
totalNoOfData = opData.shape[0]
noOfEpochs = 400
for i in range(noOfEpochs):
    # print("epoch: ", i+1)
    for j in range(0, totalNoOfData, batchSize):
        ipBatch = ipData[j:j+batchSize , :]/255.

        ipBatch = torch.from_numpy(ipBatch).float()

        optimizer_d.zero_grad()
        noise = randomNoise(batchSize)
        dNetOut_real = dNet(ipBatch)
        dNetOut_fake = dNet(gNet(noise))
        real_loss = loss(dNetOut_real, torch.ones(batchSize, 1))
        real_loss.backward()
        fake_loss = loss(dNetOut_fake, torch.zeros(batchSize, 1))
        fake_loss.backward()
        optimizer_d.step()

        optimizer_g.zero_grad()
        noise = randomNoise(batchSize)
        generatorOut = dNet(gNet(noise))
        gen_loss = loss(generatorOut, torch.ones(batchSize, 1))
        gen_loss.backward()
        optimizer_g.step()

        print(i, j, real_loss, fake_loss, gen_loss)

        noise1 = randomNoise(batchSize)
        displaOut = gNet(noise1)
        displaOut = displaOut.detach().numpy()
        todisp = np.reshape(displaOut[0], (28, 28))
        todisp = todisp * 255
        todisp = todisp.astype('int')

        if j == 200:
            imgName = './output/epoch' + str(i+1)+ '.png'
            plt.imsave(imgName, todisp, cmap='gray')
            print("image saved...")

        
