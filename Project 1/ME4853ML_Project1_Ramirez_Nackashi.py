import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import math
import time

# Author's Note (Edward Ramirez)
# this script runs a CNN model for 200 epochs. It outputs the best RMSE and also plots the actual vs predicted stiffness.

# customize the path file locations to your computer!
trainPath = "C:/Python Playground/Machine-Learning-for-Mechanical-Engineering/Project 1/Microstructure-Stiffness Train Dataset.csv"
testPath = "C:/Python Playground/Machine-Learning-for-Mechanical-Engineering/Project 1/Microstructure-Stiffness Test Dataset.csv"

def getTrainParams(n, data):
    trainName = data.iloc[n, 0]
    trainParameters = data.iloc[n, 1:16]
    print("Train Name:\n{}".format(trainName))
    print("Train Parameters:\n{}".format(trainParameters))

# for i in range(5):
#     getTrainParams(i, trainData)

class MicrostructureDataset(Dataset):

    def __init__(self, csvPath):
        self.dataFrame = pd.read_csv(csvPath)

        self.names = self.dataFrame.iloc[:, 0]
        self.features = self.dataFrame.iloc[:, 1:16].values.astype(np.float32)
        self.labels = self.dataFrame.iloc[:, 16].values.astype(np.float32)

    def __len__(self):
        return len(self.dataFrame)

    def __getitem__(self, index):
        featureTensor = torch.tensor(self.features[index]).unsqueeze(0)
        labelTensor = torch.tensor(self.labels[index])
        return featureTensor, labelTensor

trainDataset = MicrostructureDataset(trainPath)
testDataset = MicrostructureDataset(testPath)

# Dataloader:

trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=True)
testLoader = DataLoader(testDataset, batch_size=32)

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.convLayers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3),
            nn.ReLU()
        )
        self.fcLayers = nn.Sequential(
            nn.Linear(32 * 11, 64),  # 15 → conv1 → 13 → conv2 → 11 length
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.convLayers(x)
        x = x.view(x.size(0), -1)
        return self.fcLayers(x)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    

def computeRMSE(dataLoader, model, device, criterion):
    model.eval()
    runningLoss = 0.0
    totalSamples = 0
    with torch.no_grad():
        for features, labels in dataLoader:
            features, labels = features.to(device), labels.to(device)
            predictions = model(features)
            loss = criterion(predictions.squeeze(), labels)
            runningLoss += loss.item() * features.size(0)
            totalSamples += features.size(0)
    mse = runningLoss / totalSamples
    rmse = math.sqrt(mse)
    return rmse

# Training with 1 Test Per Epoch:
startTime = time.time()
epochs = 200
trainRMSEs = []
testRMSEs = []

for epoch in range(epochs):
    model.train()
    # Training step
    for features, labels in trainLoader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions.squeeze(), labels)
        loss.backward()
        optimizer.step()

    trainRMSE = computeRMSE(trainLoader, model, device, criterion)
    testRMSE = computeRMSE(testLoader, model, device, criterion)
    trainRMSEs.append(trainRMSE)
    testRMSEs.append(testRMSE)

    print(f"Epoch {epoch+1}/{epochs} - Train RMSE: {trainRMSE:.4f}, Test RMSE: {testRMSE:.4f}")
endTime = time.time()
totalTime = endTime - startTime
print(f"\nTotal Training Runtime: {totalTime:.2f} seconds")

bestTestRMSE = min(testRMSEs)
bestEpoch = testRMSEs.index(bestTestRMSE) + 1
print(f"Best Test RMSE: {bestTestRMSE:.4f} at epoch {bestEpoch}")

model.eval()
allPreds = []
allActuals = []

with torch.no_grad():
    for features, labels in testLoader:
        features, labels = features.to(device), labels.to(device)
        predictions = model(features)
        allPreds.extend(predictions.squeeze().cpu().numpy())
        allActuals.extend(labels.cpu().numpy())


plt.figure(figsize=(8,5))
plt.plot(range(1, epochs+1), trainRMSEs, label="Train RMSE", marker='o')
plt.plot(range(1, epochs+1), testRMSEs, label="Test RMSE", marker='s')
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title("Train vs Test RMSE per Epoch")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(6,6))
plt.scatter(allActuals, allPreds, color='blue', alpha=0.6)
plt.plot([min(allActuals), max(allActuals)],
         [min(allActuals), max(allActuals)],
         color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel("Actual Stiffness")
plt.ylabel("Predicted Stiffness")
plt.title("Predicted vs Actual Stiffness")
plt.legend()
plt.grid(True)
plt.show()