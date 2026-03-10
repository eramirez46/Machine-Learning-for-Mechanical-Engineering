import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import math

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

# for features, labels in trainLoader:
#     print(features.shape)
#     print(labels.shape)
#     break

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

epochs = 2
for epoch in range(epochs):
    model.train()
    runningLoss = 0.0

    for features, labels in trainLoader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions.squeeze(), labels)
        loss.backward()
        optimizer.step()

        runningLoss += loss.item() * features.size(0)

    epochLoss = runningLoss / len(trainDataset)
    RMSE = math.sqrt(epochLoss)
    print(f"Epoch {epoch+1}/{epochs} - RMSE: {RMSE:.4f}")