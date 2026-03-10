import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
testData = pd.read_csv(
    "C:/Python Playground/Machine-Learning-for-Mechanical-Engineering/Project 1/Microstructure-Stiffness Test Dataset.csv"
)
trainData = pd.read_csv(
    "C:/Python Playground/Machine-Learning-for-Mechanical-Engineering/Project 1/Microstructure-Stiffness Train Dataset.csv"
)

def getTrainParams(n, data):
    trainName = data.iloc[n, 0]
    trainParameters = data.iloc[n, 1:16]
    print("Train Name:\n{}".format(trainName))
    print("Train Parameters:\n{}".format(trainParameters))

for i in range(5):
    getTrainParams(i, trainData)


