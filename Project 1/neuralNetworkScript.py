import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import pandas as pd
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
import glob
import warnings

warnings.filterwarnings("ignore")
testData = pd.read_csv(
    "C:/Users/Edward Ramirez/Documents/Python Playground/Machine-Learning-for-Mechanical-Engineering/Project 1/Microstructure-Stiffness Test Dataset.csv"
)
# print(testData)
trainData = pd.read_csv(
    "C:/Users/Edward Ramirez/Documents/Python Playground/Machine-Learning-for-Mechanical-Engineering\Project 1/Microstructure-Stiffness Train Dataset.csv"
)
# print(trainData)

# n = 0  # starting from the first datapoint
# trainName = trainData.iloc[n, 0]
# trainParameters = trainData.iloc[n, 1:16]
# trainParameters = np.asarray(trainParameters, dtype=float).reshape(15, -1)
# print("Train Name:\n{}".format(trainName))
# print("Train Parameters:\n{}".format(trainParameters))
# print("Done!")


def getTrainParams(n, data):
    trainName = data.iloc[n, 0]
    trainParameters = data.iloc[n, 1:16]
    print("Train Name:\n{}".format(trainName))
    print("Train Parameters:\n{}".format(trainParameters))


getTrainParams(0, trainData)
