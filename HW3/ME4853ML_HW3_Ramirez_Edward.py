import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import math

# This is where we import the kernels used for creating k1, k2, k3, and k4 in getKernels()
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    WhiteKernel,
    ExpSineSquared,
    DotProduct,
    ConstantKernel,
    RationalQuadratic
)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Simplifies the use of file paths throughout the code
# Adjust these file paths as needed when testing the code
trainPath = "C:/Python Playground/Machine-Learning-for-Mechanical-Engineering/HW3/Microstructure-Stiffness Train Dataset.csv"
testPath = "C:/Python Playground/Machine-Learning-for-Mechanical-Engineering/HW3/Microstructure-Stiffness Test Dataset.csv"

# FUNCTION DEFINITIONS 

# Loads the CSV datasets for training and testing
def loadData(trainPath, testPath):
    trainDf = pd.read_csv(trainPath)
    testDf = pd.read_csv(testPath)
    return trainDf, testDf

# Returns the parameters for the nth training sample
def getTrainParams(n, data):
    trainName = data.iloc[n, 0]
    trainParameters = data.iloc[n, 1:16]
    print("Train Name:\n{}".format(trainName))
    print("Train Parameters:\n{}".format(trainParameters))

# Randomly samples a fraction of the dataset (defined as 10% by default)
def sampleFraction(df, fraction=0.1, randomState=42):
    return df.sample(frac=fraction, random_state=randomState).reset_index(drop=True)

# Separates features (X) and target (y) from the dataframe
def splitFeaturesTargets(df):
    X = df.iloc[:, 1:16].values
    y = df.iloc[:, -1].values
    return X,y

# Standardize features / apply PCA for dimensionality reduction
def applyPCA(XTrain, XTest, nComponents=5):
    scaler = StandardScaler()
    XTrainScaled = scaler.fit_transform(XTrain)
    XTestScaled = scaler.transform(XTest)

    pca = PCA(n_components=nComponents)
    XTrainPca = pca.fit_transform(XTrainScaled)
    XTestPca = pca.transform(XTestScaled)

    return XTrainPca, XTestPca, pca

# To select additional regressors based on highest variance
def selectTopVarianceFeatures(X, nFeatures=5):
    variances = np.var(X, axis=0)
    indices = np.argsort(variances)[-nFeatures:]
    return X[:, indices], indices

# Adds the selected additional features to PCA-transformed data
def augmentFeatures(XPca, XOriginal, nExtra=5):
    XExtra, indices = selectTopVarianceFeatures(XOriginal, nExtra)
    return np.hstack((XPca, XExtra))

# This is the GPR Model! We specify the kernel and optimizer settings here
def buildGprModel(kernel):
    model = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10, # 10 random starting points for hyperparameter optimization
        normalize_y=True # the target values are optimized
    )
    return model

# Runs the training and evaluation processes
# Returns performance metrics like RMSE and R2
def trainAndEvaluate(XTrain, yTrain, XTest, yTest, model):
    # This is where the Log-Marginal Likelihood Maximization happens. 
    model.fit(XTrain, yTrain)
    yPred, yStd = model.predict(XTest, return_std=True)
    mse = mean_squared_error(yTest, yPred)
    rmse = math.sqrt(mse)
    r2 = r2_score(yTest, yPred)

    print("Optimized Kernel:", model.kernel_)
    print("RMSE:", rmse)
    print("R2:", r2)

    return yPred, yStd, rmse, r2

# This is where we define the 4 kernels to be tested using the imports from earlier
# We define the kernels here to keep things parameterized and easy to adjust on the fly
def getKernels():
    # RBF is a standard kernel that assumes the target function varies smoothly
    k1 = ConstantKernel(1.0, (1e-5, 1e5)) * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5)) + WhiteKernel()
    # Dot Product for Linear Trends
    k2 = ConstantKernel(1.0, (1e-5, 1e5)) * DotProduct(sigma_0=1.0) + WhiteKernel()
    # Baseline Kernel (just white noise)
    k3 = WhiteKernel(noise_level=1, noise_level_bounds=(1e-5, 1e1))
    # The added alpha parameter helps to model multiple scales of variation
    k4 = ConstantKernel(1.0, (1e-5, 1e5)) * RationalQuadratic(length_scale=1.0, alpha=1.0) + WhiteKernel()

    return [k1,k2,k3,k4]

# RUN SECTION

# Load the datasets
trainDf, testDf, = loadData(trainPath, testPath)

# Sample just a fraction of both to keep computation time low
trainSample = sampleFraction(trainDf, 0.1, randomState=42)
testSample = sampleFraction(testDf, 0.1, randomState=24)

# Separate the features and the targets
XTrain, yTrain = splitFeaturesTargets(trainSample)
XTest, yTest = splitFeaturesTargets(testSample)

# Get the kernels we defined earlier
kernels = getKernels()

# This large for-loop iterates over difference PCA configurations (small, medium, large)
for nComp in [3,5,8]:
    plt.figure()
    # Initialize a vector to store all the results in (for plotting purposes)
    results = []

    # Apply PCA + augment with additional high-variance features
    print(f"\n--- PCA Components: {nComp} ---")
    XTrainPca, XTestPca, _ = applyPCA(XTrain, XTest, nComponents=nComp)
    XTrainFinal = augmentFeatures(XTrainPca, XTrain, nExtra=5)
    XTestFinal = augmentFeatures(XTestPca, XTest, nExtra=5)

    # this nested for-loop trains + evaluates each kernel
    # performance metrics are appended to results here
    for i, kernel in enumerate(kernels):
        print(f"\nKernel {i+1}")
        model = buildGprModel(kernel)
        yPred, yStd, rmse, r2 = trainAndEvaluate(XTrainFinal, yTrain, XTestFinal, yTest, model)
        results.append((i+1, yPred, rmse, r2))  # no need to store nComp anymore

    # For each kernel, predicted vs true stiffness is plotted
    for kernelIdx, yPred, rmse, r2 in results:
        plt.scatter(
            yTest,
            yPred,
            label=f"K{kernelIdx} (R2={r2:.3f})",
            alpha=0.6
        )

    plt.xlabel("True Stiffness")
    plt.ylabel("Predicted Stiffness")
    plt.title(f"GPR Predictions (PCA = {nComp})")
    plt.legend()
    plt.plot(
        [yTest.min(), yTest.max()],
        [yTest.min(), yTest.max()],
        'r--'
    )
    plt.show()