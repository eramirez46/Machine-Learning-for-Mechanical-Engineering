import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


def run_mlp_stiffness_model(
    train_path,
    test_path,
    hidden_layer_sizes=
    
    (32,16)
    # (64,32,16)
    ,
    activation='relu',
    solver='adam',
    ###### Change Alpha when iterating
    alpha=0.00001,
    batch_size='auto',
    learning_rate='adaptive',
    learning_rate_init=0.00125,
    max_iter=1000,
    random_state=random.randint(1, 10000),
    plot_results=True
):
    # Load data
   # Load the dataset
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)


    # Remove junk index column
    if 'Unnamed: 0' in train_df.columns:
        train_df = train_df.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in test_df.columns:
        test_df = test_df.drop(columns=['Unnamed: 0'])

    # Select features and target
    feature_cols = [f'PC{i}' for i in range(1, 16)]
    target_col = 'stiffness_value'

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values

    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    # Scale inputs
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    # Build ANN / MLP
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        batch_size=batch_size,
        learning_rate=learning_rate,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        random_state=random_state
    )

    # Train
    mlp.fit(X_train_scaled, y_train)

    # Predict
    y_pred = mlp.predict(X_test_scaled)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print("Hidden layers:", hidden_layer_sizes)
    print("Activation:", activation)
    print("Solver:", solver)
    print("RMSE:", rmse)

    # Plot loss curve
    # plt.figure(figsize=(7, 5))
    # plt.plot(mlp.loss_curve_)
    # plt.title("MLP Loss Curve")
    # plt.xlabel("Iterations")
    # plt.ylabel("Loss")
    # plt.grid(True)
    # plt.show()

    # # Plot predicted vs actual
    # if plot_results:
    #     plt.figure(figsize=(7, 7))
    #     plt.scatter(y_test, y_pred, alpha=0.5)
    #     plt.plot(
    #         [y_test.min(), y_test.max()],
    #         [y_test.min(), y_test.max()],
    #         'r--'
    #     )
    #     plt.xlabel("Actual Stiffness")
    #     plt.ylabel("Predicted Stiffness")
    #     plt.title("Predicted vs Actual Stiffness")
    #     plt.grid(True)
    #     plt.show()

    return mlp, rmse

model, rmse = run_mlp_stiffness_model(
    train_path="/Users/solomonnackashi/Downloads/Microstructure-Stiffness Train Dataset.csv",
    test_path="/Users/solomonnackashi/Downloads/Microstructure-Stiffness Test Dataset.csv"
)