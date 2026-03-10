import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# Load the dataset
train_df = pd.read_csv("/Users/solomonnackashi/Downloads/Microstructure-Stiffness Train Dataset.csv")
test_df = pd.read_csv("/Users/solomonnackashi/Downloads/Microstructure-Stiffness Test Dataset.csv")

print(train_df.shape)
print(test_df.shape)
print(train_df.head())


# Split in to inputs and outputs and reshape
X = train_df.iloc[:, 1:16].values
y = train_df.iloc[:, -1].values.reshape(-1, 1)

X_test = test_df.iloc[:, 1:16].values
y_test = test_df.iloc[:, -1].values.reshape(-1, 1)


# validation set
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Scale the data
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train = x_scaler.fit_transform(X_train)
X_val   = x_scaler.transform(X_val)
X_test  = x_scaler.transform(X_test)

y_train = y_scaler.fit_transform(y_train)
y_val   = y_scaler.transform(y_val)
y_test_scaled = y_scaler.transform(y_test)

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)

X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)

X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test_scaled, dtype=torch.float32)

# Create DataLoaders
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the ANN model
class StiffnessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            ## Best so far
            # nn.Linear(15, 32),
            # nn.ReLU(),
            # nn.Linear(32, 16),
            # nn.ReLU(),
            # nn.Linear(16, 1),

#            Current Test 
            nn.Linear(15, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)

        )

    def forward(self, x):
        return self.model(x)

model = StiffnessNet()

# Define loss and optimizer
criterion = nn.MSELoss()
######################################## CHANGE LEARNING RATE AND WEIGHT DECAY HERE ###############################################
optimizer = torch.optim.Adam(model.parameters(), lr=0.0015, weight_decay=1e-3)

# Train the model
num_epochs = 200
best_val_rmse = float("inf")
best_model_state = None

for epoch in range(num_epochs):
    model.train()

    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t).numpy()
        val_preds_unscaled = y_scaler.inverse_transform(val_preds)
        y_val_unscaled = y_scaler.inverse_transform(y_val_t.numpy())

        val_rmse = np.sqrt(mean_squared_error(y_val_unscaled, val_preds_unscaled))

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_model_state = model.state_dict()

    if (epoch + 1) % 25 == 0:
        print(f"Epoch {epoch+1}: Validation RMSE = {val_rmse:.6f}")

# Evaluate on the test set
model.load_state_dict(best_model_state)

model.eval()
with torch.no_grad():
    test_preds = model(X_test_t).numpy()

test_preds_unscaled = y_scaler.inverse_transform(test_preds)
test_rmse = np.sqrt(mean_squared_error(y_test, test_preds_unscaled))

print("Final Test RMSE:", test_rmse)

# Plot predictions vs actual
# plt.figure(figsize=(6,6))
# plt.scatter(y_test, test_preds_unscaled, alpha=0.5)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
# plt.xlabel("Actual Stiffness")
# plt.ylabel("Predicted Stiffness")
# plt.title("Predicted vs Actual Stiffness")
# plt.grid(True)
# plt.show()

