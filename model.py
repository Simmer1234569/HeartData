import torch
import torch.nn as nn
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Fetch the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
heart_disease = pd.read_csv(url, names=column_names)

# data (as pandas dataframes)
X = heart_disease[['age', 'sex', 'chol', 'thalach', 'trestbps']]
y = heart_disease['num'].apply(lambda x: 1 if x > 0 else 0)  # Ensure binary target

# Convert to tensors
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

# Split into train and test sets
X_train, X_test = X[:250], X[250:]
y_train, y_test = y[:250], y[250:]

# Define the model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(5, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# Instantiate the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Move data to the device
X_train, X_test = X_train.to(device), X_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)

# # Train the model
# epochs = 100000
# for epoch in range(epochs):
#     model.train()
#     optimizer.zero_grad()
#     y_pred = model(X_train)
#     loss = criterion(y_pred, y_train)
#     loss.backward()
#     optimizer.step()
#     # Print loss every 10 epochs
#     if epoch % 10 == 0:
#         print(f'Epoch {epoch} Loss: {loss.item():.3f}')
# torch.save(model.state_dict(), 'model.pth')
#
# # Evaluate the model
# model.eval()
# with torch.no_grad():
#     y_pred = model(X_test)
#     loss = criterion(y_pred, y_test)
#     # Convert raw scores to probabilities using sigmoid
#     y_prob = torch.sigmoid(y_pred)
#     # Apply a threshold to get binary classification
#     y_pred_tag = torch.round(y_prob)
#     # Calculate accuracy or other metrics as required
#     correct_results_sum = (y_pred_tag == y_test).sum().float()
#     acc = correct_results_sum / y_test.shape[0]
#     acc = torch.round(acc * 100)
#
#     print(f'Test loss: {loss.item():.3f}')
#     print(f'Test accuracy: {acc:.2f}%')