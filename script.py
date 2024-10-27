import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
import matplotlib.pyplot as plt
import sys


sys.path.append("/Users/andrei/Desktop/PGM/models")
from notMIWAE import *
from models import *
from utils import *


#%%

from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Fetch dataset
dataset = fetch_ucirepo(id=17)

# Data (as pandas dataframes)
X_df = dataset.data.features
y_df = dataset.data.targets  # Not used for this unsupervised task

# Convert to numpy arrays
X = X_df.values
y = y_df.values

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Introduce missing data
offset = 0.
X_train_missing, mask_train = introduce_mnar_missingness(X_train, offset)
X_val_missing, mask_val = introduce_mnar_missingness(X_val, offset)


# Convert to PyTorch tensors
X_train_missing = torch.tensor(X_train_missing, dtype=torch.float32)
X_val_missing = torch.tensor(X_val_missing, dtype=torch.float32)
mask_train = torch.tensor(mask_train, dtype=torch.float32)
mask_val = torch.tensor(mask_val, dtype=torch.float32)


# Define hyperparameters (from the paper)
d = X.shape[1]          # Input dimension
n_latent = 20        # Latent space dimension (p - 1)
n_samples = 20         # Importance samples
batch_size = 16        # Batch size
num_iterations = 100000        # Number of iterations
num_epochs = num_iterations // (X.shape[0] // batch_size)    # Number of epochs
learning_rate = 1e-3   # Learning rate

# Initialize the model
model = NotMIWAE(d, n_latent, n_hidden, activation, out_dist)

# Train the model
train_loss_history, val_loss_history = train(model, X_train_missing, mask_train, X_val_missing, mask_val, 
                                             batch_size, num_epochs, n_samples, learning_rate)

#%%

rmse_val = rmse_imputation_error(model, torch.tensor(X_val, dtype=torch.float32), mask_val)
print(f"RMSE Imputation Error (Validation): {rmse_val:.4f}")

#%%

plot_s_given_x_lines(model)

#%%

import scipy.io as sio

train_data = sio.loadmat('/Users/andrei/Desktop/PGM/train_32x32.mat')
test_data = sio.loadmat('/Users/andrei/Desktop/PGM/test_32x32.mat')

# access to the dict
X_train = rgb2gray(train_data['X'].transpose((3,0,1,2))) / 255
X_val = rgb2gray(test_data['X'].transpose((3,0,1,2))) / 255


# Introduce missing data
X_train_missing, mask_train = introduce_mnar_missingness_images(X_train)
X_val_missing, mask_val = introduce_mnar_missingness_images(X_val)


# Convert to PyTorch tensors
X_train_missing = torch.tensor(X_train_missing, dtype=torch.float32)
X_val_missing = torch.tensor(X_val_missing, dtype=torch.float32)
mask_train = torch.tensor(mask_train, dtype=torch.float32)
mask_val = torch.tensor(mask_val, dtype=torch.float32)


# Unsqueeze channel dimension
X_train_missing.unsqueeze_(1)
mask_train.unsqueeze_(1)
X_val_missing.unsqueeze_(1)
mask_val.unsqueeze_(1)


# Define hyperparameters (from the paper)
n_samples = 5         # Importance samples
batch_size = 64        # Batch size
num_iterations = 1000000        # Number of iterations
num_epochs = num_iterations // (X_train.shape[0] // batch_size)    # Number of epochs
learning_rate = 1e-3   # Learning rate

# Initialize the model
model = ImageNotMIWAE()

# Train the model
train_loss_history, val_loss_history = train(model, X_train_missing, mask_train, X_val_missing, mask_val, 
                                             batch_size, num_epochs, n_samples, learning_rate)
























