import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
import matplotlib.pyplot as plt
import sys
#%%



class ConvEncoder(nn.Module):
    def __init__(self):
        super(ConvEncoder, self).__init__()
        
        self.encoder = nn.Sequential(nn.Conv2d(1, 64, 7, padding = 3),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2),
                                     nn.Conv2d(64, 128, 3, padding = 1),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2),
                                     nn.Conv2d(128, 256, 3, padding = 1),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2),
                                     nn.Flatten()
                                     )
        self.mu_head = nn.Linear(4096, 20)
        self.std_head = nn.Linear(4096, 20)
        
        
    def forward(self, x):
        enc = self.encoder(x)
        mu = self.mu_head(enc)
        std = self.std_head(enc)
        
        return mu, std


class ConvDecoder(nn.Module):
    def __init__(self):
        super(ConvDecoder, self).__init__()
        
        self.linear = nn.Sequential(nn.Linear(20, 4096),
                                     nn.ReLU())
        self.cnn = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, stride = 2, padding = 1),
                                 nn.ReLU(),
                                 nn.ConvTranspose2d(128, 64, 4, stride = 2, padding = 1),
                                 nn.ReLU())
        self.mu_head = nn.Sequential(nn.ConvTranspose2d(64, 64, 4, stride = 2, padding = 1),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(64, 1, 3, stride = 1, padding = 1),
                                     nn.Sigmoid())
        self.std_head = nn.Sequential(nn.ConvTranspose2d(64, 64, 4, stride = 2, padding = 1),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(64, 1, 3, stride = 1, padding = 1))
    
    def forward(self, x):
        z = self.linear(x)
        z = torch.unflatten(z, -1, (256, 4, 4))
        z = torch.flatten(z, 0, 1)
        z_cnn = self.cnn(z)
        mu = self.mu_head(z_cnn)
        mu = torch.unflatten(mu, 0, (64, 5))
        std = self.std_head(z_cnn).clamp(min = 0.02)
        std = torch.unflatten(std, 0, (64, 5))
        
        return mu, std



class MissingProcessDecoder(nn.Module):
    def __init__(self, d, n_hidden, missing_process):
        super(MissingProcessDecoder, self).__init__()
        
        self.d = d
        self.n_hidden = n_hidden
        self.missing_process = missing_process
        
        # Missing process (adjust based on missing_process)
        if missing_process == 'agnostic':
            self.W = nn.Parameter(torch.randn(1, 1, self.d))
            self.b = nn.Parameter(torch.randn(1, 1, self.d))
        elif missing_process == 'selfmasking_known':
            self.W = -50
            self.b = .75
        else:
            raise ValueError("Invalid missing_process. Use 'selfmasking', 'selfmasking_known', 'linear', or 'nonlinear'")
            
    def forward(self, z):
        if self.missing_process == 'agnostic':
            logits = - self.W * (z - self.b)
        elif self.missing_process == 'selfmasking_known':
            logits = - F.softplus(self.W * (z - self.b))
        elif self.missing_process == 'linear':
            logits = nn.Linear(z.size(2), self.d)(z)  # Apply linear layer
        elif self.missing_process == 'nonlinear':
            z = torch.tanh(nn.Linear(z.size(2), self.n_hidden)(z))
            logits = nn.Linear(self.n_hidden, self.d)(z)  # Apply non-linear layers
        else:
            raise ValueError("Invalid missing_process")
        return logits



class ImageMissingProcessDecoder(nn.Module):
    """
    Missing process decoder for images.
    """
    def __init__(self, image_shape, missing_process='selfmasking'):
        super().__init__()
        self.image_shape = image_shape
        self.missing_process = missing_process
        if missing_process == 'selfmasking':
            self.W = nn.Parameter(torch.randn(image_shape))
            self.b = nn.Parameter(torch.randn(image_shape))
        elif missing_process == 'selfmasking_known':
            self.W = -50
            self.b = .75
        else:
            raise ValueError("Invalid missing_process.")

    def forward(self, x):
        if self.missing_process == 'selfmasking':
            logits = -self.W * (x - self.b)
        elif self.missing_process == 'selfmasking_known':
            logits = -F.softplus(self.W * (x - self.b))
        else:
            raise ValueError("Invalid missing_process.")
        return logits
    
    
    



#%%


def train(model, X_train, S_train, X_val, S_val, batch_size=128, num_epochs=100, n_samples=1, learning_rate=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_loss_history = []
    val_loss_history = []
    X_train = X_train.clone()[:X_train.shape[0] - (X_train.shape[0] % batch_size)]
    X_train = X_val.clone()[:X_val.shape[0] - (X_val.shape[0] % batch_size)]
    
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for i in range(0, len(X_train), batch_size):
            x_batch = X_train[i:i+batch_size]
            s_batch = S_train[i:i+batch_size]

            optimizer.zero_grad()
            mu, std, q_mu, q_log_sig2, l_z, l_out_mixed, logits_miss = model(x_batch, s_batch, n_samples)
            
            
            # Calculate loss
            loss = notmiwae_loss(mu, std, q_mu, q_log_sig2, l_z, l_out_mixed, logits_miss, x_batch, s_batch, model.out_dist, n_samples) 
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)


        train_loss /= len(X_train)
        train_loss_history.append(train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for i in range(0, len(X_val), batch_size):
                x_batch_val = X_val[i:i+batch_size]
                s_batch_val = S_val[i:i+batch_size]
                mu, std, q_mu, q_log_sig2, l_z, l_out_mixed, logits_miss = model(x_batch_val, s_batch_val, n_samples)
                val_loss += notmiwae_loss(mu, std, q_mu, q_log_sig2, l_z, l_out_mixed, logits_miss, x_batch_val, s_batch_val, model.out_dist, n_samples).item() * x_batch_val.size(0)
            val_loss /= len(X_val)
            val_loss_history.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
    return train_loss_history, val_loss_history




class ImageNotMIWAE(nn.Module):
    """
    NotMIWAE model for images of size 1x32x32.
    """
    def __init__(self, image_shape=(1, 32, 32), n_latent=20,
                 missing_process='selfmasking_known'):
        super().__init__()
        self.image_shape = image_shape
        self.n_latent = n_latent
        self.out_dist = 'gauss'
        self.encoder = ConvEncoder()
        self.decoder = ConvDecoder()
        self.missing_process_decoder = ImageMissingProcessDecoder(image_shape, missing_process)

    def forward(self, x, s, n_samples=1):
        q_mu, q_log_sig2 = self.encoder(x)
        
        # Reparameterization trick
        eps = torch.randn_like(q_mu).unsqueeze(1).repeat(1, n_samples, 1)
        l_z = q_mu.unsqueeze(1) + eps * torch.exp(0.5 * q_log_sig2).unsqueeze(1)

        # Decode and sample
        mu, std = self.decoder(l_z)
        
        # Reparameterization trick for the output
        eps_out = torch.randn_like(mu) 
        l_out_sample = mu + std * eps_out

        # Missing data process
        l_out_mixed = l_out_sample * (1 - s).unsqueeze(1) + x.unsqueeze(1)
        logits_miss = self.missing_process_decoder(l_out_mixed)

        return mu, std, q_mu, q_log_sig2, l_z, l_out_mixed, logits_miss





#%%

def rmse_imputation_error(model, X_test, mask_test, n_samples=10000):
    """Calculates the RMSE imputation error.

    Args:
      model: The trained NotMIWAE model.
      X_test: The test data with missing values.
      mask_test: The mask indicating missing values in X_test.
      n_samples: The number of samples to use for estimation.

    Returns:
      The RMSE imputation error.
    """
    model.eval()
    with torch.no_grad():
        mu, std, _, _, _, l_out_mixed, _ = model(X_test, mask_test, n_samples)

        # For Gaussian output, the imputed values are the mean of the output distribution
        if model.out_dist in ['gauss', 'normal', 'truncated_normal']:
            imputed_values = mu.mean(dim=1)  # Average over samples
        elif model.out_dist == 'bern':
            # For Bernoulli, use the probabilities (sigmoid of logits)
            imputed_values = torch.sigmoid(mu).mean(dim=1)
        elif model.out_dist in ['t', 't-distribution']:
            # For Student's t, use the mean
            imputed_values = mu.mean(dim=1)
        else:
            raise ValueError("Invalid out_dist")

        # Calculate RMSE only for the missing values
        mask_test = mask_test.int().clone()
        rmse = torch.sqrt(F.mse_loss(imputed_values[~mask_test], X_test[~mask_test]))

    return rmse.item()


def introduce_mnar_missingness(X, offset=0):
    """Introduces MNAR missingness to the data.

    Args:
      X: The data.
      offset: The offset to add to the mean for the cutoff point.

    Returns:
      X_missing: The data with MNAR missingness.
      mask: The mask indicating missing values.
    """
    X_missing = X.copy()
    mask = np.ones_like(X, dtype=bool)
    cutoff = np.mean(X, axis = 0) + offset * np.std(X, axis = 0)
    mask = X <= cutoff  # Values above cutoff are missing
    X_missing[~mask] = 0
    return X_missing, mask




def introduce_mnar_missingness_images(X):
    logits = -50 * (X - 0.75)
    mask = np.random.binomial(1, 1 / (1 + np.exp(-logits)), size = X.shape)
    X_missing = X * mask
    return X_missing, mask




def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])




def notmiwae_loss(mu, std, q_mu, q_log_sig2, l_z, l_out_mixed, logits_miss, x, s, out_dist='gauss', n_samples=1):
    # Reconstruction loss (adapted for different out_dist)
    if out_dist in ['gauss', 'normal', 'truncated_normal']:
        output_dist = dist.Normal(loc=mu, scale=std)
        if out_dist == 'truncated_normal':
            # For truncated normal, we need to handle the log_prob differently
            log_p_x_given_z = output_dist.log_prob(
                x.unsqueeze(1).clamp(0.0001, 0.9999)
            ).sum(-1)
        else:
            log_p_x_given_z = (output_dist.log_prob(x.unsqueeze(1)) * s[:,None,:]).sum((-2,-1)).squeeze(-1)

    elif out_dist == 'bern':
        logits = mu  # For Bernoulli, mu represents the logits
        output_dist = dist.Bernoulli(logits=logits)
        log_p_x_given_z = output_dist.log_prob(x.unsqueeze(1)).sum(-1)

    elif out_dist in ['t', 't-distribution']:
        df = std  # For Student's t, std represents the degrees of freedom
        output_dist = dist.StudentT(loc=mu, scale=1.0, df=df)  # Assuming scale=1 for simplicity
        log_p_x_given_z = output_dist.log_prob(x.unsqueeze(1)).sum(-1)

    else:
        raise ValueError("Invalid out_dist")

    # Missing process loss (adapted for different missing_process)
    p_s_given_x = dist.Bernoulli(logits=logits_miss)
    log_p_s_given_x = p_s_given_x.log_prob(s.unsqueeze(1)).sum((-2,-1)).squeeze(-1)
    
    q_z_given_x = dist.Normal(loc=q_mu.unsqueeze(1), scale=torch.exp(q_log_sig2.unsqueeze(1) / 2))
    log_q_z_given_x = q_z_given_x.log_prob(l_z).sum((-1))
    
    # NotMIWAE loss
    prior = dist.Normal(loc=torch.tensor(0.0), scale=torch.tensor(1.0))
    log_p_z = prior.log_prob(l_z).sum((-1))
    
    l_w = log_p_x_given_z + log_p_s_given_x + log_p_z - log_q_z_given_x 
    log_avg_weight = torch.logsumexp(l_w, dim=1) - torch.log(torch.tensor(n_samples, dtype=torch.float32))
    notmiwae_loss = -torch.mean(log_avg_weight)

    return notmiwae_loss






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
X_train_missing = torch.tensor(X_train_missing, dtype=torch.float32)[:73216]
X_val_missing = torch.tensor(X_val_missing, dtype=torch.float32)[:25984]
mask_train = torch.tensor(mask_train, dtype=torch.float32)[:73216]
mask_val = torch.tensor(mask_val, dtype=torch.float32)[:25984]


# Unsqueeze channel dimension
X_train_missing.unsqueeze_(1)
mask_train.unsqueeze_(1)
X_val_missing.unsqueeze_(1)
mask_val.unsqueeze_(1)

print(X_val_missing.shape)

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



#%%


rmse_val = rmse_imputation_error(model, torch.tensor(X_val, dtype=torch.float32)[:10000].unsqueeze_(1), mask_val)
print(f"RMSE Imputation Error (Validation): {rmse_val:.4f}")





























