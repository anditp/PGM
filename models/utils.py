import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
import matplotlib.pyplot as plt


#%%

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
            log_p_x_given_z = (output_dist.log_prob(x.unsqueeze(1)) * s[:,None,:]).sum(-1)

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
    log_p_s_given_x = p_s_given_x.log_prob(s.unsqueeze(1)).sum(-1)
    
    q_z_given_x = dist.Normal(loc=q_mu.unsqueeze(1), scale=torch.exp(q_log_sig2.unsqueeze(1) / 2))
    log_q_z_given_x = q_z_given_x.log_prob(l_z).sum(-1)
    
    # NotMIWAE loss
    prior = dist.Normal(loc=torch.tensor(0.0), scale=torch.tensor(1.0))
    log_p_z = prior.log_prob(l_z).sum(-1)
    
    
    l_w = log_p_x_given_z + log_p_s_given_x + log_p_z - log_q_z_given_x 
    log_avg_weight = torch.logsumexp(l_w, dim=1) - torch.log(torch.tensor(n_samples, dtype=torch.float32))
    notmiwae_loss = -torch.mean(log_avg_weight)

    return notmiwae_loss



def notmiwae_loss_image(mu, std, q_mu, q_log_sig2, l_z, l_out_mixed, logits_miss, x, s, out_dist='gauss', n_samples=1):
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


def train(model, X_train, S_train, X_val, S_val, batch_size=128, num_epochs=100, n_samples=1, learning_rate=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_loss_history = []
    val_loss_history = []

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



def train_image(model, X_train, S_train, X_val, S_val, batch_size=128, num_epochs=100, n_samples=1, learning_rate=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_loss_history = []
    val_loss_history = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for i in range(0, len(X_train), batch_size):
            x_batch = X_train[i:i+batch_size]
            s_batch = S_train[i:i+batch_size]

            optimizer.zero_grad()
            mu, std, q_mu, q_log_sig2, l_z, l_out_mixed, logits_miss = model(x_batch, s_batch, n_samples)
            
            
            # Calculate loss
            loss = notmiwae_loss_image(mu, std, q_mu, q_log_sig2, l_z, l_out_mixed, logits_miss, x_batch, s_batch, model.out_dist, n_samples) 
            
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
                val_loss += notmiwae_loss_image(mu, std, q_mu, q_log_sig2, l_z, l_out_mixed, logits_miss, x_batch_val, s_batch_val, model.out_dist, n_samples).item() * x_batch_val.size(0)
            val_loss /= len(X_val)
            val_loss_history.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
    return train_loss_history, val_loss_history



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



#%%

def plot_s_given_x_lines(model):
    t = torch.linspace(-2, 2, 200)
    W = model.missing_process_decoder.W
    b = model.missing_process_decoder.b
    probs = torch.zeros((model.d, 200))
    
    with torch.no_grad():
        plt.figure(figsize=(10, 5))
        for i in range(model.d):
            probs[i] = torch.sigmoid(- W[0,0,i] * (t - b[0,0,i]))
            plt.plot(t, probs[i], alpha=0.7, label=f"Feature {i+1}")
    
        plt.xlabel("Feature Value (x)")
        plt.ylabel("Probability of Missingness (s)")
        plt.title("Probability of Missingness vs. Feature Value")
        plt.legend()
        plt.show()

