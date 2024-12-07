import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

class Encoder(nn.Module):
    def __init__(self, d, n_latent=50, n_hidden=100, activation = nn.Tanh()):
        super(Encoder, self).__init__()
        
        self.d = d
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.activation = activation
        
        self.encoder = nn.Sequential(nn.Linear(self.d, self.n_hidden), 
                                     self.activation,
                                     nn.Linear(self.n_hidden, self.n_hidden),
                                     self.activation)
        self.mu_head = nn.Linear(self.n_hidden, self.n_latent)
        self.std_head = nn.Linear(self.n_hidden, self.n_latent)
        
    def forward(self, x):
        enc = self.encoder(x)
        mu = self.mu_head(enc)
        std = self.std_head(enc)
        
        return mu, torch.clamp(std, min = -10, max = 10)


class Decoder(nn.Module):
    def __init__(self, d, n_latent=50, n_hidden=100, out_dist='gauss', activation=nn.ReLU()):
        super(Decoder, self).__init__()
        
        self.d = d
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.activation = activation
        self.out_dist = out_dist
        
        # Decoder (adjust based on out_dist)
        if out_dist in ['gauss', 'normal', 'truncated_normal']:
            self.dec_gauss1 = nn.Linear(self.n_latent, self.n_hidden)
            self.dec_gauss2 = nn.Linear(self.n_hidden, self.n_hidden)
            self.mu_layer = nn.Linear(self.n_hidden, self.d)
            self.std_layer = nn.Linear(self.n_hidden, self.d)
            
        elif out_dist == 'bern':
            self.dec_bern1 = nn.Linear(self.n_latent, self.n_hidden)
            self.dec_bern2 = nn.Linear(self.n_hidden, self.n_hidden)
            self.logits_layer = nn.Linear(self.n_hidden, self.d)
            
        elif out_dist in ['t', 't-distribution']:
            self.dec1 = nn.Linear(self.n_latent, self.n_hidden)
            self.dec2 = nn.Linear(self.n_hidden, self.n_hidden)
            self.mu_layer = nn.Linear(self.n_hidden, self.d)
            self.log_sigma_layer = nn.Linear(self.n_hidden, self.d)
            self.df_layer = nn.Linear(self.n_hidden, self.d)
        else:
            raise ValueError("Invalid out_dist. Use 'gauss', 'normal', 'truncated_normal', 'bern', 't', or 't-distribution'")

        
    def forward(self, z): 
       if self.out_dist in ['gauss', 'normal', 'truncated_normal']:
           z = self.activation(self.dec_gauss1(z))
           z = self.activation(self.dec_gauss2(z))
           mu = self.mu_layer(z)
           std = F.softplus(self.std_layer(z))
           return mu, std
       elif self.out_dist == 'bern':
           z = self.activation(self.dec_bern1(z))
           z = self.activation(self.dec_bern2(z))
           logits = self.logits_layer(z)
           return logits
       elif self.out_dist in ['t', 't-distribution']:
           z = self.activation(self.dec1(z))
           z = self.activation(self.dec2(z))
           mu = self.mu_layer(z)
           log_sigma = torch.clamp(self.log_sigma_layer(z), -10, 10)
           df = self.df_layer(z)  # No activation for df
           return mu, log_sigma, df
       else:
           raise ValueError("Invalid out_dist")



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
        # TODO: Add more missing processes
        else:
            raise ValueError("Invalid missing_process. Use 'agnostic', 'selfmasking', 'selfmasking_known', 'linear', or 'nonlinear'")
            
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













