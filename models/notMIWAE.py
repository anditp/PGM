import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from models import *


#%%


class NotMIWAE(nn.Module):
    def __init__(self, d, n_latent=50, n_hidden=100,  
                 activation=torch.tanh, out_dist='gauss', 
                 missing_process='agnostic'):
        super(NotMIWAE, self).__init__()

        self.d = d
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.activation = activation
        self.out_dist = out_dist
        self.missing_process = missing_process

        # Encoder
        self.encoder = Encoder(d, n_latent, n_hidden, activation)
        
        # Decoder
        self.decoder = Decoder(d, n_latent, n_hidden, out_dist, activation)
        
        # Missing Process Decoder
        self.missing_process_decoder = MissingProcessDecoder(d, n_hidden, missing_process)


    def forward(self, x, s, n_samples=1):
        # Encoder
        q_mu, q_log_sig2 = self.encoder(x)

        # Reparameterization trick
        eps = torch.randn_like(q_mu).unsqueeze(1).repeat(1, n_samples, 1) 
        l_z = q_mu.unsqueeze(1) + torch.exp(0.5 * q_log_sig2).unsqueeze(1) * eps

        # Decoder
        if self.out_dist in ['gauss', 'normal', 'truncated_normal']:
            mu, std = self.decoder(l_z)

            # Reparameterization trick for the output
            eps_out = torch.randn_like(mu) 
            l_out_sample = mu + std * eps_out

            if self.out_dist == 'truncated_normal':
                # For truncated normal, we need to resample values outside the range
                while True:
                    mask = (l_out_sample < 0) | (l_out_sample > 1)
                    if not mask.any():
                        break
                    eps_out_new = torch.randn_like(mu)
                    l_out_sample[mask] = mu[mask] + std[mask] * eps_out_new[mask]

        elif self.out_dist == 'bern':
            logits = self.decoder(l_z)
            probs = torch.sigmoid(logits)

            # Reparameterization trick for Bernoulli (using Gumbel-Softmax)
            # (This is an approximation, but it's differentiable)
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(probs)))
            l_out_sample = torch.sigmoid((logits + gumbel_noise) / 1.0)  # Temperature = 1.0

        elif self.out_dist in ['t', 't-distribution']:
            mu, log_sig2, df = self.decoder(l_z)
            
            # Reparameterization trick for Student's t-distribution
            normal_sample = torch.randn_like(mu)
            gamma_sample = torch.distributions.Gamma(df / 2, df / 2).rsample()
            l_out_sample = mu + (normal_sample / torch.sqrt(gamma_sample)) * F.softplus(log_sig2)

        # Missing data process (using l_out_sample)
        l_out_mixed = l_out_sample * (1 - s).unsqueeze(1) + x.unsqueeze(1)
        
        # Logits for the missing process decoder
        logits_miss = self.missing_process_decoder(l_out_mixed)

        return mu, std, q_mu, q_log_sig2, l_z, l_out_mixed, logits_miss  # Return the desired outputs



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
        print(l_out_sample.shape, s.shape)
        l_out_mixed = l_out_sample * (1 - s).unsqueeze(1) + x.unsqueeze(1)
        logits_miss = self.missing_process_decoder(l_out_mixed)

        return mu, std, q_mu, q_log_sig2, l_z, l_out_mixed, logits_miss


