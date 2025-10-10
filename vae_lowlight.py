# ===============================================
# Latent and Frequency Disentanglement for 
# Low-Light Image Enhancement using Variational Autoencoders
#
# Author: Jonathan Francisco
# ===============================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- Frequency Decomposition -----
def split_freq(img):
    low = F.avg_pool2d(img, kernel_size=3, stride=1, padding=1)
    high = img - low
    return low, high

# ----- Encoders -----
class Encoder(nn.Module):
    def __init__(self, in_ch, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU()
        )
        self.fc_mu = nn.Linear(128 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(128 * 16 * 16, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)
    
# ----- Decoder -----
class Decoder(nn.Module):
    def __init__(self, latent_dim, out_ch):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 16 * 16)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, out_ch, 4, 2, 1), nn.Sigmoid()
        )

    def forward(self, z):
        z = self.fc(z).view(z.size(0), 128, 16, 16)
        return self.deconv(z)
    
# ----- Full VAE -----
class DisentangledFreqVAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.enc_content = Encoder(3, latent_dim//2)
        self.enc_illum = Encoder(3, latent_dim//2)
        self.dec = Decoder(latent_dim, 3)

    def forward(self, x):
        low, high = split_freq(x)
        mu_c, logvar_c = self.enc_content(high)
        mu_i, logvar_i = self.enc_illum(low)
        z_c = self.reparameterize(mu_c, logvar_c)
        z_i = self.reparameterize(mu_i, logvar_i)
        z = torch.cat([z_c, z_i], dim=1)
        recon = self.dec(z)
        return recon, mu_c, logvar_c, mu_i, logvar_i
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std