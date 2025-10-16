# ===============================================
# Latent and Frequency Disentanglement for 
# Low-Light Image Enhancement using Variational Autoencoders
#
# Based on:
#  - Latent Disentanglement for Low-Light Image Enhancement (Zheng & Chuah, 2023)
#  - Advanced Frequency Disentanglement Paradigm for LLIE (Zhou et al., 2024)
#
# Author: Jonathan Francisco
# ===============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ------------------------------------------------
# Frequency Decomposition
# ------------------------------------------------
def split_freq(img):
    """Split input image into low and high frequency components."""
    low = F.avg_pool2d(img, kernel_size=3, stride=1, padding=1)
    high = img - low
    return low, high

# ------------------------------------------------
# Encoder
# ------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, in_ch, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU()
        )
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
# ------------------------------------------------
# Decoder
# ------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, latent_dim, out_ch):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, out_ch, 4, 2, 1), nn.Sigmoid()
        )

    def forward(self, z):
        z = self.fc(z).view(z.size(0), 128, 8, 8)
        return self.deconv(z)
    
# ------------------------------------------------
# Full Disentangled VAE
# ------------------------------------------------
class DisentangledFreqVAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        # Half for illumination, half for content
        self.enc_content = Encoder(3, latent_dim // 2)
        self.enc_illum = Encoder(3, latent_dim // 2)
        self.dec = Decoder(latent_dim, 3)

    def reparameterize(self, mu, logvar):
        """Sample latent vector using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # Split image into low and high frequency bands
        low, high = split_freq(x)
        mu_c, logvar_c = self.enc_content(high)
        mu_i, logvar_i = self.enc_illum(low)

        # Sampple latent codes
        z_c = self.reparameterize(mu_c, logvar_c)
        z_i = self.reparameterize(mu_i, logvar_i)

        # Combine latent codes and decode
        z = torch.cat([z_c, z_i], dim=1)
        recon = self.dec(z)
        return recon, mu_c, logvar_c, mu_i, logvar_i
    
# ------------------------------------------------
# Loss Function
# ------------------------------------------------
def vae_loss(recon_x, x, mu_c, logvar_c, mu_i, logvar_i):
    """VAE loss = reconstruction + KL divergenc."""
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld_c = -0.5 * torch.sum(1 + logvar_c - mu_c.pow(2) - logvar_c.exp())
    kld_i = -0.5 * torch.sum(1 + logvar_i - mu_i.pow(2) - logvar_i.exp())

    kl_loss = kld_c + kld_i
    total_loss = recon_loss + kl_loss
    return total_loss, recon_loss, kl_loss

# ------------------------------------------------
# Training Loop (Demo)
# ------------------------------------------------
def train_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DisentangledFreqVAE(latent_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    #using MNIST as placeholder -- replace with a low-light dataset later
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    dataset = datasets.FakeData(transform=transform, size=200)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    for epoch in range(3): # short demo run
        for img, _ in loader:
            img = img.to(device)
            recon, mu_c, logvar_c, mu_i, logvar_i = model(img)
            loss, recon_loss, kl_loss = vae_loss(recon, img, mu_c, logvar_c, mu_i, logvar_i)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/3], Total Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}")

    # Visualize result
    with torch.no_grad():
        img, _ = next(iter(loader))
        img = img.to(device)
        recon, _, _, _, _ = model(img)
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)
        plt.title("Original")
        plt.imshow(img[0].permute(1,2,0).cpu())
        plt.subplot(1,2,2)
        plt.title("Reconstructed")
        plt.imshow(recon[0].permute(1,2,0).cpu())
        plt.show()

if __name__ == "__main__":
    train_demo()