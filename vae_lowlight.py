# ===============================================
# Latent and Frequency Disentanglement for 
# Low-Light Image Enhancement using Variational Autoencoders
#
# Author: Jonathan Francisco
# ===============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import kagglehub
from PIL import Image
import glob
import os

# ---------------------------
# Download LOL-v2 dataset
# ---------------------------
print("Downloading LOL-v2 dataset from KaggleHub...")
dataset_path = kagglehub.dataset_download("tanhyml/lol-v2-dataset")
print("Dataset downloaded to: ", dataset_path)

# ---------------------------
# Helper to get images
# ---------------------------
def get_images(path):
    exts = ["*.png", "*.jpg", "*.jpeg"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(path, ext)))
    return sorted(files)

# ---------------------------
# Dataset Class
# ---------------------------
class LowLightDataset(Dataset):
    def __init__(self, low_dir, normal_dir, transform=None):
        self.low_images = get_images(low_dir)
        self.normal_images = get_images(normal_dir)
        self.transform = transform

        assert len(self.low_images) == len(self.normal_images), "Mismatch in dataset sizes!"

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        low_img = Image.open(self.low_images[idx]).convert("RGB")
        normal_img = Image.open(self.normal_images[idx]).convert("RGB")

        if self.transform:
            low_img = self.transform(low_img)
            normal_img = self.transform(normal_img)

        return low_img, normal_img

# ---------------------------
# Frequency Decomposition
# ---------------------------
def split_freq(img):
    low = F.avg_pool2d(img, kernel_size=3, stride=1, padding=1)
    high = img - low
    return low, high

# ---------------------------
# Encoder
# ---------------------------
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

# ---------------------------
# Decoder
# ---------------------------
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

# ---------------------------
# Disentangled VAE
# ---------------------------
class DisentangledFreqVAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.enc_content = Encoder(3, latent_dim // 2)
        self.enc_illum = Encoder(3, latent_dim // 2)
        self.dec = Decoder(latent_dim, 3)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        low, high = split_freq(x)
        mu_c, logvar_c = self.enc_content(high)
        mu_i, logvar_i = self.enc_illum(low)
        z_c = self.reparameterize(mu_c, logvar_c)
        z_i = self.reparameterize(mu_i, logvar_i)
        z = torch.cat([z_c, z_i], dim=1)
        recon = self.dec(z)
        return recon, mu_c, logvar_c, mu_i, logvar_i

# ---------------------------
# Loss Function
# ---------------------------
def vae_loss(recon_x, target_x, mu_c, logvar_c, mu_i, logvar_i):
    recon_loss = F.mse_loss(recon_x, target_x, reduction='sum')
    kld_c = -0.5 * torch.sum(1 + logvar_c - mu_c.pow(2) - logvar_c.exp())
    kld_i = -0.5 * torch.sum(1 + logvar_i - mu_i.pow(2) - logvar_i.exp())
    kl_loss = kld_c + kld_i
    return recon_loss + kl_loss, recon_loss, kl_loss

# ---------------------------
# Training Loop
# ---------------------------
def train_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DisentangledFreqVAE(latent_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # ---------------------------
    # Training set
    # ---------------------------
    train_low = os.path.join(dataset_path, "LOL-v2", "Real_captured", "Train", "Low")
    train_normal = os.path.join(dataset_path, "LOL-v2", "Real_captured", "Train", "Normal")
    train_dataset = LowLightDataset(train_low, train_normal, transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # ---------------------------
    # Test set for visualization
    # ---------------------------
    test_low = os.path.join(dataset_path, "LOL-v2", "Real_captured", "Test", "Low")
    test_normal = os.path.join(dataset_path, "LOL-v2", "Real_captured", "Test", "Normal")
    test_dataset = LowLightDataset(test_low, test_normal, transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # ---------------------------
    # Training loop
    # ---------------------------
    for epoch in range(3):
        for low_img, normal_img in train_loader:
            low_img = low_img.to(device)
            normal_img = normal_img.to(device)
            recon, mu_c, logvar_c, mu_i, logvar_i = model(low_img)
            loss, recon_loss, kl_loss = vae_loss(recon, normal_img, mu_c, logvar_c, mu_i, logvar_i)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/3], Total Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}")

    # ---------------------------
    # Visualize some test results
    # ---------------------------
    with torch.no_grad():
        low_img, normal_img = next(iter(test_loader))
        low_img = low_img.to(device)
        recon, _, _, _, _ = model(low_img)

        plt.figure(figsize=(10,4))
        for i in range(min(4, low_img.size(0))):
            plt.subplot(2, 4, i+1)
            plt.title("Low-light")
            plt.imshow(low_img[i].permute(1,2,0).cpu())
            plt.axis("off")

            plt.subplot(2, 4, i+5)
            plt.title("Enhanced")
            plt.imshow(recon[i].permute(1,2,0).cpu())
            plt.axis("off")
        plt.show()

if __name__ == "__main__":
    train_demo()
