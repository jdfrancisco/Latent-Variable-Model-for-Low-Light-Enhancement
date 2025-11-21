# ================================================================
# Multi-Scale Retinex with Color Restoration (MSRCR)
# Applied on LOL-v2 Low-Light Image Enhancement Dataset
#
# Author: Jonathan Francisco
# ================================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision import models
from torchvision import transforms
from pytorch_msssim import ms_ssim
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from PIL import Image
from glob import glob
from datetime import datetime

# Set global random seeds for reproduciblity
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ------------------------------------------------
# Download LOL-v2 Dataset from Kaggle
# ------------------------------------------------
def load_lolv2_dataset(base_dir=None, num_samples=3, use_synthetic=True, use_train=True):
    if base_dir is None:
        try:
            base_dir = kagglehub.dataset_download("tanhyml/lol-v2-dataset")
        except Exception as e:
            print("KaggleHub download failed, please provide local dataset path.")
            raise e
        
        base_dir = os.path.join(base_dir, "LOL-v2")

    if use_train:
        # Choose between real captured or synthetic Train data
        data_type = "Synthetic" if use_synthetic else "Real_captured"
        train_dir = os.path.join(base_dir, data_type, "Train")

        low_dir = os.path.join(train_dir, "Low")
        high_dir = os.path.join(train_dir, "Normal")
    else: 
        # Choose between real captured or synthetic Test data
        data_type = "Synthetic" if use_synthetic else "Real_captured"
        test_dir = os.path.join(base_dir, data_type, "Test")

        low_dir = os.path.join(test_dir, "Low")
        high_dir = os.path.join(test_dir, "Normal")

    # Get image file lists
    low_images = sorted(glob(os.path.join(low_dir, "*.png")))[:num_samples]
    high_images = sorted(glob(os.path.join(high_dir, "*.png")))[:num_samples]

    # Load and pair images
    pairs = []
    for l, h in zip(low_images, high_images):
        low_img = cv2.cvtColor(cv2.imread(l), cv2.COLOR_BGR2RGB)
        high_img = cv2.cvtColor(cv2.imread(h), cv2.COLOR_BGR2RGB)
        if low_img is not None and high_img is not None:
            pairs.append((low_img, high_img))

    print(f"Loaded {len(pairs)} low/normal image pairs from {data_type}.")
    return pairs

# ------------------------------------------------
# Retinex Classical Baselines
# ------------------------------------------------
def single_scale_retinex(img, sigma=15):
    img = img.astype(np.float32) + 1.0
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))
    retinex = (retinex - np.min(retinex)) / (np.max(retinex) - np.min(retinex)) * 255
    return np.uint8(np.clip(retinex, 0, 255))

def multi_scale_retinex(img, sigmas=[15, 80, 250]):
    img = img.astype(np.float32) + 1.0
    retinex = np.zeros_like(img)
    for sigma in sigmas:
        blur = cv2.GaussianBlur(img, (0, 0), sigma)
        retinex += np.log10(img) - np.log10(blur)
    retinex /= len(sigmas)
    retinex = (retinex - np.min(retinex)) / (np.max(retinex) - np.min(retinex)) * 255
    return np.uint8(np.clip(retinex, 0, 255))

def color_restoration(img, alpha=125, beta=46):
    img_sum = np.sum(img, axis=2, keepdims=True)
    color = beta * (np.log10(alpha * img + 1e-3) - np.log10(img_sum + 1e-3))
    return color 

def msrcr_ycbcr(img):
    ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y, cb, cr = cv2.split(ycbcr)
    y = y.astype(np.float32) / 255.0

    y_retinex = multi_scale_retinex(y, [15, 80, 250])
    y_retinex = np.clip((y_retinex - np.min(y_retinex)) / (np.max(y_retinex) - np.min(y_retinex)), 0, 1)

    y_retinex = (y_retinex * 255).astype(np.uint8)
    enhanced = cv2.merge([y_retinex, cb, cr])
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_YCrCb2RGB)
    return enhanced_rgb

# ------------------------------------------------
# Gray / White Balance for MSRCR
# ------------------------------------------------
def gray_world_white_balance(img):
    img = img.astype(np.float32)
    mean_r = np.mean(img[:, :, 0])
    mean_g = np.mean(img[:, :, 1])
    mean_b = np.mean(img[:, :, 2])

    mean_gray = (mean_r + mean_g + mean_b) / 3
    scale_r = mean_gray / (mean_r + 1e-6)
    scale_g = mean_gray / (mean_g + 1e-6)
    scale_b = mean_gray / (mean_b + 1e-6)

    img[:, :, 0] *= scale_r
    img[:, :, 1] *= scale_g
    img[:, :, 2] *= scale_b

    img = np.clip(img / 255.0, 0, 1)
    return (img * 255).astype(np.uint8)

# ------------------------------------------------
# RetinexNet CNN class and helpers
# ------------------------------------------------
class VGGPerceptual(nn.Module):
    def __init__(self, device):
        super().__init__()

        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16] # Up to relu3_3
        for param in vgg16.parameters():
            param.requires_grad = False
        self.vgg16 = vgg16.to(device)

    def forward(self, x):  
        # Rescale input to [0,1] -> [0,1] expected by VGG normalization  
        # Assume input x in [0,1]  
        # Normalize as VGG expects  
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)  
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)  
        x = (x - mean) / std  
        return self.vgg16(x)
    
# --- Mixed dataset for synthetic + real samples ---
class MixedDataset(Dataset):
    def __init__(self, synth_list, real_list, real_ratio=0.2):
        self.synth = synth_list
        self.real = real_list
        self.real_ratio = real_ratio
        self.length = max(len(self.synth), len(self.real))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if len(self.real) > 0 and random.random() < self.real_ratio:
            return self.real[idx % len(self.real)]
        else:
            return self.synth[idx % len(self.synth)]

# --- Color consistency loss ---
def color_consistency_loss(img):
    mean_rgb = img.mean(dim=[2,3])
    mean_gray = mean_rgb.mean(dim=1, keepdim=True)
    return ((mean_rgb - mean_gray) ** 2).mean()
    
class SimpleRetinexNet(nn.Module):
    def __init__(self):
        super(SimpleRetinexNet, self).__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features
        self.encoder = nn.Sequential(*mobilenet[:14])
        for p in self.encoder.parameters():
            p.requires_grad = True

        self.fusion = nn.Sequential(
            nn.Conv2d(96, 64, 3, padding=1),  # <-- changed from 160 → 96
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.reflectance_head = nn.Conv2d(32, 3, 1)
        self.illumination_head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        f = self.encoder(x)
        f = self.fusion(f)
        R = torch.tanh(self.reflectance_head(f))
        R = (R + 1) / 2
        L = torch.sigmoid(self.illumination_head(f))
        enhanced = torch.clamp(R * L, 0, 1)

        # Upsample all outputs back to input size
        enhanced = F.interpolate(enhanced, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        R = F.interpolate(R, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        L = F.interpolate(L, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        return enhanced, R, L

def enhance_with_retinexnet(img, model, device):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    img_pil = Image.fromarray(img)
    input_tensor = transform(img_pil).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        out, R, L = model(input_tensor)
    out_np = (out.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.resize(out_np, (img.shape[1], img.shape[0]))

def enhance_hybrid_msrcr_retinexnet(low_img, model, device):
    """
    Hybrid pipeline:
        Step 1: MSRCR (Classical illumination correction)
        Step 2: RetinexNet (ML refinement)
    """

    # Step 1: Classical MSRCR
    msrcr_img = msrcr_ycbcr(low_img)

    # Convert MSRCR output to tensor for RetinexNet
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    msrcr_pil = Image.fromarray(msrcr_img)
    input_tensor = transform(msrcr_pil).unsqueeze(0).to(device)

    # Step 2: CNN Refinement
    model.eval()
    with torch.no_grad():
        out, R, L = model(input_tensor)

    # Convert back to uint8 Numpy
    out_np = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    # Resiize back to original resolution
    refined = cv2.resize(out_np, (low_img.shape[1], low_img.shape[0]))

    return msrcr_img, refined

# ------------------------------------------------
# RetinexNet Fine-Tuning on LOL-v2
# ------------------------------------------------
# --- Total Variation Loss for smoothness without blur ---
def total_variation_loss(img):
    return torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + \
           torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))

# --- Data augmentation ---
augment = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5))
])

def train_retinexnet(model, train_loader_synth, train_loader_real, device, vgg):
    """Enhanced RetinexNet training with illumination decomposition, color consistency, and mixed-data pretraining."""
    criterion_l1 = nn.L1Loss()
    real_ratio = 0.2
    phase1_epochs, phase2_epochs = 50, 100

    # Phase 1: Mixed synthetic + real pretraining
    synth_list = list(train_loader_synth.dataset)
    real_list = list(train_loader_real.dataset)
    mixed_dataset = MixedDataset(synth_list, real_list, real_ratio)
    mixed_loader = torch.utils.data.DataLoader(mixed_dataset, batch_size=12, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase1_epochs + phase2_epochs)

    print("Phase 1: Mixed Pretraining (Synthetic + Real)")
    for epoch in range(phase1_epochs):
        model.train()
        running_loss = 0.0
        for low, high in mixed_loader:
            low, high = low.to(device), high.to(device)
            optimizer.zero_grad()
            out, R, L = model(low)

            out_gamma = torch.pow(torch.clamp(out, 1e-6, 1.0), 0.9)
            high_gamma = torch.pow(torch.clamp(high, 1e-6, 1.0), 0.9)

            loss_l1 = criterion_l1(out, high)
            loss_percep = F.mse_loss(vgg(out_gamma), vgg(high_gamma))
            loss_ssim = 1 - ms_ssim(out_gamma, high_gamma, data_range=1.0).mean()
            loss_tv = total_variation_loss(L)

            loss = (1.0 * loss_l1) + (0.05 * loss_percep) + (0.5 * loss_ssim) + (0.1 * loss_tv)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        print(f"[Phase1 Epoch {epoch+1}/{phase1_epochs}] Loss: {running_loss/len(mixed_loader):.6f}")

    # Phase 2: Fine-tune on real only
    print("\nPhase 2: Fine-tuning on Real Data")
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-6, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    for epoch in range(phase2_epochs):
        model.train()
        running_loss = 0.0
        for low, high in train_loader_real:
            low, high = low.to(device), high.to(device)
            optimizer.zero_grad()
            out, R, L = model(low)

            out_gamma = torch.pow(torch.clamp(out, 1e-6, 1.0), 0.9)
            high_gamma = torch.pow(torch.clamp(high, 1e-6, 1.0), 0.9)

            loss_l1 = criterion_l1(out, high)
            loss_percep = F.mse_loss(vgg(out_gamma), vgg(high_gamma))
            loss_ssim = 1 - ms_ssim(out_gamma, high_gamma, data_range=1.0).mean()
            loss_tv = total_variation_loss(L)

            loss = (1.0 * loss_l1) + (0.05 * loss_percep) + (0.5 * loss_ssim) + (0.1 * loss_tv)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        print(f"[Phase2 Epoch {epoch+1}/{phase2_epochs}] Loss: {running_loss/len(train_loader_real):.6f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("Models", exist_ok=True)
    torch.save(model.state_dict(), f"Models/retinexnet_finetuned_{timestamp}.pth")
    print(f"Training complete. Saved model as retinexnet_finetuned_{timestamp}.pth")
    model.eval()
    return model

# ================================
# Post-processing Step
# ================================
def enhance_postprocess(img_tensor):
    """Apply CLAHE + unsharp masking to improve local contrast and sharpness."""
    img = (img_tensor.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab = cv2.merge((l_clahe, a, b))
    img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    img_sharp = cv2.detailEnhance(img_clahe, sigma_s=8, sigma_r=0.12)
    img_final = cv2.addWeighted(img_clahe, 0.8, img_sharp, 0.2, 0)
    img_final = cv2.fastNlMeansDenoisingColored(img_final, None, 3, 3, 7, 21)
    return np.clip(img_final / 255.0, 0, 1)

# ------------------------------------------------
# Visualization and Evaluation
# ------------------------------------------------
def visualize_results(low, high, enhanced_ssr, enhanced_msr, enhanced_reti, enhanced_hybrid, idx):
    plt.figure(figsize=(18, 6))
    titles = ["Low-light", "Ground Truth", "SSR", "MSRCR", "RetinexNet", "Hybrid"]
    images = [low, high, enhanced_ssr, enhanced_msr, enhanced_reti, enhanced_hybrid]
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 6, i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"Model_Comparisons/comparison_{idx}.png")
    plt.close()

# ------------------------------------------------
# Evaluation Metrics
# ------------------------------------------------
def evaluate_metrics(enhanced, reference):
    """Compute PSNR and SSIM for one image pair."""
    p = psnr(reference, enhanced, data_range=255)
    s = ssim(reference, enhanced, channel_axis=2, data_range=255)
    return p, s

# ------------------------------------------------
# Main Execution
# ------------------------------------------------
# Preprocess image pairs to tensors
def preprocess_pairs(pairs):
    data = []
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    for low, high in pairs:
        # Convert NumPy arrays (RGB) -> PIL for torchvision transforms
        low_pil = Image.fromarray(low)
        high_pil = Image.fromarray(high)

        low_t = transform(low_pil)
        high_t = transform(high_pil)
        data.append((low_t, high_t))
    return data
    
def main():
    # Load LOL-v2 dataset paths
    print("Loading training data...")
    train_pairs_synth = load_lolv2_dataset(use_synthetic=True, num_samples=64, use_train=True)
    train_pairs_real = load_lolv2_dataset(use_synthetic=False, num_samples=128, use_train=True)
    test_pairs_real = load_lolv2_dataset(use_synthetic=False, num_samples=24, use_train=False)

    test_pairs = test_pairs_real

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_synth = preprocess_pairs(train_pairs_synth)
    dataset_real_train = preprocess_pairs(train_pairs_real)

    train_loader_synth = torch.utils.data.DataLoader(dataset_synth, batch_size=16, shuffle=True)
    train_loader_real = torch.utils.data.DataLoader(dataset_real_train, batch_size=8, shuffle=True)

    # Initialize model
    retinexnet = SimpleRetinexNet().to(device)
    vgg = VGGPerceptual(device)
    retinexnet = train_retinexnet(retinexnet, train_loader_synth, train_loader_real, device, vgg)

    # ------------------------------------------------
    # Evaluation phase (using unseen real test images)
    # ------------------------------------------------
    psnr_ssr_all, ssim_ssr_all = [], []
    psnr_msr_all, ssim_msr_all = [], []
    psnr_reti_all, ssim_reti_all = [], []
    psnr_hybrid_all, ssim_hybrid_all = [], []

    for idx, (low_img, high_img) in enumerate(test_pairs):
        low = low_img
        high = high_img

        # Classical Retinex variants
        low_balanced = gray_world_white_balance(low)
        enhanced_ssr = single_scale_retinex(low)
        enhanced_msr = multi_scale_retinex(low_balanced)

        # Deep RetinexNet
        enhanced_retinexnet = enhance_with_retinexnet(low, retinexnet, device)

        # Hybrid MSRCR -> ReinexNet
        msrcr_img, hybrid_retinex = enhance_hybrid_msrcr_retinexnet(low, retinexnet, device)

        # Visualization
        visualize_results(low, high, enhanced_ssr, enhanced_msr, enhanced_retinexnet, hybrid_retinex, idx)

        # Evaluation metrics
        psnr_ssr, ssim_ssr = evaluate_metrics(enhanced_ssr, high)
        psnr_msr, ssim_msr = evaluate_metrics(enhanced_msr, high)
        psnr_reti, ssim_reti = evaluate_metrics(enhanced_retinexnet, high)
        psnr_hybrid, ssim_hybrid = evaluate_metrics(hybrid_retinex, high)

        psnr_ssr_all.append(psnr_ssr)
        ssim_ssr_all.append(ssim_ssr)
        psnr_msr_all.append(psnr_msr)
        ssim_msr_all.append(ssim_msr)
        psnr_reti_all.append(psnr_reti)
        ssim_reti_all.append(ssim_reti)
        psnr_hybrid_all.append(psnr_hybrid)
        ssim_hybrid_all.append(ssim_hybrid)

        print(f"\n Image {idx+1} Results")
        print(f"{'Method':<20}{'PSNR':>10}{'SSIM':>12}")
        print("-"*35)
        print(f"{'SSR':<20}{psnr_ssr:>10.2f}{ssim_ssr:>12.4f}")
        print(f"{'MSR':<20}{psnr_msr:>10.2f}{ssim_msr:>12.4f}")
        print(f"{'RetinexNet (FT)':<20}{psnr_reti:>10.2f}{ssim_reti:>12.4f}")
        print(f"{'Hybrid':<20}{psnr_hybrid:>10.2f}{ssim_hybrid:>12.4f}")

    print("\n" + "="*45)
    print("Average Results Across All Test Images:")
    print(f"{'Method':<20}{'PSNR':>10}{'SSIM':>12}")
    print("-"*35)
    print(f"{'SSR':<20}{np.mean(psnr_ssr_all):>10.2f}{np.mean(ssim_ssr_all):>12.4f}")
    print(f"{'MSR':<20}{np.mean(psnr_msr_all):>10.2f}{np.mean(ssim_msr_all):>12.4f}")
    print(f"{'RetinexNet (FT)':<20}{np.mean(psnr_reti_all):>10.2f}{np.mean(ssim_reti_all):>12.4f}")
    print(f"{'Hybrid':<20}{np.mean(psnr_hybrid_all):>10.2f}{np.mean(ssim_hybrid_all):>12.4f}")
    print("="*45)

if __name__ == "__main__":
    main()
