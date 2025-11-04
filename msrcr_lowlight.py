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
from torch.optim.lr_scheduler import StepLR
from torchvision import models
from torchvision import transforms
from pytorch_msssim import ssim as ssim_pytorch
from timm import create_model
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from glob import glob

# ------------------------------------------------
# Download LOL-v2 Dataset from Kaggle
# ------------------------------------------------
def load_lolv2_dataset(base_dir=None, num_samples=3, use_synthetic=True):
    if base_dir is None:
        base_dir = kagglehub.dataset_download("tanhyml/lol-v2-dataset")
        base_dir = os.path.join(base_dir, "LOL-v2")

    # Choose between real captured or synthetic data
    data_type = "Synthetic" if use_synthetic else "Real_captured"
    train_dir = os.path.join(base_dir, data_type, "Train")

    low_dir = os.path.join(train_dir, "Low")
    high_dir = os.path.join(train_dir, "Normal")

    # Get image file lists
    low_images = sorted(glob(os.path.join(low_dir, "*.png")))[:num_samples]
    high_images = sorted(glob(os.path.join(high_dir, "*.png")))[:num_samples]

    # Load and pair images
    pairs = []
    for l, h in zip(low_images, high_images):
        low_img = cv2.imread(l)
        high_img = cv2.imread(h)
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
    
class SimpleRetinexNet(nn.Module):
    def __init__(self):
        super(SimpleRetinexNet, self).__init__()
        mobilenet = models.mobilenet_v2(pretrained=True).features

        self.model = nn.Sequential(*mobilenet[:14])  # Up to the last inverted residual

        for name, param in self.model.named_parameters():
            if "13" not in name and "12" not in name:
                param.requires_grad = False

        # fusion_conv channels = sum of selected feature channels (here 192)
        self.fusion_conv = nn.Conv2d(192, 64, kernel_size=1)
        self.conv_out = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        # save original spatial size
        _, _, H_in, W_in = x.shape
        features = []
        cur = x
        for i, layer in enumerate(self.model):
            cur = layer(cur)
            if i in [4, 7, 13]:
                features.append(cur)

        # if for any reason features list shorter, fall back
        if len(features) == 0:
            # fallback: just run through a small head
            out = torch.sigmoid(self.conv_out(F.relu(self.fusion_conv(cur))))
            out = F.interpolate(out, size=(H_in, W_in), mode='bilinear', align_corners=False)
            return out

        # Multi-scale fusion with bilinear upsampling to same spatial dims (use last feature's size)
        target_h, target_w = features[-1].shape[2], features[-1].shape[3]
        upsampled = [F.interpolate(f, size=(target_h, target_w), mode='bilinear', align_corners=False)
                     for f in features]
        fused = torch.cat(upsampled, dim=1)        # channels must match fusion_conv in_channels
        fused = F.relu(self.fusion_conv(fused))
        out = torch.sigmoid(self.conv_out(fused))

        # Upsample to original input resolution so loss can be computed
        out = F.interpolate(out, size=(H_in, W_in), mode='bilinear', align_corners=False)
        return out
    
def enhance_with_retinexnet(img, model, device):
    transform = transforms.Compose([
        transforms.ToTensor(),  # outputs [0,1]
        transforms.Resize((256, 256)),
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)  # [0,1] already from sigmoid
    # Scale to [0,255] for visualization
    output_img = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output_img = np.clip(output_img * 255, 0, 255).astype(np.uint8)
    return cv2.resize(output_img, (img.shape[1], img.shape[0]))

# ------------------------------------------------
# RetinexNet Fine-Tuning on LOL-v2
# ------------------------------------------------
def train_retinexnet(model, train_loader_synth, train_loader_real, device, vgg):
    """Fine-tune RetinexNet on synthetic then real images with perceptual + SSIM loss."""
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)
    criterion_l1 = nn.L1Loss()

    print("Phase 1: Training on synthetic images...")  
    for epoch in range(3):  # Synthetic pre-training  
        running_loss = 0.0  
        for low, high in train_loader_synth:  
            low, high = low.to(device), high.to(device)  

            optimizer.zero_grad()  
            output = model(low)  

            # L1 loss  
            loss_l1 = criterion_l1(output, high)  

            # Perceptual loss (VGG features)  
            loss_percep = F.mse_loss(vgg(output), vgg(high))  

            # SSIM loss  
            loss_ssim = 1 - ssim_pytorch(output, high, data_range=1.0, size_average=True)  

            # Combined loss  
            loss = loss_l1 + 0.1*loss_percep + 0.1*loss_ssim  

            loss.backward()  
            optimizer.step()  
            running_loss += loss.item()  

        scheduler.step()  
        print(f"[Synthetic Epoch {epoch+1}/3] Loss: {running_loss/len(train_loader_synth):.4f}")  

    print("\nPhase 2: Fine-tuning on real-captured images...")  
    for epoch in range(5):  # Increased epochs from 2 → 5  
        running_loss = 0.0  
        for low, high in train_loader_real:  
            low, high = low.to(device), high.to(device)  

            optimizer.zero_grad()  
            output = model(low)  

            # L1 + perceptual + SSIM losses  
            loss_l1 = criterion_l1(output, high)  
            loss_percep = F.mse_loss(vgg(output), vgg(high))  
            loss_ssim = 1 - ssim_pytorch(output, high, data_range=1.0, size_average=True)  

            loss = loss_l1 + 0.1*loss_percep + 0.1*loss_ssim  

            loss.backward()  
            optimizer.step()  
            running_loss += loss.item()  

        print(f"[Real FT Epoch {epoch+1}/5] Loss: {running_loss/len(train_loader_real):.4f}")  

    torch.save(model.state_dict(), "retinexnet_finetuned_real_improved.pth")  
    print("Fine-tuning complete. Model saved as retinexnet_finetuned_real_improved.pth.")  

    model.eval()  
    return model  

# ------------------------------------------------
# Visualization and Evaluation
# ------------------------------------------------
def visualize_results(low, high, enhanced_ssr, enhanced_msr, enhanced_reti, idx):
    plt.figure(figsize=(15, 5))
    titles = ["Low-light", "Ground Truth", "SSR", "MSR", "RetinexNet"]
    images = [low, high, enhanced_ssr, enhanced_msr, enhanced_reti]
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 5, i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"comparison_{idx}.png")
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
def main():
    # Load LOL-v2 dataset paths
    print("Loading training data...")
    pairs_synth = load_lolv2_dataset(use_synthetic=True, num_samples=200) # Synthetic
    pairs_real = load_lolv2_dataset(use_synthetic=False, num_samples=50) # Real

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------
    # Split real images into fine-tuning and testing sets
    # ------------------------------------------------
    random.shuffle(pairs_real)
    num_real_test = 10
    test_pairs_real = pairs_real[:num_real_test]       # unseen images for evaluation
    train_pairs_real = pairs_real[num_real_test:]      # rest for fine-tuning

    # Preprocess image pairs to tensors
    def preprocess_pairs(pairs):
        data = []
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 512))
        ])
        for low, high in pairs:
            low_t = transform(cv2.cvtColor(low, cv2.COLOR_BGR2RGB))
            high_t = transform(cv2.cvtColor(high, cv2.COLOR_BGR2RGB))
            data.append((low_t, high_t))
        return data

    dataset_synth = preprocess_pairs(pairs_synth)
    dataset_real_train = preprocess_pairs(train_pairs_real)

    train_loader_synth = torch.utils.data.DataLoader(dataset_synth, batch_size=4, shuffle=True)
    train_loader_real = torch.utils.data.DataLoader(dataset_real_train, batch_size=2, shuffle=True)

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

    for idx, (low_img, high_img) in enumerate(test_pairs_real):
        low = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB)
        high = cv2.cvtColor(high_img, cv2.COLOR_BGR2RGB)

        # Classical Retinex variants
        low_balanced = gray_world_white_balance(low)
        enhanced_ssr = single_scale_retinex(low_balanced)
        enhanced_msr = multi_scale_retinex(low_balanced)

        # Deep RetinexNet
        enhanced_retinexnet = enhance_with_retinexnet(low, retinexnet, device)

        # Visualization
        visualize_results(low, high, enhanced_ssr, enhanced_msr, enhanced_retinexnet, idx)

        # Evaluation metrics
        psnr_ssr, ssim_ssr = evaluate_metrics(enhanced_ssr, high)
        psnr_msr, ssim_msr = evaluate_metrics(enhanced_msr, high)
        psnr_reti, ssim_reti = evaluate_metrics(enhanced_retinexnet, high)

        psnr_ssr_all.append(psnr_ssr)
        ssim_ssr_all.append(ssim_ssr)
        psnr_msr_all.append(psnr_msr)
        ssim_msr_all.append(ssim_msr)
        psnr_reti_all.append(psnr_reti)
        ssim_reti_all.append(ssim_reti)

        print(f"\n Image {idx+1} Results")
        print(f"{'Method':<20}{'PSNR':>10}{'SSIM':>12}")
        print("-"*35)
        print(f"{'SSR':<20}{psnr_ssr:>10.2f}{ssim_ssr:>12.4f}")
        print(f"{'MSR':<20}{psnr_msr:>10.2f}{ssim_msr:>12.4f}")
        print(f"{'RetinexNet (FT)':<20}{psnr_reti:>10.2f}{ssim_reti:>12.4f}")

    print("\n" + "="*45)
    print("Average Results Across All Test Images:")
    print(f"{'Method':<20}{'PSNR':>10}{'SSIM':>12}")
    print("-"*35)
    print(f"{'SSR':<20}{np.mean(psnr_ssr_all):>10.2f}{np.mean(ssim_ssr_all):>12.4f}")
    print(f"{'MSR':<20}{np.mean(psnr_msr_all):>10.2f}{np.mean(ssim_msr_all):>12.4f}")
    print(f"{'RetinexNet (FT)':<20}{np.mean(psnr_reti_all):>10.2f}{np.mean(ssim_reti_all):>12.4f}")
    print("="*45)

if __name__ == "__main__":
    main()
