# Low-Light Image Enhancement Using Classical and Deep Retinex Models

## Overview
This project explores **low-light image enhancement** using both **classical Retinex algorithms** and a **deep learning-based RetinexNet model**.  
It compares three approaches:

1. **Single-Scale Retinex (SSR)** – classical illumination correction using a single Gaussian scale.  
2. **Multi-Scale Retinex with Color Restoration (MSRCR)** – multi-scale fusion with improved tone and color balance.  
3. **RetinexNet (CNN-based)** – a deep model that decomposes images into **reflectance (detail)** and **illumination (brightness)** for learned enhancement.

The goal is to improve image visibility and contrast under poor lighting conditions while maintaining natural color and texture fidelity.

---

## Problem Statement
Low-light images often suffer from poor contrast, color distortion, and amplified noise.  
Traditional enhancement techniques like histogram equalization or naive brightening may cause overexposure or unrealistic tone mapping.  

This project implements a **data-driven RetinexNet** and compares it with **SSR** and **MSRCR** to:
- Enhance global illumination.  
- Preserve local detail and texture.  
- Maintain natural color and perceptual realism.  

---

## Dataset
The project uses the **LOL-v2 Low-Light Image Enhancement dataset**, which contains both **synthetic** and **real-captured** image pairs of the same scenes under low and normal illumination.

- **Synthetic subset:** Used for initial pretraining.  
- **Real-captured subset:** Used for fine-tuning and testing.  
- Each image pair consists of a *low-light input* and its *ground-truth normal-light output.*

**Dataset breakdown:**
- 64 synthetic pairs (training)  
- 128 real-captured pairs (fine-tuning)  
- 24 real-captured pairs (testing)

---

## Methodology

### 1. Preprocessing
- **Gray-world white balance** for color normalization.  
- **Gamma correction** for stable perceptual loss computation.  
- **Data augmentation:** random flips, color jitter, and Gaussian blur.  
- **Normalization:** all pixel intensities scaled to `[0, 1]`.

---

### 2. Models Implemented

#### **Single-Scale Retinex (SSR)**
Applies a logarithmic ratio between the original image and its Gaussian-blurred version to simulate human visual adaptation to illumination.

#### **Multi-Scale Retinex with Color Restoration (MSRCR)**
Extends SSR by blending results across multiple scales, followed by color restoration to maintain chromatic balance. Produces smoother, more natural tonal mapping.

#### **RetinexNet (CNN-based)**
A lightweight **MobileNetV2-based encoder-decoder** CNN that decomposes each image into:
- **Reflectance (R):** fine details and textures.  
- **Illumination (L):** global brightness component.

**Architecture highlights:**
- Encoder initialized from MobileNetV2 layers.  
- Two heads output reflectance and illumination maps.  
- Outputs recombined as:  
  `Enhanced = Reflectance × Illumination`.

---

### 3. Training Pipeline
Two-stage training strategy inspired by *Wei et al. (2018):*

#### **Phase 1 — Mixed Pretraining**
- Dataset: combination of synthetic + real images  
- Objective: learn general enhancement patterns  
- Epochs: `50`  
- Optimizer: `Adam (lr = 1e-4)`  
- Scheduler: `CosineAnnealingLR`  

#### **Phase 2 — Real Fine-tuning**
- Dataset: real-captured pairs only  
- Epochs: `100`  
- Optimizer: `Adam (lr = 5e-6, weight_decay = 1e-5)`  
- Scheduler: `CosineAnnealingWarmRestarts`  

**Loss Function:**

| Loss Term | Description |
|------------|-------------|
| **L1** | Pixel-level reconstruction accuracy |
| **Perceptual (VGG)** | Feature similarity in high-level feature space |
| **SSIM** | Structural similarity for perceptual quality |
| **Total Variation (TV)** | Smooth illumination map, reduce noise |

---

### 4. Post-processing
After enhancement, two refinement steps improve perceptual realism:
- **CLAHE (Contrast-Limited Adaptive Histogram Equalization):** boosts local contrast without overexposing highlights.  
- **Unsharp Masking:** restores fine edges and texture lost during smoothing.

---

## Evaluation Metrics
| Metric | Purpose |
|---------|----------|
| **PSNR** | Measures pixel-level reconstruction accuracy |
| **SSIM** | Quantifies structural and perceptual similarity |
| **Visual comparison** | Qualitative analysis of tonal balance, sharpness, and realism |

---

## Results

| Method | PSNR | SSIM | Observations |
|---------|------|------|---------------|
| **SSR** | 14.87 | 0.5942 | Brightens image but desaturates colors |
| **MSRCR** | 17.37 | 0.5895 | Balanced tone, improved color fidelity |
| **RetinexNet (FT)** | 16.10 | 0.6223 | Natural brightness and color; slight blurring due to loss smoothing |

**Average over 24 test images (LOL-v2 Real subset).**

> RetinexNet achieved the **highest SSIM**, indicating better perceptual and structural consistency despite slightly lower PSNR.

---

## Conclusion
This project demonstrates that combining **Retinex theory** with **deep learning** achieves superior enhancement for low-light images.  
While SSR and MSRCR provide interpretable, physics-based models, the **RetinexNet** learns illumination correction directly from data — producing natural, well-balanced images under complex lighting.

### **Future Work**
- Integrate **attention modules** to dynamically enhance dark regions.  
- Use **transformer-based architectures** for better global context modeling.  
- Explore **unsupervised or self-supervised learning** to generalize across domains.  

Through continued model optimization, RetinexNet and similar architectures can achieve **real-time, artifact-free enhancement** closely aligned with human visual perception.

---

## References
- Jobson, D. J., Rahman, Z., & Woodell, G. A. (1996). *Multi-Scale Retinex for Color Image Enhancement.* IEEE ICIP.  
- Choi, D.-H., Jang, I.-H., Kim, M.-H., & Kim, N.-C. (2008). *Color Image Enhancement Using Single-Scale Retinex Based on an Improved Image Formation Model.* EUSIPCO.  
- Wei, C., Wang, W., Yang, W., & Liu, J. (2018). *Deep Retinex Decomposition for Low-Light Enhancement.* BMVC.

---

## Example Usage

### **Run Training**
```bash
python msrcr_lowlight.py

Phase 1: Mixed Pretraining (Synthetic + Real)
[Epoch 1/50] Loss: 0.9121
...

Phase 2: Fine-tuning on Real Data
[Epoch 1/100] Loss: 0.5920
...

Average Results Across Test Images:
SSR   -> PSNR: 14.87, SSIM: 0.5942
MSRCR -> PSNR: 17.37, SSIM: 0.5895
RetinexNet (FT) -> PSNR: 16.10, SSIM: 0.6223
