# Latent and Frequency Disentanglement for Low-Light Image Enhancement using Variational Autoencoders

## Problem Statement
Low-light images often suffer from poor visibility, low contrast, and noise, degrading performance in downstream vision tasks (object detection, classification, etc.).
Traditional enhancement methods often over-brighten or lose detail.
This project aims to design a lightweight VAE-based model that separately learns:

1. Content (structure, texture) and illumination (brightness) representations in latent space

2. Low- and high-frequency features (illumination vs detail) in the image domain.

By combining these two disentanglement strategies, the model learns to enhance illumination while preserving natural textures.

## Dataset
Use a small, publicly available dataset:

- LOL (Low-Light Image Enhancement Dataset) — pairs of low-light and normal-light images
🔗 https://daooshee.github.io/LOLdataset/

- Optional fallback (smaller): See-in-the-Dark (SID) or MIT FiveK (subset)

You can train on ~1,000 image pairs (resized to 128×128 or 256×256) to fit a mid-range GPU or CPU-based training.

## Methodology
1. Preprocessing
- Resize & normalize images

- Optionally perform frequency decomposition (using a Laplacian or DCT filter) into:

- Low-frequency (LF): illumination info

- High-frequency (HF): texture/detail info

- LF → input to illumination encoder; HF → input to content encoder

2. Model Architecture
Base:
A Variational Autoencoder (VAE) with two encoders and one decoder.
a. Dual Encoder
  - Content Encoder -> latenc Zc (edges, texture)
  - Illumination Encoder -> laten Zi (brightness, color)

b. Latent Disentaglement
  - Zc and Zi concatenated into joint latent vector
  - Orthogonality / decorrelation loss Ldis = ||Zc Zi ||

c. Frequency Disentanglement
  - Feed LF image to illumination encoder, HF to content encoder
  - Optionally, add reconstruction considtency between frequency domains

d. Decoder
- Combines Zc + modified Zi -> reconstruct enhanced image
- At inference: scale or shift Zi to simulate increased illumination

3. Training Objectives
| Loss Component                  | Description                                                       |
| ------------------------------- | ----------------------------------------------------------------- |
| Reconstruction Loss             | L1/L2 between output and ground truth bright image                |
| KL Divergence                   | Standard VAE regularization for latent prior                      |
| Disentanglement Loss            | Reduce correlation between ( Zc ) and ( Zi )                      |
| Frequency Consistency Loss      | Ensure HF features preserved post-enhancement                     |
| Illumination Scaling (optional) | Encourage monotonic brightness mapping between ( Zi ) and output  |

4. Evaluation Metrics

| Metric            | Purpose                                    |
| ----------------- | ------------------------------------------ |
| PSNR / SSIM       | Image reconstruction & enhancement quality |
| NIQE / BRISQUE    | Naturalness & perceptual realism           |
| Brightness ratio  | Quantitative illumination gain             |
| Visual comparison | Side-by-side original vs enhanced images   |

5. Expected Results
- Brighter, more natural low-light images

- Textures & colors preserved better than with naive brightening

- Stable latent spaces (interpretable content vs illumination vectors)
   
