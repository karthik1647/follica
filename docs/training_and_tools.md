# Follica: ML Training Methodology, Algorithms, and Tooling Specification

## Executive Overview

This document provides a comprehensive technical reference for the machine learning training pipeline, algorithmic hyperparameters, data engineering, and software tooling used across the Follica scalp triage platform.

---

## 1. Machine Learning Algorithms & Training Formulation

### 1. Model Backbone Architecture
- **Primary Model**: **MobileNetV4** (`mobilenetv4_conv_small.e1200_r224_in1k`)
  - **Core Mechanism**: Universal Inverted Bottleneck (UIB) blocks that dynamically blend inverted bottleneck, depthwise separable convolutions, and Mobile Multi-Query Attention (MQA).
  - **Input Resolution**: $224 \times 224 \times 3$ RGB.
  - **Pre-Training**: Pre-trained on ImageNet-1K (`pretrained=True`), fine-tuned with transfer learning for 10 scalp condition classes.
  - **Footprint**: ~14.2 MB serialized ONNX model with sub-50 ms inference latency on commodity mobile/laptop CPUs.
- **Fallback Architecture**: **EfficientNet-B0**
  - Compound-scaled convolutional backbone utilizing mobile inverted bottleneck convolutions (MBConv) with squeeze-and-excitation blocks (~19 MB ONNX footprint).

### 2. Loss Function Formulation
- **Categorical Cross-Entropy Loss**:
  $$\mathcal{L}_{\text{CE}} = -\sum_{c=1}^{C} y_c \log\left(\frac{e^{z_c}}{\sum_{j=1}^{C} e^{z_j}}\right)$$
  Where $C = 10$ is the number of scalp condition classes, $y_c$ is the one-hot ground-truth label, and $z$ represents the raw unnormalized logits produced by the final classification head.

### 3. Optimization Algorithm & Regularization
- **Optimizer**: **AdamW (Decoupled Weight Decay Regularization)**
  - **Initial Learning Rate ($\eta$)**: $1 \times 10^{-4}$ ($0.0001$).
  - **Decoupled Weight Decay ($\lambda$)**: $1 \times 10^{-2}$ ($0.01$).
  - Unlike standard $L_2$ regularization in Adam (which couples weight decay with gradient variance moments), AdamW decouples weight decay directly from gradient updates, avoiding premature weight saturation and improving generalizability on dermatological image sets.

### 4. Learning Rate Schedule
- **Cosine Annealing Schedule** (`optim.lr_scheduler.CosineAnnealingLR`):
  $$\eta_t = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{max}} - \eta_{\text{min}})\left(1 + \cos\left(\frac{t}{T_{\text{max}}}\pi\right)\right)$$
  - $T_{\text{max}} = \text{total epochs}$ (20 to 25 epochs).
  - Smoothly decays the learning rate along a cosine curve without sudden step-drops, allowing the model to settle into wider, more robust loss minima.

### 5. Data Augmentation & Preprocessing Pipeline
To make the model invariant to real-world smartphone capture conditions (handheld camera tilt, ambient light shifts, and varied phone sensors), the training pipeline applies:
1. **Resizing**: Bilinear interpolation to $224 \times 224$ pixels.
2. **Random Horizontal Flip**: $p = 0.5$ (exploits bilateral scalp anatomical symmetry).
3. **Random Affine Rotation**: Angle range $[-15^\circ, +15^\circ]$ with edge reflection padding to simulate natural handheld camera tilt.
4. **Photometric Color Jitter**:
   - Brightness jitter: $\pm 15\%$
   - Contrast jitter: $\pm 15\%$
   - Saturation jitter: $\pm 15\%$
   - Pre-conditions the network to remain invariant across warm incandescent (2700K) and cool fluorescent (6500K) room lighting.
5. **Z-Score Normalization**:
   $$x_{\text{norm}} = \frac{x - \mu}{\sigma}$$
   Normalized per-channel using ImageNet parameters ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

### 6. Dataset Split & Stratified Calibration Merge
- **Split Ratio**: 85% Training, 15% Validation split using deterministic index permutation (`torch.randperm`).
- **Volunteer Calibration Set Merge**: Implemented in `training/calibration_loader.py`. Designed to integrate 100–200 consented volunteer photos across Fitzpatrick skin phototypes I–VI to calibrate against demographic biases before final export.

### 7. Quantization, Model Tracing, and Export
- **Format**: Open Neural Network Exchange (ONNX) format via `torch.onnx.export`.
- **Opset Version**: `opset_version=14`.
- **Optimization**: Constant folding enabled (`do_constant_folding=True`) to pre-compute stationary layer weights and fuse batch normalization layers into convolutional weights.
- **Dynamic Axes**: Dynamic batch size enabled (`batch_size`), allowing single-image mobile inference as well as batch inference.

### 8. Fault-Tolerant Checkpointing
- **Mechanism**: Saves a full checkpoint dictionary (`checkpoint_epoch_{N}.pt`) after every training epoch:
  ```python
  {
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'scheduler_state_dict': scheduler.state_dict(),
      'val_loss': val_loss,
      'val_acc': val_acc,
      'best_val_acc': best_val_acc,
      'arch': arch
  }
  ```
- **Survival Guarantee**: Automatically syncs to Google Drive mounts to survive cloud session disconnections without data loss.

---

## 2. Complete Software Tools & Frameworks Specification

| Tool / Technology | Version / Spec | Functional Role in Follica |
| :--- | :--- | :--- |
| **PyTorch** (`torch`) | 2.1+ (CUDA 12.x) | Deep learning model definition, tensor mathematics, backpropagation, and training loop. |
| **timm** (PyTorch Image Models) | $\ge$ 0.9.16 | MobileNetV4 UIB primitives and pre-trained ImageNet-1K checkpoint loader. |
| **Torchvision** (`torchvision`) | 0.16+ | Image transformation and data augmentation pipeline (`ColorJitter`, `RandomRotation`). |
| **ONNX Runtime** (`onnxruntime`) | 1.16+ | Cross-platform inference engine delivering sub-50 ms CPU latency without PyTorch dependencies in production. |
| **OpenCV** (`cv2`) | 4.8+ | Classical CV feature extraction (Laplacian focus filter, Top-Hat morphology, Gray-World normalization). |
| **NumPy** (`numpy`) | 1.24+ | Array vectorization, mathematical invariant verification, and fallback image transformation operations. |
| **Pillow** (`PIL.Image`) | 10.x | Image file decoding (JPEG, PNG, WEBP) and photographic EXIF metadata parsing. |
| **FastAPI** | 0.104+ | Asynchronous REST API routing, Pydantic request/response validation, and CORS security. |
| **Uvicorn** | 0.24+ | High-performance ASGI web server hosting the FastAPI application across local and LAN interfaces. |
| **ReportLab** | 4.0+ | Algorithmic generation of 1-page clinical PDF doctor hand-off sheets. |
| **SQLite** (`sqlite3`) | 3.x | Persistent local storage for GDPR/HIPAA-aligned user consent records and right-to-erasure audits. |
| **React Native / Expo** | SDK 51+ | Cross-platform mobile front-end application with offline-first held-out specimen caching. |
| **Android SDK / AGP** | API 34 / AGP 8.x | Native Android packaging, Gradle release signing (V1+V2 PKCS12 keystore), and cleartext LAN networking. |
| **Pytest** (`pytest`) | 7.x / 9.x | Automated test suite comprising 114 unit, API, invariant, and adversarial specifications. |
| **GitHub Actions** | Ubuntu 24.04 Runners | Continuous Integration (CI) and automated release APK compilation and release asset publishing. |
| **Google Colab** | Nvidia T4 GPU (16 GB) | Cloud GPU environment used for model training with zero local workstation storage consumption. |

---

## 3. Classical Computer Vision Algorithms Summary

In parallel with deep learning, Follica executes deterministic classical algorithms:

1. **Shafer Dichromatic Reflection Separation**: Detects specular flash glare ($V > 240, |R-G| + |G-B| < 25$) to isolate surface lipid gloss and prevent false-positive dandruff squame scoring.
2. **Patch-Wise Discrete Laplacian Variance**: Computes the variance of the 2D discrete Laplacian kernel across an $8 \times 8$ grid, taking the 95th-percentile patch ($\sigma^2 \ge 60.0$) as the focus gate.
3. **Constrained Gray-World White Balance**: Constrains color channel scaling factors to $[0.85, 1.15]$ to neutralize indoor lighting temperature shifts while protecting genuine vascular erythema.
4. **Top-Hat Morphological Transform**: Extracts high-contrast bright particulate contours on darker scalp backgrounds for flake severity scoring.
5. **YCrCb Biological Tissue Segmentation**: Enforces human skin ($Cr \in [128, 180], Cb \in [75, 142]$) and hair melanin boundaries, rejecting non-scalp photographs (objects, clothing, pets) with HTTP 400.
