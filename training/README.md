# Follica ML Training & Calibration Pipeline

This directory contains the machine learning training pipeline designed to run **100% in the cloud (Google Colab / Kaggle)** with **0 MB local storage impact**.

---

## 1. Zero-Cost Cloud Training Workflow

```
[Kaggle / Roboflow Base Dataset]
                 │
                 ▼
[Google Colab Free Nvidia T4 GPU] <─── [Volunteer Calibration Photos (100–200 photos)]
                 │
                 ├── Saves `checkpoint_epoch_{N}.pt` to Google Drive every epoch
                 │   (survives free-tier disconnects)
                 │
                 ▼
[Export ONNX Model (`follica_mobilenetv4.onnx` ~14 MB)]
                 │
                 ▼
[Download ONLY the ~14 MB file to `backend/models/` on your laptop]
```

---

## 2. Model Architecture: MobileNetV4 vs. EfficientNet-B0

| Model | Size | Status | Rationale |
| :--- | :--- | :--- | :--- |
| **MobileNetV4** (`mobilenetv4_conv_small`) *(Default)* | **~14 MB** | Recommended | Google's next-generation (2024) mobile vision architecture with Universal Inverted Bottleneck (UIB) blocks. Delivers superior Pareto-optimal accuracy and lowest latency on mobile NPUs/CPUs. Export to ONNX is fully verified via `timm >= 0.9.16`. |
| **EfficientNet-B0** | **~19 MB** | Fallback | Battle-tested compound scaling model with universal cross-platform compatibility across all legacy TFLite/ONNX runtimes. |

---

## 3. Merging the Volunteer Calibration Set

To calibrate model accuracy across real-world Indian scalp and hair textures (e.g., student campus environments with hard water exposure), place 100–200 consented volunteer photos in:

```
data/calibration/
├── 0_normal/
├── 1_dry_scalp_dandruff/
├── 2_seborrheic_dermatitis/
├── 3_alopecia_areata/
└── 4_folliculitis/
```

The `calibration_loader.py` script automatically merges these images into the training and validation splits with stratified sampling.

---

## 4. How to Run on Google Colab

1. Open [Google Colab](https://colab.research.google.com/) and upload `follica_training_colab.ipynb`.
2. Change Runtime Type to **T4 GPU** (Runtime $\rightarrow$ Change runtime type $\rightarrow$ T4 GPU).
3. Run all cells. Checkpoints will automatically sync to your Google Drive folder: `/MyDrive/follica_checkpoints/`.
4. Once completed, download `follica_mobilenetv4.onnx` and copy it to `follica/backend/models/`.
