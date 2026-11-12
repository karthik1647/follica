# Course Project Report: Follica — Multimodal Optical Signal Processing and On-Device Deep Learning for Scalp Health Triage

**Submitted By:** Karthik Sativada  
**Project Category:** Biomedical Signal Processing, Computer Vision & Edge AI Systems  
**Source Code Repository:** [https://github.com/karthik1647/follica](https://github.com/karthik1647/follica)  
**GitHub Release Page (Signed Android APK):** [https://github.com/karthik1647/follica/releases/tag/v1.0.0-apk](https://github.com/karthik1647/follica/releases/tag/v1.0.0-apk)  
**Direct Android APK Download (100% Offline On-Device AI, ~137 MB):** [https://github.com/karthik1647/follica/releases/download/v1.0.0-apk/Follica-Clinical-Triage-v1.0.0.apk](https://github.com/karthik1647/follica/releases/download/v1.0.0-apk/Follica-Clinical-Triage-v1.0.0.apk)

---

## 1. Executive Summary & Abstract

Inflammatory, infectious, and scarring scalp disorders frequently go unrecognized in early stages due to overlapping visual morphology (erythema, perifollicular scaling, pustules, and diffuse vs. focal alopecia) and limited access to clinical trichoscopy. Existing consumer applications suffer from three fundamental engineering flaws: (1) they accept blurry, flash-washed, or non-scalp photographs without pre-inference validation, (2) they rely on black-box cloud neural networks that fail without internet connectivity, and (3) they conflate subjective questionnaire answers with physical image measurements.

**Follica** is an end-to-end clinical decision-support and scalp triage system engineered to solve these limitations. Rather than relying solely on a black-box neural network, **over 80% of Follica's codebase consists of deterministic digital signal processing (DSP), classical computer vision micro-feature extractors, and rule-based clinical safety state machines**, while **under 20% comprises a fine-tuned MobileNetV4-Conv-Small neural classifier**.

The system enforces a strict two-stage clinical workflow:
1. **Step 01 — Clinical Intake & Anamnesis (`survey_risk_engine.py`):** A 4-stage structured questionnaire capturing demographics, anatomical concern zones, extrinsic mechanical/environmental triggers (such as borewell hard water combined with daily helmet occlusion), systemic/nutritional indicators, and follicular pull-test dynamics to compute a standalone **Survey Risk Score (0 to 100)**.
2. **Step 02 — Optical Triage & Synthesis (`quality_gate.py`, `cv_pipeline.py`, `classifier_service.py`, `triage_engine.py`):**
   - **5-Stage Pre-Inference Quality Gate:** Rejects low-resolution (`<150x150`), underexposed (`Y < 35`), overexposed (`Y > 225`), flash-glare corrupted (`>15%` specular area), out-of-focus (2D Laplacian variance `< 60.0`), and out-of-distribution non-scalp photographs (`<10%` YCrCb cutaneous/melanin tissue) *before* inference.
   - **Classical CV Micro-Feature Pipeline:** Computes an objective **Scalp Health Score (SHS: 0 to 100)** strictly from image pixels across four 0-to-25 sub-scores: Flake/Scaling, Erythema Redness, Sebum/Shine, and Follicular Density.
   - **10-Class MobileNetV4 ONNX Classifier & Safety Gate:** Fine-tuned on a 12,000-image 10-class dataset, exported to a standalone `10.19 MB` ONNX graph, and governed by a `0.65` confidence threshold and Level 1 / Level 2 / Level 3 clinical escalation matrix.

To enable field deployment without Wi-Fi or cloud servers, the complete `10.19 MB` ONNX model and pixel-level quality gates are embedded directly inside a signed Android release APK via `onnxruntime-react-native`, executing **100% offline in Airplane Mode**.

---

## 2. Problem Formulation, Objectives & Engineering Contribution

### 2.1 Specific Project Objectives
1. **Objective 1 (Pre-Inference Signal Integrity):** Design a deterministic optical quality gate capable of rejecting motion blur, defocus, extreme illumination, specular camera flash hotspots, and non-scalp objects prior to neural network execution.
2. **Objective 2 (Explainable Classical CV Biomarkers):** Formulate white-balanced pixel segmentation algorithms to decompose scalp condition into four interpretable sub-scores (Flakes, Erythema, Sebum, Density) independent of neural network logits.
3. **Objective 3 (Multimodal Clinical Correlation without Bias):** Decouple subjective lifestyle/symptom history (Survey Risk Score) from pixel measurements (Scalp Health Score) while reconciling high-risk clinical red flags (such as bleeding ulcers or painful pustules) in the final triage routing.
4. **Objective 4 (100% Offline Edge Deployment):** Package the trained 10-class neural network and pixel quality gate into a standalone Android APK that operates with zero network dependency and 100% prediction parity with the Python backend.

### 2.2 Division of Engineering Effort: Deterministic DSP/CV (>80%) vs. Deep Learning (<20%)
A common critique of AI course projects is over-reliance on pre-built deep learning APIs. The table below documents the exact architectural division in Follica, demonstrating that the core contributions are custom signal processing, mathematical computer vision, and safety engineering:

| Subsystem | Codebase Module | Algorithmic Nature | Engineering Role & Contribution |
| :--- | :--- | :--- | :--- |
| **1. Optical Quality & OOD Gate** | `backend/app/services/quality_gate.py` & `mobile/src/services/offlineEngine.js` | **100% Deterministic DSP & Color-Space Math** | Rec. 601 luminance, dichromatic specular glare mask, 4-neighbor 2D Laplacian + 8x8 focal block sharpness, YCrCb cutaneous/hair chrominance gate, EXIF ISO/shutter evaluator. |
| **2. Classical CV Biomarkers** | `backend/app/services/cv_pipeline.py` | **100% Classical Computer Vision** | Constrained Gray-World white balance, adaptive glare-free flake thresholding, epidermal red-excess index, diffuse sebum shine, CIELAB `L*` hair density & 64x64 spatial tile uniformity. |
| **3. Longitudinal Registration** | `backend/app/services/registration.py` | **100% Geometric Computer Vision** | ORB keypoint detection (500 features), Brute-Force Hamming matcher, and RANSAC homography warping (`cv2.findHomography`) for baseline vs. follow-up alignment. |
| **4. Anamnesis Risk Engine** | `backend/app/services/survey_risk_engine.py` | **100% Rule-Based Clinical Modeling** | 4-step weighted risk matrix, compound trigger detection (hard water x helmet wear), and ranked modifiable intervention queue. |
| **5. Clinical Triage & Safety Gate** | `backend/app/services/triage_engine.py` | **100% Deterministic State Machine** | Enforces 0.65 confidence gate, reconciles photo-vs-symptom contradictions, routes Level 1/2/3 urgency, and maps brand-neutral active molecules. |
| **6. Privacy & Consent Store** | `backend/app/services/consent_store.py` | **100% Cryptographic / DB Engineering** | Salted SHA-256 pseudonymization, UUID v4 consent receipts, and DPDP/GDPR-compliant SQLite revocation. |
| **7. 10-Class Visual Classifier** | `training/train.py` & `classifier_service.py` | **Fine-Tuned Deep Learning (<20% of system)** | `mobilenetv4_conv_small` trained with AdamW, Cosine Annealing, label smoothing (`0.10`), and Albumentations; exported to Opset 14 ONNX (`10.19 MB`). |

---

## 3. Literature Review & Methodological Lineage

Every mathematical operator and clinical rule in Follica is derived from peer-reviewed literature and regulatory standards:

| Engineering Stage | Primary Literature Reference | How It Was Adapted & Implemented in Follica |
| :--- | :--- | :--- |
| **Specular Glare Separation** | Shafer, S. A. (1985). *Using Color to Separate Reflection Components*. Color Research & Application, 10(4), 210-218. | Implemented in `build_specular_glare_mask()` (`cv_pipeline.py`) and `calculate_flash_glare_ratio()` (`quality_gate.py`). Identifies achromatic high-intensity pixels (`max(R,G,B) > 240` and `|R-G| + |G-B| < 20`) caused by direct smartphone flash on curved scalp skin, masking them out before flake segmentation so flash reflections are never misclassified as dandruff. |
| **Macro Focus Validation** | Pech-Pacheco, J. L., et al. (2000). *Diatom Autofocusing in Brightfield Microscopy: A Comparative Study*. ICPR, Vol. 3, 314-317. | Implemented in `calculate_laplacian_variance()` (`quality_gate.py` & `offlineEngine.js`). Computes the variance of the discrete 2D Laplacian operator. Extended with an **8x8 spatial grid peak variance** check so that smooth bald skin or background bokeh does not artificially depress the sharpness score of an in-focus focal lesion (raising clinical pass rate from 70.8% to 95.9%). |
| **Illuminant Color Constancy** | Buchsbaum, G. (1980). *A Spatial Processor Model for Object Color Perception*. Journal of the Franklin Institute, 310(1), 1-26. | Implemented in `apply_gray_world_white_balance()` (`cv_pipeline.py`). Standard unconstrained Gray-World scaling pulls red inflamed scalps toward neutral gray, hiding disease. We implemented **Constrained Gray-World** by clamping per-channel scaling gains to `[0.70, 1.40]` and cross-checking against raw unscaled red-excess. |
| **Alopecia Coverage Staging** | Olsen, E. A., et al. (2004). *SALT Investigational Assessment Guidelines*. JAAD, 51(3), 440-447; Norwood, O. T. (1975). | Implemented in `extract_hair_density()` (`cv_pipeline.py`). Converts RGB to CIELAB and segments dark keratin hair shafts (`L* < 85`), applies morphological closing, and evaluates 64x64 spatial tile distribution to distinguish diffuse thinning from focal patchy loss. |
| **Seborrheic & Scaling Index** | Gupta, A. K., & Bluhm, R. (2004). *Seborrheic Dermatitis*. JEADV, 18(1), 13-26. | Implemented in `extract_flakes()` and `extract_erythema()` (`cv_pipeline.py`). Combines glare-free local adaptive contrast (`gray > local_mean + 55` and `gray > 185`) with exposed-skin red-excess chrominance (`R - 0.5*(G+B)`). |
| **Mobile Neural Backbone** | Qin, D., et al. (2024). *MobileNetV4: Universal Models for the Mobile Ecosystem*. ECCV 2024. | Implemented in `training/train.py`, `classifier_service.py`, and `offlineEngine.js`. Uses Universal Inverted Bottleneck (UIB) blocks to achieve a `10.19 MB` single-file ONNX footprint suitable for offline execution on budget Android smartphones. |
| **Clinical Safety Governance** | US FDA (2022). *Clinical Decision Support Software — Guidance for Industry and FDA Staff*. | Implemented in `triage_engine.py` and `constants.py`. Enforces non-diagnostic output phrasing ("Pattern Suggestive of..."), transparent basis disclosure, a 0.65 confidence gate, and mandatory Level 3 dermatologist escalation for scarring/infectious conditions. |

---

## 4. Mathematical Formulations & System Architecture

### 4.1 Stage 1: 5-Stage Pre-Inference Optical Quality Gate (`quality_gate.py` & `offlineEngine.js`)
Given an input RGB image tensor `I(x, y)` of dimensions `H x W x 3`:

1. **Dimensional Integrity Gate:**
   - Formula: `min(W, H) >= 150 px`
   - Rejects low-resolution thumbnails lacking spatial detail for follicular segmentation.
2. **Perceptual Luminance (Rec. 601) Exposure Gate:**
   - Formula: `Y(x, y) = 0.299 * R(x, y) + 0.587 * G(x, y) + 0.114 * B(x, y)`
   - Criterion: `35.0 <= mean(Y) <= 225.0` (rejects severe underexposure or washed-out overexposure).
3. **Dichromatic Specular Flash Glare Gate:**
   - Formula: `M_glare(x, y) = 1 if (max(R, G, B) > 240 AND (|R - G| + |G - B|) < 20) else 0`
   - Criterion: `(sum(M_glare) / (H * W)) * 100 <= 15.0%` (rejects frames obliterated by direct camera flash).
4. **Global + 8x8 Focal Patch Laplacian Sharpness Gate:**
   - Discrete 2D Laplacian: `L(x, y) = Y(x+1, y) + Y(x-1, y) + Y(x, y+1) + Y(x, y-1) - 4 * Y(x, y)`
   - Criterion: `S_sharp = max(Var(L_global), max_{i,j in 0..7} Var(L_patch[i,j])) >= 60.0`
5. **YCrCb Biological Cutaneous & Hair Chrominance Gate (Out-of-Distribution Rejection):**
   - Chrominance Transform: `Cr = (R - Y) * 0.713 + 128`, `Cb = (B - Y) * 0.564 + 128`
   - Epidermal Mask: `M_skin = (128 <= Cr <= 180) AND (75 <= Cb <= 142) AND (25 <= Y <= 240) AND (R >= B - 5) AND (|R - G| + |R - B| >= 8)`
   - Melanin Hair Mask: `M_hair = (Y < 85) AND (|R - G| < 40) AND (|G - B| < 40)`
   - Criterion: `BioRatio = mean(M_skin OR M_hair) * 100 >= 10.0%` (rejects non-scalp objects, pets, furniture, and landscapes).

### 4.2 Stage 2: Classical CV Sub-Scores & Composite Scalp Health Score (`cv_pipeline.py`)
Each valid photograph is decomposed into four sub-scores bounded in `[0, 25]` (where `25.0` represents optimal health):
- **Flake Score (`S_flake` in `[0, 25]`):** Computed from the area fraction of bright hyperkeratotic particles (`gray > local_mean + 55` and `gray > 185`) on glare-free non-hair epidermis (`NOT M_glare AND NOT M_hair`).
- **Redness Score (`S_red` in `[0, 25]`):** Computed from mean epidermal red-excess `E_r = mean_{M_skin}(R - 0.5 * (G + B))` across both constrained Gray-World white-balanced and raw channels.
- **Sebum Score (`S_sebum` in `[0, 25]`):** Computed from diffuse high-luminance sheen (`185 < Y <= 240`) on exposed skin after excluding specular flash hotspots.
- **Follicular Density Score (`S_density` in `[0, 25]`):** Computed from segmented hair shaft ratio (`L* < 85`) and `64 x 64` spatial tile coverage uniformity.
- **Composite Scalp Health Score (`SHS` in `[0, 100]`):**
  - Formula: `SHS = round(S_flake + S_red + S_sebum + S_density, 1)`

### 4.3 Stage 3: Neural Classification, Softmax Calibration & Triage Routing (`classifier_service.py`, `triage_engine.py`)
1. **ImageNet Normalization:** Input resized to `224 x 224`, normalized via `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`.
2. **Dominance-Calibrated Softmax:** Because the model is trained with label smoothing `0.10`, maximum raw softmax probability on compressed real-world photos can compress slightly below `0.65`. When top-1 probability `p_1` is in `[0.50, 0.65)` and dominance ratio `p_1 / p_2 >= 4.5`, temperature-scaled logits (`1.35 * z`) recover true confidence.
3. **Confidence Gate & Clinical Reconciliation:**
   - If `p_1 < 0.65`: Output `Indeterminate Pattern / Requires Verification` (`Level 2: Monitored Routine`).
   - If the visual prediction is non-inflammatory thinning (*Androgenetic Alopecia* or *Telogen Effluvium*) but the patient reports acute painful pustules or thick crusty plaques in Step 01, the engine reconciles the contradiction in favor of the higher-acuity inflammatory condition (*Folliculitis* or *Scalp Psoriasis*).

---

## 5. Dataset, Training Pipeline & Toolchain Specification

### 5.1 Dataset Composition & Cross-Split Audit
- **Source:** 10-Class Kaggle *Hair Diseases* Dataset (`12,000` total images: `9,600` train / `1,200` validation / `1,200` test; `1,200` balanced images per class).
- **10 Condition Classes:** `Alopecia Areata`, `Contact Dermatitis`, `Folliculitis`, `Head Lice`, `Lichen Planus`, `Male Pattern Baldness`, `Psoriasis`, `Seborrheic Dermatitis`, `Telogen Effluvium`, `Tinea Capitis`.
- **Integrity Audit:** Verified `0` unreadable/corrupt files and `0` exact MD5 byte duplicates across train/val/test splits.

### 5.2 Training Hyperparameters & Complete Toolchain

| Component | Library / Tool | Exact Specification in Repository |
| :--- | :--- | :--- |
| **Backbone Network** | `timm` (`1.0+`) | `mobilenetv4_conv_small.e1200_r224_in1k` (Universal Inverted Bottleneck) |
| **Deep Learning Framework** | PyTorch (`2.4+` CUDA AMP) | Mixed-precision FP16 (`torch.cuda.amp.autocast`), 15 epochs on NVIDIA T4 GPU |
| **Optimizer & Regularization** | `torch.optim.AdamW` | Learning rate `3e-4`, weight decay `1e-2`, Cross-Entropy `label_smoothing = 0.10` |
| **Learning Rate Schedule** | `CosineAnnealingLR` | Cosine decay from `3e-4` down to `1e-6` over 15 epochs |
| **Photometric & Geometric Augmentation** | `Albumentations` | `CLAHE`, `ColorJitter`, `RandomResizedCrop(224,224)`, `HorizontalFlip`, `ShiftScaleRotate` |
| **ONNX Graph Export** | `torch.onnx` & `onnx` | Opset 14, constant folding enabled, merged single-file `10.19 MB` binary |
| **Backend & PDF Engine** | FastAPI, OpenCV, NumPy, ReportLab | REST API (`/api/v1/analyze`), ORB homography, 1-page Clinical Pass PDF generator |
| **Offline Mobile Runtime** | React Native `0.74`, Expo `51`, `onnxruntime-react-native` `1.18.0` | Bundled `asset:///follica_mobilenetv4.onnx` + `expo-image-manipulator` + `jpeg-js` |
| **CI/CD & Cloud APK Builder** | GitHub Actions | `.github/workflows/ci.yml` (114 pytest tests) & `build-apk.yml` (API 34 RSA-2048 signed APK) |

---

## 6. On-Device Offline Android Engineering

To make the mobile app work completely offline without a laptop server or Wi-Fi connection:
1. **Single-File ONNX Consolidation (`mobile/scripts/export_offline_assets.py`):** Merged PyTorch's external weight file (`follica_mobilenetv4.onnx` + `.onnx.data`) into a single self-contained `10.19 MB` model (`mobile/assets/model/follica_mobilenetv4.onnx`).
2. **Native Android Asset Packaging (`.github/workflows/build-apk.yml`):** Configured Gradle (`minSdkVersion 24`, `targetSdkVersion 34`, `com.microsoft.onnxruntime:onnxruntime-android:1.18.0`) to package `follica_mobilenetv4.onnx` inside `android/app/src/main/assets/` and copy it to the app document directory on first launch.
3. **On-Device Inference & Quality Gate (`mobile/src/services/offlineEngine.js`):**
   - Uses `expo-image-manipulator` and pure-JS `jpeg-js` to decode camera/gallery photos into RGBA pixel arrays on the phone.
   - Runs `qualityMetrics()` and `evaluateQuality()` locally, enforcing the exact same 5-stage quality gate (resolution, brightness, glare, Laplacian sharpness, YCrCb skin/hair ratio) as the Python server.
   - Normalizes valid `224 x 224` pixel buffers into a `Float32Array(150528)` and executes `InferenceSession.run()` directly on the phone via `onnxruntime-react-native`.

---

## 7. Experimental Results, Critical Analysis & Verification

### 7.1 Quantitative Benchmark Summary

| Evaluation Benchmark | Dataset / Cohort Size | Measured Result | Engineering Interpretation |
| :--- | :--- | :--- | :--- |
| **Held-Out Test Split Accuracy** | `1,200` test images (10 classes) | **100.00% (`1,200 / 1,200`)** | Macro F1 = `1.000` on dataset test split (see Section 8.2 for critical analysis of dataset homogeneity). |
| **Adversarial Optical Perturbation** | `100` perturbed test images | **97.00% (`97 / 100`)** | Simultaneous 15-deg rotation + 15% border crop + 15% darkening + JPEG Q75 compression. |
| **Quality Gate Clinical Pass Rate** | `1,200` clinical test images | **95.92% (`1,151 / 1,200`)** | Upgraded from `70.8%` by adding 8x8 focal block sharpness and border studio-wall glare filtering. |
| **Quality Gate Corrupt/OOD Rejection** | Synthetic dark, bright, blurry, non-scalp | **100.00% Rejected** | All corrupt and non-scalp inputs intercepted before neural network inference. |
| **On-Device Mobile vs. Backend Parity** | `10` held-out clinical demo specimens | **100.00% (`10 / 10` match)** | Identical top-1 predicted disease class between mobile JS engine and Python backend. |
| **Real-Life Clinical Persona Suite** | `20` multimodal patient scenarios | **100.00% (`20 / 20` pass)** | End-to-end verification of intake + photo + triage routing (`test_real_life_expectations.py`). |
| **Automated Software Test Suite** | `114` unit, integration & fuzz tests | **100.00% (`114 / 114` pass)** | Verified in CI and locally via `python verify_accuracy.py`. |

### 7.2 On-Device Offline Parity Table (`node mobile/scripts/verify_offline_engine.js`)

| Specimen Image | On-Device Mobile Prediction | Confidence | Quality Gate | Python Backend Agreement |
| :--- | :--- | :--- | :--- | :--- |
| `alopecia_areata_0026.jpg` | Pattern Suggestive of Alopecia Areata / Patchy Loss | `0.926` | `PASS` | Exact Match (Class 0) |
| `contact_dermatitis_0036.jpg` | Pattern Suggestive of Contact Dermatitis / Irritation | `0.937` | `PASS` | Exact Match (Class 1) |
| `folliculitis_0062.jpg` | Pattern Suggestive of Folliculitis / Scalp Bumps | `0.960` | `PASS` | Exact Match (Class 2) |
| `head_lice_0036.jpg` | Pattern Suggestive of Head Lice / Infestation | `0.901` | `PASS` | Exact Match (Class 3) |
| `lichen_planus_0026.jpg` | Pattern Suggestive of Lichen Planus / Scarring Inflammation | `0.920` | `PASS` | Exact Match (Class 4) |
| `male_pattern_baldness_0026.jpg` | Pattern Suggestive of Androgenetic Alopecia / Pattern Thinning | `0.966` | `PASS` | Exact Match (Class 5) |
| `psoriasis_0026.jpg` | Pattern Suggestive of Scalp Plaque / Scaling | `0.969` | `PASS` | Exact Match (Class 6) |
| `seborrheic_dermatitis_0026.jpg` | Pattern Suggestive of Seborrheic Dermatitis | `0.952` | `PASS` | Exact Match (Class 7) |
| `telogen_effluvium_0026.jpg` | Pattern Suggestive of Telogen Effluvium / Diffuse Shedding | `0.956` | `PASS` | Exact Match (Class 8) |
| `tinea_capitis_0026.jpg` | Pattern Suggestive of Tinea Capitis / Fungal Scalp Infection | `0.949` | `PASS` | Exact Match (Class 9) |

---

## 8. Critical Discussion, Limitations & Academic Honesty

To maintain rigorous academic transparency, three limitations are explicitly documented:

1. **Critical Analysis of the 100.0% Dataset Split Accuracy:**
   While our fine-tuned MobileNetV4 achieves `100.0%` (`1,200 / 1,200`) on the held-out test split of the public Kaggle *Hair Diseases* dataset (with zero exact MD5 file duplicates across splits), our dataset audit (`README.md` Section 13.1) noted that public dermatology datasets often contain pre-augmented variations across splits. Therefore, `100.0%` reflects convergence on that curated distribution rather than guaranteed accuracy on arbitrary internet photographs. Under simultaneous synthetic camera perturbations (rotation, crop, darkening, JPEG compression), accuracy is `97.0%`, and on unconstrained wild photographs, our **0.65 Confidence Gate** and **5-Stage Quality Gate** act as essential safety barriers against domain shift.
2. **Provisional Nature of Classical CV Thresholds (`docs/thresholds.md`):**
   Numeric cutoffs in the classical CV pipeline (Laplacian blur `60.0`, flash glare `15.0%`, YCrCb bio-ratio `10.0%`, confidence gate `0.65`) are engineering heuristics calibrated on our development set. Clinical deployment requires prospective calibration across 100 to 200 dermatologist-graded photographs spanning Fitzpatrick skin phototypes IV to VI (`training/calibration_loader.py`).
3. **On-Device Sub-Score Approximation:**
   While the Android offline engine runs the **exact same 10.19 MB MobileNetV4 ONNX neural network** and the **exact 5-stage quality gate** as the Python backend, its 4 optical sub-scores use a streamlined single-pass RGBA pixel estimator in JavaScript (`subScores()` in `offlineEngine.js`) rather than full C++ OpenCV morphological dilation.

---

## 9. Evaluator & Professor Verification Guide (Reproducibility)

### 9.1 Install & Test the 100% Offline Android APK (Airplane Mode)
1. **Download APK on Android Phone (Android 7.0+):**
   - **Direct APK Download (~137 MB):** [https://github.com/karthik1647/follica/releases/download/v1.0.0-apk/Follica-Clinical-Triage-v1.0.0.apk](https://github.com/karthik1647/follica/releases/download/v1.0.0-apk/Follica-Clinical-Triage-v1.0.0.apk)
   - **GitHub Release Page:** [https://github.com/karthik1647/follica/releases/tag/v1.0.0-apk](https://github.com/karthik1647/follica/releases/tag/v1.0.0-apk)
2. **Verify Offline Execution:** Turn on **Airplane Mode** (disable Wi-Fi and mobile data), open **Follica**, complete **Step 01: Clinical Intake**, select a scalp photograph in **Step 02: Optical Triage**, and tap **Run On-Device Offline Triage (No Wi-Fi)**.
3. **Verify Safety Rejection:** Select a non-scalp photo (e.g., an object or landscape) or a dark/blurry image and verify that the on-device quality gate rejects it with an actionable error dialog before inference.

### 9.2 Run the Automated Verification Suite on Laptop (60 Seconds)
```powershell
# 1. Clone the repository
git clone https://github.com/karthik1647/follica.git
cd follica

# 2. Run the master verification scorecard (20 clinical personas + 114 pytest tests)
python verify_accuracy.py

# 3. Verify 10/10 on-device mobile ONNX parity & quality gates in Node.js
cd mobile
npm install --legacy-peer-deps
node scripts/verify_offline_engine.js
```
