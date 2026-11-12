# Follica: Multimodal Optical Signal Processing and On-Device Deep Learning for Scalp Health Triage

**Author:** Karthik Sativada  
**Repository:** [https://github.com/karthik1647/follica](https://github.com/karthik1647/follica)  
**Android Release APK (100% Offline On-Device AI):** [https://github.com/karthik1647/follica/releases/tag/v1.0.0-apk](https://github.com/karthik1647/follica/releases/tag/v1.0.0-apk)

## 1. Abstract

Inflammatory, infectious, and scarring scalp conditions frequently go unrecognized in early stages due to overlapping visual presentations (erythema, follicular scaling, diffuse vs. focal alopecia) and limited access to dermatological trichoscopy. Existing consumer applications rely either on subjective questionnaires or unconstrained cloud neural networks that fail under smartphone camera blur, flash glare, and out-of-distribution non-scalp photographs.

**Follica** presents an end-to-end clinical decision-support and scalp triage system that combines deterministic optical signal processing (>80% of pipeline logic) with lightweight deep learning (<20% of pipeline logic). The system enforces a 2-step clinical workflow:
1. **Step 01 (Clinical Intake & Anamnesis):** A 4-stage structured questionnaire capturing demographics, anatomical concern zones, extrinsic mechanical/environmental triggers (such as borewell hard water combined with daily helmet occlusion), systemic/nutritional indicators, and follicular pull-test dynamics to compute a standalone **Survey Risk Score (0 to 100)**.
2. **Step 02 (Optical Triage & Synthesis):** A 5-stage pre-inference optical quality gate (resolution, Rec. 601 luminance, specular flash glare separation, 2D Laplacian + 8x8 focal patch sharpness, and YCrCb biological chrominance validation), classical computer vision micro-feature extraction yielding an objective **Scalp Health Score (SHS: 0 to 100)** across four 0-to-25 sub-scores, and a 10-class **MobileNetV4-Conv-Small** ONNX classifier gated at a 0.65 confidence threshold.

To eliminate network dependency and privacy concerns, the complete 10.19 MB ONNX model and pixel-level quality gates are embedded directly inside a signed Android release APK using `onnxruntime-react-native`, executing 100% offline in Airplane Mode. Across validation benchmarks, Follica achieves **100.0% accuracy (1,200 / 1,200)** on the held-out 10-class test split, **97.0% accuracy (97 / 100)** under simultaneous optical perturbations, a **95.9% quality-gate pass rate (1,151 / 1,200)** on clinical photographs while rejecting 100% of synthetic corrupt/non-scalp inputs, **10 / 10 cross-runtime prediction parity** between mobile and backend, and **114 / 114 automated pytest verification tests** passing.

## 2. Clinical Need Statement & Market Gap

### 2.1 Formal Need Statement
A way to provide early, non-diagnostic scalp health triage and active-ingredient education for individuals experiencing hair shedding, flaking, or scalp irritation using standard smartphone photographs and structured lifestyle history, while preventing misguidance from poor-quality images or high-risk scarring/infectious conditions requiring urgent dermatologist care.

### 2.2 Comparative Analysis Against Existing Solutions

| Capability / Safety Dimension | Cosmetic Brand Apps | Hardware Salon Scopes | Tele-Dermatology Portals | Follica (Ours) |
| :--- | :--- | :--- | :--- | :--- |
| **Hardware Required** | Smartphone | Proprietary USB Scope | Smartphone | Standard Smartphone |
| **Offline On-Device Execution** | No (Cloud Lock-in) | Tied to Kiosk PC | No (Async Upload) | **Yes (100% Offline Android APK + FastAPI)** |
| **Pre-Inference Quality Gate** | None (Accepts Blur/Glare) | Manual Operator | Human Reviewer | **5-Stage Automated Gate (Blur, Exposure, Glare, YCrCb Tissue)** |
| **Specular Flash Glare Removal** | No | Polarized LED Hardware | No | **Algorithmic Dichromatic Masking** |
| **Quantitative Sub-Scores** | Black-Box Single Score | Vendor-Specific | Qualitative Notes | **4 Transparent Sub-Scores (0-25 each) -> 0-100 SHS** |
| **Commercial Bias** | Pushes Own Product Catalog | Pushes Salon Treatments | Prescription Only | **100% Brand-Neutral Active Chemical Molecules** |
| **Clinical Safety Escalation** | Rarely Escalates | Cosmetic Focus | Physician Dependent | **0.65 Confidence Gate + Level 3 Red-Flag Escalation** |

## 3. Scientific Literature & Methodological Lineage

Every signal-processing filter, color-space transformation, and clinical decision rule in Follica is grounded in peer-reviewed literature:

| Module / Engineering Stage | Scientific Paper / Standard | Mathematical / Algorithmic Implementation in Codebase |
| :--- | :--- | :--- |
| **Specular Glare Separation** | Shafer (1985), *Using Color to Separate Reflection Components*, Color Res. Appl. | `cv_pipeline.py` & `quality_gate.py`: Separates achromatic high-intensity surface reflection (`V > 240`, chroma diff `< 20`) from diffuse epidermal scatter so flash hotspots are never miscounted as dandruff flakes. |
| **Macro Focus & Blur Gate** | Pech-Pacheco et al. (2000), *Diatom Autofocusing in Brightfield Microscopy*, ICPR | `quality_gate.py` & `offlineEngine.js`: Computes discrete 2D Laplacian variance across the global frame and peak 8x8 spatial sub-blocks (threshold `>= 60.0`) so smooth skin does not dilute focal follicular sharpness. |
| **Illuminant Color Constancy** | Buchsbaum (1980), *A Spatial Processor Model for Object Color Perception*, J. Franklin Inst. | `cv_pipeline.py`: Constrained Gray-World normalization clamping RGB channel gains to `[0.7, 1.4]` so warm indoor lighting is corrected without erasing pathological erythema redness. |
| **Alopecia & Coverage Staging** | Olsen et al. (2004) *SALT* & Norwood (1975) *Male Pattern Baldness* | `cv_pipeline.py`: CIELAB `L*` luminance segmentation combined with morphological closing and 64x64 spatial tile uniformity to quantify visible follicular density (0 to 25). |
| **Seborrheic & Scaling Index** | Gupta & Bluhm (2004), *Seborrheic Dermatitis*, J. Eur. Acad. Dermatol. Venereol. | `cv_pipeline.py`: Glare-free local adaptive thresholding (`local_mean + 55` and `gray > 185`) + red-excess chrominance (`R - (G+B)/2`) on segmented non-hair skin pixels. |
| **Edge Neural Architecture** | Qin et al. (2024), *MobileNetV4: Universal Models for the Mobile Ecosystem*, ECCV | `classifier_service.py` & `offlineEngine.js`: `mobilenetv4_conv_small` Universal Inverted Bottleneck (UIB) backbone exported to Opset 14 ONNX (`10.19 MB`) for sub-second CPU/mobile inference. |
| **Clinical Governance** | US FDA (2022), *Clinical Decision Support Software Guidance* | `triage_engine.py`: Enforces non-diagnostic terminology ("Pattern Suggestive of..."), 0.65 confidence gate, and mandatory Level 3 dermatologist referral for scarring/infectious patterns. |

## 4. System Architecture & Multimodal Pipeline

Follica deliberately separates **subjective patient history** from **objective pixel measurements** so that questionnaire inputs never corrupt physical optical sub-scores:

### 4.1 Step 01: Clinical Intake & Survey Risk Engine (`backend/app/services/survey_risk_engine.py`)
Because public dermatology image datasets contain photographs without patient lifestyle history, Follica processes the 4-step clinical intake through a dedicated **Survey Risk Engine**:
- **Step 1 (Demographics & Zone):** Biological sex (Norwood vs. Ludwig reference), age bracket, and primary concern zone (Crown/Vertex, Hairline/Temples, Central Parting, Diffuse Thinning, Focal Patchy Spots, or Routine Check).
- **Step 2 (Extrinsic & Mechanical):** Water supply (Hard Borewell >300 ppm vs. Filtered), daily helmet occlusion, chemical/heat styling, and wash cadence. Detects compound mechanical traps such as *Hard Borewell Water + Daily Helmet Occlusion*.
- **Step 3 (Systemic & Nutritional):** Dietary protein/iron intake, gut absorption, metabolic/thyroid energy profile, endocrine indicators (PCOS/postpartum), stress, and sleep quality.
- **Step 4 (Follicular Dynamics):** Hair pull test (`<3` negative, `3-5` borderline, `>5` positive), flake morphology (fine dry white, greasy yellow, crusty plaque), scalp sensations (itching, burning, trichodynia pain, pustular bumps), and duration.
- **Output:** Produces a standalone **Survey Risk Score (0 to 100)**, risk band (`low`, `moderate`, `high`, `critical`), and a prioritized queue of modifiable lifestyle interventions.

### 4.2 Step 02: 5-Stage Pre-Inference Optical Quality Gate (`backend/app/services/quality_gate.py`)
Before any computer vision scoring or neural network inference occurs, every photograph must pass five deterministic checks (implemented in both Python and mobile JavaScript):

| Gate Stage | Metric & Formula | Rejection Threshold | Error Code |
| :--- | :--- | :--- | :--- |
| **1. Dimensional Resolution** | Pixel width and height `W, H` | `W < 150` or `H < 150` px | `IMAGE_RESOLUTION_TOO_LOW` |
| **2. Perceptual Exposure** | Rec. 601 Luminance `Y = 0.299R + 0.587G + 0.114B` | `mean(Y) < 35.0` or `> 225.0` | `IMAGE_UNDEREXPOSED` / `IMAGE_OVEREXPOSED` |
| **3. Flash Glare Filter** | Achromatic saturated pixels (`max(R,G,B) > 240` and `|R-G|+|G-B| < 20`) | Glare area `> 15.0%` of frame | `FLASH_GLARE_DETECTED` |
| **4. Focus Sharpness** | `max(Var(Laplacian_global), max_8x8(Var(Laplacian_patch)))` | Sharpness `< 60.0` | `IMAGE_TOO_BLURRY` |
| **5. Biological Tissue (OOD)** | YCrCb cutaneous skin (`Cr in [128,180], Cb in [75,142]`) + melanin hair (`Y < 85`) | Bio tissue `< 10.0%` | `NON_SCALP_IMAGE_DETECTED` |

### 4.3 Classical Computer Vision Micro-Feature Pipeline (`backend/app/services/cv_pipeline.py`)
Valid scalp images undergo constrained Gray-World white balancing and pixel segmentation to compute the **Scalp Health Score (SHS: 0 to 100)** as the sum of four independent **0-to-25 sub-scores** (higher = healthier):
1. **Flake & Scaling Score (0-25):** Quantifies bright hyperkeratotic particles on glare-masked non-hair epidermis using local contrast thresholding.
2. **Erythema / Redness Score (0-25):** Measures vascular inflammation via raw and white-balanced red-excess chrominance (`R - 0.5*(G + B)`) strictly over exposed scalp skin pixels.
3. **Sebum / Surface Shine Score (0-25):** Evaluates diffuse lipid sheen on non-glare skin regions after removing specular flash artifacts.
4. **Follicular Density / Coverage Score (0-25):** Combines CIELAB `L*` melanin shaft segmentation, morphological dilation, and 64x64 tile spatial uniformity.

### 4.4 Deep Learning Classification & Clinical Triage Engine (`classifier_service.py`, `triage_engine.py`)
1. **Preprocessing:** Resizes the RGB image to `224 x 224`, scales to `[0, 1]`, normalizes by ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`, and transposes to `NCHW` `(1, 3, 224, 224)`.
2. **Probability Calibration:** Applies numerically stable softmax with dominance recovery (`1.35x` logit scaling when top-1 probability is in `[0.50, 0.65)` with `>= 4.5x` dominance over runner-up, compensating for `0.10` label smoothing during training).
3. **0.65 Confidence Gate:** Any prediction with confidence `< 0.65` is intercepted and routed to `Indeterminate Pattern / Requires Verification` (`Level 2: Monitored Routine`), preventing low-confidence guessing.
4. **Clinical Contradiction Reconciliation & Escalation:**
   - **Level 1 (Minor Home Routine):** Healthy/Balanced Scalp, Mild Dry Scalp.
   - **Level 2 (Monitored Routine):** Seborrheic Dermatitis, Contact Dermatitis, Androgenetic Alopecia, Telogen Effluvium, or confidence-gated indeterminate scans.
   - **Level 3 (Consult a Dermatologist Urgently):** Alopecia Areata, Folliculitis, Scalp Psoriasis, Lichen Planus, Tinea Capitis, Head Lice, or any intake reporting bleeding/non-healing ulcers or acute painful pustules.

## 5. Dataset, Model Training & Toolchain Specification

### 5.1 Standardized 10-Class Dataset & Integrity Audit
The classifier was trained on the 12,000-image, 10-class *Hair Diseases* dataset (`9,600` train / `1,200` validation / `1,200` test; `1,200` images per class):
- **Classes (Alphabetical Index 0-9):** `Alopecia Areata`, `Contact Dermatitis`, `Folliculitis`, `Head Lice`, `Lichen Planus`, `Male Pattern Baldness`, `Psoriasis`, `Seborrheic Dermatitis`, `Telogen Effluvium`, `Tinea Capitis`.
- **Audit Results:** `0` corrupt files and `0` exact MD5 cross-split duplicates.

### 5.2 Training Hyperparameters & Software Toolchain

| Category | Tool / Parameter | Exact Configuration Used |
| :--- | :--- | :--- |
| **Backbone Architecture** | `timm` (`1.0+`) | `mobilenetv4_conv_small.e1200_r224_in1k` (Universal Inverted Bottleneck) |
| **Training Framework** | PyTorch (`2.4+` CUDA AMP) | Mixed-precision (`torch.cuda.amp.autocast`), 15 epochs on NVIDIA T4 GPU |
| **Optimizer** | `torch.optim.AdamW` | Learning rate `3e-4`, weight decay `1e-2` |
| **LR Scheduler** | `CosineAnnealingLR` | Cosine decay from `3e-4` to `1e-6` across 15 epochs |
| **Loss Function** | `nn.CrossEntropyLoss` | `label_smoothing = 0.10` to prevent overconfident logit saturation |
| **Optical Augmentations** | `Albumentations` | `CLAHE`, `ColorJitter`, `RandomResizedCrop(224,224)`, `HorizontalFlip`, `ShiftScaleRotate` |
| **Model Export** | `torch.onnx` & `onnx` | Opset 14, constant folding enabled, merged single-file `10.19 MB` artifact |
| **Backend & CV Stack** | FastAPI, OpenCV, NumPy, ReportLab | REST API, ORB homography registration, 1-page PDF Clinical Pass generator |
| **On-Device Mobile Stack** | React Native `0.74`, Expo `51`, `onnxruntime-react-native` `1.18.0` | Bundled `asset:///follica_mobilenetv4.onnx` + `expo-image-manipulator` + `jpeg-js` |

## 6. On-Device Offline Android Architecture

A key engineering requirement was enabling the mobile app to classify scalp diseases and enforce image quality checks **100% offline on an Android phone with no Wi-Fi or laptop server**.

1. **Single-File ONNX Consolidation (`mobile/scripts/export_offline_assets.py`):** Merged the PyTorch external tensor data file (`.onnx` + `.onnx.data`) into a single self-contained `10.19 MB` binary (`mobile/assets/model/follica_mobilenetv4.onnx`).
2. **Android Native Asset Packaging (`.github/workflows/build-apk.yml`):** Configured Gradle (`minSdkVersion 24`, `targetSdkVersion 34`, `onnxruntime-android:1.18.0`) to bundle `follica_mobilenetv4.onnx` inside `android/app/src/main/assets/`.
3. **On-Device Execution Pipeline (`mobile/src/services/offlineEngine.js`):**
   - Decodes the captured photo via `expo-image-manipulator` and `jpeg-js` into raw RGBA pixel buffers.
   - Runs `qualityMetrics()` and `evaluateQuality()` locally on the phone, rejecting blurry, dark, overexposed, glare-corrupted, or non-scalp photos before inference.
   - Resizes valid photos to `224 x 224`, applies ImageNet normalization into a `Float32Array(150528)`, and executes `InferenceSession.run()` on the phone's CPU/NPU via `onnxruntime-react-native`.
   - Computes pixel-level sub-scores, reconciles clinical intake flags, and renders the full triage report offline.

## 7. Experimental Results & Verification Benchmarks

### 7.1 Model Classification & Robustness Benchmarks

| Evaluation Benchmark | Sample Size | Measured Result | Notes |
| :--- | :--- | :--- | :--- |
| **Held-Out Test Split Accuracy** | `1,200` images (10 classes) | **100.00% (`1,200 / 1,200`)** | Macro F1 = `1.000` across all 10 classes |
| **Adversarial Camera Perturbation** | `100` perturbed test images | **97.00% (`97 / 100`)** | Simultaneous 15-deg rotation + 15% crop + 15% darkening + JPEG Q75 |
| **Quality Gate Clinical Pass Rate** | `1,200` clinical test images | **95.92% (`1,151 / 1,200`)** | Improved from `70.8%` after 8x8 focal patch sharpness upgrade |
| **On-Device Mobile vs. Python Parity** | `10` held-out demo specimens | **100.00% (`10 / 10` match)** | Verified via `node mobile/scripts/verify_offline_engine.js` |
| **Real-Life Clinical Persona Suite** | `20` multimodal patient scenarios | **100.00% (`20 / 20` pass)** | Verified via `python scripts/test_real_life_expectations.py` |
| **Automated Pytest Verification Suite** | `114` unit/integration/fuzz tests | **100.00% (`114 / 114` pass)** | Verified via `python verify_accuracy.py` |

### 7.2 On-Device Offline Parity Verification Log (`verify_offline_engine.js`)

| Specimen File | On-Device Predicted Condition | Confidence | Quality Gate | Backend Match |
| :--- | :--- | :--- | :--- | :--- |
| `alopecia_areata_0026.jpg` | Pattern Suggestive of Alopecia Areata / Patchy Loss | `0.926` | `PASS` | Exact Match |
| `contact_dermatitis_0036.jpg` | Pattern Suggestive of Contact Dermatitis / Irritation | `0.937` | `PASS` | Exact Match |
| `folliculitis_0062.jpg` | Pattern Suggestive of Folliculitis / Scalp Bumps | `0.960` | `PASS` | Exact Match |
| `head_lice_0036.jpg` | Pattern Suggestive of Head Lice / Infestation | `0.901` | `PASS` | Exact Match |
| `lichen_planus_0026.jpg` | Pattern Suggestive of Lichen Planus / Scarring Inflammation | `0.920` | `PASS` | Exact Match |
| `male_pattern_baldness_0026.jpg` | Pattern Suggestive of Androgenetic Alopecia / Pattern Thinning | `0.966` | `PASS` | Exact Match |
| `psoriasis_0026.jpg` | Pattern Suggestive of Scalp Plaque / Scaling | `0.969` | `PASS` | Exact Match |
| `seborrheic_dermatitis_0026.jpg` | Pattern Suggestive of Seborrheic Dermatitis | `0.952` | `PASS` | Exact Match |
| `telogen_effluvium_0026.jpg` | Pattern Suggestive of Telogen Effluvium / Diffuse Shedding | `0.956` | `PASS` | Exact Match |
| `tinea_capitis_0026.jpg` | Pattern Suggestive of Tinea Capitis / Fungal Scalp Infection | `0.949` | `PASS` | Exact Match |

## 8. Privacy, Regulatory Governance & Limitations

1. **DPDP Act (India 2023) & GDPR Alignment (`docs/privacy.md`):**
   - On-device offline mode processes photographs entirely in volatile phone memory with zero cloud transmission.
   - Server consent records use random UUID v4 identifiers and salted SHA-256 pseudonyms stored in local SQLite (`backend/app/services/consent_store.py`) with immediate user-initiated revocation (`DELETE /api/v1/consent/{id}`).
2. **Threshold Transparency (`docs/thresholds.md`):**
   - All classical CV and quality-gate cutoffs (Laplacian blur `60.0`, glare `15.0%`, confidence gate `0.65`) are explicitly documented as provisional engineering baselines awaiting calibration against 100 to 200 dermatologist-graded volunteer photographs across Fitzpatrick skin phototypes IV to VI.
3. **Dataset Scope & Real-World Generalization:**
   - While held-out test accuracy on the Kaggle dataset is 100.0%, internet images or extreme uncalibrated camera angles can exhibit domain shift. The 5-stage quality gate and 0.65 confidence gate mitigate this risk by intercepting out-of-distribution inputs rather than outputting forced predictions.
