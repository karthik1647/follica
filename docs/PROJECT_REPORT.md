# Biodesign Course Project Report: Follica — Multimodal Optical Signal Processing and On-Device Deep Learning for Scalp Health Triage

**Submitted By:** Karthik Sativada  
**Course Framework:** Stanford / India Biodesign Innovation Process (*Identify -> Invent -> Implement*)  
**Source Code Repository:** [https://github.com/karthik1647/follica](https://github.com/karthik1647/follica)  
**GitHub Release Page (Signed Android APK):** [https://github.com/karthik1647/follica/releases/tag/v1.0.0-apk](https://github.com/karthik1647/follica/releases/tag/v1.0.0-apk)  
**Direct Android APK Download (100% Offline On-Device AI, ~137 MB):** [https://github.com/karthik1647/follica/releases/download/v1.0.0-apk/Follica-Clinical-Triage-v1.0.0.apk](https://github.com/karthik1647/follica/releases/download/v1.0.0-apk/Follica-Clinical-Triage-v1.0.0.apk)

---

## Executive Summary

Structured around the three phases of the **Biodesign Innovation Process (Identify, Invent, Implement)**, **Follica** addresses a widespread clinical gap in primary dermatology: young adults and student populations experiencing scalp scaling, erythema, pruritus, or hair shedding face either predatory Direct-to-Consumer (D2C) cosmetic subscription funnels (`Rs 2,500-3,500/month`) or prohibitive specialist wait times and hardware trichoscopy costs. Conversely, naive consumer AI apps accept blurry, flash-glare corrupted, or non-scalp photographs and output uncalibrated guesses while requiring constant internet connectivity.

To solve this clinical need within strict safety and accessibility constraints, we engineered a dual-pipeline clinical decision-support system where **over 80% of the architecture is deterministic digital signal processing (DSP), classical computer vision, and regulatory safety state machines**, and **under 20% is a fine-tuned 10-class MobileNetV4-Conv-Small neural network**:
- **Phase I (Identify):** Scoped the clinical need across patients, dermatologists, campus clinics, and health administrators; formulated a deconstructed Biodesign Need Statement; and audited failure modes of existing D2C funnels, AI wrappers, and clinical dermascopes.
- **Phase II (Invent):** Screened competing engineering concepts via a Biodesign Pugh Matrix and built a 2-step multimodal pipeline: **Step 01 (4-Stage Clinical Anamnesis & Survey Risk Engine)** and **Step 02 (5-Stage Pre-Inference Optical Quality Gate + 4 Classical CV Sub-Scores + 10-Class MobileNetV4 ONNX Classifier + 0.65 Confidence Gate)**. Bundled the standalone `10.19 MB` ONNX model and pixel quality gates into a signed Android APK (`onnxruntime-react-native`) that executes **100% offline in Airplane Mode**.
- **Phase III (Implement):** Verified **100.0% held-out test accuracy (`1,200 / 1,200`)**, **97.0% adversarial optical perturbation accuracy (`97 / 100`)**, **95.9% clinical quality-gate pass rate (`1,151 / 1,200`)** with 100% corrupt/non-scalp rejection, **10 / 10 mobile-to-backend prediction parity**, **20 / 20 real-life clinical persona expectations**, and **114 / 114 automated pytest tests**, governed by FDA Clinical Decision Support Software (CDSS) non-diagnostic rules and India DPDP Act (2023) privacy compliance.

---

## PHASE I: IDENTIFY — Clinical Need Finding & Stakeholder Analysis

### 1.1 Clinical Observation & Problem Identification
1. **High Prevalence & Delayed Escalation:** Scalp disorders (*Seborrheic Dermatitis, Androgenetic Alopecia, Telogen Effluvium, Folliculitis, Scalp Psoriasis, Tinea Capitis, Alopecia Areata, and Lichen Planus*) affect upwards of 50% to 70% of university students and young adults due to hormonal shifts, chronic academic stress, hard borewell water exposure, and daily helmet occlusion. Delayed triage leads to irreversible outcomes including cicatricial (scarring) alopecia, advanced follicular miniaturization (Norwood IV-VII / Ludwig II-III), and secondary bacterial superinfections.
2. **The Over-The-Counter (OTC) Trial-and-Error Trap:** Patients routinely misidentify fungal *Malassezia* proliferation or psoriatic plaques as simple dry scalp, applying harsh cosmetic shampoos or heavy occlusive oils that worsen epidermal barrier disruption instead of targeted, affordable generic molecules (such as Ketoconazole 2%, Salicylic Acid 2%, or EDTA chelators).
3. **Hyper-Local Environmental Compounding:** Student and urban populations face compound mechanical and chemical triggers rarely modeled in Western clinical tools—specifically **hard borewell water (>300 ppm calcium/magnesium carbonate) combined with daily two-wheeler helmet sweat occlusion**, which creates an alkaline, humid microenvironment that accelerates follicular inflammation.

### 1.2 The Formal Biodesign Need Statement
**A way to objectively evaluate, quantify, and triage scalp pathology and follicular density in young adults and student populations, which provides explainable optical sub-score telemetry and generic molecule-first care guidance for mild conditions while generating standardized clinical hand-off summaries for dermatologist escalation, without requiring expensive clinic hardware, internet connectivity, invasive data monetization, or commercial product subscription lock-ins.**

| Biodesign Need Component | Specific Formulation in Follica |
| :--- | :--- |
| **Target Population** | Young adults, university students, and primary-care / tele-triage patients presenting with early scalp symptoms or hair shedding. |
| **Target Problem / Action** | Objective optical quality verification, quantitative micro-feature scoring, and risk-stratified triage (Level 1 Home Care vs. Level 2 Monitored vs. Level 3 Urgent Specialist Referral). |
| **Desired Clinical Outcome** | Early differentiation of benign xerosis/dandruff from scarring, infectious, or autoimmune dermatoses; reduction of OTC trial-and-error; structured 1-page Clinical Pass (PDF) for dermatologists. |
| **Design Constraints** | Zero hardware add-ons (standard smartphone camera), 100% offline capability (Airplane Mode), zero commercial brand bias (`Rs 100-200` generic active molecules), and strict non-diagnostic safety gating. |

### 1.3 Multi-Stakeholder Cycle of Care & Value Proposition

| Stakeholder Group | Current Clinical Pain Point | Follica Biodesign Value Proposition |
| :--- | :--- | :--- |
| **Patients (Students & Public)** | Confused by predatory D2C ads (`Rs 2,500-3,500/mo` kits); embarrassed to visit clinics early; cannot tell if shedding/flaking is benign or urgent. | Free, private, 100% offline smartphone triage; transparent `0-100` Scalp Health Score; unbiased chemical molecule names available at any pharmacy for `Rs 100-200`. |
| **Dermatologists & Trichologists** | Consultations wasted on mild dandruff or bogged down by poor patient recall and blurry WhatsApp photos; late presentation of scarring alopecia. | Standardized 4-step pre-intake anamnesis, optically validated macro photographs (blur/glare checked), 4 objective CV sub-scores, and a 1-page Clinical Pass PDF. |
| **Campus & Primary Care Clinics** | Overwhelmed outpatient queues; general practitioners lack polarized trichoscopes to screen scalp complaints quickly. | Automated pre-screening that filters Level 1 home-care cases while fast-tracking Level 3 red flags (*Alopecia Areata, Folliculitis, Tinea Capitis, Lichen Planus*). |
| **Healthcare Administrators** | High cost of specialist referrals and hardware dermatoscopes (`$2,000-$15,000` per unit). | Zero-cost open-source software running locally on existing smartphones or free-tier institutional containers. |

### 1.4 Market Landscape & Gap Analysis

| Feature / Biodesign Dimension | D2C Commercial Funnels (Traya, Man Matters, Hims) | Consumer AI Wrappers / Vision Chatbots | In-Clinic Hardware Scopes (FotoFinder, Dino-Lite) | Follica (Our Biodesign Solution) |
| :--- | :--- | :--- | :--- | :--- |
| **Financial & Ethical Alignment** | Sells proprietary monthly kits (`Rs 2,500-3,500/mo`). | Subscription paywalls or ad monetization. | High-fee clinic visits + `$2k-$15k` hardware. | **100% Free, Open-Source & Brand-Neutral**: Recommends generic active chemical molecules only. |
| **Offline Rural / Hostel Access** | Requires cloud web funnel. | Requires cloud LLM API. | Locked inside specialist clinics. | **100% Offline Android APK**: Runs MobileNetV4 ONNX + Quality Gate in Airplane Mode. |
| **Optical Quality & OOD Gate** | None (accepts blurry/dark photos). | None (hallucinates on non-scalp photos). | Controlled polarized ring-light hardware. | **5-Stage Deterministic Gate**: Rejects blur, dark/bright exposure, flash glare, and non-scalp objects. |
| **Explainable Telemetry** | Marketing labels ("Gut/Stress Stage 2"). | Opaque text paragraphs. | Physician-only hair count metrics. | **4 Mathematical Sub-Scores (`0-25` each)** summing to `0-100` Scalp Health Score + separate Survey Risk Score. |
| **Clinical Safety & Hand-Off** | Retains severe cases inside sales funnel. | Text disclaimer only; no doctor report. | Direct physician management. | **0.65 Confidence Gate + Level 3 Escalation + 1-Page Clinical Pass PDF**. |

---

## PHASE II: INVENT — Concept Screening, Signal Processing & System Architecture

### 2.1 Biodesign Concept Screening (Pugh Matrix)
During the **Invent** phase, four candidate concepts were evaluated against our Need Criteria:
- **Concept A:** Clip-on polarized USB/smartphone hardware dermatoscope + companion app.
- **Concept B:** Cloud-only Multimodal Vision LLM wrapper.
- **Concept C:** Questionnaire-only clinical rule engine (no computer vision).
- **Concept D (Selected — Follica):** Dual-pipeline deterministic DSP/CV Quality Gate & Sub-Score Engine + On-Device `MobileNetV4` ONNX Classifier + Decoupled Clinical Anamnesis Engine.

| Biodesign Need Criterion (Weight) | Concept A (Clip-On Hardware Scope) | Concept B (Cloud Vision LLM) | Concept C (Survey Rule Engine) | Concept D (Follica Dual-Pipeline On-Device) |
| :--- | :--- | :--- | :--- | :--- |
| **Zero Hardware Cost (20%)** | `-1` (Requires `$30-$200` lens) | `+1` (Smartphone only) | `+1` (Smartphone only) | **`+1` (Standard smartphone camera)** |
| **100% Offline Execution (20%)** | `+1` (Local USB) | `-1` (Fails without internet) | `+1` (Local rules) | **`+1` (Bundled `10.19 MB` ONNX in APK)** |
| **Optical Blur/Glare/OOD Rejection (20%)** | `+1` (Polarized LED) | `-1` (Accepts corrupt images) | `0` (No image used) | **`+1` (5-stage deterministic pixel gate)** |
| **Explainable Sub-Scores (15%)** | `+1` (Optical metrics) | `-1` (Black-box text) | `-1` (Subjective only) | **`+1` (4 pixel sub-scores `0-25` each)** |
| **Privacy & DPDP Compliance (15%)** | `+1` (Local storage) | `-1` (Third-party cloud upload) | `+1` (Local) | **`+1` (Zero-cloud offline + salted SHA-256)** |
| **Longitudinal Tracking (10%)** | `+1` (Fixed magnification) | `-1` (No alignment) | `-1` (No visual delta) | **`+1` (ORB keypoint RANSAC homography)** |
| **Weighted Total Score** | **`+0.60`** | **`-0.60`** | **`+0.30`** | **`+1.00` (Selected Architecture)** |

### 2.2 Biodesign Design Inputs to Engineering Outputs Traceability

| Biodesign Design Input (Clinical Requirement) | Target Engineering Metric (Design Output) | Implemented Module in Codebase |
| :--- | :--- | :--- |
| **Reject out-of-focus macro captures** | 2D Laplacian variance (global or 8x8 focal peak) `>= 60.0` | `quality_gate.py` & `offlineEngine.js` |
| **Reject under/overexposed lighting** | Rec. 601 mean luminance `Y in [35.0, 225.0]` | `quality_gate.py` & `offlineEngine.js` |
| **Eliminate camera flash dandruff false positives** | Dichromatic specular hotspot mask (`max(RGB) > 240`, chroma diff `< 20`) `<= 15.0%` | `quality_gate.py` & `cv_pipeline.py` |
| **Reject non-scalp photos (objects, pets, walls)** | YCrCb cutaneous skin + melanin hair pixel ratio `>= 10.0%` | `quality_gate.py` & `offlineEngine.js` |
| **Prevent subjective survey from biasing pixel score** | Strict architectural decoupling: `SHS (0-100)` from pixels only; `Risk Score (0-100)` from intake only | `cv_pipeline.py` vs. `survey_risk_engine.py` |
| **Prevent low-confidence AI misguidance** | Server & on-device confidence gate at `0.65` -> routes to `Indeterminate (Level 2)` | `classifier_service.py` & `triage_engine.py` |
| **Run 100% offline on budget Android phones** | Single-file ONNX footprint `<= 15 MB` (`10.19 MB` achieved), `minSdkVersion 24` | `mobile/assets/model/follica_mobilenetv4.onnx` |

### 2.3 Division of Engineering Effort: Deterministic DSP/CV (>80%) vs. Deep Learning (<20%)
To ensure clinical explainability and academic rigor, Follica does not treat scalp assessment as a naive image-to-label black box. Over 80% of the engineering implementation consists of mathematical signal processing, classical computer vision, and deterministic clinical state machines:

| Subsystem | Codebase Module | Algorithmic Category | Engineering Role |
| :--- | :--- | :--- | :--- |
| **1. Pre-Inference Quality Gate** | `backend/app/services/quality_gate.py` & `mobile/src/services/offlineEngine.js` | **100% Deterministic DSP & Color-Space Math** | Resolution check, Rec. 601 luminance, dichromatic specular glare mask, 2D Laplacian + 8x8 focal block sharpness, YCrCb skin/hair OOD gate, EXIF ISO/shutter parser. |
| **2. Classical CV Biomarkers** | `backend/app/services/cv_pipeline.py` | **100% Classical Computer Vision** | Constrained Gray-World white balance (`[0.7, 1.4]` gain clamp), glare-free adaptive flake segmentation, epidermal red-excess index, diffuse sebum shine, CIELAB `L*` hair density & 64x64 tile uniformity. |
| **3. Longitudinal Registration** | `backend/app/services/registration.py` | **100% Geometric Computer Vision** | ORB keypoint detection (500 features), Brute-Force Hamming matcher, and RANSAC homography warping (`cv2.findHomography`) for Day 0 vs. Day 30 alignment. |
| **4. Anamnesis Risk Engine** | `backend/app/services/survey_risk_engine.py` | **100% Rule-Based Clinical Modeling** | 4-step weighted risk matrix, compound trigger detection (hard borewell water x helmet wear), and ranked modifiable intervention queue. |
| **5. Clinical Triage & Safety Gate** | `backend/app/services/triage_engine.py` | **100% Deterministic State Machine** | Enforces 0.65 confidence gate, reconciles photo-vs-symptom contradictions, routes Level 1/2/3 urgency, and maps brand-neutral active molecules. |
| **6. Privacy & Consent Store** | `backend/app/services/consent_store.py` | **100% Cryptographic / DB Engineering** | Salted SHA-256 pseudonymization, UUID v4 consent receipts, and DPDP/GDPR-compliant SQLite revocation. |
| **7. 10-Class Visual Classifier** | `training/train.py` & `classifier_service.py` | **Fine-Tuned Deep Learning (<20% of system)** | `mobilenetv4_conv_small` trained with AdamW, Cosine Annealing, label smoothing (`0.10`), and Albumentations; exported to Opset 14 ONNX (`10.19 MB`). |

### 2.4 Scientific Literature & Methodological Lineage

| Engineering Stage | Primary Literature Reference | Adaptation & Implementation in Follica |
| :--- | :--- | :--- |
| **Specular Glare Separation** | Shafer, S. A. (1985). *Using Color to Separate Reflection Components*. Color Res. Appl., 10(4), 210-218. | `build_specular_glare_mask()` (`cv_pipeline.py`) & `calculate_flash_glare_ratio()` (`quality_gate.py`): Separates achromatic surface reflection (`max(RGB) > 240`, chroma diff `< 20`) from diffuse epidermal scatter so flash hotspots are never counted as dandruff flakes. |
| **Macro Focus Validation** | Pech-Pacheco, J. L., et al. (2000). *Diatom Autofocusing in Brightfield Microscopy*. ICPR, Vol. 3, 314-317. | `calculate_laplacian_variance()` (`quality_gate.py` & `offlineEngine.js`): Computes discrete 2D Laplacian variance across the global frame and peak 8x8 spatial sub-blocks (`>= 60.0`), raising clinical test pass rate from 70.8% to 95.9% while rejecting blurry captures. |
| **Illuminant Color Constancy** | Buchsbaum, G. (1980). *A Spatial Processor Model for Object Color Perception*. J. Franklin Inst., 310(1), 1-26. | `apply_gray_world_white_balance()` (`cv_pipeline.py`): Constrained Gray-World normalization clamping RGB channel gains to `[0.70, 1.40]` so warm indoor lighting is corrected without washing out erythema redness. |
| **Alopecia Coverage Staging** | Olsen, E. A., et al. (2004). *SALT Assessment Guidelines*. JAAD, 51(3), 440-447; Norwood, O. T. (1975). | `extract_hair_density()` (`cv_pipeline.py`): CIELAB `L* < 85` melanin shaft segmentation, morphological closing, and 64x64 spatial tile uniformity to quantify visible follicular density (`0-25`). |
| **Seborrheic & Scaling Index** | Gupta, A. K., & Bluhm, R. (2004). *Seborrheic Dermatitis*. JEADV, 18(1), 13-26. | `extract_flakes()` & `extract_erythema()` (`cv_pipeline.py`): Glare-free local contrast thresholding (`gray > local_mean + 55` and `gray > 185`) + exposed-skin red-excess chrominance (`R - 0.5*(G+B)`). |
| **Mobile Neural Backbone** | Qin, D., et al. (2024). *MobileNetV4: Universal Models for the Mobile Ecosystem*. ECCV 2024. | `train.py`, `classifier_service.py`, `offlineEngine.js`: Universal Inverted Bottleneck (UIB) backbone exported to a `10.19 MB` Opset 14 ONNX graph for sub-second CPU/mobile execution. |
| **Clinical Safety Governance** | US FDA (2022). *Clinical Decision Support Software — Guidance for Industry and FDA Staff*. | `triage_engine.py`: Enforces non-diagnostic terminology ("Pattern Suggestive of..."), transparent basis disclosure, 0.65 confidence gate, and mandatory Level 3 dermatologist referral for scarring/infectious conditions. |

### 2.5 Mathematical Formulations of the Pipeline
1. **Perceptual Luminance & Glare Masking:**
   - `Y(x, y) = 0.299 * R(x, y) + 0.587 * G(x, y) + 0.114 * B(x, y)`, valid exposure: `35.0 <= mean(Y) <= 225.0`.
   - `M_glare(x, y) = 1 if (max(R, G, B) > 240 AND (|R - G| + |G - B|) < 20) else 0`, valid glare ratio: `mean(M_glare) * 100 <= 15.0%`.
2. **Discrete 2D Laplacian & 8x8 Focal Patch Sharpness:**
   - `L(x, y) = Y(x+1, y) + Y(x-1, y) + Y(x, y+1) + Y(x, y-1) - 4 * Y(x, y)`
   - `S_sharp = max(Var(L_global), max_{i,j in 0..7} Var(L_patch[i,j])) >= 60.0`
3. **YCrCb Biological Cutaneous & Melanin Chrominance Gate:**
   - `Cr = (R - Y) * 0.713 + 128`, `Cb = (B - Y) * 0.564 + 128`
   - `M_skin = (128 <= Cr <= 180) AND (75 <= Cb <= 142) AND (25 <= Y <= 240) AND (R >= B - 5) AND (|R - G| + |R - B| >= 8)`
   - `M_hair = (Y < 85) AND (|R - G| < 40) AND (|G - B| < 40)`
   - `BioRatio = mean(M_skin OR M_hair) * 100 >= 10.0%`
4. **Composite Scalp Health Score (`SHS` in `[0, 100]`):**
   - `SHS = round(S_flake + S_red + S_sebum + S_density, 1)`, where each sub-score is bounded in `[0, 25]`.

### 2.6 Model Training & On-Device Offline Android Architecture
- **Dataset:** 12,000-image, 10-class Kaggle *Hair Diseases* dataset (`9,600` train / `1,200` val / `1,200` test across *Alopecia Areata, Contact Dermatitis, Folliculitis, Head Lice, Lichen Planus, Male Pattern Baldness, Psoriasis, Seborrheic Dermatitis, Telogen Effluvium, Tinea Capitis*). Audited with `0` corrupt files and `0` exact MD5 cross-split duplicates.
- **Training Configuration:** `timm` `mobilenetv4_conv_small.e1200_r224_in1k`, mixed-precision FP16 (`torch.cuda.amp.autocast`), `AdamW` (`lr = 3e-4`, `weight_decay = 1e-2`), `CosineAnnealingLR` (15 epochs on NVIDIA T4), Cross-Entropy with `label_smoothing = 0.10`, and Albumentations (`CLAHE`, `ColorJitter`, `ShiftScaleRotate`).
- **On-Device Offline Android Execution (`mobile/src/services/offlineEngine.js`):**
  - Consolidated PyTorch's external tensor weights into a single `10.19 MB` file (`mobile/assets/model/follica_mobilenetv4.onnx`) bundled inside `android/app/src/main/assets/`.
  - Decodes camera/gallery JPEGs on-device via `expo-image-manipulator` and `jpeg-js`, runs the 5-stage pixel quality gate locally, normalizes valid `224 x 224` tensors into `Float32Array(150528)`, and executes `InferenceSession.run()` on the smartphone via `onnxruntime-react-native` (`1.18.0`) with zero internet connection.

---

## PHASE III: IMPLEMENT — Verification, Regulatory Governance & Clinical Translation

### 3.1 Design Verification & Validation (V&V) Benchmarks

| Verification Dimension | Test Cohort / Protocol | Measured Result | Status |
| :--- | :--- | :--- | :--- |
| **Held-Out Test Split Accuracy** | `1,200` test images (10 classes) | **100.00% (`1,200 / 1,200`, Macro F1 = `1.000`)** | Verified |
| **Adversarial Camera Perturbation** | `100` perturbed test images (15-deg rot + 15% crop + 15% dark + JPEG Q75) | **97.00% (`97 / 100`)** | Verified |
| **Quality Gate Clinical Pass Rate** | `1,200` clinical test images | **95.92% (`1,151 / 1,200`)** (up from `70.8%` via 8x8 focal sharpness) | Verified |
| **Quality Gate Corrupt/OOD Rejection** | Synthetic black, white, blurry, and non-scalp images | **100.00% Rejected Prior to Inference** | Verified |
| **On-Device Mobile vs. Backend Parity** | `10` held-out clinical specimens (`verify_offline_engine.js`) | **100.00% (`10 / 10` exact class match)** | Verified |
| **Real-Life Clinical Persona Suite** | `20` multimodal patient scenarios (`test_real_life_expectations.py`) | **100.00% (`20 / 20` pass)** | Verified |
| **Automated Software Verification** | `114` unit, integration, privacy & fuzz tests (`verify_accuracy.py`) | **100.00% (`114 / 114` pass)** | Verified |

### 3.2 On-Device Offline Parity Verification (`node mobile/scripts/verify_offline_engine.js`)

| Specimen Image | On-Device Mobile Prediction | Confidence | Quality Gate | Backend Agreement |
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

### 3.3 Regulatory Pathway, Risk Management (ISO 14971) & Privacy Governance
1. **US FDA CDSS & India CDSCO Positioning:**
   - Follica is architected strictly as a **Non-Device Clinical Decision Support / Educational Triage Tool** under the 2022 FDA CDSS Guidance (Criterion 4: enabling the user/clinician to independently review the basis of recommendations via transparent optical sub-scores, EXIF quality metrics, and explicit symptom correlation notes).
   - **ISO 14971 Hazard Mitigation:**
     - *Hazard 1 (False reassurance on scarring/infectious pathology):* Mitigated by routing *Alopecia Areata, Folliculitis, Scalp Psoriasis, Lichen Planus, Tinea Capitis, Head Lice*, and any reported bleeding/non-healing sore or painful pustule directly to **Level 3: Urgent Dermatologist Escalation**.
     - *Hazard 2 (Hallucination on blurry or non-scalp photo):* Mitigated by the 5-stage Pre-Inference Quality Gate and the `0.65` Confidence Gate.
2. **DPDP Act (India 2023) & GDPR Data Privacy (`docs/privacy.md`):**
   - **Zero-Cloud Offline Mode:** In Android offline mode, photographs are decoded and analyzed strictly in volatile device RAM and never leave the phone.
   - **Revocable Pseudonymized Consent (`consent_store.py`):** Web/server consent records use random UUID v4 receipts and salted SHA-256 pseudonyms with instant user-initiated deletion (`DELETE /api/v1/consent/{id}`).

### 3.4 Critical Discussion, Limitations & Clinical Translation Roadmap
1. **Critical Analysis of Held-Out Dataset Accuracy (`100.0%`) vs. Wild Domain Shift:**
   While our model achieves `100.0%` (`1,200 / 1,200`) on the Kaggle held-out test split (with zero exact MD5 duplicates across splits), our dataset audit noted that public academic splits exhibit high lighting and framing homogeneity. Under simultaneous synthetic camera perturbations, accuracy is `97.0%`, and on arbitrary web photos, the **5-Stage Quality Gate** and **0.65 Confidence Gate** serve as essential guardrails against domain shift.
2. **Provisional Threshold Calibration (`docs/thresholds.md`):**
   Classical CV cutoffs (`blur = 60.0`, `glare = 15.0%`, `confidence = 0.65`) are documented as provisional engineering baselines. The next phase of clinical translation uses `training/calibration_loader.py` to calibrate these thresholds against 100 to 200 dermatologist-graded smartphone captures across Fitzpatrick skin phototypes IV to VI.

---

## 4. Professor & Evaluator Verification Guide (Reproducibility)

### 4.1 Install & Test the 100% Offline Android APK (Airplane Mode)
1. **Download the Signed Release APK (Android 7.0+ / API 24+):**
   - **Direct APK Download (~137 MB):** [https://github.com/karthik1647/follica/releases/download/v1.0.0-apk/Follica-Clinical-Triage-v1.0.0.apk](https://github.com/karthik1647/follica/releases/download/v1.0.0-apk/Follica-Clinical-Triage-v1.0.0.apk)
   - **GitHub Release Page:** [https://github.com/karthik1647/follica/releases/tag/v1.0.0-apk](https://github.com/karthik1647/follica/releases/tag/v1.0.0-apk)
2. **Test Offline Disease Prediction:** Turn on **Airplane Mode** (Wi-Fi and mobile data off), open **Follica**, complete **Step 01: Clinical Intake**, pick a scalp photograph in **Step 02: Optical Triage**, and tap **Run On-Device Offline Triage (No Wi-Fi)**.
3. **Test Pre-Inference Safety Rejection:** Pick a non-scalp image (e.g., an object or landscape) or a dark/blurry image and verify that the on-device quality gate rejects it with an explanatory dialog before running the neural network.

### 4.2 Run the Automated Verification Suite on Laptop (60 Seconds)
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
