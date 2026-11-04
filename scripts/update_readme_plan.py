readme_text = """# Follica: AI-Powered Scalp Health Assessment and Clinical Triage System

Follica is an open-source, dual-pipeline scalp health assessment and clinical triage platform. It evaluates user-uploaded scalp photographs and contextual clinical questionnaires to generate objective health scoring (Scalp Health Score 0-100), recommend targeted active ingredient care routines for minor conditions, and produce standardized clinical hand-off summaries when medical escalation to a dermatologist is warranted.

---

## 1. Need Scoping & Clinical Need Statement

### The Formal Need Statement
> **A way to objectively evaluate, quantify, and triage scalp pathology and follicular density in young adults and student populations, which provides explainable sub-score telemetry and generic molecule-first care guidance for mild conditions while generating standardized clinical hand-off summaries for dermatologist escalation, without requiring expensive clinic hardware, invasive data monetization, or commercial product subscription lock-ins.**

---

Following the standard medical device and clinical engineering need scoping methodology, Follica addresses the problem of neglected scalp disorders through seven structured dimensions:

### a. Clinical Relevance
- **Epidemic Scale and High Prevalence**: Scalp disorders (Seborrheic Dermatitis, Androgenetic Alopecia, Telogen Effluvium, Folliculitis, and Scalp Psoriasis) affect over 2 billion individuals globally and upwards of 50% to 70% of university students and young adults due to hormonal fluctuations, chronic academic stress, hard water exposure, and mechanical friction.
- **Prevention of Irreversible Complications**: Early objective detection and continuous tracking prevent irreversible clinical outcomes, including permanent cicatricial (scarring) alopecia, advanced androgenetic follicular miniaturization (Norwood Stages IV-VII / Ludwig Stages II-III), secondary bacterial superinfections from excoriation, and severe psycho-dermatological morbidity (social anxiety, clinical depression).
- **The Over-The-Counter Trial-and-Error Trap**: Patients frequently misidentify inflammatory or fungal flaking as simple dry scalp, using generic cosmetic shampoos that exacerbate stratum corneum barrier disruption. Non-adherence to condition-specific active molecules (e.g., Ketoconazole 2% for Malassezia yeast overgrowth vs. Salicylic Acid 2% for hyperkeratotic plaque descaling vs. chelating agents for mineral crust) worsens clinical trajectories.

### b. Stakeholder Perspectives
- **Patients (Students and General Public)**: Require accessible, private, non-judgmental, immediate, and zero-cost objective scoring. They need clear active-ingredient chemical schedules rather than commercial brand marketing, and transparent guidance on when self-care is safe versus when specialist consultation is imperative.
- **Doctors (Dermatologists and Trichologists)**: Require standardized clinical pre-intake, high-resolution scalp photography (with EXIF metadata assessed for optical quality indicators), and objective mathematical sub-scores (Flake, Erythema, Sebum, Follicular Density) to fast-track in-person differential diagnosis and eliminate patient recall bias.
- **Healthcare Systems and Campus Clinics (e.g., Institutional Health Services)**: Require scalable, automated triage to filter out benign xerosis and mild dandruff from active autoimmune or infectious presentations (Alopecia Areata, dissecting cellulitis), reducing outpatient wait queues and preventing clinician burnout.
- **Payers and Institutional Administrators**: Require low-overhead, evidence-based triage infrastructure that prevents unnecessary specialist referrals while ensuring timely escalation of complex dermatoses.

### c. Market and Population Potential
- **Large and Growing Addressable Population**: Scalp disorders are among the most prevalent dermatological conditions globally, and demand for accessible, affordable scalp-health screening tools is increasing. (Market size claims have been removed; no independently verified figure was available.)
- **High-Density Vulnerable Demographics**: High-density student and urban populations represent a high-friction demographic exposed to unique environmental aggravators: hard borewell water mineral crystallization, high atmospheric PM2.5 particulate deposition, daily motorcycle helmet friction and sweat occlusion, academic exam stress, and dietary micronutrient deficits (Iron/Ferritin, Vitamin D3, protein synthesis gaps).

### d. Technical Feasibility and Engineering Mitigations
- **Dual-Pipeline Architectural Approach**: Combines lightweight Deep Learning classification (MobileNetV4 / EfficientNet-B0) with deterministic Classical Computer Vision micro-feature extractors (Laplacian variance focus filtering, CLAHE dehazing, Gray-World White Balance color temperature normalization, and HSV/Lab color space analysis for vascular erythema and sebum lipid clustering) alongside a Multi-Factor Clinical Correlation Matrix.
- **Technical Challenges**: Smartphone optical variability, camera lens fingerprint oil and dust haze, ambient lighting color temperature shifts (2700K warm incandescent vs. 6500K daylight), diverse skin phototypes (Fitzpatrick I-VI), and lack of specialized polarizing dermascopes.
- **Engineering Mitigations**:
  - Pre-inference quality gate rejecting blurred captures (Laplacian variance < 60.0) and extreme exposure (brightness < 35 or > 225) as configured in core/config.py (provisional thresholds).
  - EXIF metadata parser evaluating sensor ISO noise rating, shutter speed motion blur risk, and flash firing status to compute an objective Optical Quality Score (0-100).
  - Gray-World White Balance color normalization to prevent warm ambient illumination from producing false-positive erythema readings.
  - Server-side confidence gate (< 0.65) strictly rerouting low-confidence predictions to Level 2 (Monitored Routine).

### e. Regulatory Position and Clinical Safety Governance
- **Software as a Medical Device (SaMD) vs. Informational Triage**: Follica is engineered and positioned strictly as an **educational assessment and clinical triage system**, not a certified diagnostic medical device.
- **Non-Diagnostic Linguistic Governance**: Hardcoded architectural constraints enforce probabilistic, non-diagnostic terminology (*"suggests visual pattern of"*, never *"diagnoses"* or *"prescribes"*).
- **Clinical Safety Protocols**: Mandatory universal medical disclaimers across all API payloads and automated generation of a 1-page structured clinical hand-off pass (PDF) for licensed physician review.

### f. Implementation and Clinical Adoption Strategy
- **Smartphone-Native Capture**: Direct camera integration with forced flash guidance and 4-rule visual capture instructions (Clean Lens, Flash ON, Expose Parting, 10-15 cm Distance).
- **Multi-Platform Deployment**: Zero-install responsive web application + cross-platform React Native / Expo mobile architecture.
- **Zero-Cost Footprint**: Ultra-low latency and zero local storage burden (<15 MB compressed ONNX model footprint, serverless/free-tier execution).
- **Professional Clinical Interface**: High-contrast, accessible design with zero decorative emojis, formal clinical typography, and 1-click clinical pass PDF generation.

---

## 2. Market Landscape: Existing Solutions vs. What We Are Building

### Market Solutions Breakdown & Systemic Shortcomings

The current scalp care and trichology market is bifurcated into three distinct categories, each characterized by critical structural deficiencies:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                CURRENT MARKET SOLUTIONS & FAILURE MODES                         │
├───────────────────────────────┬──────────────────────────────────┬───────────────────────────────┤
│ 1. D2C E-Commerce Funnels     │ 2. Consumer AI Chatbots/Apps     │ 3. Specialized Dermascopes    │
│ (Traya, Man Matters, Hims)    │ (SkinVision, GPT Wrappers)       │ (FotoFinder, Dino-Lite)       │
├───────────────────────────────┼──────────────────────────────────┼───────────────────────────────┤
│ - Primary Goal: Sell private- │ - Primary Goal: User engagement  │ - Primary Goal: In-clinic Rx  │
│   label kits (Rs 2500-3500/mo)│   via opaque text generation     │   by private specialists      │
│ - Zero Computer Vision triage │ - Black-box single class output  │ - Prohibitive cost ($2k-$15k) │
│ - Cannot prescribe cheap OTC  │ - Zero optical white-balancing   │ - Locked behind clinic walls  │
│   generic active molecules    │ - No clinical doctor hand-off    │ - Inaccessible to students    │
└───────────────────────────────┴──────────────────────────────────┴───────────────────────────────┘
```

### Comparative Analysis Matrix

| Feature / Dimension | D2C Commercial Funnels (e.g. Traya) | Consumer AI Wrappers / Apps | In-Clinic Hardware (FotoFinder) | Follica (What We Are Building) |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Financial Alignment** | E-commerce sales of proprietary subscription product bundles (Rs 2,000-3,500/month). | Advertising or recurring app subscription paywalls ($10-30/month). | High-fee specialist consultations ($50-150/visit) and hardware leasing. | **100% Free & Open-Source**: Zero-cost cloud footprint; no brand sponsorships or monetized kits. |
| **Visual Analysis Engine** | Form questionnaire; photo uploads are optional, non-standardized marketing attachments. | Single-pass Black-Box CNN or generic Multimodal Vision LLM without telemetry. | Multi-spectral polarized optical magnification (50x-200x) hardware sensor. | **Dual-Pipeline**: MobileNetV4 / EfficientNet-B0 + OpenCV Classical CV micro-feature extraction. |
| **Optical Invariance & Quality Gate** | None: Blurry, dark, or warm yellow room lighting photos are ingested without validation. | None: High vulnerability to color temperature shifts (2700K yellow light -> false erythema). | Controlled polarized ring-light hardware enclosure. | **Rigorous Pre-Inference & EXIF Gate**: Laplacian blur check, ISO sensor noise evaluation, and Gray-World White Balance normalization. |
| **Scoring Explainability** | Marketing categorizations ("Root Cause: Gut/Stress", "Stage 2 Thinning") without math. | Opaque text generation without numeric sub-score telemetry. | Proprietary physician metrics (hair count/cm2) requiring specialized training. | **Transparent Composite Index (0-100 SHS)** derived from 4 discrete sub-scores (0-25 each for Flake, Redness, Sebum, Density). |
| **Treatment Guidance Strategy** | Mandates purchase of private-label proprietary serums, ayurvedic oils, and internal supplements. | Generic lifestyle tips or general advisory text without chemical specificity. | Prescription-only pharmaceutical compounds (Oral Finasteride, Spironolactone). | **Molecule-First Generic Guidance**: Evidence-based active chemical molecules (Ketoconazole 2%, Salicylic Acid 2%, Zinc Pyrithione 1%, EDTA chelators) available at any pharmacy for Rs 100-200. |
| **Medical Safety & Triage Escalation** | Tries to retain high-risk presentations within the supplement purchasing funnel. | Disclaimers embedded in text, but no structured physician hand-off mechanism. | Direct in-person physician management. | **Strict Confidence Gate (<0.65) & Clinic Pass (PDF)**: Automatically halts consumer self-care and generates a standardized 1-page clinical hand-off report for certified dermatologists. |
| **Longitudinal Tracking Rigor** | Uncalibrated gallery photos subject to distance, angle, and lighting shifts. | Basic photo logging without pixel-level registration. | Automated serial dermatoscopy with fixed mechanical camera arm. | **ORB Homography Keypoint Registration**: Algorithmic perspective warping of Day 14/30 follow-up onto Day 0 coordinates for spot-by-spot delta analysis. |
| **Hyper-Local Context Modeling** | Generic questionnaires ignoring hostel and campus environmental realities. | Global generic prompts. | Clinical anamnesis recorded in electronic medical record (EMR). | **Multi-Factor Correlation Engine**: Models hard borewell water mineral crusting, motorcycle helmet sweat/friction occlusion, exam stress, and student protein deficits. |

---

### What We Are Building: The 5 Strategic Pillars

Follica bridges the chasm between inaccessible high-cost clinical hardware and commercialized D2C product sales funnels through five core engineering pillars:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                THE 5 STRATEGIC PILLARS OF FOLLICA                                │
├───────────────────────────────┬──────────────────────────────────┬───────────────────────────────┤
│ 1. Dual-Pipeline CV Engine    │ 2. Optical Normalization & EXIF  │ 3. Molecule-First Regimens    │
│ - Parallel DL Classifier      │ - Pre-inference Laplacian gate   │ - Pure chemical actives       │
│ - 4 Deterministic Sub-Scores  │ - EXIF sensor noise evaluation   │ - No commercial brand bias    │
│ - Explainable 0-100 SHS score │ - Gray-World White Balance       │ - Rs 100-200 generic options  │
├───────────────────────────────┼──────────────────────────────────┼───────────────────────────────┤
│ 4. Multi-Factor Correlations  │ 5. Clinical Safety & Clinic Pass                                 │
│ - Hard water mineral crust    │ - Server-side confidence gate (<0.65 reroutes to Level 2)        │
│ - Daily helmet sweat trap     │ - 1-Click standardized Clinical Summary PDF for dermatologist    │
│ - Pull test & shedding rate   │ - ORB homography longitudinal tracking (Day 0 vs Day 14/30)      │
└───────────────────────────────┴──────────────────────────────────────────────────────────────────┘
```

1. **Dual-Pipeline Computer Vision Engine (DL + Classical CV)**:
   - Eliminates the black-box opacity of standard AI. The Deep Learning classifier (MobileNetV4 / EfficientNet-B0) identifies broader pathological visual patterns, while classical computer vision algorithms extract four discrete, mathematically bounded micro-feature sub-scores (0.0 to 25.0 each):
     - `flake_score` (Stratum corneum cluster density)
     - `redness_score` (Vascular erythema intensity)
     - `sebum_score` (Surface lipid distribution)
     - `density_score` (Follicular unit spacing index)
   - The arithmetic sum forms the **Scalp Health Score (0-100 SHS)**.

2. **Optical Invariance and Sensor-Noise Normalization**:
   - Consumer smartphone photography is notoriously prone to optical corruption. Follica executes a pre-inference quality gate that intercepts blurred photos (Laplacian variance < 60.0, provisional threshold configured in core/config.py) and uncalibrated lighting.
   - Parses EXIF metadata (ISO rating, shutter speed, flash status) to compute an objective **Optical Quality Score (0-100)**.
   - Applies **Gray-World White Balance color normalization** to ensure that warm yellow incandescent room lights (2700K) are neutralized before calculating erythema sub-scores.

3. **Molecule-First Care Schedules (Zero Commercial Bias)**:
   - Unlike commercial funnels that require proprietary subscriptions, Follica outputs pure active chemical molecules with exact concentrations, mechanisms of action, and leave-on application durations:
     - *Seborrheic Dermatitis*: Ketoconazole 2% wash (5-minute contact time, 2x/week) + Zinc Pyrithione alternate wash + Salicylic Acid 2% exfoliant.
     - *Hard Water Mineral Crust*: EDTA chelating wash + Salicylic Acid descaler + Ceramide barrier lipid serum.
   - Enables users to obtain effective treatments from any generic pharmacy for Rs 100-200.

4. **Multi-Factor Clinical & Environmental Correlation Matrix**:
   - Synthesizes user lifestyle data with visual CV metrics to explain root drivers:
     - *Hard Water + Sebum* -> Insoluble calcium-fatty acid soap scum deposition.
     - *Daily Helmet Wear + Itching* -> Friction and sweat occlusion accelerating *Malassezia* fungal proliferation.
     - *Positive Hair Pull Test (>5 hairs) + Sudden Shedding + Exam Stress* -> Acute Telogen Effluvium premature follicle transition.
     - *Family History + Vertex Thinning* -> Androgenetic miniaturization predisposition.

5. **Clinical Hand-Off Pass & Longitudinal ORB Registration**:
   - For high-risk or unresolving cases (Level 3 Escalation), Follica generates a standardized **1-Page Clinical Summary PDF** containing macro/micro photos, sub-score telemetry, symptom duration, attempted treatments, and physician notes.
   - For longitudinal tracking, the ORB keypoint registration engine aligns follow-up captures (Day 14/30) onto Day 0 coordinates via homography perspective transformations, providing objective delta tracking invariant to user camera movement.

---

## 3. Strategic Engineering Roadmap

```
==================================================================================================
PHASE 1: DUAL-PIPELINE CORE & WEB STUDIO [COMPLETED]
├── FastAPI dual-pipeline inference service (/api/v1/analyze)
├── Pre-inference Laplacian quality gate and EXIF optical analyzer (/api/v1/quality/check)
├── Classical CV micro-feature extractors with Gray-World white balancing
├── Multi-factor clinical correlation matrix & server confidence gate (<0.65)
├── Responsive web application with shimmering skeleton UI & 1-click clinical PDF pass
└── 46-test automated integration and adversarial stress suite (pytest)

PHASE 2: MOBILE APPLICATION WITH HARDWARE-LOCKED CAPTURE [IN PROGRESS]
├── React Native (Expo) cross-platform iOS/Android mobile client (mobile/)
├── Hardware camera viewfinder with locked flash and distance crosshair overlays
├── Local SQLite scan diary with offline baseline caching (recommends OS-level disk encryption)
└── Adherence push notifications for active wash leave-on schedules (e.g. 2x/week)

PHASE 3: ADVANCED TRICHOSCOPIC BIOMARKERS [PLANNED]
├── Hair Shaft Caliber Heterogeneity Ratio (OpenCV Hough/Gabor filter for miniaturization index)
├── Total Dissolved Solids (TDS) regional hard water chelating cadence calculator
└── Multi-angle composite stitching (Crown, Hairline, and Parting in unified 3D mesh)

PHASE 4: CLINICIAN TELE-TRIAGE & QR INTEGRATION [PLANNED]
├── Cryptographically signed QR code on Clinical Pass PDF for instant physician browser review
├── Zero-install doctor view rendering high-resolution zoomable trichoscopy and sub-score metrics
└── Anonymous federated learning pipeline for privacy-preserving model calibration
==================================================================================================
```

---

## 4. Core Design Principles

Every engineering decision in Follica is governed by five strict principles:

- **Zero-Cost Infrastructure**:
  The entire pipeline is engineered to run on free-tier services. Heavy deep-learning training runs on free Google Colab/Kaggle cloud GPUs (0 MB storage burden on local laptops). Model weights are compressed to lightweight ONNX formats (~14 MB).
- **Explainable Scoring (No Black Boxes)**:
  Rather than outputting an opaque label, Follica executes a Dual-Pipeline: a Deep Learning classifier runs alongside a deterministic Classical Computer Vision pipeline that extracts four discrete, mathematically bounded sub-scores (Flake, Redness, Sebum, Density).
- **Strict Non-Diagnostic Language**:
  Software analyzing consumer photos must not make authoritative medical diagnoses. The system strictly uses probabilistic terminology (*"suggests"*, *"patterns consistent with"*, *"recommends an active ingredient schedule"*) and never uses prescriptive verbs (*"diagnoses"*, *"prescribes"*).
- **Institutional Neutrality**:
  The system contains no hardcoded clinic, university, or hospital names. Escalations are addressed generically (*"For your treating physician / dermatologist"*) and provide a self-triggered local search link rather than a locked-in booking vendor.
- **Safety-First Confidence Gating**:
  Any classifier prediction with confidence below 0.65 is intercepted server-side and automatically rerouted to Level 2 (Monitored Routine) to prevent improper dismissal or unnecessary panic.

---

## 5. System Architecture at a Glance

```
                         [User Upload: Scalp Photo + Multi-Factor Clinical Intake]
                                                     │
                                                     ▼
                                     [Pre-Inference Image Quality Gate]
                                     - Laplacian Variance Focus / Blur Check
                                     - Luminance & Illumination Distribution Check
                                     - EXIF Metadata Parser (ISO Noise, Shutter, Flash)
                                     - Dimension Integrity Check (>=150x150px)
                                                     │
                               ┌─────────────────────┴─────────────────────┐
                               ▼ (Failed Quality Gate)                     ▼ (Passed Quality Gate)
                     [HTTP 400 Bad Request]                      [Dual-Pipeline Assessment]
                     Actionable Guidance:                                  │
                     - Blur / Retake Focus              ┌──────────────────┴──────────────────┐
                     - Dark / Flash Required            ▼                                     ▼
                     - Overexposure / Glare    [Pipeline 1: Deep Learning]           [Pipeline 2: Classical CV]
                                               MobileNetV4 / EfficientNet-B0         OpenCV / NumPy Micro-Features
                                                        │                            - Gray-World White Balance
                                                        │                            - Flake & Scaling (0-25)
                                                        │                            - Erythema Redness (0-25)
                                                        │                            - Sebum & Lipids (0-25)
                                                        │                            - Follicular Density (0-25)
                                                        ▼                                     │
                                               Raw Condition + Conf (0.0-1.0)                 ▼
                                                        │                            4 Discrete Sub-Scores (0-25)
                                                        │                                     │
                                                        └──────────────────┬──────────────────┘
                                                                           ▼
                                                             [Multi-Factor Correlation Engine]
                                                             - Hard Water Mineral Deposition
                                                             - Helmet Friction / Sweat Occlusion
                                                             - Acute Telogen Effluvium Risk
                                                             - Androgenetic Miniaturization
                                                             - Perifollicular Trichodynia
                                                             - Nutritional Synthesis Deficits
                                                                           │
                                                                           ▼
                                                             [Server-Side Triage Engine]
                                                                           │
                                                ┌──────────────────────────┴──────────────────────────┐
                                                ▼                                                     ▼
                                       [Confidence < 0.65]                                   [Confidence >= 0.65]
                                   Strict Server Reroute:                                Evaluate Severity:
                                   Level 2: Monitored Routine                            - Level 1: Minor Home Routine
                                   (Inconclusive visual signals)                         - Level 2: Monitored Active Routine
                                                                                         - Level 3: Escalation (Dermatologist)
```

---

## 6. Core Features & User Journey

### A. Multi-Angle Photo Scanner
- **Guided Capture Mode**: Viewfinder overlays for Crown/Vertex, Frontal Hairline, Left/Right Temples, and Parting Line.
- **Pre-Inference Quality Validation**: Real-time flash, focus distance, and Laplacian blur gating (`/api/v1/quality/check`) preventing blurry or poorly illuminated uploads from skewing scores.
- **Optical Invariance**: Gray-World White Balance normalization neutralizes warm incandescent room lighting (2700K) to prevent false-positive erythema readings.

### B. Multi-Section Clinical Context Engine
Captures physiological and environmental drivers through 4 progressive sections:
1. **Demographics & Concern Area**: Age bracket (<18 to 50+), biological sex model (Norwood vs. Ludwig), primary concern zone.
2. **Environment, Friction & Habits**: Hard borewell vs. treated water, daily motorcycle helmet wear (sweat/friction occlusion), chemical/heat styling, washing cadence.
3. **Nutrition, Sleep & Biomarkers**: Protein intake, sleep quality, known deficiencies (Vitamin D3, Iron/Ferritin, Thyroid imbalance), exam/work stress history.
4. **Scalp Dynamics & Sensation**: Sensation (itching, burning, pain), self-administered hair pull test (<3, 3-5, >5 hairs shed), flake presentation, and symptom duration slider.

### C. The "Scalp Health Score" (SHS: 0–100)
- **Transparent Composite Index** calculated across 4 discrete sub-scores (0–25 each, where 25 represents optimal health):
  - **Flake & Scaling Index (0–25)**: Stratum corneum desquamation density and cluster area.
  - **Redness & Erythema Index (0–25)**: Micro-vascular inflammation and vascular tone.
  - **Sebum & Lipid Balance (0–25)**: Lipid overproduction vs. stratum corneum barrier dehydration.
  - **Follicular Density Index (0–25)**: Hairline and parting coverage comparison against standardized density boundaries.

### D. Ingredient-First Care Schedule (No Random Brands)
- Recommends scientific active chemical molecules with explicit biochemical mechanisms and application cadences rather than sponsored cosmetic brands:
  - *Oily Fungal Scaling*: Ketoconazole 2% wash 2x/week (5-min leave-on) + Zinc Pyrithione alternate wash + Salicylic Acid 2% solution.
  - *Hard Water Mineral Buildup*: EDTA chelating wash + Salicylic Acid exfoliant + Ceramide lipid replenishment.
  - *Acute Effluvium Support*: Gentle sulfate-free daily cleanser + barrier soothing serum (Panthenol/Aloe) + nutritional evaluation.

### E. Progress Tracker & Comparison Split Slider (Hair Diary)
- 14-day and 30-day longitudinal re-scan comparison.
- Homography keypoint alignment engine (`backend/app/services/registration.py`) calculating spot-by-spot follicle coverage deltas and erythema reduction.
- Interactive before/after split comparison slider in the web interface.

### F. Clinical Doctor Hand-off (Clinic Pass)
- For Level 3 escalations or unresolving symptoms:
- 1-Click generation of a standardized, 1-page clinical summary PDF (`/api/v1/report/pdf`) detailing images, timeline, duration, attempted washes, and objective sub-score metrics for seamless physician/dermatologist consultation.

### G. Future Scalability: DermaIQ Multi-Domain Architecture
- Modular domain registry (`backend/app/core/domains.py`) enabling pluggable expansion from Scalp Health to Facial Acne and Eczema/Skin Rashes with zero breaking changes to the core engine.

---

## 7. Current Status: What Is Built vs. Stubbed

To help new contributors navigate the codebase without ambiguity:

| Component / File | Implementation Status | Description & Notes |
| :--- | :--- | :--- |
| `backend/app/api/endpoints.py` | **Fully Implemented** | Exposes `/analyze`, `/consent`, `/system/status`, `/report/pdf`, `/quality/check`, `/progress/compare`, `/domains`. |
| `backend/app/services/quality_gate.py` | **Fully Implemented** | Pre-inference gate with Laplacian variance blur check, luminance distribution, EXIF analyzer, and dual OpenCV/NumPy fallback. |
| `backend/app/services/registration.py` | **Fully Implemented** | Longitudinal ORB keypoint homography alignment and spot-by-spot delta calculations. |
| `backend/app/core/domains.py` | **Fully Implemented** | DermaIQ modular domain architecture registry (Scalp, Facial Acne, Eczema). |
| `backend/app/static/index.html` | **Fully Implemented** | Interactive single-page web UI with guided capture rules, 4-step clinical intake, shimmering skeleton loading, before/after slider, and PDF export. |
| `backend/app/services/cv_pipeline.py` | **Fully Implemented** | Production-ready classical CV micro-feature extractors with Gray-World White Balancing and pure NumPy fallback. |
| `backend/app/services/triage_engine.py` | **Fully Implemented** | Enforces 0.65 confidence gate, multi-factor risk correlation matrix, non-diagnostic phrasing, and sub-score summation. |
| `backend/app/services/pdf_service.py` | **Fully Implemented** | Generates generic 1-page clinical PDF hand-off report using ReportLab. |
| `backend/app/core/constants.py` | **Fully Implemented** | Active ingredient guides, non-diagnostic terms, and mandatory universal disclaimer. |
| `backend/app/models/schemas.py` | **Fully Implemented** | Pydantic data contracts for separate sub-scores, multi-factor symptoms, progress comparison, and consent. |
| `backend/app/services/classifier_service.py` | **Functional / Heuristic Stub** | Ready to load `.onnx` weights from `models/`. Runs an embedded heuristic classifier when local model weights are not present. |
| `training/follica_training_colab.ipynb` | **Fully Implemented** | Cloud training notebook with Google Drive checkpointing to survive disconnects. |
| `training/calibration_loader.py` | **Fully Implemented / Data Placeholder** | Pipeline logic ready; awaiting physical addition of 100-200 volunteer photos. |
| `docs/supabase_storage_security.sql` | **Reference Template** | Example PostgreSQL schema for private storage buckets and automated 30-day retention purge (no application-level encryption at rest; recommends OS-level disk encryption). |
| `mobile/` | **Not Started** | Reserved for React Native / Expo cross-platform mobile frontend. |

---

## 8. Next Steps for Contributors

If you are joining this repository, pick up the following tasks in priority order:

1. **Task 1: Mobile UI (React Native / Expo)** (`mobile/`):
   - Initialize the Expo application.
   - Build the camera capture interface with crosshair/quadrant overlays for Crown, Hairline, and Parting.
   - Build the Consent Screen adhering to the `/api/v1/consent` contract.
   - Build the interactive Scalp Health Score visualization dashboard with discrete sub-scores and before/after slider.
2. **Task 2: Volunteer Calibration Dataset Collection**:
   - Collect and organize 100-200 consented scalp photos across Fitzpatrick skin types into `data/calibration/` and execute training via `follica_training_colab.ipynb`.
3. **Task 3: Push Notification Adherence Reminders**:
   - Add local push notification hooks in the mobile app for wash schedules (e.g. Ketoconazole 2x/week) and 14-day re-scan reminders.

---

## 9. How to Run Locally

### Setup and Start Backend Server
```bash
# 1. Navigate to backend directory
cd backend

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Windows PowerShell:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Start FastAPI on localhost:8000
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Interactive API Documentation
- Swagger UI: `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/api/v1/system/status`

### Run Automated Tests
```bash
pytest tests/ -v
```
"""

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme_text.strip() + '\n')
print('SUCCESS: README.md updated with Need Statement and Market Landscape Plan')