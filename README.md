# Follica: AI-Powered Scalp Health Diagnostic and Triage System

Follica is an open-source, dual-pipeline scalp health assessment and clinical triage platform. It evaluates user-uploaded scalp photographs and contextual symptom questionnaires to provide objective health scoring, active ingredient care routines for minor issues, and structured clinical hand-off summaries when medical escalation is warranted.

---

## 1. The Problem Being Solved

Hair thinning, dandruff, and scalp inflammation affect a large percentage of college students and young adults, yet early intervention is consistently hindered by three systemic issues:

1. **The Over-The-Counter Guessing Trap**:
   Dry scalp flaking, fungal *Seborrheic Dermatitis*, and *Scalp Psoriasis* all present superficially as "white flakes". When students or general clinic staff treat all flaking with generic commercial shampoos (or random anti-dandruff products), it frequently aggravates the underlying barrier damage or yeast proliferation. Each distinct condition requires specific active molecules (e.g., Ketoconazole 2% vs. Salicylic Acid vs. Barrier Lipids vs. Chelating washes).

2. **Campus and Environmental Aggravators**:
   Hostel borewell hard water (calcium and magnesium buildup), academic stress, irregular sleep, and nutritional changes accelerate scalp barrier disruption. Standard general health centers often lack specialized trichoscopy tools to differentiate benign flaking from active follicular inflammation.

3. **Delayed Escalation for High-Risk Conditions**:
   Conditions such as *Alopecia Areata* (coin-shaped bald patches) or *Folliculitis* (painful follicular pustules) require prompt dermatological evaluation. Because booking a specialist can be intimidating, expensive, or delayed, students often wait until irreversible follicular miniaturization or scarring occurs.

**Follica bridges this gap**: It acts as an objective, non-diagnostic triage layer that recommends science-backed active ingredient schedules for minor self-resolving cases, and immediately generates a standardized clinical hand-off document for dermatologists when red flags appear.

---

## 2. Core Design Principles

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

## 3. System Architecture at a Glance

```
                         [User Upload: Scalp Photo + Questionnaire]
                                             │
                  ┌──────────────────────────┴──────────────────────────┐
                  ▼                                                     ▼
     [Pipeline 1: Deep Learning]                           [Pipeline 2: Classical CV]
     MobileNetV4 / EfficientNet-B0                         OpenCV / NumPy Micro-Features
                  │                                                     │
                  ▼                                                     ▼
     Raw Condition + Confidence (0.00-1.00)                4 Discrete Sub-Scores (0-25):
                  │                                        - Flake & Scaling Score
                  │                                        - Erythema / Redness Score
                  │                                        - Sebum / Lipid Balance Score
                  │                                        - Follicular Density Score
                  │                                                     │
                  └──────────────────────────┬──────────────────────────┘
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

## 4. Current Status: What Is Built vs. Stubbed

To help new contributors navigate the codebase without ambiguity:

| Component / File | Implementation Status | Description & Notes |
| :--- | :--- | :--- |
| `backend/app/api/endpoints.py` | **Fully Implemented** | Exposes `/analyze`, `/consent`, `/system/status`, `/report/pdf`. |
| `backend/app/services/cv_pipeline.py` | **Fully Implemented** | Production-ready classical CV micro-feature extractors with pure NumPy fallback. |
| `backend/app/services/triage_engine.py` | **Fully Implemented** | Enforces 0.65 confidence gate, non-diagnostic phrasing, and sub-score summation. |
| `backend/app/services/pdf_service.py` | **Fully Implemented** | Generates generic 1-page clinical PDF hand-off report using ReportLab. |
| `backend/app/core/constants.py` | **Fully Implemented** | Active ingredient guides, non-diagnostic terms, and mandatory universal disclaimer. |
| `backend/app/models/schemas.py` | **Fully Implemented** | Pydantic data contracts for separate sub-scores, symptoms, and consent. |
| `backend/app/services/classifier_service.py` | **Functional / Heuristic Stub** | Ready to load `.onnx` weights from `models/`. Runs an embedded heuristic classifier when local model weights are not present. |
| `training/follica_training_colab.ipynb` | **Fully Implemented** | Cloud training notebook with Google Drive checkpointing to survive disconnects. |
| `training/calibration_loader.py` | **Fully Implemented / Data Placeholder** | Pipeline logic ready; awaiting physical addition of 100-200 volunteer photos. |
| `docs/supabase_storage_security.sql` | **Fully Implemented** | SQL schema for AES-256 encrypted bucket and automated 30-day retention purge. |
| `mobile/` | **Not Started** | Reserved for React Native / Expo cross-platform mobile frontend. |

---

## 5. Next Steps for Contributors

If you are joining this repository, pick up the following tasks in priority order:

1. **Task 1: Image Quality Pre-Inference Gate** (`backend/app/services/quality_gate.py`):
   - Implement server-side Laplacian variance blur detection and illumination distribution checks before running inference. If an image is severely blurred or under-exposed, return a structured 400 error asking the user to retake the photo with flash.
2. **Task 2: Progress-Tracking Image Registration** (`backend/app/services/registration.py`):
   - Build an image alignment/homography service using ORB/SIFT feature matching to align photos taken 14–30 days apart, enabling exact spot-by-spot follicle density delta calculations.
3. **Task 3: Mobile UI (React Native / Expo)** (`mobile/`):
   - Initialize the Expo app.
   - Build the camera capture interface with crosshair/quadrant overlays for Crown, Hairline, and Parting.
   - Build the Consent Screen adhering to the `/api/v1/consent` contract.
   - Build the interactive Scalp Health Score visualization dashboard.
4. **Task 4: Volunteer Calibration Dataset Collection**:
   - Collect and organize 100–200 consented scalp photos across Fitzpatrick skin types into `data/calibration/` and execute training via `follica_training_colab.ipynb`.
5. **Task 5: Push Notification Adherence Reminders**:
   - Add local push notification hooks in the mobile app for wash schedules (e.g. Ketoconazole 2x/week) and 14-day re-scan reminders.

---

## 6. How to Run Locally

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
pytest backend/tests/test_api.py
```
