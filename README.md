# Follica 🌿
### AI-Powered Hair & Scalp Health Diagnostic & Triage System

**Follica** is a dual-pipeline, non-diagnostic hair and scalp assessment platform built to bridge the gap between neglected scalp disorders, ineffective over-the-counter guessing, and appropriate clinical triage.

---

## 🌟 Key Architecture & Capabilities

1. **Dual-Pipeline Analysis**:
   - **Deep Learning Classifier**: MobileNetV4 (with EfficientNet-B0 fallback) for condition pattern recognition.
   - **Classical Computer Vision**: OpenCV micro-feature extractors returning 4 separate sub-scores:
     - ❄️ **Flake & Scaling Score (0–25)**
     - 🔴 **Erythema / Redness Score (0–25)**
     - 💧 **Sebum & Lipid Balance Score (0–25)**
     - 💇 **Hair Density & Follicle Visibility Score (0–25)**
     - 🏆 **Composite Scalp Health Score (0–100)**

2. **Server-Side Safety & Confidence Gate**:
   - Automatically detects low-confidence classifications (`< 0.65`) and reroutes them to **Level 2 (Monitored Routine)** to avoid improper escalation or self-care errors.
   - Enforces strictly non-diagnostic phrasing across all responses (*"suggests"*, *"recommends an ingredient schedule"*, never *"diagnoses"* or *"prescribes"*).
   - Mandatory universal disclaimer included on every API response.

3. **Generic Clinical Hand-off (No Named Institutions)**:
   - Exports a 1-page printable clinical summary PDF addressed generically to *"For your treating physician / dermatologist"* with a self-triggered local search link.

4. **Zero-Cost Cloud Training Pipeline**:
   - Trains on free Google Colab Nvidia T4 GPUs with per-epoch checkpointing (surviving disconnects) and zero local storage burden on your laptop.

---

## 📁 Repository Structure

```
follica/
├── backend/                  # FastAPI REST API & Dual-Pipeline Engine
│   ├── app/
│   │   ├── api/             # API Endpoints (/analyze, /consent, /system/status, /report/pdf)
│   │   ├── core/            # Config, Constants, Non-Diagnostic Phrasing, Disclaimers
│   │   ├── models/          # Pydantic Schemas (SubScores, SymptomContext, Consent)
│   │   └── services/        # CV Pipeline, ML Classifier, Triage Engine, PDF Service
│   ├── tests/               # Pytest Suite (Sub-scores, Confidence Gate, Consent, PDF)
│   └── requirements.txt
├── training/                 # Cloud ML Training & Calibration Pipeline
│   ├── follica_training_colab.ipynb # Ready-to-run Google Colab Notebook with Drive Mount
│   ├── train.py             # Standalone Python training script with checkpointing
│   ├── calibration_loader.py# Volunteer calibration dataset merge module
│   └── README.md
├── docs/                     # Documentation & Database Schemas
│   ├── architecture.md      # Dual-pipeline architecture specification
│   └── supabase_storage_security.sql # AES-256 encrypted storage & 30-day purge RLS
├── mobile/                   # Mobile app directory (Reserved for Phase 3)
└── README.md
```

---

## 🚀 Running the Backend Locally on `localhost:8000`

### 1. Set Up Virtual Environment & Dependencies
```bash
cd backend
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Start the FastAPI Server
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 3. Access Interactive API Docs & Health Checks
- **Interactive Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Alternative ReDoc UI**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Cold-Start Warmup Hook**: [http://127.0.0.1:8000/api/v1/system/status](http://127.0.0.1:8000/api/v1/system/status)

### 4. Run Automated Test Suite
```bash
pytest tests/
```

---

## 🔮 Next-Phase Implementation TODOs (Explicitly Tracked)

The following components are scheduled for the next phase once the backend and scoring pipeline verification is complete:

- [ ] **Mobile UI (React Native / Expo)**: Build the cross-platform mobile client with camera capture overlays (Crown, Hairline, Parting, Macro close-up) and interactive Scalp Health Score visualization.
- [ ] **Image-Quality Gate**: Pre-inference client-side and server-side checks for blur (Laplacian variance), insufficient lighting, and extreme distance.
- [ ] **Progress-Tracking Image Registration**: Computer vision alignment/homography algorithm to match scalp photos taken 14–30 days apart for exact spot-to-spot follicle density comparison.
- [ ] **Adherence & Routine Reminders**: Push notification scheduling for wash routines (e.g. Ketoconazole 2x/week) and 14-day check-in scans.
- [ ] **Dermatology Expansion (DermaIQ)**: Extend classifier heads to facial acne, eczema, and skin lesions.
