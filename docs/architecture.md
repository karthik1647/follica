# Follica Technical Architecture & Dual-Pipeline Specification

## 1. System Overview
Follica utilizes a **Dual-Pipeline Architecture** that combines:
1. **Deep Learning Classification** (MobileNetV4 / EfficientNet-B0) for condition pattern recognition.
2. **Classical Computer Vision** (OpenCV) for objective micro-feature extraction (Flake area, Erythema chrominance, Specular oil reflection, Texture edge density).

```
                      [Scalp Image & Questionnaire Context]
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             ▼                                                     ▼
 [Pipeline 1: Deep Learning]                          [Pipeline 2: Classical CV]
 MobileNetV4 / EfficientNet-B0                        OpenCV Micro-Feature Analysis
             │                                                     │
             ▼                                                     ▼
 [Condition + Raw Confidence]                         [4 Discrete Sub-Scores (0-25)]
             │                                        - Flake Score
             │                                        - Redness Score
             │                                        - Sebum Score
             │                                        - Density Score
             │                                                     │
             └──────────────────────────┬──────────────────────────┘
                                        ▼
                         [Server-Side Triage Engine]
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             ▼                                                     ▼
 [Confidence < 0.65 Gate]                              [Confidence >= 0.65]
 Enforce Level 2: Monitored Routine                    Determine Severity (Level 1, 2, or 3)
```

---

## 2. Server-Side Confidence Gate
- **Threshold**: `0.65` (65% confidence).
- If the deep learning model confidence is below this threshold, the system **strictly overrides** any Level 1 or Level 3 classification and assigns:
  - `suggested_condition = "Indeterminate Pattern / Requires Verification"`
  - `triage_level = "Level 2: Monitored Routine"`
  - `confidence_gated = true`
- This ensures no critical misdiagnosis or improper escalation occurs from low-quality or ambiguous photos.

---

## 3. Sub-Scores Mathematical Formulation
Composite Scalp Health Score (SHS):
$$\text{SHS} = S_{\text{flake}} + S_{\text{redness}} + S_{\text{sebum}} + S_{\text{density}} \in [0, 100]$$

| Metric | Range | Extraction Method |
| :--- | :--- | :--- |
| **Flake Score ($S_{\text{flake}}$)** | 0.0 – 25.0 | Top-Hat morphological transform + adaptive thresholding for high-contrast particle contour ratio. |
| **Redness Score ($S_{\text{redness}}$)** | 0.0 – 25.0 | CIELAB color space $a^*$ channel deviation from uninflamed skin baseline mask. |
| **Sebum Score ($S_{\text{sebum}}$)** | 0.0 – 25.0 | Specular reflection / luster hotspot detection in HSV Value channel ($V > 230, S < 80$). |
| **Density Score ($S_{\text{density}}$)** | 0.0 – 25.0 | Multi-directional Sobel gradient magnitude and high-frequency edge energy. |

---

## 4. Medical Phrasing & Safety Guidelines
- **Universal Disclaimer**: Appended to every response.
- **Non-Diagnostic Vocabulary**: Uses "suggests", "patterns consistent with", and "recommends an active ingredient schedule".
- **Generic Hand-off**: Addressed to *"For your treating physician / dermatologist"*, with generic local clinic search URLs and no named institutions.
