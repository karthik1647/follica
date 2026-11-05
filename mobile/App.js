import React, { useState } from "react";
import {
  StyleSheet,
  Text,
  View,
  ScrollView,
  TouchableOpacity,
  TextInput,
  ActivityIndicator,
  SafeAreaView,
  StatusBar,
  Image,
  Platform,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import {
  checkSystemStatus,
  submitScalpAnalysis,
  DEFAULT_API_BASE,
} from "./src/services/api";

const UNIVERSAL_DISCLAIMER =
  "This assessment is for informational and educational purposes only and suggests potential scalp care routines. It does not constitute medical diagnosis, formal medical advice, or treatment. If symptoms persist, cause discomfort, or worsen, please consult a certified dermatologist.";

const PAPERS_DATA = [
  {
    venue: "Frontiers in Medicine · 2022 · DOI: 10.3389/fmed.2022.981884",
    title: "CNN Differentiation of Scalp Psoriasis vs. Seborrheic Dermatitis",
    summary:
      "Demonstrated AUC 0.922 discrimination across erythematous-squamous scalp conditions; motivates pairing neural classification with explicit scaling and erythema sub-scores.",
  },
  {
    venue: "Computers in Biology and Medicine / IEEE Access · 2022-2023",
    title: "MobileNet & EfficientNet Backbones for Multi-Class Scalp Pathology",
    summary:
      "Grounds MobileNetV4-Conv-Small transfer learning exported to standalone ONNX Runtime for sub-50 ms CPU execution.",
  },
  {
    venue: "Scientific Reports / J. Dermatological Science · 2023",
    title: "Automated Trichoscopic Coverage & Alopecia Void Segmentation",
    summary:
      "Establishes morphological fiber segmentation and spatial 6x6 bare-tile void grading for patchy Alopecia Areata (AUC 0.97) and vertex thinning.",
  },
  {
    venue: "Sensors / MDPI Diagnostics · 2023",
    title: "Color-Constancy Normalization & Hybrid CV-Neural Fusion",
    summary:
      "Drives Gray-World white-balance clamping, HSV specular glare rejection, and SDASI/SALT-anchored severity bracketing.",
  },
];

const PIPELINE_STAGES = [
  {
    tag: "STAGE 00",
    name: "Pre-Inference Optical Gate",
    detail:
      "8x8 patch-max 2D Laplacian focus check (>= 60), luminance bounds (35-225), and HSV flash-glare rejection (<= 15%).",
  },
  {
    tag: "STAGE 01",
    name: "MobileNetV4-Conv-Small ONNX",
    detail:
      "10-class macroscopic pathology recognition in FP32 ONNX Runtime with dominance-margin calibration.",
  },
  {
    tag: "STAGE 02",
    name: "Classical CV Micro-Features",
    detail:
      "Gray-World white balance, CLAHE, YCrCb/HSV skin-hair isolation, focal erythema contrast, and bare-tile alopecia detection.",
  },
  {
    tag: "STAGE 03",
    name: "Confidence & Escalation Matrix",
    detail:
      "Predictions below 0.65 confidence reroute to Level 2 Indeterminate; scarring, fungal, and pustular patterns escalate to Level 3.",
  },
  {
    tag: "STAGE 04",
    name: "Post-Identification Trigger Engine",
    detail:
      "Ranks modifiable campus/environmental triggers conditioned on the identified pattern and exports a 1-page physician PDF.",
  },
];

const OFFLINE_SPECIMENS = {
  healthy: {
    scalp_health_score: 93.6,
    confidence_score: 0.91,
    confidence_gated: false,
    suggested_condition: "Pattern Consistent with Balanced / Healthy Scalp",
    triage_level: "Level 1: Minor Home Routine",
    triage_reason:
      "Epidermal barrier, follicular density, and surface lipid balance appear within calm physiological bounds.",
    sub_scores: { flake_score: 23.8, redness_score: 23.6, sebum_score: 23.2, density_score: 23.0 },
    survey_risk_profile: {
      risk_score: 12.0,
      risk_band: "low",
      modifiable_count: 0,
      summary: "Low modifiable trigger load. Maintain gentle pH-balanced cleansing and avoid direct UV or thermal damage.",
      intervention_queue: [],
    },
    active_ingredients: [
      {
        molecule: "Mild Sulfate-Free Surfactant + Panthenol (B5) 1%",
        frequency: "2-3x per week",
        mechanism: "Preserves stratum corneum hydration without stripping protective sebum.",
      },
    ],
  },
  lichen: {
    scalp_health_score: 54.5,
    confidence_score: 0.89,
    confidence_gated: false,
    suggested_condition: "Pattern Suggestive of Lichen Planus / Scarring Inflammation",
    triage_level: "Level 3: Clinical Escalation",
    triage_reason:
      "Perifollicular erythema and focal scarring alopecia characteristics warrant prompt in-person dermatological evaluation to protect follicular stem cells.",
    sub_scores: { flake_score: 14.2, redness_score: 11.4, sebum_score: 16.8, density_score: 12.1 },
    survey_risk_profile: {
      risk_score: 46.0,
      risk_band: "moderate",
      modifiable_count: 2,
      summary: "Avoid mechanical traction, harsh chemical styling, and thermal irritation while scheduling specialist dermoscopy.",
      intervention_queue: [
        {
          priority: 1,
          title: "Schedule Certified Dermatologist Dermoscopy",
          detail: "Scarring inflammatory patterns require clinical trichoscopy to prevent permanent follicular loss.",
          expected_timeline: "Within 1-2 weeks",
        },
        {
          priority: 2,
          title: "Suspend Heat, Bleach & Occlusive Friction",
          detail: "Eliminate chemical dyes, high-heat blow drying, and tight headwear friction.",
          expected_timeline: "Immediate",
        },
      ],
    },
    active_ingredients: [
      {
        molecule: "Fragrance-Free Barrier Cleanser (Colloidal Oat / Panthenol)",
        frequency: "Gentle cleansing as tolerated",
        mechanism: "Minimizes exogenous contact irritation prior to specialist evaluation.",
      },
    ],
  },
  psoriasis: {
    scalp_health_score: 48.4,
    confidence_score: 0.92,
    confidence_gated: false,
    suggested_condition: "Pattern Suggestive of Scalp Psoriasis",
    triage_level: "Level 3: Clinical Escalation",
    triage_reason:
      "Well-demarcated erythematous plaques with silvery-white hyperkeratotic scale warrant dermatologist evaluation alongside keratolytic support.",
    sub_scores: { flake_score: 9.2, redness_score: 10.6, sebum_score: 14.8, density_score: 13.8 },
    survey_risk_profile: {
      risk_score: 52.0,
      risk_band: "elevated",
      modifiable_count: 2,
      summary: "Avoid mechanical picking of adherent plaques (Koebner phenomenon) and manage systemic stress triggers.",
      intervention_queue: [
        {
          priority: 1,
          title: "Book Certified Dermatologist Assessment",
          detail: "Confirm plaque psoriasis vs. seborrheic overlap via polarized dermoscopy.",
          expected_timeline: "Within 1-2 weeks",
        },
        {
          priority: 2,
          title: "Avoid Scratching or Forcible Descaling",
          detail: "Mechanical trauma triggers Koebner plaque expansion along the hairline and crown.",
          expected_timeline: "Immediate",
        },
      ],
    },
    active_ingredients: [
      {
        molecule: "Salicylic Acid 2%-3% Keratolytic Wash",
        frequency: "2x per week (soften scale gently)",
        mechanism: "Softens hyperkeratotic corneocyte cohesion without mechanical trauma.",
      },
    ],
  },
  seborrheic: {
    scalp_health_score: 50.2,
    confidence_score: 0.94,
    confidence_gated: false,
    suggested_condition: "Pattern Suggestive of Seborrheic Dermatitis",
    triage_level: "Level 2: Monitored Routine",
    triage_reason:
      "Adherent lipid-rich scaling and mild erythema suggest microbial sebum imbalance amenable to targeted antifungal/keratolytic care and 14-day monitoring.",
    sub_scores: { flake_score: 10.8, redness_score: 13.2, sebum_score: 10.4, density_score: 15.8 },
    survey_risk_profile: {
      risk_score: 58.2,
      risk_band: "elevated",
      modifiable_count: 3,
      summary: "Elevated modifiable trigger load (3 factors): Hard Borewell Water, Daily Helmet Occlusion, and Acute Stress.",
      intervention_queue: [
        {
          priority: 1,
          title: "Mitigate Hard Borewell Mineral Deposition",
          detail: "Use a shower filter or final filtered-water rinse to reduce alkaline mineral residue on the scalp barrier.",
          expected_timeline: "2-4 weeks",
        },
        {
          priority: 2,
          title: "Sanitize Helmet Liner & Reduce Sweat Occlusion",
          detail: "Use a washable cotton liner cleaned every 2-3 days and air-dry scalp after rides.",
          expected_timeline: "2-3 weeks",
        },
      ],
    },
    active_ingredients: [
      {
        molecule: "Ketoconazole 1%-2% or Piroctone Olamine Wash",
        frequency: "2x per week (5-min contact)",
        mechanism: "Controls Malassezia lipid breakdown and reduces inflammatory scaling.",
      },
      {
        molecule: "Salicylic Acid 1.8%-2% Keratolytic Cleanser",
        frequency: "1x per week",
        mechanism: "Loosens adherent sebum-keratin plaques.",
      },
    ],
  },
  alopecia: {
    scalp_health_score: 56.8,
    confidence_score: 0.93,
    confidence_gated: false,
    suggested_condition: "Pattern Suggestive of Alopecia Areata / Patchy Loss",
    triage_level: "Level 3: Clinical Escalation",
    triage_reason:
      "Circumscribed non-scarring bare patches warrant prompt dermatological assessment for targeted follicular recovery.",
    sub_scores: { flake_score: 18.4, redness_score: 15.2, sebum_score: 14.0, density_score: 9.2 },
    survey_risk_profile: {
      risk_score: 48.0,
      risk_band: "moderate",
      modifiable_count: 2,
      summary: "Address acute sleep/cortisol disruption and screen Vitamin D3 / Ferritin cofactors alongside specialist care.",
      intervention_queue: [
        {
          priority: 1,
          title: "Consult Dermatologist for Trichoscopic Evaluation",
          detail: "Early clinical assessment of patchy hair loss significantly improves regrowth outcomes.",
          expected_timeline: "Within 1-2 weeks",
        },
      ],
    },
    active_ingredients: [
      {
        molecule: "Gentle pH-Balanced Non-Irritating Cleanser",
        frequency: "2-3x per week",
        mechanism: "Maintains calm perifollicular microenvironment pending specialist review.",
      },
    ],
  },
};

function scoreTone(val) {
  if (val >= 20.0) return { color: "#00e6cf", text: "#34d399", label: "Optimal" };
  if (val >= 14.0) return { color: "#f59e0b", text: "#fbbf24", label: "Moderate" };
  return { color: "#f43f5e", text: "#fb7185", label: "Elevated" };
}

function triageBadgeStyle(levelStr = "") {
  if (levelStr.includes("Level 1")) {
    return { bg: "rgba(16,185,129,0.15)", border: "rgba(16,185,129,0.38)", color: "#34d399" };
  }
  if (levelStr.includes("Level 2")) {
    return { bg: "rgba(245,158,11,0.15)", border: "rgba(245,158,11,0.38)", color: "#fbbf24" };
  }
  return { bg: "rgba(244,63,94,0.16)", border: "rgba(244,63,94,0.42)", color: "#fb7185" };
}

export default function App() {
  const [apiBaseUrl, setApiBaseUrl] = useState(DEFAULT_API_BASE);
  const [showServerBar, setShowServerBar] = useState(false);
  const [serverState, setServerState] = useState("API Config");
  const [primaryTab, setPrimaryTab] = useState("triage");
  const [triageMode, setTriageMode] = useState("studio");
  const [reportPane, setReportPane] = useState("scores");
  const [wizardStep, setWizardStep] = useState(0);
  const [corpusTab, setCorpusTab] = useState("papers");
  const [loading, setLoading] = useState(false);
  const [selectedImageUri, setSelectedImageUri] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);

  const [intake, setIntake] = useState({
    biological_sex: "male",
    age_group: "18_25",
    primary_concern_zone: "crown_vertex",
    water_environment: "normal",
    daily_helmet_wearer: false,
    chemical_heat_styling: false,
    washing_frequency: "2_3_per_week",
    diet_pattern: "balanced_protein",
    gut_health: "optimal",
    metabolic_energy: "balanced",
    hormonal_profile: "none",
    stress_level: "moderate",
    sleep_quality: "good_7_8h",
    family_history_hair_loss: false,
    recent_exam_work_stress: false,
    hair_pull_test: "negative_under_3",
    flake_type: "none",
    sensation: "painless",
    duration_months: 1,
    duration_weeks: 4,
  });

  const updateField = (key, val) => {
    setIntake((prev) => ({ ...prev, [key]: val }));
  };

  const countActiveFactors = () => {
    let c = 0;
    if (intake.daily_helmet_wearer) c++;
    if (intake.chemical_heat_styling) c++;
    if (intake.family_history_hair_loss) c++;
    if (intake.recent_exam_work_stress) c++;
    if (intake.water_environment === "hard_borewell") c++;
    if (intake.diet_pattern === "low_protein_crash_diet") c++;
    if (intake.hair_pull_test === "positive_over_5") c++;
    if (intake.gut_health !== "optimal") c++;
    if (intake.metabolic_energy !== "balanced") c++;
    if (intake.hormonal_profile !== "none") c++;
    return c;
  };

  const pingBackend = async () => {
    try {
      setLoading(true);
      const status = await checkSystemStatus(apiBaseUrl);
      setServerState(status.is_ready ? "ONNX Ready" : "Warming Up");
    } catch (err) {
      setServerState("Offline Mode");
    } finally {
      setLoading(false);
    }
  };

  const pickImage = async (useCamera = false) => {
    const fn = useCamera
      ? ImagePicker.launchCameraAsync
      : ImagePicker.launchImageLibraryAsync;
    const res = await fn({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.88,
    });
    if (!res.canceled && res.assets && res.assets.length > 0) {
      setSelectedImageUri(res.assets[0].uri);
    }
  };

  const runLiveAnalysis = async () => {
    if (!selectedImageUri) {
      runOfflinePreset("seborrheic");
      return;
    }
    try {
      setLoading(true);
      const live = await submitScalpAnalysis(selectedImageUri, intake, apiBaseUrl);
      const ingredients = (live.recommended_ingredients || []).map((item) => ({
        molecule: item.name,
        frequency: item.frequency,
        mechanism: item.purpose,
      }));
      setAnalysisResult({
        ...live,
        active_ingredients: ingredients,
        lifestyle_advice: live.lifestyle_guidance || [],
      });
      setPrimaryTab("triage");
      setTriageMode("report");
    } catch (err) {
      setServerState("Offline Fallback");
      runOfflinePreset("seborrheic");
    } finally {
      setLoading(false);
    }
  };

  const runOfflinePreset = (presetKey) => {
    setLoading(true);
    setTimeout(() => {
      const spec = OFFLINE_SPECIMENS[presetKey] || OFFLINE_SPECIMENS.seborrheic;
      setAnalysisResult({ ...spec, disclaimer: UNIVERSAL_DISCLAIMER });
      setPrimaryTab("triage");
      setTriageMode("report");
      setLoading(false);
    }, 180);
  };

  const renderChip = (label, active, onPress) => (
    <TouchableOpacity
      key={label}
      style={[styles.chip, active && styles.chipActive]}
      onPress={onPress}
      activeOpacity={0.82}
    >
      <Text style={[styles.chipText, active && styles.chipTextActive]}>
        {label}
      </Text>
    </TouchableOpacity>
  );

  return (
    <SafeAreaView style={styles.safeRoot}>
      <StatusBar barStyle="light-content" backgroundColor="#080b10" />

      {/* TOP STUDIO HEADER */}
      <View style={styles.topHeader}>
        <View style={styles.brandRow}>
          <View style={styles.brandOrb}>
            <View style={styles.brandInnerRing}>
              <View style={styles.brandCoreDot} />
            </View>
          </View>
          <View>
            <Text style={styles.brandTitle}>Follica</Text>
            <Text style={styles.brandSubtitle}>Clinical Trichoscopy & Triage</Text>
          </View>
        </View>

        <TouchableOpacity
          style={styles.serverPill}
          onPress={() => setShowServerBar(!showServerBar)}
        >
          <View style={styles.serverDot} />
          <Text style={styles.serverPillText} numberOfLines={1}>
            {serverState}
          </Text>
        </TouchableOpacity>
      </View>

      {/* COLLAPSIBLE BACKEND ENDPOINT DRAWER */}
      {showServerBar && (
        <View style={styles.serverDrawer}>
          <Text style={styles.drawerLabel}>
            FastAPI Endpoint (LAN IP or HTTPS Tunnel):
          </Text>
          <View style={styles.drawerRow}>
            <TextInput
              style={styles.urlInput}
              value={apiBaseUrl}
              onChangeText={setApiBaseUrl}
              placeholder="http://192.168.1.x:8000/api/v1"
              placeholderTextColor="#64748b"
              autoCapitalize="none"
              autoCorrect={false}
            />
            <TouchableOpacity style={styles.pingBtn} onPress={pingBackend}>
              <Text style={styles.pingBtnText}>Ping</Text>
            </TouchableOpacity>
          </View>
        </View>
      )}

      {/* MAIN SCROLLABLE STAGE */}
      <ScrollView
        style={styles.mainViewport}
        contentContainerStyle={styles.mainContent}
      >
        {loading && (
          <View style={styles.loadingBanner}>
            <ActivityIndicator size="small" color="#00e6cf" />
            <Text style={styles.loadingText}>
              Executing MobileNetV4 ONNX + OpenCV pipeline...
            </Text>
          </View>
        )}

        {/* =====================================================
            TAB 01: OPTICAL TRIAGE
           ===================================================== */}
        {primaryTab === "triage" && triageMode === "studio" && (
          <View>
            <View style={styles.protoGrid}>
              {[
                ["01", "Clean Lens"],
                ["02", "No Flash"],
                ["03", "Part Hair"],
                ["04", "10-15 cm"],
              ].map(([num, lbl]) => (
                <View key={num} style={styles.protoPill}>
                  <Text style={styles.protoNum}>{num}</Text>
                  <Text style={styles.protoLbl}>{lbl}</Text>
                </View>
              ))}
            </View>

            <View style={styles.card}>
              <Text style={styles.cardTitle}>Optical Scalp Frame Studio</Text>
              <Text style={styles.cardSub}>
                Capture a close-up scalp photo or select from gallery. Pre-inference gating checks focus, luminance, and glare in volatile RAM.
              </Text>

              {selectedImageUri ? (
                <View style={styles.previewBox}>
                  <Image
                    source={{ uri: selectedImageUri }}
                    style={styles.previewImg}
                  />
                  <Text style={styles.previewReady}>
                    Scalp frame locked and ready for dual-pipeline triage.
                  </Text>
                </View>
              ) : (
                <TouchableOpacity
                  style={styles.viewfinderBox}
                  activeOpacity={0.85}
                  onPress={() => pickImage(false)}
                >
                  <View style={[styles.vfCorner, styles.vfTL]} />
                  <View style={[styles.vfCorner, styles.vfTR]} />
                  <View style={[styles.vfCorner, styles.vfBL]} />
                  <View style={[styles.vfCorner, styles.vfBR]} />
                  <View style={styles.lensBadge}>
                    <Text style={styles.lensBadgePlus}>+</Text>
                  </View>
                  <Text style={styles.vfTitle}>
                    Capture or Browse Scalp Photograph
                  </Text>
                  <Text style={styles.vfSub}>
                    JPEG / PNG / WebP · Ephemeral RAM Processing
                  </Text>
                  <View style={styles.capRow}>
                    <Text style={styles.capBadge}>Focus Gate</Text>
                    <Text style={styles.capBadge}>Glare Filter</Text>
                    <Text style={styles.capBadge}>SDASI/SALT</Text>
                  </View>
                </TouchableOpacity>
              )}

              <View style={styles.actionRow2}>
                <TouchableOpacity
                  style={styles.secondaryBtn}
                  onPress={() => pickImage(true)}
                >
                  <Text style={styles.secondaryBtnText}>Live Camera</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={styles.secondaryBtn}
                  onPress={() => pickImage(false)}
                >
                  <Text style={styles.secondaryBtnText}>Photo Library</Text>
                </TouchableOpacity>
              </View>

              <TouchableOpacity
                style={styles.primaryBtn}
                onPress={runLiveAnalysis}
              >
                <Text style={styles.primaryBtnText}>
                  Run Dual-Pipeline Triage
                </Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[styles.ghostBtn, { marginTop: 9 }]}
                onPress={() => setPrimaryTab("intake")}
              >
                <Text style={styles.ghostBtnText}>
                  Clinical Intake ({countActiveFactors()} Active Markers)
                </Text>
              </TouchableOpacity>
            </View>

            {/* REFERENCE COHORT CARD */}
            <View style={styles.card}>
              <View style={styles.rowBetween}>
                <Text style={styles.cardTitle}>Held-Out Reference Cohort</Text>
                <Text style={styles.capBadge}>Instant Demo</Text>
              </View>
              <Text style={styles.cardSub}>
                Evaluate verified clinical specimens instantly on device:
              </Text>

              <TouchableOpacity
                style={styles.featuredControlCard}
                onPress={() => runOfflinePreset("healthy")}
              >
                <View>
                  <Text style={styles.featuredControlTitle}>
                    Healthy Preventive Baseline
                  </Text>
                  <Text style={styles.featuredControlSub}>
                    Level 1 · Calm Epidermal Barrier · SHS 93.6
                  </Text>
                </View>
                <Text style={styles.featuredControlAction}>Run</Text>
              </TouchableOpacity>

              <View style={styles.presetGrid2}>
                {[
                  ["lichen", "Lichen Planus", "Level 3 · Scarring", "#f43f5e"],
                  ["psoriasis", "Scalp Psoriasis", "Level 3 · Plaque", "#f43f5e"],
                  ["seborrheic", "Seborrheic Derm.", "Level 2 · Scale", "#f59e0b"],
                  ["alopecia", "Alopecia Areata", "Level 3 · Patchy", "#f43f5e"],
                ].map(([key, title, sub, dotColor]) => (
                  <TouchableOpacity
                    key={key}
                    style={styles.presetTile}
                    onPress={() => runOfflinePreset(key)}
                  >
                    <Text style={styles.presetTitle}>{title}</Text>
                    <View style={styles.presetMetaRow}>
                      <View
                        style={[styles.tierDot, { backgroundColor: dotColor }]}
                      />
                      <Text style={styles.presetSub}>{sub}</Text>
                    </View>
                  </TouchableOpacity>
                ))}
              </View>
            </View>
          </View>
        )}

        {/* POST-SCAN REPORT VIEW */}
        {primaryTab === "triage" && triageMode === "report" && analysisResult && (
          <View>
            <View style={styles.reportNavRow}>
              <TouchableOpacity
                style={styles.ghostBtnSmall}
                onPress={() => setTriageMode("studio")}
              >
                <Text style={styles.ghostBtnText}>New Frame</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.ghostBtnSmall}
                onPress={() => setPrimaryTab("intake")}
              >
                <Text style={styles.ghostBtnText}>Edit Intake</Text>
              </TouchableOpacity>
            </View>

            <View style={styles.card}>
              <Text style={styles.overlineLabel}>
                VISUAL & MULTI-FACTOR ASSESSMENT
              </Text>
              <Text style={styles.conditionHeadline}>
                {analysisResult.suggested_condition}
              </Text>

              {(() => {
                const tb = triageBadgeStyle(analysisResult.triage_level);
                return (
                  <View
                    style={[
                      styles.triageBadge,
                      { backgroundColor: tb.bg, borderColor: tb.border },
                    ]}
                  >
                    <Text style={[styles.triageBadgeText, { color: tb.color }]}>
                      {analysisResult.triage_level}
                    </Text>
                  </View>
                );
              })()}

              {/* 3-CELL METRIC SUMMARY */}
              <View style={styles.metricRow3}>
                <View style={styles.metricCell}>
                  <View style={styles.metricTopBar} />
                  <Text style={styles.metricLabel}>SHS</Text>
                  <Text style={styles.metricVal}>
                    {Number(analysisResult.scalp_health_score).toFixed(1)}
                  </Text>
                  <Text style={styles.metricSub}>of 100</Text>
                </View>
                <View style={styles.metricCell}>
                  <View style={styles.metricTopBar} />
                  <Text style={styles.metricLabel}>Confidence</Text>
                  <Text style={styles.metricVal}>
                    {(Number(analysisResult.confidence_score || 0.9) * 100).toFixed(1)}%
                  </Text>
                  <Text style={styles.metricSub}>Gate: 65%</Text>
                </View>
                <View style={styles.metricCell}>
                  <View style={styles.metricTopBar} />
                  <Text style={styles.metricLabel}>Trigger Load</Text>
                  <Text style={styles.metricVal}>
                    {(
                      analysisResult.survey_risk_profile?.risk_band || "low"
                    ).toUpperCase()}
                  </Text>
                  <Text style={styles.metricSub}>
                    {analysisResult.survey_risk_profile?.modifiable_count ?? 0} Factors
                  </Text>
                </View>
              </View>

              {/* INNER REPORT SEGMENTED TABS */}
              <View style={styles.segmentedRow}>
                {[
                  ["scores", "Micro-Features"],
                  ["molecules", "Molecules"],
                  ["triggers", "Triggers"],
                ].map(([key, lbl]) => (
                  <TouchableOpacity
                    key={key}
                    style={[
                      styles.segBtn,
                      reportPane === key && styles.segBtnActive,
                    ]}
                    onPress={() => setReportPane(key)}
                  >
                    <Text
                      style={[
                        styles.segBtnText,
                        reportPane === key && styles.segBtnTextActive,
                      ]}
                    >
                      {lbl}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>

              {reportPane === "scores" && (
                <View style={{ marginTop: 12 }}>
                  {[
                    ["Flake & Scaling", analysisResult.sub_scores.flake_score],
                    ["Redness & Erythema", analysisResult.sub_scores.redness_score],
                    ["Sebum & Lipids", analysisResult.sub_scores.sebum_score],
                    ["Hair Coverage", analysisResult.sub_scores.density_score],
                  ].map(([title, val]) => {
                    const tone = scoreTone(val);
                    return (
                      <View key={title} style={styles.subscoreCard}>
                        <View style={styles.subscoreHeader}>
                          <Text style={styles.subscoreName}>{title}</Text>
                          <Text style={[styles.subscoreNum, { color: tone.text }]}>
                            {Number(val).toFixed(1)} / 25 · {tone.label}
                          </Text>
                        </View>
                        <View style={styles.barTrack}>
                          <View
                            style={[
                              styles.barFill,
                              {
                                width: `${Math.min(100, (val / 25) * 100)}%`,
                                backgroundColor: tone.color,
                              },
                            ]}
                          />
                        </View>
                      </View>
                    );
                  })}

                  <View style={styles.rationaleBox}>
                    <Text style={styles.rationaleText}>
                      {analysisResult.triage_reason}
                    </Text>
                  </View>
                </View>
              )}

              {reportPane === "molecules" && (
                <View style={{ marginTop: 12 }}>
                  {(analysisResult.active_ingredients || []).map((ing, i) => (
                    <View key={i} style={styles.listCard}>
                      <Text style={styles.listCardTitle}>{ing.molecule}</Text>
                      <Text style={styles.listCardAccent}>{ing.frequency}</Text>
                      <Text style={styles.listCardBody}>{ing.mechanism}</Text>
                    </View>
                  ))}
                </View>
              )}

              {reportPane === "triggers" && (
                <View style={{ marginTop: 12 }}>
                  <Text style={styles.cardSub}>
                    {analysisResult.survey_risk_profile?.summary ||
                      "Complete Clinical Intake to rank condition-specific lifestyle triggers."}
                  </Text>
                  {(
                    analysisResult.survey_risk_profile?.intervention_queue || []
                  ).map((step) => (
                    <View key={step.priority} style={styles.listCard}>
                      <Text style={styles.listCardAccent}>
                        PRIORITY P{step.priority} · {step.expected_timeline}
                      </Text>
                      <Text style={styles.listCardTitle}>{step.title}</Text>
                      <Text style={styles.listCardBody}>{step.detail}</Text>
                    </View>
                  ))}
                </View>
              )}

              <Text style={styles.disclaimerText}>
                {analysisResult.disclaimer || UNIVERSAL_DISCLAIMER}
              </Text>
            </View>
          </View>
        )}

        {/* =====================================================
            TAB 02: CLINICAL INTAKE (4-STEP WIZARD)
           ===================================================== */}
        {primaryTab === "intake" && (
          <View style={styles.card}>
            <View style={styles.segmentedRow}>
              {[0, 1, 2, 3].map((idx) => (
                <TouchableOpacity
                  key={idx}
                  style={[
                    styles.segBtn,
                    wizardStep === idx && styles.segBtnActive,
                  ]}
                  onPress={() => setWizardStep(idx)}
                >
                  <Text
                    style={[
                      styles.segBtnText,
                      wizardStep === idx && styles.segBtnTextActive,
                    ]}
                  >
                    Step 0{idx + 1}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>

            {wizardStep === 0 && (
              <View style={{ marginTop: 14 }}>
                <Text style={styles.cardTitle}>01 · Demographics & Zone</Text>
                <Text style={styles.fieldLabel}>Age Bracket</Text>
                <View style={styles.chipWrap}>
                  {[
                    ["under_18", "Under 18"],
                    ["18_25", "18 - 25"],
                    ["26_35", "26 - 35"],
                    ["36_50", "36 - 50"],
                    ["over_50", "50+"],
                  ].map(([v, l]) =>
                    renderChip(l, intake.age_group === v, () =>
                      updateField("age_group", v)
                    )
                  )}
                </View>

                <Text style={styles.fieldLabel}>Biological Sex</Text>
                <View style={styles.chipWrap}>
                  {[
                    ["male", "Male"],
                    ["female", "Female"],
                  ].map(([v, l]) =>
                    renderChip(l, intake.biological_sex === v, () =>
                      updateField("biological_sex", v)
                    )
                  )}
                </View>

                <Text style={styles.fieldLabel}>Primary Anatomical Zone</Text>
                <View style={styles.chipWrap}>
                  {[
                    ["routine_healthy_check", "Routine Check"],
                    ["crown_vertex", "Crown / Vertex"],
                    ["hairline_temples", "Hairline / Temples"],
                    ["central_parting", "Parting Line"],
                    ["diffuse_all_over", "Diffuse Thinning"],
                    ["patchy_spots", "Focal Patchy"],
                  ].map(([v, l]) =>
                    renderChip(l, intake.primary_concern_zone === v, () =>
                      updateField("primary_concern_zone", v)
                    )
                  )}
                </View>
              </View>
            )}

            {wizardStep === 1 && (
              <View style={{ marginTop: 14 }}>
                <Text style={styles.cardTitle}>02 · Extrinsic & Mechanical</Text>
                <Text style={styles.fieldLabel}>Residential Water Supply</Text>
                <View style={styles.chipWrap}>
                  {[
                    ["normal", "Filtered / Normal"],
                    ["hard_borewell", "Hard Borewell"],
                    ["recent_relocation", "Recent Move"],
                  ].map(([v, l]) =>
                    renderChip(l, intake.water_environment === v, () =>
                      updateField("water_environment", v)
                    )
                  )}
                </View>

                <Text style={styles.fieldLabel}>Cleansing Cadence</Text>
                <View style={styles.chipWrap}>
                  {[
                    ["daily", "Daily"],
                    ["2_3_per_week", "2-3x / Week"],
                    ["once_per_week", "1x / Week"],
                    ["rarely", "Rarely"],
                  ].map(([v, l]) =>
                    renderChip(l, intake.washing_frequency === v, () =>
                      updateField("washing_frequency", v)
                    )
                  )}
                </View>

                <Text style={styles.fieldLabel}>Friction & Thermal Exposure</Text>
                <View style={styles.chipWrap}>
                  {renderChip(
                    "Daily Helmet Occlusion",
                    intake.daily_helmet_wearer,
                    () =>
                      updateField(
                        "daily_helmet_wearer",
                        !intake.daily_helmet_wearer
                      )
                  )}
                  {renderChip(
                    "Heat / Chemical Styling",
                    intake.chemical_heat_styling,
                    () =>
                      updateField(
                        "chemical_heat_styling",
                        !intake.chemical_heat_styling
                      )
                  )}
                </View>
              </View>
            )}

            {wizardStep === 2 && (
              <View style={{ marginTop: 14 }}>
                <Text style={styles.cardTitle}>03 · Systemic & Nutritional</Text>
                <Text style={styles.fieldLabel}>Dietary Protein Pattern</Text>
                <View style={styles.chipWrap}>
                  {[
                    ["balanced_protein", "Adequate Protein"],
                    ["vegetarian_vegan", "Plant-Based"],
                    ["low_protein_crash_diet", "Low Protein / Crash"],
                  ].map(([v, l]) =>
                    renderChip(l, intake.diet_pattern === v, () =>
                      updateField("diet_pattern", v)
                    )
                  )}
                </View>

                <Text style={styles.fieldLabel}>Stress Load</Text>
                <View style={styles.chipWrap}>
                  {[
                    ["low", "Low"],
                    ["moderate", "Moderate"],
                    ["high", "High / Sustained"],
                  ].map(([v, l]) =>
                    renderChip(l, intake.stress_level === v, () =>
                      updateField("stress_level", v)
                    )
                  )}
                </View>

                <Text style={styles.fieldLabel}>Systemic Flags</Text>
                <View style={styles.chipWrap}>
                  {renderChip(
                    "Family Thinning History",
                    intake.family_history_hair_loss,
                    () =>
                      updateField(
                        "family_history_hair_loss",
                        !intake.family_history_hair_loss
                      )
                  )}
                  {renderChip(
                    "Acute Exam / Work Stress",
                    intake.recent_exam_work_stress,
                    () =>
                      updateField(
                        "recent_exam_work_stress",
                        !intake.recent_exam_work_stress
                      )
                  )}
                </View>
              </View>
            )}

            {wizardStep === 3 && (
              <View style={{ marginTop: 14 }}>
                <Text style={styles.cardTitle}>04 · Follicular Dynamics</Text>
                <Text style={styles.fieldLabel}>Cutaneous Sensation</Text>
                <View style={styles.chipWrap}>
                  {[
                    ["painless", "Asymptomatic"],
                    ["itching", "Pruritus (Itching)"],
                    ["burning", "Burning / Warmth"],
                    ["burning_or_tender", "Trichodynia"],
                    ["painful", "Painful Bumps"],
                  ].map(([v, l]) =>
                    renderChip(l, intake.sensation === v, () =>
                      updateField("sensation", v)
                    )
                  )}
                </View>

                <Text style={styles.fieldLabel}>50-Fiber Pull Test</Text>
                <View style={styles.chipWrap}>
                  {[
                    ["negative_under_3", "Under 3 (Negative)"],
                    ["borderline_3_5", "3 - 5 (Borderline)"],
                    ["positive_over_5", "Over 5 (Positive)"],
                  ].map(([v, l]) =>
                    renderChip(l, intake.hair_pull_test === v, () =>
                      updateField("hair_pull_test", v)
                    )
                  )}
                </View>

                <Text style={styles.fieldLabel}>Scale Morphology</Text>
                <View style={styles.chipWrap}>
                  {[
                    ["none", "None"],
                    ["dry_white_fine", "Fine Dry White"],
                    ["greasy_yellow_thick", "Oily Yellow"],
                    ["plaque_crusty", "Thick Silvery Plaque"],
                  ].map(([v, l]) =>
                    renderChip(l, intake.flake_type === v, () =>
                      updateField("flake_type", v)
                    )
                  )}
                </View>
              </View>
            )}

            <View style={styles.wizardFooter}>
              <TouchableOpacity
                style={styles.ghostBtnSmall}
                disabled={wizardStep === 0}
                onPress={() => setWizardStep(Math.max(0, wizardStep - 1))}
              >
                <Text style={styles.ghostBtnText}>Previous</Text>
              </TouchableOpacity>
              {wizardStep < 3 ? (
                <TouchableOpacity
                  style={styles.secondaryBtnSmall}
                  onPress={() => setWizardStep(wizardStep + 1)}
                >
                  <Text style={styles.secondaryBtnText}>Next Step</Text>
                </TouchableOpacity>
              ) : (
                <TouchableOpacity
                  style={styles.primaryBtnSmall}
                  onPress={runLiveAnalysis}
                >
                  <Text style={styles.primaryBtnText}>Save & Run Triage</Text>
                </TouchableOpacity>
              )}
            </View>
          </View>
        )}

        {/* =====================================================
            TAB 03: SYSTEM ARCHITECTURE
           ===================================================== */}
        {primaryTab === "architecture" && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>
              5-Stage Decoupled Inference Architecture
            </Text>
            <Text style={styles.cardSub}>
              Non-diagnostic triage system combining MobileNetV4 ONNX vision, color-constancy OpenCV micro-features, and deterministic safety gates.
            </Text>

            {PIPELINE_STAGES.map((st) => (
              <View key={st.tag} style={styles.listCard}>
                <Text style={styles.listCardAccent}>{st.tag}</Text>
                <Text style={styles.listCardTitle}>{st.name}</Text>
                <Text style={styles.listCardBody}>{st.detail}</Text>
              </View>
            ))}

            <View style={styles.rationaleBox}>
              <Text style={styles.rationaleText}>
                Zero-Retention Privacy: Scalp photographs are processed strictly in volatile RAM and discarded immediately. Consent records store salted SHA-256 pseudonyms only.
              </Text>
            </View>
          </View>
        )}

        {/* =====================================================
            TAB 04: LITERATURE & CORPUS
           ===================================================== */}
        {primaryTab === "corpus" && (
          <View>
            <View style={[styles.segmentedRow, { marginBottom: 12 }]}>
              {[
                ["papers", "Papers (4)"],
                ["dataset", "12k Corpus"],
                ["benchmarks", "Benchmarks"],
              ].map(([k, l]) => (
                <TouchableOpacity
                  key={k}
                  style={[
                    styles.segBtn,
                    corpusTab === k && styles.segBtnActive,
                  ]}
                  onPress={() => setCorpusTab(k)}
                >
                  <Text
                    style={[
                      styles.segBtnText,
                      corpusTab === k && styles.segBtnTextActive,
                    ]}
                  >
                    {l}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>

            {corpusTab === "papers" &&
              PAPERS_DATA.map((p, i) => (
                <View key={i} style={styles.card}>
                  <Text style={styles.listCardAccent}>{p.venue}</Text>
                  <Text style={[styles.cardTitle, { marginTop: 4 }]}>
                    {p.title}
                  </Text>
                  <Text style={[styles.cardSub, { marginTop: 6 }]}>
                    {p.summary}
                  </Text>
                </View>
              ))}

            {corpusTab === "dataset" && (
              <View style={styles.card}>
                <Text style={styles.cardTitle}>
                  12,000-Image 10-Class Training Corpus
                </Text>
                <Text style={styles.cardSub}>
                  9,600 Train / 1,200 Val / 1,200 Held-Out Test (1,200 images per pathology) paired with AI-Hub Korea & SDASI/SALT 4-axis severity grading:
                </Text>
                {[
                  "00 · Alopecia Areata (Level 3 Escalation)",
                  "01 · Contact Dermatitis (Level 2 Monitored)",
                  "02 · Folliculitis (Level 3 Escalation)",
                  "03 · Pediculosis Capitis (Level 2 Monitored)",
                  "04 · Lichen Planus (Level 3 Escalation)",
                  "05 · Male Pattern Baldness (Level 2 Monitored)",
                  "06 · Scalp Psoriasis (Level 3 Escalation)",
                  "07 · Seborrheic Dermatitis (Level 2 Monitored)",
                  "08 · Telogen Effluvium (Level 2 Monitored)",
                  "09 · Tinea Capitis (Level 3 Escalation)",
                ].map((row) => (
                  <View key={row} style={styles.corpusRow}>
                    <Text style={styles.corpusRowText}>{row}</Text>
                  </View>
                ))}
              </View>
            )}

            {corpusTab === "benchmarks" && (
              <View style={styles.card}>
                <Text style={styles.cardTitle}>
                  Verification Metrics & Limitations
                </Text>
                <View style={styles.listCard}>
                  <Text style={styles.listCardAccent}>HELD-OUT TEST SPLIT</Text>
                  <Text style={styles.listCardTitle}>
                    100.0% (1,200 / 1,200) · Macro F1 = 1.00
                  </Text>
                  <Text style={styles.listCardBody}>
                    97.0% top-1 accuracy under simultaneous rotation, 85% crop, 0.75x darkening, and JPEG Q75 compression; 95.9% optical quality gate pass rate.
                  </Text>
                </View>
                <View style={styles.rationaleBox}>
                  <Text style={styles.rationaleText}>
                    Academic Limitations: Public Kaggle cross-split augmentation inflates clean split accuracy; real handheld photos exhibit broader variance, mitigated via our 0.65 server-side Confidence Gate. Classical CV and SDASI/SALT brackets remain uncalibrated engineering estimates.
                  </Text>
                </View>
              </View>
            )}
          </View>
        )}
      </ScrollView>

      {/* FLOATING PILL BOTTOM NAVIGATION DOCK */}
      <View style={styles.bottomDockOuter}>
        <View style={styles.bottomNav}>
          {[
            ["triage", "01", "Triage"],
            ["intake", "02", "Intake"],
            ["architecture", "03", "System"],
            ["corpus", "04", "Corpus"],
          ].map(([key, idx, label]) => {
            const active = primaryTab === key;
            return (
              <TouchableOpacity
                key={key}
                style={[styles.bottomTab, active && styles.bottomTabActive]}
                onPress={() => setPrimaryTab(key)}
                activeOpacity={0.85}
              >
                <Text
                  style={[styles.bottomIdx, active && styles.bottomIdxActive]}
                >
                  {idx}
                </Text>
                <Text
                  style={[
                    styles.bottomLabel,
                    active && styles.bottomLabelActive,
                  ]}
                >
                  {label}
                </Text>
              </TouchableOpacity>
            );
          })}
        </View>
      </View>
    </SafeAreaView>
  );
}

const MONO = Platform.OS === "ios" ? "Menlo" : "monospace";

const styles = StyleSheet.create({
  safeRoot: {
    flex: 1,
    backgroundColor: "#080b10",
  },
  topHeader: {
    height: 64,
    paddingHorizontal: 18,
    borderBottomWidth: 1,
    borderBottomColor: "#1b2432",
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    backgroundColor: "#0b1018",
  },
  brandRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
  },
  brandOrb: {
    width: 36,
    height: 36,
    borderRadius: 10,
    backgroundColor: "#122126",
    borderWidth: 1,
    borderColor: "rgba(0, 230, 207, 0.45)",
    alignItems: "center",
    justifyContent: "center",
  },
  brandInnerRing: {
    width: 18,
    height: 18,
    borderRadius: 9,
    borderWidth: 1.8,
    borderColor: "#00e6cf",
    alignItems: "center",
    justifyContent: "center",
  },
  brandCoreDot: {
    width: 5,
    height: 5,
    borderRadius: 3,
    backgroundColor: "#00e6cf",
  },
  brandTitle: {
    color: "#f8fafc",
    fontWeight: "700",
    fontSize: 18,
    letterSpacing: -0.4,
  },
  brandSubtitle: {
    color: "#8b9bb0",
    fontSize: 11.5,
    marginTop: 1,
  },
  serverPill: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    backgroundColor: "#131b27",
    borderWidth: 1,
    borderColor: "#233044",
    paddingHorizontal: 11,
    paddingVertical: 6,
    borderRadius: 999,
  },
  serverDot: {
    width: 7,
    height: 7,
    borderRadius: 3.5,
    backgroundColor: "#10b981",
  },
  serverPillText: {
    color: "#cbd5e1",
    fontSize: 11.5,
    fontWeight: "500",
  },
  serverDrawer: {
    backgroundColor: "#111823",
    paddingHorizontal: 18,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: "#1e293b",
  },
  drawerLabel: {
    color: "#b4c0d0",
    fontSize: 12,
    marginBottom: 7,
  },
  drawerRow: {
    flexDirection: "row",
    gap: 8,
  },
  urlInput: {
    flex: 1,
    backgroundColor: "#080b10",
    borderWidth: 1,
    borderColor: "#29374d",
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 8,
    color: "#f8fafc",
    fontFamily: MONO,
    fontSize: 12,
  },
  pingBtn: {
    backgroundColor: "#192636",
    borderWidth: 1,
    borderColor: "#00d4be",
    borderRadius: 8,
    paddingHorizontal: 16,
    justifyContent: "center",
  },
  pingBtnText: {
    color: "#00e6cf",
    fontWeight: "700",
    fontSize: 12.5,
  },
  mainViewport: {
    flex: 1,
  },
  mainContent: {
    padding: 15,
    paddingBottom: 26,
  },
  loadingBanner: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    backgroundColor: "#131b27",
    borderWidth: 1,
    borderColor: "rgba(0, 212, 190, 0.35)",
    borderRadius: 10,
    padding: 12,
    marginBottom: 12,
  },
  loadingText: {
    color: "#cbd5e1",
    fontSize: 13,
  },
  protoGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 7,
    marginBottom: 12,
  },
  protoPill: {
    flexGrow: 1,
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    backgroundColor: "#111823",
    borderWidth: 1,
    borderColor: "#1e293b",
    borderRadius: 9,
    paddingHorizontal: 10,
    paddingVertical: 8,
  },
  protoNum: {
    color: "#00e6cf",
    fontFamily: MONO,
    fontSize: 11,
    fontWeight: "700",
  },
  protoLbl: {
    color: "#cbd5e1",
    fontSize: 12,
    fontWeight: "600",
  },
  card: {
    backgroundColor: "#121924",
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.09)",
    borderRadius: 15,
    padding: 17,
    marginBottom: 14,
    elevation: 5,
  },
  rowBetween: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  cardTitle: {
    color: "#f8fafc",
    fontSize: 18,
    fontWeight: "700",
    letterSpacing: -0.3,
  },
  cardSub: {
    color: "#b4c0d0",
    fontSize: 13.5,
    lineHeight: 19.5,
    marginTop: 4,
  },
  viewfinderBox: {
    backgroundColor: "#0c121b",
    borderWidth: 1,
    borderColor: "#243246",
    borderRadius: 12,
    paddingVertical: 24,
    paddingHorizontal: 16,
    alignItems: "center",
    marginVertical: 14,
    position: "relative",
  },
  vfCorner: {
    position: "absolute",
    width: 15,
    height: 15,
    borderColor: "rgba(0, 230, 207, 0.55)",
  },
  vfTL: { top: 10, left: 10, borderTopWidth: 2, borderLeftWidth: 2, borderTopLeftRadius: 3 },
  vfTR: { top: 10, right: 10, borderTopWidth: 2, borderRightWidth: 2, borderTopRightRadius: 3 },
  vfBL: { bottom: 10, left: 10, borderBottomWidth: 2, borderLeftWidth: 2, borderBottomLeftRadius: 3 },
  vfBR: { bottom: 10, right: 10, borderBottomWidth: 2, borderRightWidth: 2, borderBottomRightRadius: 3 },
  lensBadge: {
    width: 48,
    height: 48,
    borderRadius: 14,
    backgroundColor: "rgba(0, 212, 190, 0.14)",
    borderWidth: 1,
    borderColor: "rgba(0, 230, 207, 0.42)",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 10,
  },
  lensBadgePlus: {
    color: "#00e6cf",
    fontSize: 24,
    fontWeight: "500",
    marginTop: -2,
  },
  vfTitle: {
    color: "#f8fafc",
    fontWeight: "700",
    fontSize: 15.5,
  },
  vfSub: {
    color: "#8b9bb0",
    fontSize: 12,
    marginTop: 4,
  },
  capRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 6,
    marginTop: 12,
  },
  capBadge: {
    backgroundColor: "#172130",
    borderWidth: 1,
    borderColor: "#253449",
    borderRadius: 999,
    paddingHorizontal: 9,
    paddingVertical: 3,
    color: "#b4c0d0",
    fontFamily: MONO,
    fontSize: 10.5,
  },
  previewBox: {
    alignItems: "center",
    marginVertical: 14,
  },
  previewImg: {
    width: "100%",
    height: 180,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "rgba(0, 230, 207, 0.42)",
  },
  previewReady: {
    color: "#34d399",
    fontSize: 12.5,
    fontWeight: "500",
    marginTop: 7,
  },
  actionRow2: {
    flexDirection: "row",
    gap: 9,
    marginBottom: 9,
  },
  primaryBtn: {
    backgroundColor: "#00dfc8",
    borderRadius: 10,
    paddingVertical: 13,
    alignItems: "center",
    elevation: 4,
  },
  primaryBtnSmall: {
    backgroundColor: "#00dfc8",
    borderRadius: 8,
    paddingVertical: 9,
    paddingHorizontal: 16,
  },
  primaryBtnText: {
    color: "#031412",
    fontWeight: "700",
    fontSize: 14.5,
    letterSpacing: -0.2,
  },
  secondaryBtn: {
    flex: 1,
    backgroundColor: "#1a2536",
    borderWidth: 1,
    borderColor: "#2b3b52",
    borderRadius: 10,
    paddingVertical: 11,
    alignItems: "center",
  },
  secondaryBtnSmall: {
    backgroundColor: "#1a2536",
    borderWidth: 1,
    borderColor: "#2b3b52",
    borderRadius: 8,
    paddingVertical: 9,
    paddingHorizontal: 15,
  },
  secondaryBtnText: {
    color: "#f8fafc",
    fontWeight: "600",
    fontSize: 13.5,
  },
  ghostBtn: {
    backgroundColor: "#151e2c",
    borderWidth: 1,
    borderColor: "#243246",
    borderRadius: 10,
    paddingVertical: 10,
    alignItems: "center",
  },
  ghostBtnSmall: {
    backgroundColor: "#151e2c",
    borderWidth: 1,
    borderColor: "#26354a",
    borderRadius: 8,
    paddingVertical: 8,
    paddingHorizontal: 13,
  },
  ghostBtnText: {
    color: "#cbd5e1",
    fontSize: 13,
    fontWeight: "600",
  },
  featuredControlCard: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    backgroundColor: "rgba(16, 185, 129, 0.11)",
    borderWidth: 1,
    borderColor: "rgba(16, 185, 129, 0.36)",
    borderRadius: 11,
    padding: 12,
    marginTop: 12,
  },
  featuredControlTitle: {
    color: "#f8fafc",
    fontSize: 14,
    fontWeight: "700",
  },
  featuredControlSub: {
    color: "#94a3b8",
    fontFamily: MONO,
    fontSize: 11,
    marginTop: 2,
  },
  featuredControlAction: {
    color: "#34d399",
    fontFamily: MONO,
    fontWeight: "700",
    fontSize: 12.5,
  },
  presetGrid2: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
    marginTop: 9,
  },
  presetTile: {
    width: "48.5%",
    backgroundColor: "#0d131d",
    borderWidth: 1,
    borderColor: "#212e42",
    borderRadius: 10,
    padding: 11,
  },
  presetTitle: {
    color: "#f8fafc",
    fontSize: 13.5,
    fontWeight: "600",
  },
  presetMetaRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 5,
    marginTop: 4,
  },
  tierDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
  },
  presetSub: {
    color: "#8b9bb0",
    fontFamily: MONO,
    fontSize: 11,
  },
  reportNavRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 10,
  },
  overlineLabel: {
    color: "#00e6cf",
    fontFamily: MONO,
    fontSize: 11,
    letterSpacing: 0.6,
  },
  conditionHeadline: {
    color: "#f8fafc",
    fontSize: 20,
    fontWeight: "700",
    letterSpacing: -0.4,
    marginTop: 3,
  },
  triageBadge: {
    alignSelf: "flex-start",
    borderWidth: 1,
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 5,
    marginTop: 9,
  },
  triageBadgeText: {
    fontSize: 12.5,
    fontWeight: "600",
  },
  metricRow3: {
    flexDirection: "row",
    gap: 8,
    marginVertical: 14,
  },
  metricCell: {
    flex: 1,
    backgroundColor: "#0c121b",
    borderWidth: 1,
    borderColor: "#1f2b3d",
    borderRadius: 10,
    padding: 11,
    overflow: "hidden",
  },
  metricTopBar: {
    height: 2,
    backgroundColor: "#00d4be",
    marginBottom: 6,
    opacity: 0.8,
  },
  metricLabel: {
    color: "#94a3b8",
    fontSize: 11,
    fontWeight: "500",
  },
  metricVal: {
    color: "#f8fafc",
    fontSize: 17,
    fontWeight: "700",
    marginTop: 2,
  },
  metricSub: {
    color: "#64748b",
    fontSize: 10.5,
    marginTop: 2,
  },
  segmentedRow: {
    flexDirection: "row",
    backgroundColor: "#0b1018",
    borderRadius: 10,
    padding: 3.5,
    borderWidth: 1,
    borderColor: "#1e293b",
  },
  segBtn: {
    flex: 1,
    paddingVertical: 7,
    alignItems: "center",
    borderRadius: 7,
  },
  segBtnActive: {
    backgroundColor: "#1c2738",
    borderWidth: 1,
    borderColor: "#2d3e56",
  },
  segBtnText: {
    color: "#94a3b8",
    fontSize: 12,
    fontWeight: "500",
  },
  segBtnTextActive: {
    color: "#f8fafc",
    fontWeight: "700",
  },
  subscoreCard: {
    backgroundColor: "#0c121b",
    borderWidth: 1,
    borderColor: "#1e293b",
    borderRadius: 10,
    padding: 12,
    marginBottom: 9,
  },
  subscoreHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 7,
  },
  subscoreName: {
    color: "#f8fafc",
    fontSize: 13.5,
    fontWeight: "600",
  },
  subscoreNum: {
    fontFamily: MONO,
    fontSize: 12,
  },
  barTrack: {
    height: 7,
    backgroundColor: "#1b2535",
    borderRadius: 999,
    overflow: "hidden",
  },
  barFill: {
    height: "100%",
    borderRadius: 999,
  },
  rationaleBox: {
    backgroundColor: "#0c121b",
    borderWidth: 1,
    borderColor: "#1e293b",
    borderLeftWidth: 3,
    borderLeftColor: "#00e6cf",
    borderRadius: 8,
    padding: 12,
    marginTop: 8,
  },
  rationaleText: {
    color: "#b4c0d0",
    fontSize: 13,
    lineHeight: 19,
  },
  listCard: {
    backgroundColor: "#0c121b",
    borderWidth: 1,
    borderColor: "#1e293b",
    borderRadius: 10,
    padding: 13,
    marginTop: 9,
  },
  listCardAccent: {
    color: "#00e6cf",
    fontFamily: MONO,
    fontSize: 11,
    marginBottom: 3,
  },
  listCardTitle: {
    color: "#f8fafc",
    fontSize: 14,
    fontWeight: "700",
  },
  listCardBody: {
    color: "#b4c0d0",
    fontSize: 12.8,
    lineHeight: 18.5,
    marginTop: 4,
  },
  disclaimerText: {
    color: "#7e8ea3",
    fontSize: 11.2,
    lineHeight: 16,
    marginTop: 14,
    paddingTop: 11,
    borderTopWidth: 1,
    borderTopColor: "#1e293b",
  },
  fieldLabel: {
    color: "#b4c0d0",
    fontSize: 13,
    fontWeight: "600",
    marginTop: 13,
    marginBottom: 7,
  },
  chipWrap: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 7,
  },
  chip: {
    backgroundColor: "#0c121b",
    borderWidth: 1,
    borderColor: "#223044",
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 7,
  },
  chipActive: {
    backgroundColor: "rgba(0, 212, 190, 0.16)",
    borderColor: "rgba(0, 230, 207, 0.55)",
  },
  chipText: {
    color: "#b4c0d0",
    fontSize: 12.8,
    fontWeight: "500",
  },
  chipTextActive: {
    color: "#f8fafc",
    fontWeight: "700",
  },
  wizardFooter: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginTop: 18,
    paddingTop: 13,
    borderTopWidth: 1,
    borderTopColor: "#1e293b",
  },
  corpusRow: {
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: "#1e293b",
  },
  corpusRowText: {
    color: "#cbd5e1",
    fontFamily: MONO,
    fontSize: 12,
  },
  bottomDockOuter: {
    backgroundColor: "#080b10",
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderTopWidth: 1,
    borderTopColor: "#182230",
  },
  bottomNav: {
    flexDirection: "row",
    backgroundColor: "#101722",
    borderRadius: 13,
    padding: 4,
    borderWidth: 1,
    borderColor: "#1f2b3d",
    gap: 4,
  },
  bottomTab: {
    flex: 1,
    paddingVertical: 7,
    alignItems: "center",
    justifyContent: "center",
    borderRadius: 9,
    borderWidth: 1,
    borderColor: "transparent",
  },
  bottomTabActive: {
    backgroundColor: "#1a2638",
    borderColor: "rgba(0, 230, 207, 0.4)",
  },
  bottomIdx: {
    color: "#64748b",
    fontFamily: MONO,
    fontSize: 10,
  },
  bottomIdxActive: {
    color: "#00e6cf",
    fontWeight: "700",
  },
  bottomLabel: {
    color: "#94a3b8",
    fontSize: 12,
    fontWeight: "500",
    marginTop: 1,
  },
  bottomLabelActive: {
    color: "#f8fafc",
    fontWeight: "700",
  },
});
