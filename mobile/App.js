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
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import { checkSystemStatus, submitScalpAnalysis, DEFAULT_API_BASE } from "./src/services/api";

const UNIVERSAL_DISCLAIMER =
  "This assessment is for informational and educational purposes only and suggests potential scalp care routines. It does not constitute medical diagnosis, formal medical advice, or treatment. If symptoms persist, cause discomfort, or worsen, please consult a certified dermatologist.";

function scoreAccent(val) {
  if (val >= 20.0) return { color: "#00d2be", label: "OPTIMAL" };
  if (val >= 14.0) return { color: "#f59e0b", label: "MODERATE" };
  return { color: "#ef4444", label: "ELEVATED" };
}

export default function App() {
  const [apiBaseUrl, setApiBaseUrl] = useState(DEFAULT_API_BASE);
  const [currentStep, setCurrentStep] = useState(1);
  const [loading, setLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");
  const [selectedImageUri, setSelectedImageUri] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);

  const [intake, setIntake] = useState({
    biological_sex: "male",
    age_group: "18_25",
    primary_concern_zone: "routine_healthy_check",
    water_environment: "normal",
    daily_helmet_wearer: false,
    washing_frequency: "2_3_per_week",
    hair_pull_test: "negative_under_3",
    flake_type: "none",
    sensation: "painless",
    stress_level: "moderate",
    diet_pattern: "balanced_protein",
    duration_months: 1,
    duration_weeks: 4,
    gut_health: "optimal",
    metabolic_energy: "balanced",
    hormonal_profile: "none",
  });

  const testServerConnection = async () => {
    try {
      setLoading(true);
      setStatusMessage("Pinging telemetry backend...");
      const status = await checkSystemStatus(apiBaseUrl);
      setStatusMessage(
        `ONLINE | ${status.model_architecture || "ONNX"} Ready (${status.model_loaded ? "Loaded" : "Standby"})`
      );
    } catch (err) {
      setStatusMessage(`OFFLINE | ${err.message}`);
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
      setStatusMessage("Scalp frame locked and ready for dual-pipeline triage.");
    }
  };

  const runLiveOrMockAnalysis = async () => {
    if (selectedImageUri) {
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
        setCurrentStep(3);
        return;
      } catch (err) {
        setStatusMessage(`API Error: ${err.message}`);
      } finally {
        setLoading(false);
      }
    }
    runMockAnalysis();
  };

  const runMockAnalysis = () => {
    setLoading(true);
    setTimeout(() => {
      setAnalysisResult({
        scalp_health_score: 54.6,
        confidence_score: 0.95,
        suggested_condition: "Pattern Suggestive of Seborrheic Dermatitis",
        triage_level: "Level 2: Monitored Routine",
        triage_reason:
          "Visual indications suggest active sebum/microbial flaking. Recommends an active ingredient schedule and structured 14-day progress tracking.",
        sub_scores: {
          flake_score: 12.4,
          redness_score: 14.8,
          sebum_score: 11.2,
          density_score: 16.2,
        },
        survey_risk_profile: {
          risk_score: 58.2,
          risk_band: "elevated",
          modifiable_count: 3,
          summary:
            "Condition-anchored survey analysis indicates an ELEVATED modifiable trigger load (3 actionable factors). Primary addressable drivers: Hard Water Mineral Load, Daily Helmet / Sweat Occlusion.",
          intervention_queue: [
            {
              priority: 1,
              title: "Eliminate High-Impact Trigger: Hard Water Mineral Load",
              detail:
                "Install a shower softening filter or incorporate a weekly chelating rinse to clear mineral buildup.",
              expected_timeline: "2-4 weeks",
            },
            {
              priority: 2,
              title: "Eliminate High-Impact Trigger: Daily Helmet / Sweat Occlusion",
              detail:
                "Use a washable antibacterial helmet liner, wash every 2-3 days, and air-dry scalp after rides.",
              expected_timeline: "2-4 weeks",
            },
          ],
        },
        active_ingredients: [
          {
            molecule: "Ketoconazole 1%-2% or Piroctone Olamine Cleanser",
            frequency: "2x per week (5-min scalp contact)",
            mechanism: "Targets Malassezia lipid hydrolysis and reduces microbial scaling.",
          },
          {
            molecule: "Salicylic Acid 1.8%-2% Scalp Wash",
            frequency: "1x per week",
            mechanism: "Keratolytic descaling to clear adherent sebum-keratin buildup.",
          },
        ],
        lifestyle_advice: [
          "Wash helmet liners every 2-3 days to reduce occluded heat and sweat retention.",
          "Retake standardized photo in 14 days under identical natural lighting.",
        ],
        disclaimer: UNIVERSAL_DISCLAIMER,
      });
      setLoading(false);
      setCurrentStep(3);
    }, 650);
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#06080a" />

      {/* Top Carbon AMG Header */}
      <View style={styles.header}>
        <View style={styles.brandRow}>
          <View style={styles.logoMark}>
            <Text style={styles.logoMarkText}>F</Text>
          </View>
          <View>
            <Text style={styles.title}>FOLLICA</Text>
            <Text style={styles.subtitle}>CLINICAL SCALP TELEMETRY</Text>
          </View>
        </View>
        <TouchableOpacity style={styles.demoChip} onPress={runMockAnalysis}>
          <Text style={styles.demoChipText}>DEMO</Text>
        </TouchableOpacity>
      </View>

      {/* Segmented Navigation Bar */}
      <View style={styles.stepBar}>
        {[
          { id: 1, label: "01 CAPTURE" },
          { id: 2, label: "02 SURVEY" },
          { id: 3, label: "03 TELEMETRY" },
        ].map((tab) => (
          <TouchableOpacity
            key={tab.id}
            style={[styles.stepItem, currentStep === tab.id && styles.stepItemActive]}
            onPress={() => setCurrentStep(tab.id)}
          >
            <Text style={[styles.stepText, currentStep === tab.id && styles.stepTextActive]}>
              {tab.label}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      <ScrollView contentContainerStyle={styles.scrollContent}>
        {/* STEP 1: CAPTURE & CONNECTION */}
        {currentStep === 1 && (
          <View>
            <View style={styles.heroStrip}>
              <Text style={styles.heroOverline}>ONNX VISION + SDASI CALIBRATION</Text>
              <Text style={styles.heroHeadline}>PRECISION SCALP TRIAGE</Text>
              <View style={styles.tealRule} />
              <Text style={styles.heroSub}>
                Capture a clear, flash-free scalp frame for 10-class pattern triage and Layer-2 survey analysis.
              </Text>
            </View>

            <View style={styles.card}>
              <View style={styles.cardTitleRow}>
                <Text style={styles.cardSectionTag}>OPTICAL PROTOCOL</Text>
                <Text style={styles.cardStepBadge}>STEP 01</Text>
              </View>

              <View style={styles.protocolGrid}>
                <View style={styles.protocolTile}>
                  <Text style={styles.protocolNum}>01</Text>
                  <Text style={styles.protocolHead}>Clean Lens</Text>
                  <Text style={styles.protocolSub}>Wipe oil smudges first.</Text>
                </View>
                <View style={styles.protocolTile}>
                  <Text style={styles.protocolNum}>02</Text>
                  <Text style={styles.protocolHead}>No Flash</Text>
                  <Text style={styles.protocolSub}>Daylight or room light.</Text>
                </View>
                <View style={styles.protocolTile}>
                  <Text style={styles.protocolNum}>03</Text>
                  <Text style={styles.protocolHead}>Part Hair</Text>
                  <Text style={styles.protocolSub}>Expose scalp clearly.</Text>
                </View>
                <View style={styles.protocolTile}>
                  <Text style={styles.protocolNum}>04</Text>
                  <Text style={styles.protocolHead}>10-15 cm</Text>
                  <Text style={styles.protocolSub}>Tap to lock focus.</Text>
                </View>
              </View>

              {selectedImageUri && (
                <View style={styles.previewFrame}>
                  <Image source={{ uri: selectedImageUri }} style={styles.previewImage} resizeMode="cover" />
                  <Text style={styles.previewCaption}>SCALP FRAME LOCKED</Text>
                </View>
              )}

              <View style={styles.dualBtnRow}>
                <TouchableOpacity
                  style={[styles.outlineButton, { flex: 1, marginRight: 6 }]}
                  onPress={() => pickImage(true)}
                  disabled={loading}
                >
                  <Text style={styles.outlineButtonText}>CAMERA</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.outlineButton, { flex: 1, marginLeft: 6 }]}
                  onPress={() => pickImage(false)}
                  disabled={loading}
                >
                  <Text style={styles.outlineButtonText}>GALLERY</Text>
                </TouchableOpacity>
              </View>

              {selectedImageUri && (
                <TouchableOpacity
                  style={styles.primaryButton}
                  onPress={runLiveOrMockAnalysis}
                  disabled={loading}
                >
                  {loading ? (
                    <ActivityIndicator color="#041010" />
                  ) : (
                    <Text style={styles.primaryButtonText}>ANALYZE PHOTO NOW</Text>
                  )}
                </TouchableOpacity>
              )}

              <TouchableOpacity
                style={[styles.secondaryButton, { marginTop: 10 }]}
                onPress={() => setCurrentStep(2)}
              >
                <Text style={styles.secondaryButtonText}>NEXT: CONFIGURE CLINICAL SURVEY</Text>
              </TouchableOpacity>
            </View>

            <View style={styles.card}>
              <Text style={styles.cardSectionTag}>TELEMETRY ENDPOINT</Text>
              <TextInput
                style={styles.input}
                value={apiBaseUrl}
                onChangeText={setApiBaseUrl}
                placeholder="http://10.0.2.2:8000/api/v1"
                placeholderTextColor="#5b6675"
                autoCapitalize="none"
                autoCorrect={false}
              />
              <TouchableOpacity style={styles.secondaryButton} onPress={testServerConnection} disabled={loading}>
                <Text style={styles.secondaryButtonText}>VERIFY SERVER LINK</Text>
              </TouchableOpacity>
              {statusMessage ? <Text style={styles.statusText}>{statusMessage}</Text> : null}
            </View>
          </View>
        )}

        {/* STEP 2: CLINICAL & LIFESTYLE SURVEY */}
        {currentStep === 2 && (
          <View style={styles.card}>
            <View style={styles.cardTitleRow}>
              <Text style={styles.cardSectionTag}>LAYER-2 INTAKE SURVEY</Text>
              <Text style={styles.cardStepBadge}>STEP 02</Text>
            </View>
            <Text style={styles.cardDescription}>
              Post-identification survey signals refine modifiable triggers and prioritise your care queue.
            </Text>

            <Text style={styles.fieldLabel}>PRIMARY CONCERN ZONE</Text>
            <View style={styles.chipRow}>
              {[
                { l: "Routine Baseline", v: "routine_healthy_check" },
                { l: "Crown / Vertex", v: "crown_vertex" },
                { l: "Hairline / Temples", v: "hairline_temples" },
                { l: "Parting Line", v: "central_parting" },
                { l: "Diffuse Thinning", v: "diffuse_all_over" },
                { l: "Patchy Spots", v: "patchy_spots" },
              ].map((item) => (
                <TouchableOpacity
                  key={item.v}
                  style={[styles.chip, intake.primary_concern_zone === item.v && styles.chipActive]}
                  onPress={() => setIntake({ ...intake, primary_concern_zone: item.v })}
                >
                  <Text style={[styles.chipText, intake.primary_concern_zone === item.v && styles.chipTextActive]}>
                    {item.l}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>

            <Text style={styles.fieldLabel}>SCALP SENSATION</Text>
            <View style={styles.chipRow}>
              {[
                { l: "Painless", v: "painless" },
                { l: "Itching", v: "itching" },
                { l: "Burning / Tender", v: "burning_or_tender" },
                { l: "Painful Bumps", v: "painful_bumps" },
              ].map((item) => (
                <TouchableOpacity
                  key={item.v}
                  style={[styles.chip, intake.sensation === item.v && styles.chipActive]}
                  onPress={() => setIntake({ ...intake, sensation: item.v })}
                >
                  <Text style={[styles.chipText, intake.sensation === item.v && styles.chipTextActive]}>
                    {item.l}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>

            <Text style={styles.fieldLabel}>FLAKE PRESENTATION</Text>
            <View style={styles.chipRow}>
              {[
                { l: "None", v: "none" },
                { l: "Dry White", v: "dry_white_fine" },
                { l: "Greasy Yellow", v: "greasy_yellow_thick" },
                { l: "Thick Plaque", v: "plaque_crusty" },
              ].map((item) => (
                <TouchableOpacity
                  key={item.v}
                  style={[styles.chip, intake.flake_type === item.v && styles.chipActive]}
                  onPress={() => setIntake({ ...intake, flake_type: item.v })}
                >
                  <Text style={[styles.chipText, intake.flake_type === item.v && styles.chipTextActive]}>
                    {item.l}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>

            <Text style={styles.fieldLabel}>WATER SOURCE &amp; HEADWEAR</Text>
            <View style={styles.chipRow}>
              {[
                { l: "Normal Tap", v: "normal" },
                { l: "Hard Borewell", v: "hard_borewell" },
                { l: "Recent Move", v: "recent_relocation" },
              ].map((item) => (
                <TouchableOpacity
                  key={item.v}
                  style={[styles.chip, intake.water_environment === item.v && styles.chipActive]}
                  onPress={() => setIntake({ ...intake, water_environment: item.v })}
                >
                  <Text style={[styles.chipText, intake.water_environment === item.v && styles.chipTextActive]}>
                    {item.l}
                  </Text>
                </TouchableOpacity>
              ))}
              <TouchableOpacity
                style={[styles.chip, intake.daily_helmet_wearer && styles.chipActive]}
                onPress={() => setIntake({ ...intake, daily_helmet_wearer: !intake.daily_helmet_wearer })}
              >
                <Text style={[styles.chipText, intake.daily_helmet_wearer && styles.chipTextActive]}>
                  Daily Helmet: {intake.daily_helmet_wearer ? "YES" : "NO"}
                </Text>
              </TouchableOpacity>
            </View>

            <Text style={styles.fieldLabel}>STRESS &amp; DIET PROFILE</Text>
            <View style={styles.chipRow}>
              {[
                { l: "Low Stress", v: "low" },
                { l: "Moderate Stress", v: "moderate" },
                { l: "High Stress", v: "high" },
              ].map((item) => (
                <TouchableOpacity
                  key={item.v}
                  style={[styles.chip, intake.stress_level === item.v && styles.chipActive]}
                  onPress={() => setIntake({ ...intake, stress_level: item.v })}
                >
                  <Text style={[styles.chipText, intake.stress_level === item.v && styles.chipTextActive]}>
                    {item.l}
                  </Text>
                </TouchableOpacity>
              ))}
              {[
                { l: "Balanced Protein", v: "balanced_protein" },
                { l: "Veg / Vegan", v: "vegetarian_vegan" },
                { l: "Low Protein", v: "low_protein_crash_diet" },
              ].map((item) => (
                <TouchableOpacity
                  key={item.v}
                  style={[styles.chip, intake.diet_pattern === item.v && styles.chipActive]}
                  onPress={() => setIntake({ ...intake, diet_pattern: item.v })}
                >
                  <Text style={[styles.chipText, intake.diet_pattern === item.v && styles.chipTextActive]}>
                    {item.l}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>

            <Text style={styles.fieldLabel}>GUT &amp; ENDOCRINE AXIS</Text>
            <View style={styles.chipRow}>
              {[
                { l: "Optimal Gut", v: "optimal" },
                { l: "Bloating / Acidity", v: "frequent_bloating_acidity" },
                { l: "IBS / Irregular", v: "irregular_ibs" },
              ].map((item) => (
                <TouchableOpacity
                  key={item.v}
                  style={[styles.chip, intake.gut_health === item.v && styles.chipActive]}
                  onPress={() => setIntake({ ...intake, gut_health: item.v })}
                >
                  <Text style={[styles.chipText, intake.gut_health === item.v && styles.chipTextActive]}>
                    {item.l}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>

            <TouchableOpacity style={styles.primaryButton} onPress={runLiveOrMockAnalysis} disabled={loading}>
              {loading ? (
                <ActivityIndicator color="#041010" />
              ) : (
                <Text style={styles.primaryButtonText}>RUN DUAL-PIPELINE TELEMETRY</Text>
              )}
            </TouchableOpacity>
          </View>
        )}

        {/* STEP 3: CLINICAL TELEMETRY REPORT */}
        {currentStep === 3 && (
          <View>
            {!analysisResult ? (
              <View style={styles.card}>
                <Text style={styles.cardSectionTag}>NO TELEMETRY RECORDED</Text>
                <Text style={[styles.cardDescription, { marginTop: 8 }]}>
                  Upload a scalp photograph in Step 01 or launch the Demo telemetry scenario.
                </Text>
                <TouchableOpacity style={styles.primaryButton} onPress={() => setCurrentStep(1)}>
                  <Text style={styles.primaryButtonText}>GO TO STEP 01 CAPTURE</Text>
                </TouchableOpacity>
              </View>
            ) : (
              <View style={styles.card}>
                <View style={styles.cardTitleRow}>
                  <Text style={styles.cardSectionTag}>CLINICAL TELEMETRY REPORT</Text>
                  <Text style={styles.cardStepBadge}>SDASI / SALT</Text>
                </View>

                {/* SHS Speedometer Hero */}
                <View style={styles.scoreHero}>
                  <View>
                    <Text style={styles.scoreOverline}>SCALP HEALTH SCORE</Text>
                    <View style={{ flexDirection: "row", alignItems: "baseline" }}>
                      <Text style={styles.scoreNumber}>
                        {Number(analysisResult.scalp_health_score).toFixed(1)}
                      </Text>
                      <Text style={styles.scoreDenom}>/100</Text>
                    </View>
                  </View>
                  <View style={styles.triagePill}>
                    <Text style={styles.triagePillText}>{analysisResult.triage_level}</Text>
                  </View>
                </View>

                {/* Condition Pattern Box */}
                <View style={styles.triageBox}>
                  <Text style={styles.patternOverline}>VISUAL ASSESSMENT PATTERN</Text>
                  <Text style={styles.conditionText}>{analysisResult.suggested_condition}</Text>
                  {analysisResult.confidence_score !== undefined && (
                    <Text style={styles.confText}>
                      Classifier Confidence: {(analysisResult.confidence_score * 100).toFixed(1)}%
                    </Text>
                  )}
                  <Text style={styles.triageReason}>{analysisResult.triage_reason}</Text>
                </View>

                {/* 4-Axis CV Sub-Scores */}
                <Text style={styles.fieldLabel}>MICRO-FEATURE SUB-SCORES (0-25)</Text>
                <View style={styles.subScoreGrid}>
                  {[
                    { k: "flake_score", n: "FLAKE & SCALING", v: analysisResult.sub_scores.flake_score },
                    { k: "redness_score", n: "ERYTHEMA REDNESS", v: analysisResult.sub_scores.redness_score },
                    { k: "sebum_score", n: "SEBUM & LIPIDS", v: analysisResult.sub_scores.sebum_score },
                    { k: "density_score", n: "HAIR COVERAGE", v: analysisResult.sub_scores.density_score },
                  ].map((ax) => {
                    const acc = scoreAccent(ax.v);
                    return (
                      <View key={ax.k} style={styles.subScoreItem}>
                        <Text style={styles.subScoreName}>{ax.n}</Text>
                        <View style={styles.subScoreValRow}>
                          <Text style={styles.subScoreVal}>{Number(ax.v).toFixed(1)}/25</Text>
                          <Text style={[styles.subScoreGrade, { color: acc.color }]}>{acc.label}</Text>
                        </View>
                        <View style={styles.barTrack}>
                          <View
                            style={[
                              styles.barFill,
                              { width: `${Math.min(100, (ax.v / 25) * 100)}%`, backgroundColor: acc.color },
                            ]}
                          />
                        </View>
                      </View>
                    );
                  })}
                </View>

                {/* Layer-2 Post-Diagnosis Survey Pipeline */}
                {analysisResult.survey_risk_profile && (
                  <View style={styles.surveyBox}>
                    <View style={styles.cardTitleRow}>
                      <Text style={styles.cardSectionTag}>POST-DX SURVEY PIPELINE</Text>
                      <Text style={styles.cardStepBadge}>
                        {String(analysisResult.survey_risk_profile.risk_band || "").toUpperCase()} LOAD (
                        {Math.round(analysisResult.survey_risk_profile.risk_score || 0)}/100)
                      </Text>
                    </View>
                    <Text style={styles.surveySummary}>{analysisResult.survey_risk_profile.summary}</Text>
                    {(analysisResult.survey_risk_profile.intervention_queue || []).map((step) => (
                      <View key={step.priority} style={styles.queueRow}>
                        <View style={styles.queueBadge}>
                          <Text style={styles.queueBadgeText}>P{step.priority}</Text>
                        </View>
                        <View style={{ flex: 1 }}>
                          <Text style={styles.queueTitle}>{step.title}</Text>
                          <Text style={styles.queueDetail}>{step.detail}</Text>
                          <Text style={styles.queueWindow}>WINDOW: {step.expected_timeline}</Text>
                        </View>
                      </View>
                    ))}
                  </View>
                )}

                {/* Molecule-First Care Schedule */}
                <Text style={[styles.fieldLabel, { marginTop: 16 }]}>MOLECULE-FIRST CARE SCHEDULE</Text>
                {(analysisResult.active_ingredients || []).map((item, idx) => (
                  <View key={idx} style={styles.moleculeCard}>
                    <View style={styles.cardTitleRow}>
                      <Text style={styles.moleculeTitle}>{item.molecule}</Text>
                    </View>
                    <Text style={styles.moleculeCadence}>{item.frequency}</Text>
                    <Text style={styles.moleculeMech}>{item.mechanism}</Text>
                  </View>
                ))}

                <Text style={styles.disclaimerText}>
                  {analysisResult.disclaimer || UNIVERSAL_DISCLAIMER}
                </Text>

                <TouchableOpacity style={styles.secondaryButton} onPress={() => setCurrentStep(1)}>
                  <Text style={styles.secondaryButtonText}>NEW SCALP TELEMETRY SCAN</Text>
                </TouchableOpacity>
              </View>
            )}
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#06080a",
  },
  header: {
    paddingHorizontal: 18,
    paddingVertical: 14,
    backgroundColor: "#090c10",
    borderBottomWidth: 1,
    borderBottomColor: "#1b222c",
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  brandRow: {
    flexDirection: "row",
    alignItems: "center",
  },
  logoMark: {
    width: 34,
    height: 34,
    backgroundColor: "#00d2be",
    borderRadius: 3,
    alignItems: "center",
    justifyContent: "center",
    marginRight: 10,
  },
  logoMarkText: {
    color: "#031010",
    fontWeight: "900",
    fontSize: 17,
  },
  title: {
    fontSize: 18,
    fontWeight: "900",
    color: "#ffffff",
    letterSpacing: 1.4,
  },
  subtitle: {
    fontSize: 9,
    fontWeight: "800",
    color: "#00d2be",
    letterSpacing: 2,
    marginTop: 1,
  },
  demoChip: {
    borderWidth: 1,
    borderColor: "rgba(0,210,190,0.45)",
    backgroundColor: "rgba(0,210,190,0.10)",
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 3,
  },
  demoChipText: {
    color: "#00d2be",
    fontSize: 11,
    fontWeight: "900",
    letterSpacing: 1,
  },
  stepBar: {
    flexDirection: "row",
    backgroundColor: "#090c10",
    borderBottomWidth: 1,
    borderBottomColor: "#1b222c",
  },
  stepItem: {
    flex: 1,
    paddingVertical: 12,
    alignItems: "center",
    borderBottomWidth: 2,
    borderBottomColor: "transparent",
  },
  stepItemActive: {
    borderBottomColor: "#00d2be",
  },
  stepText: {
    fontSize: 11,
    color: "#5b6675",
    fontWeight: "800",
    letterSpacing: 1.2,
  },
  stepTextActive: {
    color: "#00d2be",
  },
  scrollContent: {
    padding: 16,
    paddingBottom: 40,
  },
  heroStrip: {
    marginBottom: 16,
    paddingVertical: 8,
  },
  heroOverline: {
    fontSize: 10,
    fontWeight: "800",
    color: "#00d2be",
    letterSpacing: 2.2,
    marginBottom: 6,
  },
  heroHeadline: {
    fontSize: 26,
    fontWeight: "900",
    color: "#ffffff",
    letterSpacing: -0.5,
  },
  tealRule: {
    width: 56,
    height: 3,
    backgroundColor: "#00d2be",
    marginVertical: 10,
  },
  heroSub: {
    fontSize: 12,
    color: "#94a0b0",
    lineHeight: 18,
  },
  card: {
    backgroundColor: "#0e1217",
    borderRadius: 4,
    padding: 16,
    borderWidth: 1,
    borderColor: "#1b222c",
    marginBottom: 16,
  },
  cardTitleRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 12,
  },
  cardSectionTag: {
    fontSize: 11,
    fontWeight: "900",
    color: "#00d2be",
    letterSpacing: 1.6,
  },
  cardStepBadge: {
    fontSize: 10,
    fontWeight: "800",
    color: "#c8cfd8",
    backgroundColor: "#151b24",
    borderWidth: 1,
    borderColor: "#26303d",
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 2,
    letterSpacing: 0.8,
  },
  cardDescription: {
    fontSize: 12,
    color: "#94a0b0",
    lineHeight: 18,
    marginBottom: 14,
  },
  protocolGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "space-between",
    marginBottom: 12,
  },
  protocolTile: {
    width: "48.5%",
    backgroundColor: "#090c10",
    borderWidth: 1,
    borderColor: "#1b222c",
    padding: 10,
    borderRadius: 3,
    marginBottom: 8,
  },
  protocolNum: {
    fontSize: 10,
    fontWeight: "900",
    color: "#00d2be",
    marginBottom: 2,
  },
  protocolHead: {
    fontSize: 12,
    fontWeight: "800",
    color: "#f3f5f8",
  },
  protocolSub: {
    fontSize: 11,
    color: "#94a0b0",
    marginTop: 2,
  },
  previewFrame: {
    borderWidth: 1,
    borderColor: "rgba(0,210,190,0.45)",
    backgroundColor: "#090c10",
    padding: 8,
    borderRadius: 3,
    marginBottom: 12,
    alignItems: "center",
  },
  previewImage: {
    width: "100%",
    height: 190,
    borderRadius: 2,
  },
  previewCaption: {
    fontSize: 10,
    fontWeight: "800",
    color: "#00d2be",
    letterSpacing: 1.5,
    marginTop: 8,
  },
  dualBtnRow: {
    flexDirection: "row",
    marginTop: 4,
  },
  outlineButton: {
    borderWidth: 1,
    borderColor: "#00d2be",
    backgroundColor: "rgba(0,210,190,0.08)",
    paddingVertical: 12,
    borderRadius: 3,
    alignItems: "center",
  },
  outlineButtonText: {
    color: "#00d2be",
    fontWeight: "900",
    fontSize: 12,
    letterSpacing: 1,
  },
  primaryButton: {
    backgroundColor: "#00d2be",
    paddingVertical: 14,
    borderRadius: 3,
    alignItems: "center",
    marginTop: 14,
  },
  primaryButtonText: {
    color: "#031010",
    fontWeight: "900",
    fontSize: 13,
    letterSpacing: 1.2,
  },
  secondaryButton: {
    backgroundColor: "#141a22",
    borderWidth: 1,
    borderColor: "#26303d",
    paddingVertical: 12,
    borderRadius: 3,
    alignItems: "center",
    marginTop: 10,
  },
  secondaryButtonText: {
    color: "#f3f5f8",
    fontWeight: "800",
    fontSize: 12,
    letterSpacing: 1,
  },
  input: {
    backgroundColor: "#090c10",
    borderWidth: 1,
    borderColor: "#26303d",
    borderRadius: 3,
    paddingHorizontal: 12,
    paddingVertical: 10,
    color: "#f3f5f8",
    fontSize: 12,
    marginTop: 10,
  },
  statusText: {
    fontSize: 11,
    color: "#00d2be",
    fontWeight: "700",
    marginTop: 10,
    textAlign: "center",
    letterSpacing: 0.5,
  },
  fieldLabel: {
    fontSize: 10,
    fontWeight: "800",
    color: "#94a0b0",
    letterSpacing: 1.3,
    marginBottom: 8,
    marginTop: 6,
  },
  chipRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    marginBottom: 10,
  },
  chip: {
    backgroundColor: "#090c10",
    borderWidth: 1,
    borderColor: "#26303d",
    borderRadius: 3,
    paddingHorizontal: 11,
    paddingVertical: 7,
    marginRight: 7,
    marginBottom: 7,
  },
  chipActive: {
    backgroundColor: "rgba(0,210,190,0.14)",
    borderColor: "#00d2be",
  },
  chipText: {
    fontSize: 11,
    color: "#94a0b0",
    fontWeight: "700",
  },
  chipTextActive: {
    color: "#00d2be",
    fontWeight: "800",
  },
  scoreHero: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingVertical: 14,
    borderTopWidth: 1,
    borderBottomWidth: 1,
    borderColor: "#1b222c",
    marginBottom: 14,
  },
  scoreOverline: {
    fontSize: 9,
    fontWeight: "800",
    color: "#94a0b0",
    letterSpacing: 1.5,
  },
  scoreNumber: {
    fontSize: 44,
    fontWeight: "900",
    color: "#ffffff",
  },
  scoreDenom: {
    fontSize: 15,
    fontWeight: "700",
    color: "#5b6675",
    marginLeft: 4,
  },
  triagePill: {
    borderWidth: 1,
    borderColor: "#00d2be",
    backgroundColor: "rgba(0,210,190,0.10)",
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 3,
    maxWidth: "48%",
  },
  triagePillText: {
    color: "#00d2be",
    fontSize: 10,
    fontWeight: "900",
    letterSpacing: 0.6,
    textTransform: "uppercase",
  },
  triageBox: {
    backgroundColor: "#090c10",
    padding: 12,
    borderRadius: 3,
    borderLeftWidth: 3,
    borderLeftColor: "#00d2be",
    borderWidth: 1,
    borderColor: "#1b222c",
    marginBottom: 16,
  },
  patternOverline: {
    fontSize: 9,
    fontWeight: "800",
    color: "#00d2be",
    letterSpacing: 1.5,
  },
  conditionText: {
    fontSize: 15,
    fontWeight: "900",
    color: "#ffffff",
    marginTop: 4,
  },
  confText: {
    fontSize: 11,
    fontWeight: "700",
    color: "#c8cfd8",
    marginTop: 3,
  },
  triageReason: {
    fontSize: 12,
    color: "#94a0b0",
    marginTop: 6,
    lineHeight: 17,
  },
  subScoreGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "space-between",
    marginBottom: 10,
  },
  subScoreItem: {
    width: "48.5%",
    backgroundColor: "#090c10",
    padding: 11,
    borderRadius: 3,
    marginBottom: 9,
    borderWidth: 1,
    borderColor: "#1b222c",
  },
  subScoreName: {
    fontSize: 9,
    fontWeight: "800",
    color: "#94a0b0",
    letterSpacing: 0.8,
  },
  subScoreValRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginVertical: 5,
  },
  subScoreVal: {
    fontSize: 14,
    fontWeight: "900",
    color: "#ffffff",
  },
  subScoreGrade: {
    fontSize: 9,
    fontWeight: "900",
    letterSpacing: 0.6,
  },
  barTrack: {
    height: 4,
    backgroundColor: "#1b222c",
    borderRadius: 2,
    overflow: "hidden",
  },
  barFill: {
    height: "100%",
    borderRadius: 2,
  },
  surveyBox: {
    backgroundColor: "#090c10",
    borderWidth: 1,
    borderColor: "#1b222c",
    borderLeftWidth: 3,
    borderLeftColor: "#00d2be",
    padding: 12,
    borderRadius: 3,
    marginTop: 6,
  },
  surveySummary: {
    fontSize: 11,
    color: "#94a0b0",
    lineHeight: 16,
    marginBottom: 10,
  },
  queueRow: {
    flexDirection: "row",
    alignItems: "flex-start",
    backgroundColor: "#0e1217",
    borderWidth: 1,
    borderColor: "#1b222c",
    padding: 10,
    borderRadius: 3,
    marginBottom: 7,
  },
  queueBadge: {
    backgroundColor: "#00d2be",
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 2,
    marginRight: 9,
    marginTop: 1,
  },
  queueBadgeText: {
    color: "#031010",
    fontSize: 10,
    fontWeight: "900",
  },
  queueTitle: {
    color: "#ffffff",
    fontSize: 12,
    fontWeight: "800",
    marginBottom: 2,
  },
  queueDetail: {
    color: "#94a0b0",
    fontSize: 11,
    lineHeight: 16,
  },
  queueWindow: {
    color: "#00d2be",
    fontSize: 9,
    fontWeight: "800",
    letterSpacing: 0.8,
    marginTop: 4,
  },
  moleculeCard: {
    backgroundColor: "#090c10",
    padding: 12,
    borderRadius: 3,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: "#1b222c",
  },
  moleculeTitle: {
    fontSize: 13,
    fontWeight: "800",
    color: "#f3f5f8",
  },
  moleculeCadence: {
    fontSize: 10,
    fontWeight: "800",
    color: "#00d2be",
    letterSpacing: 0.6,
    marginBottom: 4,
    textTransform: "uppercase",
  },
  moleculeMech: {
    fontSize: 11,
    color: "#94a0b0",
    lineHeight: 16,
  },
  disclaimerText: {
    fontSize: 10,
    color: "#5b6675",
    lineHeight: 15,
    marginTop: 12,
    marginBottom: 6,
  },
});
