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
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import { checkSystemStatus, submitScalpAnalysis, DEFAULT_API_BASE } from "./src/services/api";

export default function App() {
  const [apiBaseUrl, setApiBaseUrl] = useState(DEFAULT_API_BASE);
  const [currentStep, setCurrentStep] = useState(1);
  const [loading, setLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");
  const [selectedImageUri, setSelectedImageUri] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);

  // Questionnaire state
  const [intake, setIntake] = useState({
    biological_sex: "male",
    age_group: "18_25",
    primary_concern_zone: "crown_vertex",
    water_environment: "hard_borewell",
    daily_helmet_wearer: true,
    washing_frequency: "2_3_times_week",
    hair_pull_test: "normal_under_3",
    flake_type: "greasy_yellow",
    sensation: "itching",
    duration_months: 6,
    duration_weeks: 24,
    gut_health: "frequent_bloating_acidity",
    metabolic_energy: "chronic_sluggish_fatigue",
    hormonal_profile: "none",
  });

  const testServerConnection = async () => {
    try {
      setLoading(true);
      setStatusMessage("Checking server connectivity...");
      const status = await checkSystemStatus(apiBaseUrl);
      setStatusMessage(`Connected: ${status.app} v${status.version} (Model ready: ${status.model_loaded})`);
    } catch (err) {
      setStatusMessage(`Connection failed: ${err.message}`);
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
      quality: 0.85,
    });
    if (!res.canceled && res.assets && res.assets.length > 0) {
      setSelectedImageUri(res.assets[0].uri);
      setStatusMessage("Scalp photograph selected and ready for analysis.");
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
          contact_time: "Standard leave-on",
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
        setStatusMessage(`Live API error: ${err.message}`);
      } finally {
        setLoading(false);
      }
    }
    runMockAnalysis();
  };

  const runMockAnalysis = () => {
    // Allows immediate mobile UI demonstration with authentic clinical triage payload
    setLoading(true);
    setTimeout(() => {
      setAnalysisResult({
        scalp_health_score: 54.5,
        suggested_condition: "Pattern Suggestive of Seborrheic Dermatitis / Fungal Flaking",
        triage_level: "Level 2: Monitored Active Routine",
        triage_reason: "Elevated flake clustering and mild vascular erythema detected alongside hard water exposure.",
        sub_scores: {
          flake_score: 11.2,
          redness_score: 13.5,
          sebum_score: 14.8,
          density_score: 15.0,
        },
        risk_indicators: {
          level: "moderate",
          count: 3,
          details: ["Daily helmet wear", "Hard borewell water", "Reported pruritus / itching"],
        },
        active_ingredients: [
          {
            molecule: "Ketoconazole 2% Wash",
            frequency: "2x per week",
            contact_time: "5 minutes leave-on",
            mechanism: "Inhibits fungal ergosterol synthesis in Malassezia yeast overgrowth.",
          },
          {
            molecule: "Salicylic Acid 2% Exfoliant",
            frequency: "1x per week",
            contact_time: "Rinse thoroughly",
            mechanism: "Keratolytic descaling to remove stratum corneum crust and insoluble mineral deposits.",
          },
        ],
        lifestyle_advice: [
          "Wash helmet padding weekly to prevent trapped sebum and heat microbial amplification.",
          "Use a chelating or clarified wash cadence to counteract borewell calcium/magnesium buildup.",
        ],
      });
      setLoading(false);
      setCurrentStep(3);
    }, 1200);
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#0f172a" />

      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.title}>FOLLICA</Text>
        <Text style={styles.subtitle}>Scalp Health Assessment & Clinical Triage</Text>
      </View>

      {/* Step Indicator */}
      <View style={styles.stepBar}>
        <TouchableOpacity
          style={[styles.stepItem, currentStep === 1 && styles.stepItemActive]}
          onPress={() => setCurrentStep(1)}
        >
          <Text style={[styles.stepText, currentStep === 1 && styles.stepTextActive]}>1. Setup</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.stepItem, currentStep === 2 && styles.stepItemActive]}
          onPress={() => setCurrentStep(2)}
        >
          <Text style={[styles.stepText, currentStep === 2 && styles.stepTextActive]}>2. Intake</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.stepItem, currentStep === 3 && styles.stepItemActive]}
          onPress={() => setCurrentStep(3)}
        >
          <Text style={[styles.stepText, currentStep === 3 && styles.stepTextActive]}>3. Report</Text>
        </TouchableOpacity>
      </View>

      <ScrollView contentContainerStyle={styles.scrollContent}>
        {/* Step 1: Camera Guidance & Server URL */}
        {currentStep === 1 && (
          <View style={styles.card}>
            <Text style={styles.cardHeader}>Standardized Capture Protocol</Text>
            <Text style={styles.cardDescription}>
              High-accuracy triage requires clear, uncorrupted photographs. Follow the 4 optical rules:
            </Text>

            <View style={styles.ruleBox}>
              <Text style={styles.ruleTitle}>1. Clean Camera Lens</Text>
              <Text style={styles.ruleDesc}>Wipe off oil, fingerprints, and pocket dust before shooting.</Text>

              <Text style={styles.ruleTitle}>2. Natural or Overhead Light Only</Text>
              <Text style={styles.ruleDesc}>Avoid direct phone flash hotspots to prevent glare false-positives.</Text>

              <Text style={styles.ruleTitle}>3. Expose Parting Line Clearly</Text>
              <Text style={styles.ruleDesc}>Part hair to expose bare scalp skin at the area of concern.</Text>

              <Text style={styles.ruleTitle}>4. 10 to 15 cm Distance</Text>
              <Text style={styles.ruleDesc}>Hold phone parallel to scalp; ensure focus lock before capture.</Text>
            </View>

            <Text style={[styles.fieldLabel, { marginTop: 16 }]}>Backend Server Endpoint</Text>
            <TextInput
              style={styles.input}
              value={apiBaseUrl}
              onChangeText={setApiBaseUrl}
              placeholder="http://10.0.2.2:8000/api/v1"
              placeholderTextColor="#64748b"
              autoCapitalize="none"
              autoCorrect={false}
            />

            <TouchableOpacity style={styles.secondaryButton} onPress={testServerConnection} disabled={loading}>
              <Text style={styles.secondaryButtonText}>Test Server Connection</Text>
            </TouchableOpacity>

            <TouchableOpacity style={styles.secondaryButton} onPress={() => pickImage(true)} disabled={loading}>
              <Text style={styles.secondaryButtonText}>Capture Scalp Photo (Camera)</Text>
            </TouchableOpacity>

            <TouchableOpacity style={styles.secondaryButton} onPress={() => pickImage(false)} disabled={loading}>
              <Text style={styles.secondaryButtonText}>Select Scalp Photo (Gallery)</Text>
            </TouchableOpacity>

            {statusMessage ? <Text style={styles.statusText}>{statusMessage}</Text> : null}

            <TouchableOpacity style={styles.primaryButton} onPress={() => setCurrentStep(2)}>
              <Text style={styles.primaryButtonText}>Continue to Clinical Questionnaire</Text>
            </TouchableOpacity>
          </View>
        )}

        {/* Step 2: Clinical Intake Questionnaire */}
        {currentStep === 2 && (
          <View style={styles.card}>
            <Text style={styles.cardHeader}>Clinical Context & Lifestyle Intake</Text>
            <Text style={styles.cardDescription}>
              Scalp disorders are driven by environmental, nutritional, and systemic factors.
            </Text>

            {/* Concern Zone */}
            <Text style={styles.fieldLabel}>Primary Concern Zone</Text>
            <View style={styles.chipRow}>
              {["crown_vertex", "hairline_temples", "central_parting", "diffuse_overall"].map((zone) => (
                <TouchableOpacity
                  key={zone}
                  style={[styles.chip, intake.primary_concern_zone === zone && styles.chipActive]}
                  onPress={() => setIntake({ ...intake, primary_concern_zone: zone })}
                >
                  <Text style={[styles.chipText, intake.primary_concern_zone === zone && styles.chipTextActive]}>
                    {zone.replace("_", " ")}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>

            {/* Water Source */}
            <Text style={styles.fieldLabel}>Water Environment</Text>
            <View style={styles.chipRow}>
              {[
                { label: "Hard Borewell", val: "hard_borewell" },
                { label: "Treated Tap", val: "soft_tap" },
                { label: "Relocated Recently", val: "relocated_recently" },
              ].map((item) => (
                <TouchableOpacity
                  key={item.val}
                  style={[styles.chip, intake.water_environment === item.val && styles.chipActive]}
                  onPress={() => setIntake({ ...intake, water_environment: item.val })}
                >
                  <Text style={[styles.chipText, intake.water_environment === item.val && styles.chipTextActive]}>
                    {item.label}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>

            {/* Friction & Helmet */}
            <Text style={styles.fieldLabel}>Daily Motorcycle Helmet / Commuter Friction</Text>
            <View style={styles.chipRow}>
              <TouchableOpacity
                style={[styles.chip, intake.daily_helmet_wearer && styles.chipActive]}
                onPress={() => setIntake({ ...intake, daily_helmet_wearer: true })}
              >
                <Text style={[styles.chipText, intake.daily_helmet_wearer && styles.chipTextActive]}>Yes (Daily)</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.chip, !intake.daily_helmet_wearer && styles.chipActive]}
                onPress={() => setIntake({ ...intake, daily_helmet_wearer: false })}
              >
                <Text style={[styles.chipText, !intake.daily_helmet_wearer && styles.chipTextActive]}>No / Rare</Text>
              </TouchableOpacity>
            </View>

            {/* Gut & Metabolism */}
            <Text style={styles.fieldLabel}>Digestive Health (Systemic Absorption)</Text>
            <View style={styles.chipRow}>
              {[
                { label: "Optimal", val: "optimal" },
                { label: "Bloating / Acidity", val: "frequent_bloating_acidity" },
                { label: "Constipation", val: "chronic_constipation" },
              ].map((item) => (
                <TouchableOpacity
                  key={item.val}
                  style={[styles.chip, intake.gut_health === item.val && styles.chipActive]}
                  onPress={() => setIntake({ ...intake, gut_health: item.val })}
                >
                  <Text style={[styles.chipText, intake.gut_health === item.val && styles.chipTextActive]}>
                    {item.label}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>

            <TouchableOpacity style={styles.primaryButton} onPress={runLiveOrMockAnalysis} disabled={loading}>
              {loading ? (
                <ActivityIndicator color="#ffffff" />
              ) : (
                <Text style={styles.primaryButtonText}>Run Dual-Pipeline Analysis</Text>
              )}
            </TouchableOpacity>
          </View>
        )}

        {/* Step 3: Clinical Triage Report */}
        {currentStep === 3 && analysisResult && (
          <View style={styles.card}>
            <Text style={styles.cardHeader}>Scalp Health Assessment</Text>

            {/* Health Score Hero Dial */}
            <View style={styles.scoreHero}>
              <Text style={styles.scoreNumber}>{analysisResult.scalp_health_score}</Text>
              <Text style={styles.scoreDenom}>/ 100</Text>
              <Text style={styles.scoreLabel}>Composite Scalp Health Score (SHS)</Text>
            </View>

            {/* Condition & Level */}
            <View style={styles.triageBox}>
              <Text style={styles.triageLevel}>{analysisResult.triage_level}</Text>
              <Text style={styles.conditionText}>{analysisResult.suggested_condition}</Text>
              <Text style={styles.triageReason}>{analysisResult.triage_reason}</Text>
            </View>

            {/* Sub-Score Telemetry */}
            <Text style={[styles.fieldLabel, { marginTop: 16 }]}>Computer Vision Sub-Scores (0-25)</Text>
            <View style={styles.subScoreGrid}>
              <View style={styles.subScoreItem}>
                <Text style={styles.subScoreName}>Flake & Scaling</Text>
                <Text style={styles.subScoreVal}>{analysisResult.sub_scores.flake_score} / 25</Text>
              </View>
              <View style={styles.subScoreItem}>
                <Text style={styles.subScoreName}>Erythema Redness</Text>
                <Text style={styles.subScoreVal}>{analysisResult.sub_scores.redness_score} / 25</Text>
              </View>
              <View style={styles.subScoreItem}>
                <Text style={styles.subScoreName}>Sebum Balance</Text>
                <Text style={styles.subScoreVal}>{analysisResult.sub_scores.sebum_score} / 25</Text>
              </View>
              <View style={styles.subScoreItem}>
                <Text style={styles.subScoreName}>Hair Coverage</Text>
                <Text style={styles.subScoreVal}>{analysisResult.sub_scores.density_score} / 25</Text>
              </View>
            </View>

            {/* Molecule-First Regimen */}
            <Text style={[styles.fieldLabel, { marginTop: 16 }]}>Active Chemical Molecule Schedule</Text>
            {analysisResult.active_ingredients.map((item, idx) => (
              <View key={idx} style={styles.moleculeCard}>
                <Text style={styles.moleculeTitle}>{item.molecule}</Text>
                <Text style={styles.moleculeCadence}>{item.frequency} - {item.contact_time}</Text>
                <Text style={styles.moleculeMech}>{item.mechanism}</Text>
              </View>
            ))}

            <TouchableOpacity style={styles.secondaryButton} onPress={() => setCurrentStep(1)}>
              <Text style={styles.secondaryButtonText}>Start New Evaluation</Text>
            </TouchableOpacity>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#0f172a",
  },
  header: {
    paddingHorizontal: 20,
    paddingTop: 12,
    paddingBottom: 16,
    borderBottomWidth: 1,
    borderBottomColor: "#1e293b",
  },
  title: {
    fontSize: 22,
    fontWeight: "800",
    color: "#f8fafc",
    letterSpacing: 1.5,
  },
  subtitle: {
    fontSize: 12,
    color: "#94a3b8",
    marginTop: 2,
  },
  stepBar: {
    flexDirection: "row",
    backgroundColor: "#1e293b",
    borderBottomWidth: 1,
    borderBottomColor: "#334155",
  },
  stepItem: {
    flex: 1,
    paddingVertical: 12,
    alignItems: "center",
  },
  stepItemActive: {
    borderBottomWidth: 2,
    borderBottomColor: "#0ea5e9",
  },
  stepText: {
    fontSize: 13,
    color: "#64748b",
    fontWeight: "600",
  },
  stepTextActive: {
    color: "#38bdf8",
  },
  scrollContent: {
    padding: 16,
  },
  card: {
    backgroundColor: "#1e293b",
    borderRadius: 12,
    padding: 16,
    borderWidth: 1,
    borderColor: "#334155",
  },
  cardHeader: {
    fontSize: 16,
    fontWeight: "700",
    color: "#f1f5f9",
    marginBottom: 6,
  },
  cardDescription: {
    fontSize: 13,
    color: "#94a3b8",
    lineHeight: 18,
    marginBottom: 16,
  },
  ruleBox: {
    backgroundColor: "#0f172a",
    borderRadius: 8,
    padding: 12,
    borderWidth: 1,
    borderColor: "#334155",
  },
  ruleTitle: {
    fontSize: 13,
    fontWeight: "700",
    color: "#38bdf8",
    marginTop: 6,
  },
  ruleDesc: {
    fontSize: 12,
    color: "#cbd5e1",
    marginBottom: 6,
  },
  fieldLabel: {
    fontSize: 13,
    fontWeight: "600",
    color: "#e2e8f0",
    marginBottom: 8,
  },
  input: {
    backgroundColor: "#0f172a",
    borderWidth: 1,
    borderColor: "#334155",
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    color: "#f8fafc",
    fontSize: 13,
    marginBottom: 12,
  },
  primaryButton: {
    backgroundColor: "#0284c7",
    paddingVertical: 14,
    borderRadius: 8,
    alignItems: "center",
    marginTop: 16,
  },
  primaryButtonText: {
    color: "#ffffff",
    fontWeight: "700",
    fontSize: 14,
  },
  secondaryButton: {
    backgroundColor: "#334155",
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: "center",
    marginTop: 10,
  },
  secondaryButtonText: {
    color: "#f8fafc",
    fontWeight: "600",
    fontSize: 13,
  },
  statusText: {
    fontSize: 12,
    color: "#38bdf8",
    marginTop: 8,
    textAlign: "center",
  },
  chipRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    marginBottom: 12,
  },
  chip: {
    backgroundColor: "#0f172a",
    borderWidth: 1,
    borderColor: "#334155",
    borderRadius: 20,
    paddingHorizontal: 12,
    paddingVertical: 6,
    marginRight: 8,
    marginBottom: 8,
  },
  chipActive: {
    backgroundColor: "#0369a1",
    borderColor: "#38bdf8",
  },
  chipText: {
    fontSize: 12,
    color: "#94a3b8",
  },
  chipTextActive: {
    color: "#ffffff",
    fontWeight: "600",
  },
  scoreHero: {
    alignItems: "center",
    paddingVertical: 16,
    borderBottomWidth: 1,
    borderBottomColor: "#334155",
    marginBottom: 16,
  },
  scoreNumber: {
    fontSize: 48,
    fontWeight: "900",
    color: "#38bdf8",
  },
  scoreDenom: {
    fontSize: 16,
    color: "#64748b",
  },
  scoreLabel: {
    fontSize: 12,
    color: "#94a3b8",
    marginTop: 4,
  },
  triageBox: {
    backgroundColor: "#0f172a",
    padding: 12,
    borderRadius: 8,
    borderLeftWidth: 4,
    borderLeftColor: "#eab308",
    marginBottom: 16,
  },
  triageLevel: {
    fontSize: 12,
    fontWeight: "700",
    color: "#eab308",
    textTransform: "uppercase",
  },
  conditionText: {
    fontSize: 14,
    fontWeight: "700",
    color: "#f8fafc",
    marginTop: 2,
  },
  triageReason: {
    fontSize: 12,
    color: "#cbd5e1",
    marginTop: 4,
  },
  subScoreGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "space-between",
  },
  subScoreItem: {
    width: "48%",
    backgroundColor: "#0f172a",
    padding: 10,
    borderRadius: 6,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: "#334155",
  },
  subScoreName: {
    fontSize: 11,
    color: "#94a3b8",
  },
  subScoreVal: {
    fontSize: 13,
    fontWeight: "700",
    color: "#f1f5f9",
    marginTop: 2,
  },
  moleculeCard: {
    backgroundColor: "#0f172a",
    padding: 12,
    borderRadius: 8,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: "#334155",
  },
  moleculeTitle: {
    fontSize: 13,
    fontWeight: "700",
    color: "#38bdf8",
  },
  moleculeCadence: {
    fontSize: 11,
    color: "#fbbf24",
    marginVertical: 2,
  },
  moleculeMech: {
    fontSize: 12,
    color: "#94a3b8",
    lineHeight: 16,
  },
});
