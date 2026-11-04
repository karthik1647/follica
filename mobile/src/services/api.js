/**
 * Follica Mobile API Client Service.
 * Connects the mobile frontend to the FastAPI dual-pipeline inference engine.
 */

// Default to local development host; replace with tunnel or cloud URL for remote testing
export const DEFAULT_API_BASE = "http://10.0.2.2:8000/api/v1"; // 10.0.2.2 maps to host localhost in Android Emulator

export async function checkSystemStatus(baseUrl = DEFAULT_API_BASE) {
  try {
    const res = await fetch(`${baseUrl}/system/status`);
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  } catch (error) {
    console.error("Health check error:", error);
    throw error;
  }
}

export async function submitScalpAnalysis(imageUri, questionnaireData, baseUrl = DEFAULT_API_BASE) {
  const formData = new FormData();

  formData.append("image", {
    uri: imageUri,
    type: "image/jpeg",
    name: "scalp_scan.jpg",
  });

  // Append symptom and lifestyle variables
  Object.keys(questionnaireData).forEach((key) => {
    const val = questionnaireData[key];
    if (val !== undefined && val !== null) {
      formData.append(key, String(val));
    }
  });

  const res = await fetch(`${baseUrl}/analyze`, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "multipart/form-data",
    },
    body: formData,
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Analysis failed (${res.status}): ${errText}`);
  }

  return await res.json();
}

export async function recordConsent(userId, retentionDays = 30, baseUrl = DEFAULT_API_BASE) {
  const res = await fetch(`${baseUrl}/consent`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      user_id: userId,
      biometric_image_consent: true,
      research_opt_in: false,
      data_retention_days: retentionDays,
    }),
  });

  if (!res.ok) throw new Error("Consent recording failed");
  return await res.json();
}
