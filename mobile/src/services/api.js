/**
 * Follica Mobile API Client Service.
 * Connects the mobile frontend to the FastAPI dual-pipeline inference engine.
 */

// Configured for local Wi-Fi pairing; update if laptop IPv4 changes
export const DEFAULT_API_BASE = "http://10.146.13.169:8000/api/v1";

export async function checkSystemStatus(baseUrl = DEFAULT_API_BASE) {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 6000);
    const res = await fetch(`${baseUrl}/system/status`, { signal: controller.signal });
    clearTimeout(timeoutId);
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

  let res;
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 15000);
    res = await fetch(`${baseUrl}/analyze`, {
      method: "POST",
      headers: {
        Accept: "application/json",
      },
      body: formData,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
  } catch (netErr) {
    const isTimeout = netErr?.name === "AbortError";
    const err = new Error(
      isTimeout
        ? `Request timed out after 15s. Ensure phone and laptop are on the same Wi-Fi (${baseUrl}).`
        : `Cannot connect to clinical backend (${baseUrl}). Make sure your phone and laptop are connected to the same Wi-Fi and verify the IP in API Config.`
    );
    err.isNetworkError = true;
    throw err;
  }

  if (!res.ok) {
    let errData = {};
    try {
      errData = await res.json();
    } catch (_) {
      const errText = await res.text().catch(() => "");
      errData = { detail: errText };
    }
    const detail = errData.detail || {};
    const errorCode = typeof detail === "object" ? detail.error_code : "";
    const rejectionReason = typeof detail === "object" ? (detail.rejection_reason || detail.error) : (typeof detail === "string" ? detail : "");
    const userGuidance = typeof detail === "object" ? detail.user_guidance : "";

    const err = new Error(rejectionReason || `Analysis failed (${res.status})`);
    err.status = res.status;
    err.errorCode = errorCode;
    err.rejectionReason = rejectionReason;
    err.userGuidance = userGuidance;
    throw err;
  }

  return await res.json();
}
