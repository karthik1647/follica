/**
 * Follica on-device engine: runs the trained MobileNetV4 ONNX model and the
 * image-quality gates entirely on the phone (no network).
 * Mirrors backend/app/services/quality_gate.py and classifier_service.py.
 */
import { Buffer } from "buffer";
import jpeg from "jpeg-js";
import GUIDES from "../data/offline_guides.json";

export const CLASSES = GUIDES.classes; // 10 model outputs, training order
const CONF_GATE = 0.65;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const T = { minBlur: 60, minBright: 35, maxBright: 225, minDim: 150, minBio: 10 };

/* ---------- pure-JS pixel analysis (testable in Node) ---------- */

export function qualityMetrics(rgba, w, h) {
  const n = w * h;
  const gray = new Float32Array(n);
  let sum = 0, glare = 0, bio = 0, nonBio = 0;
  for (let i = 0; i < n; i++) {
    const r = rgba[i * 4], g = rgba[i * 4 + 1], b = rgba[i * 4 + 2];
    const y = 0.299 * r + 0.587 * g + 0.114 * b;
    gray[i] = 0.2989 * r + 0.587 * g + 0.114 * b;
    sum += gray[i];
    if (Math.max(r, g, b) > 240 && Math.abs(r - g) + Math.abs(g - b) < 20) glare++;
    const cr = (r - y) * 0.713 + 128, cb = (b - y) * 0.564 + 128;
    const cd = Math.abs(r - g) + Math.abs(r - b);
    const skin = cr >= 128 && cr <= 180 && cb >= 75 && cb <= 142 && y >= 25 && y <= 240 && r >= b - 5 && cd >= 8;
    const hair = y < 85 && Math.abs(r - g) < 40 && Math.abs(g - b) < 40;
    if (skin || hair) bio++;
    if ((b > r + 35 && b > g + 15 && b > 60) || (g > r + 30 && g > b + 25 && g > 60) ||
        (r > g + 30 && b > g + 30 && b > 60)) nonBio++;
  }
  // Laplacian (4-neighbour, edge padded), global + best 8x8 patch variance
  const lap = new Float32Array(n);
  const at = (x, yy) => gray[Math.min(h - 1, Math.max(0, yy)) * w + Math.min(w - 1, Math.max(0, x))];
  let s = 0, s2 = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const v = at(x + 1, y) + at(x - 1, y) + at(x, y + 1) + at(x, y - 1) - 4 * at(x, y);
    lap[y * w + x] = v; s += v; s2 += v * v;
  }
  const globalVar = s2 / n - (s / n) * (s / n);
  const bh = Math.max(1, Math.floor(h / 8)), bw = Math.max(1, Math.floor(w / 8));
  let focal = 0;
  for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) {
    let a = 0, a2 = 0, c = 0;
    for (let y = i * bh; y < (i + 1) * bh && y < h; y++)
      for (let x = j * bw; x < (j + 1) * bw && x < w; x++) { const v = lap[y * w + x]; a += v; a2 += v * v; c++; }
    if (c) focal = Math.max(focal, a2 / c - (a / c) * (a / c));
  }
  return {
    brightness: sum / n, blur: Math.max(globalVar, focal), glarePct: (glare / n) * 100,
    bioPct: (bio / n) * 100, nonBioPct: (nonBio / n) * 100,
  };
}

export function evaluateQuality(m, origW, origH) {
  const fail = (code, reason, guidance) => ({ ok: false, code, reason, guidance, metrics: m });
  if (origW < T.minDim || origH < T.minDim)
    return fail("IMAGE_RESOLUTION_TOO_LOW", `Image is ${origW}x${origH}px; minimum is ${T.minDim}x${T.minDim}px.`, "Take a closer, higher-resolution photo.");
  if (m.brightness < T.minBright)
    return fail("IMAGE_UNDEREXPOSED", `Brightness ${m.brightness.toFixed(1)} indicates severe underexposure.`, "Retake in a well-lit room without flash.");
  if (m.brightness > T.maxBright)
    return fail("IMAGE_OVEREXPOSED", `Brightness ${m.brightness.toFixed(1)} indicates overexposure or glare.`, "Move away from harsh direct light and retake.");
  if (m.glarePct > 15)
    return fail("FLASH_GLARE_DETECTED", `Flash glare covers ${m.glarePct.toFixed(1)}% of the image (limit 15%).`, "Turn off flash/torch; use natural or ceiling light.");
  if (m.blur < T.minBlur)
    return fail("IMAGE_TOO_BLURRY", `Sharpness score ${m.blur.toFixed(1)} is below ${T.minBlur}.`, "Hold the phone steady, let it focus, and retake.");
  if (m.bioPct < T.minBio || (m.nonBioPct > 40 && m.bioPct < 20))
    return fail("NON_SCALP_IMAGE_DETECTED", `Only ${m.bioPct.toFixed(1)}% of pixels look like skin or hair.`, "Take a close-up photo of your scalp or hair parting.");
  return { ok: true, metrics: m };
}

const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));

/** Simplified pixel-derived sub-scores (0-25 each, higher = healthier). */
export function subScores(rgba, w, h) {
  const n = w * h;
  let skinN = 0, redSum = 0, flake = 0, shine = 0, hair = 0;
  for (let i = 0; i < n; i++) {
    const r = rgba[i * 4], g = rgba[i * 4 + 1], b = rgba[i * 4 + 2];
    const y = 0.299 * r + 0.587 * g + 0.114 * b;
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    const sat = mx ? (mx - mn) / mx : 0;
    if (y < 85) hair++;
    else if (mx <= 240) {
      skinN++; redSum += r - (g + b) / 2;
      if (y > 175 && sat < 0.18) flake++;
      else if (y > 200) shine++;
    }
  }
  const sk = Math.max(1, skinN);
  const r1 = (v) => Math.round(v * 10) / 10;
  const redness = r1(25 * (1 - clamp((redSum / sk - 5) / 45)));
  const flakeS = r1(25 * (1 - clamp(flake / sk / 0.15)));
  const sebum = r1(25 * (1 - clamp(shine / sk / 0.2)));
  const density = r1(25 * clamp(0.25 + (hair / n) * 1.2 + 0.0));
  return { flake_score: flakeS, redness_score: redness, sebum_score: sebum, density_score: Math.min(25, density) };
}

export function softmax(logits) {
  const mx = Math.max(...logits);
  const e = logits.map((v) => Math.exp(v - mx));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export function decide(logits, intake) {
  let probs = softmax(logits);
  const order = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
  let conf = order[0][0];
  const dom = conf / Math.max(order[1][0], 1e-6);
  if (conf >= 0.5 && conf < CONF_GATE && dom >= 4.5) conf = Math.max(...softmax(logits.map((v) => v * 1.35)));
  const healthy = !intake.itching && !intake.burning_or_pain && intake.flake_type === "none" &&
    intake.hair_pull_test === "negative_under_3" &&
    (intake.primary_concern_zone === "routine_healthy_check" ||
      (intake.shedding_pattern === "stable_no_shedding" && !intake.family_history_hair_loss && !(intake.duration_months > 0)));
  const top3 = order.slice(0, 3).map(([p, i]) => ({ condition: CLASSES[i], probability: Math.round(p * 1000) / 1000 }));
  if (healthy && conf < CONF_GATE) return { condition: GUIDES.normal, conf: 0.94, gated: false, top3 };
  if (conf < CONF_GATE) return { condition: GUIDES.indeterminate, conf: Math.round(conf * 1000) / 1000, gated: true, top3 };
  let cond = CLASSES[order[0][1]];
  const pain = intake.burning_or_pain && /bump|pustule|painful/.test(intake.sensation || "");
  if (/Androgenetic|Telogen/.test(cond)) {
    if (pain) cond = GUIDES.folliculitis;
    else if (intake.flake_type === "plaque_crusty") cond = GUIDES.psoriasis;
    else if (intake.flake_type === "greasy_yellow_thick" && intake.itching) cond = GUIDES.seborrheic;
  }
  return { condition: cond, conf: Math.round(conf * 1000) / 1000, gated: false, top3 };
}

export function buildResult(dec, scores, quality, intake, factorCount) {
  const guide = GUIDES.guides[dec.condition] || GUIDES.guides[GUIDES.normal];
  const level = GUIDES.levels[dec.condition] || GUIDES.levels[GUIDES.indeterminate];
  const shs = Math.round((scores.flake_score + scores.redness_score + scores.sebum_score + scores.density_score) * 10) / 10;
  const risk = Math.min(100, factorCount * 12 + (intake.family_history_hair_loss ? 8 : 0));
  return {
    suggested_condition: dec.condition,
    confidence_score: dec.conf,
    confidence_gated: dec.gated,
    scoring_status: "active",
    scalp_health_score: shs,
    triage_level: level,
    triage_reason: dec.gated
      ? "On-device model confidence is below the 0.65 gate; pattern is indeterminate. Retake in better light or consult a clinician."
      : `On-device MobileNetV4 prediction (${Math.round(dec.conf * 100)}% confidence) from your photo, passed all image quality checks.`,
    sub_scores: scores,
    active_ingredients: (guide.ingredients || []).map((i) => ({ molecule: i.name, frequency: i.frequency, mechanism: i.purpose })),
    lifestyle_advice: guide.lifestyle || [],
    clinical_correlations: [
      { factor: "On-device model, top predictions", impact_level: "low",
        clinical_note: dec.top3.map((t) => `${t.condition} ${(t.probability * 100).toFixed(0)}%`).join("; ") },
      { factor: "On-device image quality", impact_level: "low",
        clinical_note: `Sharpness ${quality.blur.toFixed(0)}, brightness ${quality.brightness.toFixed(0)}, glare ${quality.glarePct.toFixed(1)}%, skin/hair ${quality.bioPct.toFixed(0)}%. Sub-scores are simplified pixel estimates.` },
    ],
    survey_risk_profile: {
      condition: dec.condition, risk_score: risk,
      risk_band: risk >= 60 ? "high" : risk >= 30 ? "moderate" : "low",
      modifiable_count: factorCount,
      summary: "Estimated on-device from your intake. Connect to the server for the full risk engine.",
      intervention_queue: [],
    },
    disclaimer: GUIDES.disclaimer,
    analysis_timestamp: new Date().toISOString(),
    is_offline_eval: true,
  };
}

/* ---------- React Native glue (device only) ---------- */

let sessionPromise = null;

async function loadSession() {
  const { InferenceSession } = require("onnxruntime-react-native");
  const FileSystem = require("expo-file-system");
  const dest = FileSystem.documentDirectory + "follica_mobilenetv4.onnx";
  const info = await FileSystem.getInfoAsync(dest);
  if (!info.exists || info.size < 1000000) {
    await FileSystem.copyAsync({ from: "asset:///follica_mobilenetv4.onnx", to: dest });
  }
  return InferenceSession.create(dest.replace("file://", ""));
}

export function getSession() {
  if (!sessionPromise) sessionPromise = loadSession().catch((e) => { sessionPromise = null; throw e; });
  return sessionPromise;
}

async function decodeResized(uri, size) {
  const IM = require("expo-image-manipulator");
  const out = await IM.manipulateAsync(uri, [{ resize: size }], { compress: 1, format: IM.SaveFormat.JPEG, base64: true });
  const img = jpeg.decode(Buffer.from(out.base64, "base64"), { useTArray: true, formatAsRGBA: true });
  return { data: img.data, w: img.width, h: img.height };
}

export async function runOnDevice(uri, origW, origH, intake, factorCount) {
  const view = await decodeResized(uri, origW >= origH ? { width: 640 } : { height: 640 });
  const q = evaluateQuality(qualityMetrics(view.data, view.w, view.h), origW, origH);
  if (!q.ok) return { rejected: q };
  const net = await decodeResized(uri, { width: 224, height: 224 });
  const x = new Float32Array(3 * 224 * 224);
  for (let i = 0; i < 224 * 224; i++)
    for (let c = 0; c < 3; c++) x[c * 224 * 224 + i] = (net.data[i * 4 + c] / 255 - MEAN[c]) / STD[c];
  const { Tensor } = require("onnxruntime-react-native");
  const session = await getSession();
  const out = await session.run({ [session.inputNames[0]]: new Tensor("float32", x, [1, 3, 224, 224]) });
  const logits = Array.from(out[session.outputNames[0]].data);
  const dec = decide(logits, intake);
  return { result: buildResult(dec, subScores(view.data, view.w, view.h), q.metrics, intake, factorCount) };
}
