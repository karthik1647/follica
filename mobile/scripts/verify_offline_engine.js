// Node verification of the exact on-device pipeline (same engine code, onnxruntime-node).
const fs = require("fs"), path = require("path"), jpeg = require("jpeg-js");
const _r = require("@babel/register"); (_r.default || _r.register || _r)({ presets: [], plugins: ["@babel/plugin-transform-modules-commonjs"], only: [/offlineEngine/] });
const E = require("../src/services/offlineEngine.js");
const ort = require("onnxruntime-node");
const gold = require("./golden.json");

function resize(img, W, H) { // bilinear
  const out = new Uint8Array(W * H * 4);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const fx = ((x + 0.5) * img.width) / W - 0.5, fy = ((y + 0.5) * img.height) / H - 0.5;
    const x0 = Math.max(0, Math.floor(fx)), y0 = Math.max(0, Math.floor(fy));
    const x1 = Math.min(img.width - 1, x0 + 1), y1 = Math.min(img.height - 1, y0 + 1);
    const ax = Math.max(0, fx - x0), ay = Math.max(0, fy - y0);
    for (let c = 0; c < 4; c++) {
      const p = (xx, yy) => img.data[(yy * img.width + xx) * 4 + c];
      out[(y * W + x) * 4 + c] = p(x0, y0) * (1 - ax) * (1 - ay) + p(x1, y0) * ax * (1 - ay) + p(x0, y1) * (1 - ax) * ay + p(x1, y1) * ax * ay;
    }
  }
  return out;
}
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
const intake = { primary_concern_zone: "crown_vertex", flake_type: "none", hair_pull_test: "negative_under_3", sensation: "painless" };

(async () => {
  const s = await ort.InferenceSession.create(path.join(__dirname, "../assets/model/follica_mobilenetv4.onnx"));
  const dir = path.join(__dirname, "../../training/demo_samples");
  let ok = 0, n = 0;
  for (const f of fs.readdirSync(dir)) {
    const img = jpeg.decode(fs.readFileSync(path.join(dir, f)), { useTArray: true, formatAsRGBA: true });
    const d = resize(img, 224, 224), x = new Float32Array(3 * 224 * 224);
    for (let i = 0; i < 224 * 224; i++) for (let c = 0; c < 3; c++) x[c * 224 * 224 + i] = (d[i * 4 + c] / 255 - MEAN[c]) / STD[c];
    const o = await s.run({ [s.inputNames[0]]: new ort.Tensor("float32", x, [1, 3, 224, 224]) });
    const lg = Array.from(o[s.outputNames[0]].data), dec = E.decide(lg, intake);
    const idx = lg.indexOf(Math.max(...lg)); n++; if (idx === gold[f]) ok++;
    const q = E.evaluateQuality(E.qualityMetrics(img.data, img.width, img.height), img.width, img.height);
    console.log(f.padEnd(34), "-> raw:", E.CLASSES[idx].padEnd(22), "conf", dec.conf, "| gate:", q.ok ? "PASS" : q.code);
  }
  console.log(`MODEL AGREEMENT WITH PYTHON BACKEND: ${ok}/${n}`);
  // synthetic quality-gate checks
  const mk = (fn) => { const w = 300, h = 300, d = new Uint8Array(w * h * 4); for (let i = 0; i < w * h; i++) { const [r, g, b] = fn(i); d[i * 4] = r; d[i * 4 + 1] = g; d[i * 4 + 2] = b; d[i * 4 + 3] = 255; } return E.evaluateQuality(E.qualityMetrics(d, w, h), w, h); };
  console.log("black image   ->", mk(() => [5, 5, 5]).code);
  console.log("white image   ->", mk(() => [250, 250, 250]).code);
  console.log("blue (non-scalp) ->", mk(() => [40, 90, 200]).code);
  console.log("flat skin (blurry) ->", mk(() => [200, 150, 130]).code);
})();
