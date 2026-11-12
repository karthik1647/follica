"""Export offline assets for the mobile app: single-file ONNX, guides JSON, golden fixtures."""
import json, os, sys, io
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "backend"))
import onnx
from app.core import constants as C
from app.core.constants import INGREDIENT_GUIDES
from app.services.classifier_service import ScalpClassifierService as S

m = onnx.load(os.path.join(ROOT, "backend/models/follica_mobilenetv4.onnx"), load_external_data=True)
out = os.path.join(ROOT, "mobile/assets/model/follica_mobilenetv4.onnx")
onnx.save_model(m, out, save_as_external_data=False)
print("onnx MB", round(os.path.getsize(out) / 1e6, 2))

L3 = [C.CONDITION_ALOPECIA_AREATA, C.CONDITION_FOLLICULITIS, C.CONDITION_SCALP_PSORIASIS,
      C.CONDITION_HEAD_LICE, C.CONDITION_LICHEN_PLANUS, C.CONDITION_TINEA_CAPITIS]
L2 = [C.CONDITION_ANDROGENETIC_ALOPECIA, C.CONDITION_CONTACT_DERMATITIS, C.CONDITION_TELOGEN_EFFLUVIUM,
      C.CONDITION_SEBORRHEIC_DERMATITIS, C.CONDITION_UNVERIFIED_MONITORED]
levels = {c: C.TRIAGE_LEVEL_3_ESCALATION for c in L3}
levels.update({c: C.TRIAGE_LEVEL_2_MONITORED for c in L2})
levels[C.CONDITION_NORMAL] = C.TRIAGE_LEVEL_1_MINOR
levels[C.CONDITION_DRY_SCALP] = C.TRIAGE_LEVEL_1_MINOR
data = {
    "classes": S.MODEL_CLASSES, "normal": C.CONDITION_NORMAL, "indeterminate": C.CONDITION_UNVERIFIED_MONITORED,
    "folliculitis": C.CONDITION_FOLLICULITIS, "psoriasis": C.CONDITION_SCALP_PSORIASIS,
    "seborrheic": C.CONDITION_SEBORRHEIC_DERMATITIS, "levels": levels,
    "guides": {k: {"ingredients": v.get("ingredients", []), "lifestyle": v.get("lifestyle", [])}
               for k, v in INGREDIENT_GUIDES.items()},
    "disclaimer": C.UNIVERSAL_DISCLAIMER,
}
json.dump(data, open(os.path.join(ROOT, "mobile/src/data/offline_guides.json"), "w"))

import onnxruntime as ort
sess = ort.InferenceSession(out)
gold = {}
d = os.path.join(ROOT, "training/demo_samples")
for f in sorted(os.listdir(d)):
    img = Image.open(os.path.join(d, f)).convert("RGB").resize((224, 224))
    x = (np.array(img, np.float32) / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x = np.expand_dims(np.transpose(x.astype(np.float32), (2, 0, 1)), 0)
    lg = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    gold[f] = int(np.argmax(lg))
json.dump(gold, open(os.path.join(ROOT, "mobile/scripts/golden.json"), "w"))
print(gold)
