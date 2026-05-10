
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import timm
import cv2
import numpy as np
import json
from pathlib import Path
from huggingface_hub import hf_hub_download

# ── Model definition ──────────────────────────────────────────
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)

class DRModel(nn.Module):
    def __init__(self, backbone, drop=0.5):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False,
                                          num_classes=0, global_pool="")
        feat = self.backbone.num_features
        self.pool = GeM()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(drop),
            nn.Linear(256, 1),
        )
    def forward(self, x):
        f = self.backbone(x)
        p = f.unsqueeze(-1).unsqueeze(-1) if f.dim() == 2 else self.pool(f)
        return self.head(p).squeeze(-1)

# ── Load model ────────────────────────────────────────────────
REPO_ID = "its-karthick1/dr-grading-effv2l"
DEVICE  = torch.device("cpu")

print("Loading model from HF Hub...")
ckpt_path = hf_hub_download(REPO_ID, "models/effv2l_s42_best.pt")
ckpt      = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model     = DRModel(ckpt["backbone"])
model.load_state_dict(ckpt["model"], strict=False)
model.eval()
print("Model loaded.")

# Load thresholds
try:
    thr_path   = hf_hub_download(REPO_ID, "thresholds.json")
    THRESHOLDS = json.loads(Path(thr_path).read_text())
    if isinstance(THRESHOLDS, dict):
        THRESHOLDS = THRESHOLDS.get("thresholds", [0.5, 1.5, 2.5, 3.5])
except Exception:
    THRESHOLDS = [0.5, 1.5, 2.5, 3.5]

GRADE_LABELS = [
    "Grade 0 — No DR",
    "Grade 1 — Mild DR",
    "Grade 2 — Moderate DR",
    "Grade 3 — Severe DR",
    "Grade 4 — Proliferative DR",
]

# ── Preprocessing ─────────────────────────────────────────────
def preprocess(img_bgr, size=384):
    img = cv2.resize(img_bgr, (size, size))
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = (img - mean) / std
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor

def score_to_grade(score):
    for g, t in enumerate(THRESHOLDS):
        if score < t:
            return g
    return 4

# ── Inference ─────────────────────────────────────────────────
def predict(image):
    if image is None:
        return "Please upload a fundus image.", {}

    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    tensor  = preprocess(img_bgr)

    with torch.no_grad():
        score = model(tensor).item()

    grade = score_to_grade(score)
    label = GRADE_LABELS[grade]

    # Soft probabilities
    scores  = [score] * 5
    probs   = torch.softmax(torch.tensor(
        [-abs(score - i) for i in range(5)]
    ), dim=0).tolist()
    conf    = {GRADE_LABELS[i]: round(probs[i], 4) for i in range(5)}

    result = f"**{label}**\n\nRegression score: {score:.3f}"
    return result, conf

# ── Gradio UI ─────────────────────────────────────────────────
with gr.Blocks(title="DR Grading") as demo:
    gr.Markdown("# 🔬 Diabetic Retinopathy Grading")
    gr.Markdown("Upload a fundus photograph to get an automated DR severity grade (0–4).")

    with gr.Row():
        inp = gr.Image(type="pil", label="Fundus Image")
        with gr.Column():
            out_label = gr.Markdown(label="Prediction")
            out_conf  = gr.Label(label="Confidence per grade", num_top_classes=5)

    btn = gr.Button("Predict", variant="primary")
    btn.click(fn=predict, inputs=inp, outputs=[out_label, out_conf])

    gr.Examples(examples=[], inputs=inp)

demo.launch()
