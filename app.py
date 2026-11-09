import os
import uuid
from datetime import datetime

import torch
import timm
from PIL import Image, ImageDraw
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from torchvision import transforms

# =========================
# PATH & FOLDERS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH   = os.path.join(BASE_DIR, "models", "best_model.pth")
CLASSES_PATH = os.path.join(BASE_DIR, "models", "classes.pt")

UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CONF_THRESHOLD = 0.80   # kalau confidence < ini -> UNCERTAIN

# =========================
# SARAN PENANGANAN (DITAMPILKAN DI WEB)
# =========================
TREATMENT_TIPS = {
    "Bercak Daun": {
        "judul": "Bercak Daun",
        "ringkas": "Umumnya terkait infeksi jamur + kelembapan. Fokus: cegah penyebaran & lindungi daun baru.",
        "langkah": [
            "Jika memungkinkan, buang/isolasi daun yang sudah parah (jangan ditinggal di sekitar tanaman).",
            "Kurangi kelembapan daun: hindari penyiraman yang membasahi daun, terutama sore/malam.",
            "Jaga sirkulasi udara (tanaman tidak terlalu rapat).",
            "Jika menyebar cepat: pertimbangkan fungisida sesuai label & anjuran penyuluh/ahli setempat."
        ],
        "pencegahan": [
            "Rotasi tanam (jangan jagung terus menerus).",
            "Kelola sisa tanaman/residu (sumber penyakit bisa bertahan).",
            "Gunakan varietas/hibrida yang lebih tahan untuk musim berikutnya."
        ]
    },

    "Hawar Daun": {
        "judul": "Hawar Daun",
        "ringkas": "Lesi memanjang/‘hawar’. Fokus: kontrol dini + manajemen residu + varietas tahan.",
        "langkah": [
            "Pantau rutin. Jika muncul dini (sebelum fase berbunga), biasanya risiko lebih tinggi.",
            "Kelola residu tanaman (bersihkan/olah sisa tanaman bila memungkinkan).",
            "Kurangi kelembapan & rapat tanaman agar daun cepat kering.",
            "Jika meluas cepat: fungisida bisa dipertimbangkan sesuai label & anjuran setempat."
        ],
        "pencegahan": [
            "Varietas/hibrida tahan.",
            "Rotasi tanaman dan manajemen residu."
        ]
    },

    "Karat Daun": {
        "judul": "Karat Daun",
        "ringkas": "Bercak/pustule seperti karat. Sering menyebar lewat spora angin, cepat saat cuaca lembap.",
        "langkah": [
            "Jika ringan: monitoring rutin biasanya cukup.",
            "Jika berat/menutup banyak permukaan daun: pertimbangkan fungisida sesuai label & anjuran setempat.",
            "Jaga tanaman tetap sehat (nutrisi cukup) agar tidak mudah turun produksinya."
        ],
        "pencegahan": [
            "Varietas yang lebih tahan.",
            "Monitoring saat cuaca lembap/berkabut."
        ]
    },

    "Daun Sehat": {
        "judul": "Daun Sehat",
        "ringkas": "Tidak ada gejala dominan terdeteksi.",
        "langkah": [
            "Lanjut monitoring berkala.",
        ],
        "pencegahan": [
            "Jaga jarak tanam & sirkulasi udara, nutrisi seimbang."
        ]
    }
}

# =========================
# FLASK APP
# =========================
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB

# =========================
# DEVICE + LOAD MODEL ONCE
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

def torch_load_safe(path, map_location=None):
    """
    Biar aman di berbagai versi PyTorch:
    - kalau mendukung weights_only=True -> pakai itu
    - kalau tidak -> fallback
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

# Load classes
classes = torch_load_safe(CLASSES_PATH)
num_classes = len(classes)
print("Classes:", classes)

# Preprocess sama seperti val/test
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
infer_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# Load model
model = timm.create_model("convnext_small", pretrained=False, num_classes=num_classes)
state = torch_load_safe(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.to(device).eval()

# =========================
# HELPERS
# =========================
def allowed_file(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXT

def annotate_center_crop(img: Image.Image, label: str, conf: float, crop_size=224):
    """Gambar kotak area center-crop + label & confidence."""
    w, h = img.size
    left = (w - crop_size) // 2
    top  = (h - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size

    draw = ImageDraw.Draw(img)
    draw.rectangle([left, top, right, bottom], width=4)

    text = f"{label} ({conf*100:.1f}%)"
    pad = 6
    tx, ty = left, max(0, top - 28)
    bbox = draw.textbbox((0, 0), text)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    draw.rectangle([tx, ty, tx + tw + 2*pad, ty + th + 2*pad], fill=(0, 0, 0))
    draw.text((tx + pad, ty + pad), text, fill=(255, 255, 255))
    return img

@torch.no_grad()
def predict_pil(img: Image.Image, topk=3):
    x = infer_tf(img).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]

    k = min(topk, num_classes)
    top_probs, top_idx = torch.topk(probs, k=k)

    results = []
    for p, i in zip(top_probs, top_idx):
        results.append({"class": classes[i.item()], "confidence": float(p.item())})
    return results

# =========================
# ROUTES
# =========================
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "":
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        return "File type tidak didukung. Upload jpg/png/webp.", 400

    # Save upload
    filename = secure_filename(file.filename)
    uid = uuid.uuid4().hex[:10]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = os.path.splitext(filename)[1].lower()

    saved_name = f"{stamp}_{uid}{ext}"
    upload_path = os.path.join(UPLOAD_DIR, saved_name)
    file.save(upload_path)

    # Predict
    img = Image.open(upload_path).convert("RGB")
    preds = predict_pil(img, topk=3)

    best = preds[0]
    label = best["class"]
    conf = best["confidence"]
    status = "OK" if conf >= CONF_THRESHOLD else "UNCERTAIN"

    # Treatment tips based on predicted label
    tips = TREATMENT_TIPS.get(label)

    # Save annotated output
    annotated = annotate_center_crop(img.copy(), label, conf, crop_size=224)
    out_name = f"{stamp}_{uid}__{label.replace(' ', '_')}__{conf:.3f}.jpg"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    annotated.save(out_path, quality=95)

    return render_template(
        "result.html",
        status=status,
        label=label,
        confidence=conf,
        preds=preds,
        threshold=CONF_THRESHOLD,
        treatment=tips,
        upload_url=url_for("static", filename=f"uploads/{saved_name}"),
        output_url=url_for("static", filename=f"outputs/{out_name}")
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

