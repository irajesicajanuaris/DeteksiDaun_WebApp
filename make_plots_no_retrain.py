import os
import re
import json
import math
import numpy as np
import torch
import timm
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# =========================
# PATHS (auto dari lokasi file ini)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_IPYNB = os.path.join(BASE_DIR, "train.ipynb")
LOG_TXT     = os.path.join(BASE_DIR, "train_log.txt")  # optional fallback

MODEL_PATH   = os.path.join(BASE_DIR, "models", "best_model.pth")
CLASSES_PATH = os.path.join(BASE_DIR, "models", "classes.pt")

# test dir (kalau ada), kalau tidak ada pakai val
TEST_DIR = os.path.join(BASE_DIR, "dataset", "split", "test")
VAL_DIR  = os.path.join(BASE_DIR, "dataset", "split", "val")

PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# =========================
# SAFE TORCH LOAD (biar kompatibel berbagai versi)
# =========================
def torch_load_safe(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

# =========================
# 1) PARSE HISTORY DARI train.ipynb (tanpa retrain)
# =========================
EPOCH_LINE_RE = re.compile(
    r"\[Epoch\s*(\d+)\]\s*"
    r"train_loss=([0-9]*\.?[0-9]+|nan)\s*train_acc=([0-9]*\.?[0-9]+|nan)\s*\|\s*"
    r"val_loss=([0-9]*\.?[0-9]+|nan)\s*val_acc=([0-9]*\.?[0-9]+|nan)\s*\|\s*"
    r"lr=([0-9eE\+\-\.]+)"
)

def _to_float(x: str):
    x = x.strip().lower()
    if x == "nan":
        return float("nan")
    return float(x)

def extract_lines_from_ipynb(ipynb_path: str):
    if not os.path.exists(ipynb_path):
        return []

    with open(ipynb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    lines = []
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            # stream output (print)
            if out.get("output_type") == "stream":
                text = out.get("text", "")
                if isinstance(text, list):
                    lines.extend(text)
                elif isinstance(text, str):
                    lines.extend(text.splitlines(True))

            # execute_result / display_data kadang juga ada "text/plain"
            data = out.get("data", {})
            tp = data.get("text/plain")
            if tp:
                if isinstance(tp, list):
                    lines.extend(tp)
                elif isinstance(tp, str):
                    lines.extend(tp.splitlines(True))

    clean = []
    for ln in lines:
        ln = ln.replace("\r", "")
        for part in ln.split("\n"):
            if part.strip():
                clean.append(part.strip())
    return clean

def extract_lines_from_txt(txt_path: str):
    if not os.path.exists(txt_path):
        return []
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        return [ln.strip() for ln in f.read().splitlines() if ln.strip()]

def parse_history(lines):
    epochs = []
    train_loss, val_loss, train_acc, val_acc, lr = [], [], [], [], []

    for ln in lines:
        m = EPOCH_LINE_RE.search(ln)
        if not m:
            continue
        ep = int(m.group(1))
        tl = _to_float(m.group(2))
        ta = _to_float(m.group(3))
        vl = _to_float(m.group(4))
        va = _to_float(m.group(5))
        lrv = float(m.group(6))

        # skip baris yang nan biar plot ga rusak
        if any(math.isnan(x) for x in [tl, ta, vl, va]):
            continue

        epochs.append(ep)
        train_loss.append(tl)
        train_acc.append(ta)
        val_loss.append(vl)
        val_acc.append(va)
        lr.append(lrv)

    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "lr": lr,
    }

def save_line_plot(x, ys, labels, title, xlabel, ylabel, outpath):
    plt.figure()
    for y, lab in zip(ys, labels):
        plt.plot(x, y, label=lab)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    if labels:
        plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

# =========================
# 2) CONFUSION MATRIX + METRIK TANPA sklearn
# =========================
def confusion_matrix_np(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def metrics_from_cm(cm):
    # cm: rows=true, cols=pred
    eps = 1e-12
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)  # per true class
    pred_sum = cm.sum(axis=0).astype(np.float64) # per predicted class

    recall = tp / (support + eps)
    precision = tp / (pred_sum + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    acc = tp.sum() / (cm.sum() + eps)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support
    }

def plot_confusion_matrix(cm, class_names, outpath, title="Confusion Matrix", normalize=False):
    """
    cm: rows=true, cols=pred
    normalize=True -> tampilkan proporsi per true-class (row-normalized)
    """
    cm_show = cm.astype(np.float64)

    if normalize:
        row_sum = cm_show.sum(axis=1, keepdims=True) + 1e-12
        cm_show = cm_show / row_sum

    plt.figure(figsize=(10, 8))
    # colormap cerah & kontras bagus
    plt.imshow(cm_show, interpolation="nearest", cmap="YlGnBu")
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=35, ha="right")
    plt.yticks(tick_marks, class_names)

    # warna teks (putih/hitam) biar kebaca
    thresh = cm_show.max() * 0.55 if cm_show.max() > 0 else 0.5

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                txt = f"{cm_show[i, j]:.2f}"
            else:
                txt = str(int(cm[i, j]))
            plt.text(
                j, i, txt,
                ha="center", va="center",
                color="white" if cm_show[i, j] > thresh else "black",
                fontsize=10,
                fontweight="bold"
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()

def save_metrics_table(class_names, metrics_dict, out_csv, out_md, out_txt):
    precision = metrics_dict["precision"]
    recall = metrics_dict["recall"]
    f1 = metrics_dict["f1"]
    support = metrics_dict["support"]
    acc = metrics_dict["accuracy"]

    # CSV
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("class,precision,recall,f1,support\n")
        for i, name in enumerate(class_names):
            f.write(f"{name},{precision[i]:.6f},{recall[i]:.6f},{f1[i]:.6f},{int(support[i])}\n")

    # Markdown table
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(f"**Overall Accuracy:** {acc:.6f}\n\n")
        f.write("| Class | Precision | Recall | F1 | Support |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for i, name in enumerate(class_names):
            f.write(f"| {name} | {precision[i]:.4f} | {recall[i]:.4f} | {f1[i]:.4f} | {int(support[i])} |\n")

    # TXT rapi
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"Overall Accuracy: {acc:.6f}\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Class':25s}  {'Prec':>8s}  {'Rec':>8s}  {'F1':>8s}  {'Supp':>6s}\n")
        f.write("-" * 60 + "\n")
        for i, name in enumerate(class_names):
            f.write(f"{name[:25]:25s}  {precision[i]:8.4f}  {recall[i]:8.4f}  {f1[i]:8.4f}  {int(support[i]):6d}\n")

# =========================
# 3) LOAD MODEL + RUN EVAL (test/val) untuk CM
# =========================
def load_model_and_classes():
    classes = torch_load_safe(CLASSES_PATH)
    num_classes = len(classes)

    model = timm.create_model("convnext_small", pretrained=False, num_classes=num_classes)
    state = torch_load_safe(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, classes

@torch.no_grad()
def run_inference_dataset(model, dataset_dir, infer_tf, batch_size=16):
    ds = ImageFolder(dataset_dir, transform=infer_tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    y_true, y_pred = [], []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().numpy()
        y_true.extend(y.numpy().tolist())
        y_pred.extend(pred.tolist())

    return np.array(y_true), np.array(y_pred), ds.classes

# =========================
# MAIN
# =========================
def main():
    # ---- A) history dari ipynb / txt
    lines = extract_lines_from_ipynb(TRAIN_IPYNB)
    if not lines:
        print("⚠️ Tidak nemu output di train.ipynb. Coba fallback ke train_log.txt ...")
        lines = extract_lines_from_txt(LOG_TXT)

    history = parse_history(lines)

    if len(history["epochs"]) == 0:
        print("❌ Tidak ketemu baris log epoch di train.ipynb / train_log.txt.")
        print("Solusi cepat tanpa retrain: copy semua output training (yang ada [Epoch xx] ...) ke file train_log.txt, lalu jalankan script ini lagi.")
        return

    epochs = history["epochs"]

    # Plot: Training vs Validation Loss
    save_line_plot(
        epochs,
        [history["train_loss"], history["val_loss"]],
        ["Train Loss", "Val Loss"],
        "Loss vs Epoch",
        "Epoch", "Loss",
        os.path.join(PLOTS_DIR, "loss_vs_epoch.png")
    )

    # Plot: Training vs Validation Accuracy
    save_line_plot(
        epochs,
        [history["train_acc"], history["val_acc"]],
        ["Train Acc", "Val Acc"],
        "Accuracy vs Epoch",
        "Epoch", "Accuracy",
        os.path.join(PLOTS_DIR, "acc_vs_epoch.png")
    )

    # Plot LR
    save_line_plot(
        epochs,
        [history["lr"]],
        ["LR"],
        "Learning Rate vs Epoch",
        "Epoch", "Learning Rate",
        os.path.join(PLOTS_DIR, "lr_vs_epoch.png")
    )

    print("✅ Training curves saved:", PLOTS_DIR)

    # ---- B) Confusion matrix + per-class table (dari model + dataset)
    model, classes = load_model_and_classes()

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    infer_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    dataset_dir = TEST_DIR if os.path.isdir(TEST_DIR) else VAL_DIR
    if not os.path.isdir(dataset_dir):
        print("❌ Tidak ada folder test/val ditemukan di:", TEST_DIR, "atau", VAL_DIR)
        return

    y_true, y_pred, class_names = run_inference_dataset(model, dataset_dir, infer_tf, batch_size=16)
    cm = confusion_matrix_np(y_true, y_pred, num_classes=len(class_names))
    m = metrics_from_cm(cm)

    # Confusion matrix (angka asli)
    plot_confusion_matrix(
        cm, class_names,
        os.path.join(PLOTS_DIR, "confusion_matrix.png"),
        title=f"Confusion Matrix ({os.path.basename(dataset_dir)})",
        normalize=False
    )

    # Confusion matrix normalized (lebih kebaca kalau class imbalance)
    plot_confusion_matrix(
        cm, class_names,
        os.path.join(PLOTS_DIR, "confusion_matrix_normalized.png"),
        title=f"Confusion Matrix Normalized ({os.path.basename(dataset_dir)})",
        normalize=True
    )

    # Simpan metrics jadi tabel (CSV + Markdown + TXT)
    csv_path = os.path.join(PLOTS_DIR, "per_class_metrics.csv")
    md_path  = os.path.join(PLOTS_DIR, "per_class_metrics.md")
    txt_path = os.path.join(PLOTS_DIR, "per_class_metrics_table.txt")

    save_metrics_table(class_names, m, csv_path, md_path, txt_path)

    # (opsional) ringkasan metrik lama tetap disimpan juga
    report_path = os.path.join(PLOTS_DIR, "metrics_summary.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {dataset_dir}\n")
        f.write(f"Overall Accuracy: {m['accuracy']:.6f}\n\n")
        f.write("Per-class metrics:\n")
        for i, name in enumerate(class_names):
            f.write(
                f"- {name}: "
                f"precision={m['precision'][i]:.4f}, "
                f"recall(acc)={m['recall'][i]:.4f}, "
                f"f1={m['f1'][i]:.4f}, "
                f"support={int(m['support'][i])}\n"
            )

    print("✅ Confusion matrix saved:", os.path.join(PLOTS_DIR, "confusion_matrix.png"))
    print("✅ Confusion matrix normalized saved:", os.path.join(PLOTS_DIR, "confusion_matrix_normalized.png"))
    print("✅ Metrics tables saved:", csv_path, md_path, txt_path)
    print("✅ Metrics text saved:", report_path)
    print("\nSelesai ✅")

if __name__ == "__main__":
    main()
