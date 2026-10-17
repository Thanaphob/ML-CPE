from pathlib import Path
import argparse
import json
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
OUT = Path(__file__).resolve().parent / "outputs"
MODEL_PATH = OUT / "model.keras"
CLASSES_PATH = OUT / "classes.json"

IMG_SIZE = 64
IMG_EXT = (".jpg", ".jpeg", ".png")
ALIASES = {"train": ["train"], "val": ["val", "valid", "validation"], "test": ["test"]}

def find_base():
    if (DATASET / "data.yaml").exists():
        return DATASET
    for y in DATASET.rglob("data.yaml"):
        return y.parent
    return DATASET

BASE = find_base()

def pairs_from(img_dir, lbl_dir):
    pairs = []
    for f in sorted(img_dir.iterdir()):
        lbl = lbl_dir / (f.stem + ".txt")
        if f.suffix.lower() in IMG_EXT and lbl.exists():
            pairs.append((f, lbl))
    return pairs

def find_pairs(split):
    names = ALIASES.get(split, [split])
    if split == "test":
        names = names + ALIASES["val"]
    bases = (BASE / "dataset", BASE, DATASET / "dataset", DATASET)

    for name in names:
        for base in bases:
            for img, lbl in ((base / "images" / name, base / "labels" / name),
                             (base / name / "images", base / name / "labels"),
                             (base / name, base / name)):
                if img.is_dir() and lbl.is_dir():
                    pairs = pairs_from(img, lbl)
                    if pairs:
                        if name != split:
                            print(f"หมายเหตุ: ไม่มี '{split}' จึงใช้ '{name}' แทน")
                        return pairs
    raise FileNotFoundError(f"ไม่พบรูป/label ของ '{split}' ใน {DATASET}")

def parse_label_line(line):
    p = line.split()
    try:
        cls = int(float(p[0]))
        v = list(map(float, p[1:]))
    except (ValueError, IndexError):
        return None
    if len(v) == 4:
        return (cls, *v)
    if len(v) >= 6 and len(v) % 2 == 0:
        xs, ys = v[0::2], v[1::2]
        return (cls, (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2,
                max(xs) - min(xs), max(ys) - min(ys))
    return None

def read_boxes(label_path, w, h, num_classes, min_size=8):
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        r = parse_label_line(line)
        if r is None:
            continue
        cls, xc, yc, bw, bh = r
        if not 0 <= cls < num_classes:
            continue
        x1 = max(0, int((xc - bw / 2) * w))
        y1 = max(0, int((yc - bh / 2) * h))
        x2 = min(w, int((xc + bw / 2) * w))
        y2 = min(h, int((yc + bh / 2) * h))
        if x2 - x1 < min_size or y2 - y1 < min_size:
            continue
        boxes.append((cls, x1, y1, x2, y2))
    return boxes

def load_crops(split, limit, num_classes):
    pairs = find_pairs(split)
    if limit > 0:
        pairs = pairs[:limit]
    xs, ys = [], []
    for img_path, label_path in pairs:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        for cls, x1, y1, x2, y2 in read_boxes(label_path, w, h, num_classes):
            crop = cv2.resize(image[y1:y2, x1:x2], (IMG_SIZE, IMG_SIZE))
            xs.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            ys.append(cls)
    if not xs:
        raise RuntimeError(f"ไม่พบ crop ที่มี label ของ split '{split}'")
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int32)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--batch", type=int, default=64)
    args = p.parse_args()

    if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
        raise FileNotFoundError("ไม่พบโมเดลหรือ classes.json (รัน train.py ก่อน)")
    classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    n = len(classes)

    x, y = load_crops(args.split, args.limit, n)
    print("Total labeled crops:", len(y))

    model = tf.keras.models.load_model(MODEL_PATH)
    y_pred = np.argmax(model.predict(x, batch_size=args.batch, verbose=1), axis=1)

    accuracy = accuracy_score(y, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, average="macro", labels=list(range(n)), zero_division=0
    )

    print(f"\nAccuracy: {accuracy:.4f}\n")
    print(classification_report(y, y_pred, labels=list(range(n)),
                                target_names=classes, zero_division=0))

    OUT.mkdir(parents=True, exist_ok=True)
    metrics = {
        "split": args.split,
        "crops": len(y),
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }
    (OUT / "evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Saved:", OUT / "evaluation.json")

if __name__ == "__main__":
    main()