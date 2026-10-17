from pathlib import Path
import argparse
import json
import cv2
import numpy as np
import tensorflow as tf
import yaml
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

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
    """คืน [(path รูป, path label)] ถ้าไม่มี test จะใช้ val แทน"""
    names = ALIASES.get(split, [split])
    if split == "test":
        names = names + ALIASES["val"]
    bases = (BASE / "dataset", BASE, DATASET / "dataset", DATASET)

    for name in names:
        for base in bases:
            for img, lbl in ((base / "images" / name, base / "labels" / name),
                             (base / name / "images", base / name / "labels")):
                if img.is_dir() and lbl.is_dir():
                    pairs = pairs_from(img, lbl)
                    if pairs:
                        if name != split:
                            print(f"หมายเหตุ: ไม่มี '{split}' จึงใช้ '{name}' แทน")
                        return pairs
    raise FileNotFoundError(f"ไม่พบรูป/label ของ '{split}' ใน {DATASET}")

def load_class_names():
    for y in (BASE / "data.yaml", DATASET / "data.yaml", ROOT / "data.yaml"):
        if y.exists():
            names = (yaml.safe_load(y.read_text(encoding="utf-8")) or {}).get("names")
            if names:
                if isinstance(names, dict):
                    names = [names[k] for k in sorted(names)]
                return list(names)
    for c in (BASE / "dataset" / "classes.txt", BASE / "classes.txt"):
        if c.exists():
            return [l.strip() for l in c.read_text(encoding="utf-8").splitlines() if l.strip()]
    raise FileNotFoundError("ไม่พบ data.yaml หรือ classes.txt")

def parse_label_line(line):
    p = line.split()
    try:
        cls = int(float(p[0]))
        v = list(map(float, p[1:]))
    except (ValueError, IndexError):
        return None
    if len(v) == 4:
        return (cls, *v)
    if len(v) >= 6 and len(v) % 2 == 0:  # polygon
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
    return np.asarray(xs, dtype=np.uint8), np.asarray(ys, dtype=np.int32)

def make_model(num_classes):
    L = tf.keras.layers
    model = tf.keras.Sequential([
        L.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        L.Rescaling(1.0 / 255),
        L.RandomFlip("horizontal"),
        L.RandomRotation(0.05),
        L.RandomZoom(0.1),
        L.Conv2D(32, 3, padding="same", activation="relu"), L.BatchNormalization(), L.MaxPooling2D(),
        L.Conv2D(64, 3, padding="same", activation="relu"), L.BatchNormalization(), L.MaxPooling2D(),
        L.Conv2D(128, 3, padding="same", activation="relu"), L.BatchNormalization(), L.MaxPooling2D(),
        L.Conv2D(256, 3, padding="same", activation="relu"),
        L.GlobalAveragePooling2D(),
        L.Dense(128, activation="relu"),
        L.Dropout(0.4),
        L.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0, help="จำนวนรูป (0 = ทั้งหมด)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=64)
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    classes = load_class_names()
    n = len(classes)
    print("Classes:", classes)

    x, y = load_crops("train", args.limit, n)
    counts = np.bincount(y, minlength=n)
    print("Training crops:", len(y), "| per class:", counts.tolist())

    try:
        xtr, xva, ytr, yva = train_test_split(
            x, y, test_size=0.1, random_state=42, stratify=y)
    except ValueError:  # บางคลาสมีน้อยเกินไปสำหรับ stratify
        xtr, xva, ytr, yva = train_test_split(x, y, test_size=0.1, random_state=42)

    present = np.unique(ytr)
    w = compute_class_weight("balanced", classes=present, y=ytr)
    class_weight = {int(c): float(v) for c, v in zip(present, w)}

    model = make_model(n)
    model.fit(
        xtr.astype("float32"), ytr,
        validation_data=(xva.astype("float32"), yva),
        epochs=args.epochs, batch_size=args.batch,
        class_weight=class_weight, shuffle=True,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True)],
    )

    model.save(MODEL_PATH)
    CLASSES_PATH.write_text(json.dumps(classes, ensure_ascii=False), encoding="utf-8")
    print("Saved:", MODEL_PATH)

if __name__ == "__main__":
    main()
