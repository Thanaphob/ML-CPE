from pathlib import Path
import argparse
import json
import random
import cv2
import numpy as np
import tensorflow as tf

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
                             (base / name / "images", base / name / "labels")):
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

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, default=None, help="ชื่อไฟล์รูปใน split ที่เลือก")
    p.add_argument("--split", type=str, default="test")
    args = p.parse_args()

    if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
        raise FileNotFoundError("ไม่พบโมเดลหรือ classes.json (รัน train.py ก่อน)")
    classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))

    pairs = find_pairs(args.split)
    if args.image:
        stem = Path(args.image).stem
        match = next(((i, l) for i, l in pairs if i.stem == stem), None)
        if match is None:
            raise FileNotFoundError("ไม่พบรูปหรือ label ของรูปนี้ใน split ที่เลือก")
        image_path, label_path = match
    else:
        image_path, label_path = random.choice(pairs)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(str(image_path))

    h, w = image.shape[:2]
    boxes = read_boxes(label_path, w, h, len(classes))
    if not boxes:
        raise RuntimeError("ไม่มีวัตถุที่ใช้ได้ในรูปนี้ ลองรันใหม่เพื่อสุ่มรูปอื่น")

    crops = []
    for _, x1, y1, x2, y2 in boxes:
        c = cv2.resize(image[y1:y2, x1:x2], (IMG_SIZE, IMG_SIZE))
        crops.append(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
    x = np.asarray(crops, dtype=np.float32)

    model = tf.keras.models.load_model(MODEL_PATH)
    preds = model.predict(x, verbose=0)

    for (true_cls, x1, y1, x2, y2), pred in zip(boxes, preds):
        c = int(np.argmax(pred))
        color = (0, 255, 0) if c == true_cls else (0, 0, 255)  # เขียว=ถูก แดง=ผิด
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{classes[c]} {pred[c]:.2f}", (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    OUT.mkdir(parents=True, exist_ok=True)
    save_path = OUT / "image_result.jpg"
    cv2.imwrite(str(save_path), image)
    print("Input:", image_path, "| Objects:", len(boxes), "| Saved:", save_path)

if __name__ == "__main__":
    main()
