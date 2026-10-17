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

DEFAULT_CLASSES = ['Person', 'Car', 'Bus', 'Truck', 'Motorcycle', 'Bicycle', 'Vegetation']

ALIASES = {
    "train": ["train"],
    "val": ["val", "valid", "validation"],
    "test": ["test"],
}

def find_base():
    if (DATASET / "data.yaml").exists():
        return DATASET

    if DATASET.exists():
        for yaml_path in sorted(DATASET.rglob("data.yaml")):
            return yaml_path.parent

    return DATASET

BASE = find_base()

def pairs_from(img_dir, lbl_dir):
    pairs = []
    for image_path in sorted(img_dir.iterdir()):
        label_path = lbl_dir / (image_path.stem + ".txt")
        if image_path.suffix.lower() in IMG_EXT and label_path.exists():
            pairs.append((image_path, label_path))
    return pairs

def find_pairs(split):
    names = ALIASES.get(split, [split])

    if split == "test":
        names = names + ALIASES["val"]

    bases = (
        BASE / "dataset",
        BASE,
        DATASET / "dataset",
        DATASET,
    )

    for name in names:
        for base in bases:
            possible_dirs = (
                (base / "images" / name, base / "labels" / name),
                (base / name / "images", base / name / "labels"),
                (base / name, base / name),
            )

            for image_dir, label_dir in possible_dirs:
                if image_dir.is_dir() and label_dir.is_dir():
                    pairs = pairs_from(image_dir, label_dir)
                    if pairs:
                        if name != split:
                            print(f"หมายเหตุ: ไม่พบ split '{split}' จึงใช้ '{name}' แทน")
                        return pairs

    raise FileNotFoundError(f"ไม่พบโฟลเดอร์ภาพและ label ของ split '{split}' ภายใต้ {DATASET}")

def load_class_names():
    yaml_candidates = (
        BASE / "data.yaml",
        DATASET / "data.yaml",
        ROOT / "data.yaml",
    )

    for yaml_path in yaml_candidates:
        if yaml_path.exists():
            try:
                data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
                names = data.get("names")
                if names:
                    if isinstance(names, dict):
                        names = [names[key] for key in sorted(names, key=lambda k: int(k))]
                    return [str(name) for name in names]
            except Exception:
                pass

    classes_candidates = (
        BASE / "dataset" / "classes.txt",
        BASE / "classes.txt",
        DATASET / "classes.txt",
    )

    for classes_path in classes_candidates:
        if classes_path.exists():
            names = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if names:
                return names

    print("ไม่พบ data.yaml หรือ classes.txt -> ใช้งาน Default Class List สำหรับ Person Vehicle Dataset")
    return DEFAULT_CLASSES

def parse_label_line(line):
    parts = line.split()
    try:
        class_id = int(float(parts[0]))
        values = list(map(float, parts[1:]))
    except (ValueError, IndexError):
        return None

    if len(values) == 4:
        return (class_id, *values)

    if len(values) >= 6 and len(values) % 2 == 0:
        xs = values[0::2]
        ys = values[1::2]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        return (
            class_id,
            (x_min + x_max) / 2,
            (y_min + y_max) / 2,
            x_max - x_min,
            y_max - y_min,
        )

    return None

def read_boxes(label_path, image_width, image_height, num_classes, min_size=8):
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parsed = parse_label_line(line)
        if parsed is None:
            continue

        class_id, x_center, y_center, box_width, box_height = parsed

        if not 0 <= class_id < num_classes:
            continue

        x1 = max(0, int((x_center - box_width / 2) * image_width))
        y1 = max(0, int((y_center - box_height / 2) * image_height))
        x2 = min(image_width, int((x_center + box_width / 2) * image_width))
        y2 = min(image_height, int((y_center + box_height / 2) * image_height))

        if x2 - x1 < min_size or y2 - y1 < min_size:
            continue

        boxes.append((class_id, x1, y1, x2, y2))

    return boxes

def load_crops(split, limit, num_classes):
    pairs = find_pairs(split)

    if limit > 0:
        pairs = pairs[:limit]

    images, labels = [], []

    for image_path, label_path in pairs:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        image_height, image_width = image.shape[:2]
        boxes = read_boxes(label_path, image_width, image_height, num_classes)

        for class_id, x1, y1, x2, y2 in boxes:
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            images.append(crop)
            labels.append(class_id)

    if not images:
        raise RuntimeError(f"ไม่พบ crop ที่มี label ใช้ได้ใน split '{split}'")

    return np.asarray(images, dtype=np.uint8), np.asarray(labels, dtype=np.int32)

def make_model(num_classes):
    layers = tf.keras.layers
    model = tf.keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Rescaling(1.0 / 255),

        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(256, 3, padding="same", activation="relu"),
        layers.GlobalAveragePooling2D(),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    classes = load_class_names()
    num_classes = len(classes)

    print("Dataset directory:", BASE)
    print("Classes:", classes)

    x, y = load_crops("train", args.limit, num_classes)
    class_counts = np.bincount(y, minlength=num_classes)

    print("Training crops:", len(y))
    print("Crops per class:", class_counts.tolist())

    try:
        x_train, y_train = x, y
        x_val, y_val = load_crops("val", args.limit, num_classes)
        print("ใช้ validation split ที่มากับชุดข้อมูล")
    except (FileNotFoundError, RuntimeError):
        print("ไม่พบ validation split ที่ใช้ได้ จะแบ่งข้อมูล train 10% เป็น validation")
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=0.1, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )

    present_classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=present_classes, y=y_train)
    class_weight = {int(cls): float(w) for cls, w in zip(present_classes, weights)}

    model = make_model(num_classes)
    model.fit(
        x_train.astype(np.float32),
        y_train,
        validation_data=(x_val.astype(np.float32), y_val),
        epochs=args.epochs,
        batch_size=args.batch,
        class_weight=class_weight,
        shuffle=True,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
    )

    model.save(MODEL_PATH)
    CLASSES_PATH.write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")
    print("บันทึกโมเดลเรียบร้อย:", MODEL_PATH)

if __name__ == "__main__":
    main()