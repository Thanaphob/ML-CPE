import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from preprocessing import preprocess_image

cv2.setNumThreads(1)

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")
SPLITS = ("train", "validation", "test")
WORKERS = max(1, min(8, os.cpu_count() or 1))


def find_dataset_root(data_path):

    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"Dataset not found: {os.path.abspath(data_path)}\n"
        )

    for root, dirs, _ in os.walk(data_path):
        lowered = {d.lower(): d for d in dirs}
        if all(s in lowered for s in SPLITS):
            return root

    raise FileNotFoundError(
        f"No train/validation/test folders found under: "
        f"{os.path.abspath(data_path)}"
    )


def detect_classes(split_path):
    return sorted(
        folder for folder in os.listdir(split_path)
        if os.path.isdir(os.path.join(split_path, folder))
    )


def read_image(path, img_size):
    return preprocess_image(cv2.imread(path), img_size)


def load_split(split_path, classes, img_size=128, max_per_class=None):

    images = []
    labels = []

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for label, class_name in enumerate(classes):
            class_path = os.path.join(split_path, class_name)
            if not os.path.isdir(class_path):
                print(f"  Warning: class '{class_name}' missing in {split_path}")
                continue

            filenames = sorted(
                f for f in os.listdir(class_path)
                if f.lower().endswith(VALID_EXT)
            )

            loaded = 0
            skipped = 0
            position = 0
            while position < len(filenames):
                if max_per_class:
                    take = max_per_class - loaded
                    if take <= 0:
                        break
                else:
                    take = len(filenames) - position

                chunk = filenames[position:position + take]
                position += len(chunk)
                paths = [os.path.join(class_path, f) for f in chunk]

                for image in pool.map(read_image, paths,
                                      [img_size] * len(paths)):
                    if image is None:
                        skipped += 1
                    else:
                        images.append(image)
                        labels.append(label)
                        loaded += 1

            print(f"  {class_name:<14}: {loaded} images ({skipped} skipped)")

    return np.stack(images), np.array(labels)


def load_data(data_path, img_size=128, max_per_class=None):

    root = find_dataset_root(data_path)
    print("Dataset root:", root)

    folders = {d.lower(): d for d in os.listdir(root)
               if os.path.isdir(os.path.join(root, d))}

    classes = detect_classes(os.path.join(root, folders["train"]))
    print(f"Detected {len(classes)} classes:", classes)

    data = {}
    for split in SPLITS:
        print(f"\nLoading {split}...")
        data[split] = load_split(
            os.path.join(root, folders[split]),
            classes, img_size, max_per_class
        )

    X_train, y_train = data["train"]
    X_val, y_val = data["validation"]
    X_test, y_test = data["test"]

    return X_train, y_train, X_val, y_val, X_test, y_test, classes