import os
import cv2
import numpy as np

from preprocessing import preprocess_image

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")
SPLITS = ("train", "validation", "test")


def find_dataset_root(data_path):
    """Find the folder that contains train/, validation/ and test/.

    Kaggle's zip often nests the data one or two levels deep
    (e.g. 'Vegetable Image Dataset/Vegetable Images/train'),
    so search below data_path instead of assuming a fixed layout.
    """

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


def load_split(split_path, classes, img_size=128, max_per_class=None):
    """Load one split (train / validation / test) using a fixed class order."""

    images = []
    labels = []

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
        for filename in filenames:
            if max_per_class and loaded >= max_per_class:
                break

            image = cv2.imread(os.path.join(class_path, filename))

            # Resize here so full-size images are not all kept in memory
            image = preprocess_image(image, img_size)

            # Skip unreadable or damaged images
            if image is None:
                skipped += 1
                continue

            images.append(image)
            labels.append(label)
            loaded += 1

        print(f"  {class_name:<14}: {loaded} images ({skipped} skipped)")

    return np.stack(images), np.array(labels)


def load_data(data_path, img_size=128, max_per_class=None):
    """Load the pre-split dataset.

    Returns (X_train, y_train, X_val, y_val, X_test, y_test, classes).
    """

    root = find_dataset_root(data_path)
    print("Dataset root:", root)

    # Folder names may differ in case (Train / train), so map them first
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
