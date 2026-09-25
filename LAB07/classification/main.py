import json
import os

import numpy as np

from sklearn.model_selection import train_test_split

from data_loader import load_data
from preprocessing import to_features
from cnn_model import train_model, predict_model
from evaluate import evaluate_model, plot_history

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "..", "data", "seg_train", "seg_train")
TEST_PATH = os.path.join(BASE_DIR, "..", "data", "seg_test", "seg_test")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

IMG_SIZE = 64
VAL_SIZE = 0.1
MAX_PER_CLASS = 1500
EPOCHS = 15
BATCH_SIZE = 64


def main():

    print("--" * 30)
    print("CNN Image Recognition: Intel Image Classification")
    print("--" * 30)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[Step 1] Loading training dataset...")
    train_images, train_labels, classes = load_data(TRAIN_PATH, IMG_SIZE, MAX_PER_CLASS)

    print("\n[Step 1b] Loading test dataset...")
    test_images, test_labels, test_classes = load_data(TEST_PATH, IMG_SIZE, MAX_PER_CLASS)

    if test_classes != classes:
        raise ValueError(f"Class mismatch between train and test: {classes} vs {test_classes}")

    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    print("\nDataset loaded successfully.")
    print(f"Training images : {len(train_images)}")
    print(f"Testing images  : {len(test_images)}")
    print(f"Classes         : {classes}")

    print("\n[Step 2] Preprocessing images...")

    X_train_full = to_features(train_images)
    y_train_full = train_labels
    X_test = to_features(test_images)
    y_test = test_labels

    np.save(f"{OUTPUT_DIR}/features.npy", X_train_full)
    np.save(f"{OUTPUT_DIR}/labels.npy", y_train_full)

    print(f"Feature shape: {X_train_full.shape}")

    print("\n[Step 3] Splitting training data into train/validation...")

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=VAL_SIZE,
        random_state=42,
        stratify=y_train_full
    )

    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/X_val.npy", X_val)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_val.npy", y_val)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    print(f"Training samples  : {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Testing samples   : {len(X_test)}")

    print("\n[Step 4] Training model...")

    model, history = train_model(
        X_train, y_train, X_val, y_val, len(classes),
        OUTPUT_DIR, EPOCHS, BATCH_SIZE
    )

    print("Training completed.")

    print("\n[Step 5] Testing model...")
    predictions = predict_model(model, X_test)

    print("\n[Step 6] Evaluating model...")
    evaluate_model(y_test, predictions, classes,
                   save_path=f"{OUTPUT_DIR}/confusion_matrix.png")
    plot_history(history, f"{OUTPUT_DIR}/training_history.png")


if __name__ == "__main__":
    main()
