import json
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np

from data_loader import load_data
from preprocessing import to_features
from vgg_model import train_model, predict_model, VGG16, VGG_SMALL
from evaluate import evaluate_model, plot_history

# Paths are relative to this file, so the script runs from any directory.
# Put the extracted Kaggle dataset anywhere under ../Vegetable Image Dataset
# (the loader searches for the folder containing train/validation/test).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

IMG_SIZE = 48       # dataset is 224x224; 128 keeps detail but trains faster
MAX_PER_CLASS = None    # None = use all images (train 1000 / val 200 / test 200 per class)
EPOCHS = 12
BATCH_SIZE = 128

# VGG16 = full 13-conv layout. VGG_SMALL = lighter, for CPU training.
BLOCKS = VGG16


def main():

    print("--" * 30)
    print("VGG (Deep CNN) Image Recognition: Vegetable (15 classes)")
    print("--" * 30)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load Dataset (already split into train / validation / test)
    print("\n[Step 1] Loading dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test, classes = load_data(
        DATA_PATH, IMG_SIZE, MAX_PER_CLASS
    )

    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    print("\nDataset loaded successfully.")
    print(f"Classes ({len(classes)}): {classes}")

    # Step 2: Preprocessing
    print("\n[Step 2] Preprocessing images...")

    X_train = to_features(X_train)
    X_val = to_features(X_val)
    X_test = to_features(X_test)

    print(f"Feature shape: {X_train.shape}")

    # Step 3: Save splits (used by test_vgg.py)
    print("\n[Step 3] Saving splits...")

    np.save(f"{OUTPUT_DIR}/X_val.npy", X_val)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_val.npy", y_val)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    print(f"Training samples  : {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Testing samples   : {len(X_test)}")

    # Step 4: Train Model
    print("\n[Step 4] Training model...")

    model, history = train_model(
        X_train, y_train, X_val, y_val, len(classes),
        OUTPUT_DIR, EPOCHS, BATCH_SIZE, BLOCKS
    )

    print("Training completed.")

    # Step 5: Prediction
    print("\n[Step 5] Testing model...")
    predictions = predict_model(model, X_test)

    # Step 6: Evaluation
    print("\n[Step 6] Evaluating model...")
    evaluate_model(y_test, predictions, classes,
                   save_path=f"{OUTPUT_DIR}/confusion_matrix.png")
    plot_history(history, f"{OUTPUT_DIR}/training_history.png")


if __name__ == "__main__":
    main()
