import json
import os

import numpy as np

from data_loader import load_data
from preprocessing import to_features
from vgg_model import train_model, predict_model, VGG16, VGG_SMALL
from evaluate import evaluate_model, plot_history

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR,"data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

IMG_SIZE = 64
MAX_PER_CLASS = 500
EPOCHS = 12
BATCH_SIZE = 128

BLOCKS = VGG_SMALL


def main():

    print("--" * 30)
    print("VGG (Deep CNN) Image Recognition: Vegetable (15 classes)")
    print("--" * 30)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[Step 1] Loading dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test, classes = load_data(
        DATA_PATH, IMG_SIZE, MAX_PER_CLASS
    )

    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    print("\nDataset loaded successfully.")
    print(f"Classes ({len(classes)}): {classes}")

    print("\n[Step 2] Preprocessing images...")

    X_train = to_features(X_train)
    X_val = to_features(X_val)
    X_test = to_features(X_test)

    print(f"Feature shape: {X_train.shape}")

    print("\n[Step 3] Saving splits...")

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
        OUTPUT_DIR, EPOCHS, BATCH_SIZE, BLOCKS
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