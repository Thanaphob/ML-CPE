import json
import os

import joblib

from data_loader import load_data
from preprocessing import preprocess_data
from split_data import split_dataset
from svm_model import train_svm
from evaluate import evaluate_model

DATA_PATH = "../data_adult_income/adult.csv"
OUTPUT_DIR = "outputs"
TEST_SIZE = 0.2


def main():

    print("--" * 30)
    print('SVM Classification on Adult Income Dataset')
    print("--" * 30)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load Dataset
    print("\n[Step 1] Loading dataset...")
    X, y, classes = load_data(DATA_PATH)

    # Step 2: Preprocessing
    print("\n[Step 2] preprocess features...")
    X_scaled, y_encoded, scaler, classes = preprocess_data(X, y)

    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(list(classes), f)
    print(f'Data shape after encoding: {X_scaled.shape} rows')
    print(f'Classes: {classes}')

    # Step 3: Split Dataset
    print("\n[Step 3] Splitting dataset...")

    X_train, X_test, y_train, y_test = split_dataset(X_scaled, y_encoded, test_size=TEST_SIZE)

    print(f'Training sample: {len(X_train)}')
    print(f'Testing sample: {len(X_test)}')

    # Step 4: Train SVM
    print("\n[Step 4] Training SVM...")

    model = train_svm(X_train, y_train, kernel ='rbf', C=1.0)

    joblib.dump(model, f"{OUTPUT_DIR}/svm_model.pkl")
    joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.pkl")

    print("SVM training completed.")

    # Step 5: Prediction
    print("\n[Step 5] Testing model...")
    predictions = model.predict(X_test)

    # Step 6: Evaluation
    print("\n[Step 6] Evaluating model...")
    evaluate_model(
        y_test,
        predictions,
        classes,
        save_path=f'{OUTPUT_DIR}/confusion_matrix.png'
    )
    print(f'\nAll outputs successfully saved to \'{OUTPUT_DIR}\' folder')


if __name__ == "__main__":
    main()