import json
import os
import numpy as np
from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

N_SAMPLES = 5

def test_nn(n_samples=N_SAMPLES):
    model = keras.models.load_model(os.path.join(OUTPUT_DIR, "nn_model.keras"))
    X_test = np.load(os.path.join(OUTPUT_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(OUTPUT_DIR, "y_test.npy"))
    with open(os.path.join(OUTPUT_DIR, "classes.json")) as f:
        classes = json.load(f)
    index = np.random.choice(len(X_test), n_samples, replace=False)
    X_sample = X_test[index]
    y_sample = y_test[index]
    probabilities = model.predict(X_sample, verbose=0)
    if probabilities.shape[-1] == 1:
        probabilities = probabilities.ravel()
        predictions = (probabilities > 0.5).astype(int)
        confidence = np.where(predictions == 1, probabilities, 1 - probabilities)
    else:
        predictions = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)
    print(f"\n{'='*55}")
    print(f"{'No.':<4} | {'Predicted':<10} | {'True':<10} | {'Conf':<8} | {'Status'}")
    print(f"{'-'*55}")
    for i in range(n_samples):
        pred_label = classes[predictions[i]]
        true_label = classes[y_sample[i]]
        correct = predictions[i] == y_sample[i]
        status = "OK" if correct else "WRONG"
        print(f"[{i + 1:2d}] | {pred_label:<10} | {true_label:<10} | {confidence[i] * 100:5.1f}%  | {status}")
    print(f"{'='*55}")
    correct_total = int((predictions == y_sample).sum())
    print(f"Correct: {correct_total}/{n_samples} ({(correct_total/n_samples)*100:.1f}%)\n")

if __name__ == "__main__":
    test_nn()