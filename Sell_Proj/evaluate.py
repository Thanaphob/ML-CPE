import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

def evaluate_model(model, X_test, y_test, output_path="outputs/confusion_matrix.png"):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("=" * 50)
    print(f"Accuracy: {acc:.4f}")
    print("=" * 50)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Attack"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix - SSH Brute Force Detection")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nบันทึก confusion matrix ไว้ที่ {output_path}")

    return acc, y_pred
