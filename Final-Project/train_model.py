import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

DATA_PATH = os.path.join("data", "gesture_data.csv")
MODEL_PATH = "gesture_model.pkl"


def main():
    if not os.path.isfile(DATA_PATH):
        print(f"ไม่พบไฟล์ {DATA_PATH} — กรุณารัน collect_data.py เพื่อเก็บข้อมูลก่อน")
        return

    # 1. โหลดข้อมูล
    df = pd.read_csv(DATA_PATH)
    print(f"โหลดข้อมูลทั้งหมด {len(df)} แถว")
    print("จำนวนข้อมูลต่อท่า:")
    print(df["label"].value_counts())

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values
    y = df["label"].values

    if len(df) < 30 or df["label"].nunique() < 2:
        print("ข้อมูลน้อยเกินไป หรือมีท่าทางน้อยกว่า 2 แบบ ลองเก็บข้อมูลเพิ่มก่อนเทรน")
        return

    # 2. แบ่ง train / test (80/20) แบบ stratify เพื่อให้สัดส่วนแต่ละท่าเท่ากันทั้งสองชุด
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain set: {len(X_train)} แถว | Test set: {len(X_test)} แถว")

    # 3. เทรนโมเดล SVM (RBF kernel เหมาะกับข้อมูลที่ไม่ได้แยกเป็นเส้นตรง)
    model = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
    model.fit(X_train, y_train)

    # 4. ประเมินผล
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n=== ผลการประเมินโมเดล ===")
    print(f"Accuracy บน test set: {acc * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix (แถว = จริง, คอลัมน์ = ทำนาย):")
    labels_sorted = sorted(df["label"].unique())
    print("labels:", labels_sorted)
    print(confusion_matrix(y_test, y_pred, labels=labels_sorted))

    # 5. บันทึกโมเดล
    joblib.dump({"model": model, "classes": labels_sorted}, MODEL_PATH)
    print(f"\nบันทึกโมเดลลงไฟล์ {MODEL_PATH} แล้ว")
    print("นำไปใช้งานได้ทันทีผ่าน gesture_classifier.py (โหลดอัตโนมัติถ้าเจอไฟล์นี้)")


if __name__ == "__main__":
    main()
