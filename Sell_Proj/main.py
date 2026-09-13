from data_loader import load_raw_logs
from preprocessing import build_features
from split_data import split_and_scale
from svm_model import build_model, train_model
from evaluate import evaluate_model


def main():
    print(">> โหลด raw log ...")
    raw_df = load_raw_logs("data/ssh_anomaly_dataset.csv")
    print(f"   โหลดมาแล้ว {len(raw_df)} แถว")

    print("\n>> แปลงเป็น feature vector (ต่อ IP ต่อ 1 นาที) ...")
    feature_df = build_features(raw_df)
    print(f"   ได้ {len(feature_df)} แถว")
    print(f"   {feature_df['label'].value_counts().to_dict()}")

    print("\n>> แบ่ง train/test และ scale ...")
    X_train, X_test, y_train, y_test, scaler = split_and_scale(feature_df)
    print(f"   train: {len(X_train)} แถว, test: {len(X_test)} แถว")

    print("\n>> เทรน SVM ...")
    model = build_model()
    model = train_model(model, X_train, y_train)

    print("\n>> ประเมินผล ...")
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
