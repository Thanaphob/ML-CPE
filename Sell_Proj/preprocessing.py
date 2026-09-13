import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["failed_attempts", "avg_time_between", "unique_usernames", "success_ratio"]

LABEL_MAP = {
    "normal": 0,
    "brute_force": 1,
    "brute_force_connection_issue": 1,
}

WINDOW = "1min"

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["label"].isin(LABEL_MAP.keys())].copy()
    df["binary_label"] = df["label"].map(LABEL_MAP)

    rows = []
    grouped = df.groupby(["source_ip", pd.Grouper(key="timestamp", freq=WINDOW)])

    for (ip, window_start), group in grouped:
        if len(group) == 0:
            continue

        auth_events = group[group["status"].isin(["auth_fail", "success"])]
        if len(auth_events) == 0:
            continue

        failed_attempts = (auth_events["status"] == "auth_fail").sum()
        success_count = (auth_events["status"] == "success").sum()
        total_attempts = failed_attempts + success_count
        success_ratio = success_count / total_attempts if total_attempts > 0 else 0.0

        unique_usernames = auth_events["username"].nunique()

        timestamps = auth_events["timestamp"].sort_values().values
        if len(timestamps) > 1:
            diffs = np.diff(timestamps).astype("timedelta64[ms]").astype(float) / 1000.0
            avg_time_between = diffs.mean()
        else:
            avg_time_between = 60.0 

        majority_label = group["binary_label"].mode().iloc[0]

        rows.append(
            {
                "source_ip": ip,
                "window_start": window_start,
                "failed_attempts": failed_attempts,
                "avg_time_between": avg_time_between,
                "unique_usernames": unique_usernames,
                "success_ratio": success_ratio,
                "label": majority_label,
            }
        )

    feature_df = pd.DataFrame(rows)
    return feature_df


if __name__ == "__main__":
    from data_loader import load_raw_logs

    raw_df = load_raw_logs("data/ssh_anomaly_dataset.csv")
    feat_df = build_features(raw_df)
    print(f"สร้าง feature vector ได้ {len(feat_df)} แถว (จาก IP+window)")
    print(feat_df["label"].value_counts())
    print(feat_df.head(10))
    feat_df.to_csv("data/features.csv", index=False)
