import pandas as pd

RAW_COLUMNS = ["timestamp", "source_ip", "username", "event_type", "status", "label", "detail"]


def load_raw_logs(path: str) -> pd.DataFrame:
    """โหลดไฟล์ log ดิบ แปลง timestamp ให้เป็น datetime"""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = load_raw_logs("data/ssh_anomaly_dataset.csv")
    print(f"โหลดข้อมูลทั้งหมด {len(df)} แถว")
    print(df["label"].value_counts())
    print(df.head())
