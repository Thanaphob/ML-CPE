import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_tabular(df, target_col="income"):
    df_clean = df.dropna().copy()
    y = df_clean[target_col].astype(str).str.contains(">50K").astype(int).values
    X_raw = df_clean.drop(columns=[target_col])
    X_encoded = pd.get_dummies(X_raw, drop_first=True)
    return X_encoded, y

def to_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return (
        np.ascontiguousarray(X_train_scaled, dtype=np.float32),
        np.ascontiguousarray(X_test_scaled, dtype=np.float32),
        scaler
    )