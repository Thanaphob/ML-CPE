import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def preprocess_data(X, y):
    """Preprocess the data by scaling features and encoding labels."""

    X_encoded = pd.get_dummies(X, drop_first=True)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    # Encode the labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X_scaled, y_encoded, scaler, label_encoder.classes_