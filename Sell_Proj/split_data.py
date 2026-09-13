from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from preprocessing import FEATURE_COLUMNS

def split_and_scale(feature_df, test_size=0.25, random_state=42):
    X = feature_df[FEATURE_COLUMNS].values
    y = feature_df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
