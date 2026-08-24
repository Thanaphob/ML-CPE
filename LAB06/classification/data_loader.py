import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(data_path, max_per_class=None):
    if os.path.isdir(data_path):
        file_path = os.path.join(data_path, "adult.csv")
    else:
        file_path = data_path
    df = pd.read_csv(file_path, na_values="?", skipinitialspace=True)
    df = df.dropna()
    target_col = "income" if "income" in df.columns else df.columns[-1]
    classes = sorted(df[target_col].unique().tolist())
    print("Detected classes:", classes)

    data_list = []
    labels_list = []

    for label, class_name in enumerate(classes):
        class_subset = df[df[target_col] == class_name]
        if max_per_class:
            class_subset = class_subset.head(max_per_class)
        loaded = len(class_subset)
        data_list.append(class_subset.drop(columns=[target_col]))
        labels_list.extend([label] * loaded)
        print(f"Loaded class {class_name}: {loaded} samples (0 skipped)")
        
    combined_df = pd.concat(data_list, axis=0)
    combined_df = pd.get_dummies(combined_df, drop_first=True)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(combined_df.values)
    return np.array(scaled_features), np.array(labels_list), classes