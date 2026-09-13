from sklearn.preprocessing import StandardScaler

FEATURES = [
    'duration',
    'src_bytes',
    'dst_bytes',
    'count',
    'srv_count',
    'diff_srv_rate',
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
]


def preprocess_data(df):
  # กรองเฉพาะ Normal และ Port Scan (Probe)
  probe_attacks = ['portsweep', 'ipsweep', 'satan', 'nmap']
  df_filtered = df[
      (df['attack_type'] == 'normal') | (df['attack_type'].isin(probe_attacks))
  ].copy()

  # แปลง Label: 0 = Normal, 1 = Port Scan
  df_filtered['label'] = df_filtered['attack_type'].apply(
      lambda x: 0 if x == 'normal' else 1
  )

  X = df_filtered[FEATURES]
  y = df_filtered['label']
  return X, y


def scale_features(X_train, X_test):
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  return X_train_scaled, X_test_scaled