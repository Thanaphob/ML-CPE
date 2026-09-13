from sklearn.svm import SVC


def train_svm(X_train, y_train, kernel='rbf', C=1.0):
  model = SVC(kernel=kernel, C=C, random_state=42)
  model.fit(X_train, y_train)
  return model