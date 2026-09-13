from sklearn.svm import SVC

def build_model(kernel="rbf", C=1.0, gamma="scale", random_state=42):
    return SVC(kernel=kernel, C=C, gamma=gamma, random_state=random_state)

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model
