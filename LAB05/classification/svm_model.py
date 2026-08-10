from sklearn.svm import SVC


def train_svm(X_train, y_train, kernel='rbf', C=1.0, gamma='scale'):
    
    print(f"Training SVM with kernel={kernel}, C={C}, gamma={gamma}...")

    # Create SVM model
    model = SVC(
        kernel=kernel, 
        C=C, 
        gamma=gamma, 
        random_state=42,
        cache_size=1000
    )

    # Train model
    model.fit(X_train, y_train)

    return model


def predict_svm(model, X_test):

    # Predict
    predictions = model.predict(X_test)

    return predictions