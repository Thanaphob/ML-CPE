import importlib.util
import json
import os

os.environ.setdefault(
    "KERAS_BACKEND",
    "torch" if importlib.util.find_spec("torch") else "tensorflow"
)

import keras
from keras import layers

VGG16 = [(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)]

VGG_SMALL = [(32, 2), (64, 2), (128, 3), (256, 3)]


def vgg_block(model, filters, n_conv):

    for _ in range(n_conv):
        model.add(layers.Conv2D(filters, 3, padding="same", activation="relu"))
        model.add(layers.BatchNormalization(momentum=0.9))

    model.add(layers.MaxPooling2D(2))


def build_model(input_shape, num_classes, blocks=VGG16):

    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))

    model.add(layers.Rescaling(1.0 / 255))

    model.add(layers.RandomFlip("horizontal"))
    model.add(layers.RandomRotation(0.1))
    model.add(layers.RandomZoom(0.1))

    for filters, n_conv in blocks:
        vgg_block(model, filters, n_conv)

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(
        1 if num_classes == 2 else num_classes,
        activation="sigmoid" if num_classes == 2 else "softmax"
    ))

    model.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss="binary_crossentropy" if num_classes == 2
             else "sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_model(X_train, y_train, X_val, y_val, num_classes,
                output_dir=None, epochs=50, batch_size=32, blocks=VGG16):

    model = build_model(X_train.shape[1:], num_classes, blocks)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max",
            patience=6, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", mode="max",
            factor=0.5, patience=3, min_lr=1e-6
        ),
    ]

    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        model.save(os.path.join(output_dir, "vgg_model.keras"))
        with open(os.path.join(output_dir, "history.json"), "w") as f:
            json.dump({k: [float(v) for v in vs]
                       for k, vs in history.history.items()}, f)

        print(f"Saved: {os.path.join(output_dir, 'vgg_model.keras')}")

    return model, history


def predict_model(model, X_test):

    probabilities = model.predict(X_test, batch_size=128, verbose=0)

    if probabilities.shape[-1] == 1:
        return (probabilities.ravel() > 0.5).astype(int)

    return probabilities.argmax(axis=1)