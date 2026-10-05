import csv
import gc
import importlib.util
import json
import os
import time

os.environ.setdefault(
    "KERAS_BACKEND",
    "torch" if importlib.util.find_spec("torch") else "tensorflow"
)

import keras
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from keras import layers

from data_loader import load_data
from preprocessing import to_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR,"data")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "experiments")

MAX_PER_CLASS = None
BATCH_SIZE = 64
SEED = 42
CHECKPOINTS = (3, 6, 10)

SCRATCH_IMG_SIZE = 48
SCRATCH_LR = 1e-3

SCRATCH_CONFIGS = {
    "Small (4 conv, 64 neurons)": (
        [(32, 1), (64, 1), (128, 1), (256, 1)], (64,)),
    "Medium (7 conv, 256 neurons)": (
        [(32, 2), (64, 2), (128, 2), (256, 1)], (256,)),
    "Large (10 conv, 512-256 neurons)": (
        [(32, 2), (64, 2), (128, 3), (256, 3)], (512, 256)),
}

PRETRAINED_IMG_SIZE = 128
PRETRAINED_LR = 1e-3

PRETRAINED_ALL = {
    "VGG16": ("VGG16", "caffe"),
    "VGG19": ("VGG19", "caffe"),
    "InceptionV3": ("InceptionV3", "tf"),
    "ResNet101": ("ResNet101", "caffe"),
    "ResNet152": ("ResNet152", "caffe"),
    "InceptionResNetV2": ("InceptionResNetV2", "tf"),
    "MobileNetV1": ("MobileNet", "tf"),
    "MobileNetV2": ("MobileNetV2", "tf"),
    "EfficientNetB0": ("EfficientNetB0", "none"),
}

PRETRAINED_RUN = ["VGG16", "MobileNetV2", "EfficientNetB0"]

CAFFE_MEAN = np.array([103.939, 116.779, 123.68], dtype="float32")


def slug(name):
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def build_scratch(input_shape, num_classes, blocks, dense_units):

    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))
    model.add(layers.Rescaling(1.0 / 255))

    model.add(layers.RandomFlip("horizontal"))
    model.add(layers.RandomRotation(0.1))
    model.add(layers.RandomZoom(0.1))

    for filters, n_conv in blocks:
        for _ in range(n_conv):
            model.add(layers.Conv2D(filters, 3, padding="same",
                                    activation="relu"))
            model.add(layers.BatchNormalization(momentum=0.9))
        model.add(layers.MaxPooling2D(2))

    model.add(layers.GlobalAveragePooling2D())
    for units in dense_units:
        model.add(layers.Dense(units, activation="relu"))
        model.add(layers.Dropout(0.3))

    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=keras.optimizers.Adam(SCRATCH_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class CheckpointEval(keras.callbacks.Callback):

    def __init__(self, X_val, y_val, X_test, y_test, checkpoints):
        super().__init__()
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test
        self.checkpoints = set(checkpoints)
        self.records = {}

    def on_epoch_end(self, epoch, logs=None):
        n = epoch + 1
        if n not in self.checkpoints:
            return
        _, val_acc = self.model.evaluate(
            self.X_val, self.y_val, batch_size=256, verbose=0)
        _, test_acc = self.model.evaluate(
            self.X_test, self.y_test, batch_size=256, verbose=0)
        self.records[n] = {
            "train_acc": float(logs["accuracy"]),
            "val_acc": float(val_acc),
            "test_acc": float(test_acc),
        }


def make_rows(name, info, n_params, n_trainable, records, seconds):

    rows = []
    for epochs in sorted(records):
        r = records[epochs]
        rows.append({
            "config": name,
            "group": info["group"],
            "img_size": info["img_size"],
            "conv_layers": info["conv_layers"],
            "dense_units": info["dense_units"],
            "params": n_params,
            "trainable_params": n_trainable,
            "epochs": epochs,
            "train_acc": round(r["train_acc"], 4),
            "val_acc": round(r["val_acc"], 4),
            "test_acc": round(r["test_acc"], 4),
            "train_time_s": round(seconds, 1),
        })
    return rows


def save_history(name, hist):
    with open(os.path.join(OUT_DIR, f"history_{slug(name)}.json"), "w") as f:
        json.dump({k: [float(v) for v in vs] for k, vs in hist.items()}, f)


def train_scratch(name, model, info, data):

    X_train, y_train, X_val, y_val, X_test, y_test = data

    n_params = model.count_params()
    n_trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    print(f"\n=== {name} | {info['group']} | params: {n_params:,} "
          f"(trainable {n_trainable:,}) | image {info['img_size']}px ===")

    cb = CheckpointEval(X_val, y_val, X_test, y_test, CHECKPOINTS)

    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=max(CHECKPOINTS),
        batch_size=BATCH_SIZE,
        callbacks=[cb],
        verbose=2,
    )
    seconds = time.time() - start

    model.save(os.path.join(OUT_DIR, f"model_{slug(name)}.keras"))
    save_history(name, history.history)

    rows = make_rows(name, info, n_params, n_trainable, cb.records, seconds)

    hist = history.history
    keras.backend.clear_session()
    gc.collect()

    return rows, hist


def preprocess_batch(batch, mode):

    batch = batch.astype("float32")
    if mode == "tf":
        return batch / 127.5 - 1.0
    if mode == "caffe":
        return np.ascontiguousarray(batch[..., ::-1]) - CAFFE_MEAN
    return batch


def extract_features(base, X, mode, flip, tag, batch_size=64):

    outputs = []
    total = len(X)

    for start in range(0, total, batch_size):
        batch = X[start:start + batch_size]
        if flip:
            batch = np.ascontiguousarray(batch[:, :, ::-1, :])
        outputs.append(np.asarray(
            base.predict_on_batch(preprocess_batch(batch, mode))))
        if (start // batch_size) % 20 == 0:
            print(f"\r  {tag}: {min(start + batch_size, total)}/{total}",
                  end="", flush=True)

    print(f"\r  {tag}: {total}/{total}")
    return np.concatenate(outputs)


def train_pretrained(label, app_name, mode, data, num_classes):

    X_train, y_train, X_val, y_val, X_test, y_test = data
    name = f"{label} (pretrained)"
    info = {
        "group": "pretrained",
        "img_size": PRETRAINED_IMG_SIZE,
        "conv_layers": "-",
        "dense_units": "-",
    }

    print(f"\n=== {name} | extracting features | "
          f"image {PRETRAINED_IMG_SIZE}px ===")
    start = time.time()

    base = getattr(keras.applications, app_name)(
        weights="imagenet", include_top=False,
        input_shape=X_train.shape[1:], pooling="avg")
    base.trainable = False
    base_params = base.count_params()

    F_train = extract_features(base, X_train, mode, False, "train")
    F_train_flip = extract_features(base, X_train, mode, True, "train flipped")
    F_val = extract_features(base, X_val, mode, False, "validation")
    F_test = extract_features(base, X_test, mode, False, "test")

    del base
    keras.backend.clear_session()
    gc.collect()

    keras.utils.set_random_seed(SEED)
    head = keras.Sequential([
        keras.Input(shape=(F_train.shape[1],)),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    head.compile(
        optimizer=keras.optimizers.Adam(PRETRAINED_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    n_trainable = head.count_params()
    n_params = base_params + n_trainable
    print(f"  params: {n_params:,} (trainable {n_trainable:,})")

    total_epochs = max(CHECKPOINTS)
    rng = np.random.default_rng(SEED)
    hist = {"accuracy": [], "loss": [], "val_accuracy": [], "val_loss": []}
    records = {}

    for epoch in range(1, total_epochs + 1):
        flip_mask = rng.random(len(y_train)) < 0.5
        X_epoch = np.where(flip_mask[:, None], F_train_flip, F_train)
        h = head.fit(
            X_epoch, y_train,
            validation_data=(F_val, y_val),
            batch_size=BATCH_SIZE,
            epochs=1,
            verbose=0,
        ).history
        for key in hist:
            hist[key].append(float(h[key][0]))

        print(f"Epoch {epoch}/{total_epochs} - "
              f"loss: {hist['loss'][-1]:.4f} - "
              f"accuracy: {hist['accuracy'][-1]:.4f} - "
              f"val_loss: {hist['val_loss'][-1]:.4f} - "
              f"val_accuracy: {hist['val_accuracy'][-1]:.4f}")

        if epoch in CHECKPOINTS:
            _, test_acc = head.evaluate(F_test, y_test,
                                        batch_size=256, verbose=0)
            records[epoch] = {
                "train_acc": hist["accuracy"][-1],
                "val_acc": hist["val_accuracy"][-1],
                "test_acc": float(test_acc),
            }

    seconds = time.time() - start
    save_history(name, hist)
    rows = make_rows(name, info, n_params, n_trainable, records, seconds)

    del head, F_train, F_train_flip, F_val, F_test
    keras.backend.clear_session()
    gc.collect()

    return rows, hist


def load_split_data(img_size):
    X_train, y_train, X_val, y_val, X_test, y_test, classes = load_data(
        DATA_PATH, img_size, MAX_PER_CLASS)
    data = (to_features(X_train), y_train,
            to_features(X_val), y_val,
            to_features(X_test), y_test)
    print(f"\ntrain {len(y_train)} | val {len(y_val)} | test {len(y_test)} "
          f"| classes {len(classes)} | image {img_size}px")
    return data, len(classes)


def plot_curves(histories, save_path):

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for name, h in histories.items():
        axes[0].plot(range(1, len(h["val_accuracy"]) + 1),
                     h["val_accuracy"], label=name)
        axes[1].plot(range(1, len(h["val_loss"]) + 1),
                     h["val_loss"], label=name)

    axes[0].set_title("Validation accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=7)

    axes[1].set_title("Validation loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_checkpoints(results, save_path):

    names = list(dict.fromkeys(r["config"] for r in results))
    epochs_list = sorted({r["epochs"] for r in results})
    width = 0.8 / len(epochs_list)

    fig, ax = plt.subplots(figsize=(max(10, 1.6 * len(names)), 5.5))
    for i, ep in enumerate(epochs_list):
        values = [next(r["test_acc"] for r in results
                       if r["config"] == n and r["epochs"] == ep)
                  for n in names]
        xs = np.arange(len(names)) + i * width
        bars = ax.bar(xs, values, width, label=f"{ep} epochs")
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v,
                    f"{v * 100:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(np.arange(len(names)) + width * (len(epochs_list) - 1) / 2)
    ax.set_xticklabels(names, fontsize=8, rotation=25, ha="right")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1.08)
    ax.set_title("Test accuracy by model and number of epochs")
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def save_csv(results):
    with open(os.path.join(OUT_DIR, "results.csv"), "w",
              newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def main():

    os.makedirs(OUT_DIR, exist_ok=True)

    results = []
    histories = {}

    if SCRATCH_CONFIGS:
        print("\n##### Part A: from-scratch models #####")
        data, num_classes = load_split_data(SCRATCH_IMG_SIZE)

        for name, (blocks, dense_units) in SCRATCH_CONFIGS.items():
            keras.utils.set_random_seed(SEED)
            model = build_scratch(data[0].shape[1:], num_classes,
                                  blocks, dense_units)
            info = {
                "group": "scratch",
                "img_size": SCRATCH_IMG_SIZE,
                "conv_layers": sum(n for _, n in blocks),
                "dense_units": "-".join(str(u) for u in dense_units),
            }
            rows, hist = train_scratch(name, model, info, data)
            del model
            results.extend(rows)
            histories[name] = hist
            save_csv(results)

        del data
        gc.collect()

    if PRETRAINED_RUN:
        print("\n##### Part B: pretrained models (frozen body) #####")
        data, num_classes = load_split_data(PRETRAINED_IMG_SIZE)

        for label in PRETRAINED_RUN:
            app_name, mode = PRETRAINED_ALL[label]
            rows, hist = train_pretrained(label, app_name, mode,
                                          data, num_classes)
            results.extend(rows)
            histories[f"{label} (pretrained)"] = hist
            save_csv(results)

        del data
        gc.collect()

    if not results:
        print("Nothing to run: SCRATCH_CONFIGS and PRETRAINED_RUN are empty.")
        return

    print("\n" + "=" * 84)
    print(f"{'Model':<40}{'Params(M)':>10}{'Epochs':>7}"
          f"{'Train':>9}{'Val':>9}{'Test':>9}")
    print("-" * 84)
    for r in results:
        print(f"{r['config']:<40}{r['params'] / 1e6:>10.2f}{r['epochs']:>7}"
              f"{r['train_acc'] * 100:>8.2f}%{r['val_acc'] * 100:>8.2f}%"
              f"{r['test_acc'] * 100:>8.2f}%")
    print("=" * 84)

    print(f"\nSaved: {os.path.join(OUT_DIR, 'results.csv')}")
    plot_curves(histories, os.path.join(OUT_DIR, "accuracy_curves.png"))
    plot_checkpoints(results, os.path.join(OUT_DIR, "checkpoint_comparison.png"))


if __name__ == "__main__":
    main()