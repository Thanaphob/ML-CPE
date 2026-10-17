from pathlib import Path
import argparse
import json
import time
import cv2
import numpy as np
import tensorflow as tf

OUT = Path(__file__).resolve().parent / "outputs"
MODEL_PATH = OUT / "model.keras"
CLASSES_PATH = OUT / "classes.json"
IMG_SIZE = 64

def make_windows(w, h, max_regions, scales=(0.25, 0.4, 0.6)):
    wins = []
    for s in scales:
        ww, wh = int(w * s), int(h * s)
        sx, sy = max(1, ww // 2), max(1, wh // 2)
        for y in range(0, h - wh + 1, sy):
            for x in range(0, w - ww + 1, sx):
                wins.append((x, y, x + ww, y + wh))
    if len(wins) > max_regions:
        idx = np.linspace(0, len(wins) - 1, max_regions).astype(int)
        wins = [wins[i] for i in idx]
    return wins

def detect(model, frame, windows, threshold):
    crops = []
    for x1, y1, x2, y2 in windows:
        crop = cv2.resize(frame[y1:y2, x1:x2], (IMG_SIZE, IMG_SIZE))
        crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    batch = np.asarray(crops, dtype=np.float32)

    probs = model(batch, training=False).numpy()
    cls = probs.argmax(1)
    conf = probs.max(1)

    keep = [i for i in range(len(windows)) if conf[i] >= threshold]
    if not keep:
        return []

    rects = [[windows[i][0], windows[i][1],
              windows[i][2] - windows[i][0],
              windows[i][3] - windows[i][1]] for i in keep]
    scores = [float(conf[i]) for i in keep]
    sel = np.array(cv2.dnn.NMSBoxes(rects, scores, threshold, 0.4)).flatten()
    return [(int(cls[keep[k]]), scores[k], windows[keep[k]]) for k in sel]

def draw(frame, preds, classes, fps):
    for c, conf, (x1, y1, x2, y2) in preds:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{classes[c]} {conf:.2f}", (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return frame

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.85)
    p.add_argument("--regions", type=int, default=25, help="Smaller is faster on CPU")
    p.add_argument("--every", type=int, default=3,
                   help="Run detection every N frames (larger is faster)")
    args = p.parse_args()

    if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
        raise FileNotFoundError("Model or classes.json not found (run train.py first)")
    model = tf.keras.models.load_model(MODEL_PATH)
    classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")

    print("Press Q to quit. Sliding-window crop classification demo, not real-time RPN detection.")

    windows, preds, n, fps, t0 = None, [], 0, 0.0, time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if windows is None:
                h, w = frame.shape[:2]
                windows = make_windows(w, h, args.regions)

            if n % max(1, args.every) == 0:
                preds = detect(model, frame, windows, args.threshold)
            n += 1

            now = time.time()
            inst = 1 / max(now - t0, 1e-6)
            fps = 0.9 * fps + 0.1 * inst if fps else inst
            t0 = now

            cv2.imshow("Region classifier demo (Q to quit)",
                       draw(frame, preds, classes, fps))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
