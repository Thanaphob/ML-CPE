from pathlib import Path
import argparse
import json
import cv2
import numpy as np
import tensorflow as tf

OUT = Path(__file__).resolve().parent / "outputs"
MODEL_PATH = OUT / "model.keras"
CLASSES_PATH = OUT / "classes.json"
IMG_SIZE = 64

def preprocess(crop):
    return cv2.cvtColor(cv2.resize(crop, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

def propose_regions(frame, max_regions):
    h, w = frame.shape[:2]
    boxes = []
    for scale in (.2, .35, .5):
        bw = max(16, int(w * scale))
        bh = max(16, int(h * scale))
        for cy in np.linspace(bh // 2, max(bh // 2, h - bh // 2), 4):
            for cx in np.linspace(bw // 2, max(bw // 2, w - bw // 2), 5):
                x1 = max(0, int(cx - bw / 2))
                y1 = max(0, int(cy - bh / 2))
                x2 = min(w, x1 + bw)
                y2 = min(h, y1 + bh)
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2, y2))
                if len(boxes) >= max_regions:
                    return boxes
    return boxes

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--threshold", type=float, default=.85)
    p.add_argument("--regions", type=int, default=25, help="Smaller is faster on CPU")
    args = p.parse_args()

    if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
        raise FileNotFoundError("ไม่พบโมเดลหรือ classes.json (รัน train.py ก่อน)")
    classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(MODEL_PATH)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")

    print("Press Q to quit. Sliding-window region classification demo; NOT Fast R-CNN detection.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            boxes = propose_regions(frame, max(1, args.regions))
            crops = np.asarray([preprocess(frame[y1:y2, x1:x2]) for x1, y1, x2, y2 in boxes], np.float32)
            scores = model.predict(crops, verbose=0)
            for (x1, y1, x2, y2), pred in zip(boxes, scores):
                cls = int(np.argmax(pred))
                conf = float(np.max(pred))
                if conf < args.threshold:
                    continue
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{classes[cls]} {conf:.2f}", (x1, max(20, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
            cv2.imshow("Fast-R-CNN educational demo (Q to quit)", frame)
            if cv2.waitKey(1) & 0xff == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()