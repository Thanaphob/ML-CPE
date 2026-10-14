from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DEVICE = 0 if torch.cuda.is_available() else 'cpu'


def main():
    weights = ROOT / 'runs' / 'detect' / 'runs' / 'person_vehicle' / 'weights' / 'best.pt'

    if not weights.exists():
        raise FileNotFoundError(f'Model not found: {weights}')

    model = YOLO(str(weights))

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError('Cannot open webcam (camera index 0)')

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            results = model.predict(
                source=frame,
                conf=0.25,
                imgsz=640,
                device=DEVICE,
                verbose=False
            )

            annotated_frame = results[0].plot()

            cv2.imshow(
                'Person Vehicle YOLO Real-time Detection',
                annotated_frame
            )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
