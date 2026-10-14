from pathlib import Path
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DEVICE = 0 if torch.cuda.is_available() else 'cpu'


def main():
    weights = ROOT / 'runs' / 'detect' / 'runs' / 'person_vehicle' / 'weights' / 'best.pt'

    if not weights.exists():
        raise FileNotFoundError(f'Model not found: {weights}')

    model = YOLO(str(weights))

    metrics = model.val(
        data=str(ROOT / 'dataset' / 'data.yaml'),
        split='test',
        imgsz=480,
        batch=8,
        device=DEVICE,
        workers=0,
        plots=True,
        project=str(ROOT / 'runs'),
        name='evaluation',
        exist_ok=True,
    )

    print('\n===== Evaluation Results =====')
    print(f'Precision: {metrics.box.mp:.4f}')
    print(f'Recall: {metrics.box.mr:.4f}')
    print(f'mAP50: {metrics.box.map50:.4f}')
    print(f'mAP50-95: {metrics.box.map:.4f}')


if __name__ == '__main__':
    main()
