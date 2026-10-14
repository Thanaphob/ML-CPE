import csv
from pathlib import Path
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DEVICE = 0 if torch.cuda.is_available() else "cpu"
DATA = str(ROOT / "dataset" / "data.yaml")
WORKERS = 4

CONFIGS = [
    {"name": "e5_img480_b8", "epochs": 5, "imgsz": 480, "batch": 8},
    {"name": "e10_img480_b8", "epochs": 10, "imgsz": 480, "batch": 8, "skip_train": True},
    {"name": "e10_img320_b8", "epochs": 10, "imgsz": 320, "batch": 8},
    {"name": "e10_img480_b16", "epochs": 10, "imgsz": 480, "batch": 16},
]


def read_last_row(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = [
            {k.strip(): v.strip() for k, v in row.items()}
            for row in csv.DictReader(f)
        ]
    return rows[-1]


def run_config(cfg):
    if cfg.get("skip_train"):
        save_dir = ROOT / "runs" / cfg["name"]
        if not (save_dir / "weights" / "best.pt").exists():
            raise FileNotFoundError(f"No finished run found: {save_dir}")
    else:
        model = YOLO("yolov8n.pt")
        model.train(
            data=DATA,
            epochs=cfg["epochs"],
            imgsz=cfg["imgsz"],
            batch=cfg["batch"],
            device=DEVICE,
            workers=WORKERS,
            amp=True,
            cache=False,
            project=str(ROOT / "runs"),
            name=cfg["name"],
            exist_ok=True,
            save=True,
            plots=True,
            patience=15,
        )
        save_dir = Path(model.trainer.save_dir)

    last = read_last_row(save_dir / "results.csv")

    best = YOLO(str(save_dir / "weights" / "best.pt"))
    metrics = best.val(
        data=DATA,
        split="test",
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        device=DEVICE,
        workers=WORKERS,
        plots=True,
        project=str(ROOT / "runs"),
        name=cfg["name"] + "_test",
        exist_ok=True,
    )

    train_loss = sum(
        float(last[k]) for k in ("train/box_loss", "train/cls_loss", "train/dfl_loss")
    )
    val_loss = sum(
        float(last[k]) for k in ("val/box_loss", "val/cls_loss", "val/dfl_loss")
    )

    return {
        "name": cfg["name"],
        "epochs": cfg["epochs"],
        "epochs_run": int(float(last["epoch"])),
        "imgsz": cfg["imgsz"],
        "batch": cfg["batch"],
        "train_time_min": round(float(last["time"]) / 60, 1),
        "train_loss": round(train_loss, 4),
        "val_loss": round(val_loss, 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50-95": round(float(metrics.box.map), 4),
        "best_weights": str(save_dir / "weights" / "best.pt"),
    }


def print_table(results):
    headers = [
        "name", "epochs_run", "imgsz", "batch", "train_time_min",
        "train_loss", "val_loss", "precision", "recall", "mAP50", "mAP50-95",
    ]
    widths = [
        max(len(h), *(len(str(r[h])) for r in results)) for h in headers
    ]
    line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    print("\n" + line)
    print("-" * len(line))
    for r in results:
        print(" | ".join(str(r[h]).ljust(w) for h, w in zip(headers, widths)))


def main():
    results = []

    for cfg in CONFIGS:
        print(f"\n===== {cfg['name']} =====")
        try:
            results.append(run_config(cfg))
        except Exception as error:
            print(f"Config {cfg['name']} failed: {error}")

    if not results:
        raise RuntimeError("No configuration finished successfully")

    summary = ROOT / "runs" / "summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)

    with open(summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print_table(results)

    top = max(results, key=lambda r: r["mAP50-95"])
    print(f"\nBest configuration: {top['name']}")
    print(f"Best weights: {top['best_weights']}")
    print(f"Summary saved to: {summary}")


if __name__ == "__main__":
    main()