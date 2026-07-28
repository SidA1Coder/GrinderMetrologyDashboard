"""Train a YOLO object-detection model (Ultralytics) for corner defects.

Expects a data.yaml pointing at a YOLO-detection dataset:
    datasets/corner/
        images/{train,val,test}/*.jpg
        labels/{train,val,test}/*.txt

Usage (CPU-friendly defaults):
    python scripts/train.py --data data.yaml
    python scripts/train.py --data data.yaml --epochs 100 --imgsz 640 \
        --model yolo11n.pt --batch 8

Outputs are saved under runs/detect/<name>/; best weights at
runs/detect/<name>/weights/best.pt
"""

import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", default="data.yaml", help="Path to data.yaml")
    p.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Pretrained detection model (nano = fastest on CPU)",
    )
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Train image size. Use 1024/1280 for small defects on high-res "
        "images (much slower on CPU -- prefer a GPU at those sizes).",
    )
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cpu", help="'cpu' or GPU id like '0'")
    p.add_argument("--name", default="corner", help="Run name under runs/detect/")
    p.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early-stop patience (epochs without improvement)",
    )
    # --- Augmentation (tuned for a SMALL, imbalanced defect dataset) ---
    # These synthetically multiply the few BrokenChips examples we have.
    p.add_argument(
        "--copy-paste",
        type=float,
        default=0.5,
        help="Copy-paste augment prob: pastes real defect instances "
        "into other images -> effectively more chip examples.",
    )
    p.add_argument(
        "--mosaic",
        type=float,
        default=1.0,
        help="Mosaic augment prob (combines 4 images).",
    )
    p.add_argument("--mixup", type=float, default=0.1, help="MixUp augment prob.")
    p.add_argument(
        "--degrees", type=float, default=10.0, help="Max random rotation (deg)."
    )
    p.add_argument(
        "--translate", type=float, default=0.1, help="Max random translation fraction."
    )
    p.add_argument("--scale", type=float, default=0.5, help="Random scale gain.")
    p.add_argument(
        "--hsv-v",
        type=float,
        default=0.4,
        help="HSV-Value (brightness) augment -> lighting robustness.",
    )
    p.add_argument("--hsv-s", type=float, default=0.7, help="HSV-Saturation augment.")
    return p.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO  # imported here so --help works without the dep

    if not Path(args.data).exists():
        raise SystemExit(
            f"data.yaml not found: {args.data}. "
            "Run scripts/prepare_dataset.py first and check data.yaml."
        )

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        patience=args.patience,
        # Augmentation to get the most out of a small, imbalanced dataset.
        copy_paste=args.copy_paste,
        mosaic=args.mosaic,
        mixup=args.mixup,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        hsv_v=args.hsv_v,
        hsv_s=args.hsv_s,
    )

    # Validate and report detection metrics.
    metrics = model.val()
    print(f"\nmAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Best weights: runs/detect/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
