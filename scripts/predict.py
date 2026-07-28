"""Run corner-defect detection inference with a trained YOLO model.

Usage:
    python scripts/predict.py --weights runs/detect/corner/weights/best.pt \
        --source path/to/image.jpg
    python scripts/predict.py --weights runs/detect/corner/weights/best.pt \
        --source path/to/folder --save

Prints each detected defect (class + confidence) per image. An image with no
detections is reported as GOOD (no defect).
"""

import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--weights", required=True, help="Path to trained .pt weights")
    p.add_argument("--source", required=True, help="Image file or folder")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--save",
        action="store_true",
        help="Save annotated images under runs/detect/predict/",
    )
    return p.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO

    if not Path(args.weights).exists():
        raise SystemExit(f"Weights not found: {args.weights}")

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=args.save,
    )

    for r in results:
        name = Path(r.path).name
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            print(f"{name:<40} -> GOOD (no defect)")
            continue
        parts = []
        for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
            label = r.names[int(cls_id)]
            parts.append(f"{label} {conf:.2%}")
        print(f"{name:<40} -> {len(parts)} defect(s): " + ", ".join(parts))


if __name__ == "__main__":
    main()
