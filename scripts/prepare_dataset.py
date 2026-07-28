"""Split a CVAT 'YOLO 1.1' export into YOLO-detection train/val/test folders.

CVAT 'YOLO 1.1' export typically looks like:
    cvat_export/
        obj.names                 # one class name per line (order = class id)
        obj.data
        obj_train_data/
            img001.jpg
            img001.txt            # <cls> <x> <y> <w> <h> (normalized); may be
            img002.jpg            #   absent/empty for good (no-defect) images
            ...

Output layout (what data.yaml points at):
    dst/
        images/{train,val,test}/*.jpg
        labels/{train,val,test}/*.txt   (empty label = good panel, kept)

Usage:
    python scripts/prepare_dataset.py --cvat-export cvat_export --dst datasets/corner
    python scripts/prepare_dataset.py --cvat-export cvat_export --dst datasets/corner \
        --train 0.8 --val 0.1 --test 0.1 --seed 42
"""

import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--cvat-export", required=True, help="Root of the CVAT 'YOLO 1.1' export"
    )
    p.add_argument("--dst", required=True, help="Output dataset root")
    p.add_argument("--train", type=float, default=0.8)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def find_images(root: Path):
    return [f for f in root.rglob("*") if f.is_file() and f.suffix.lower() in IMG_EXTS]


def read_class_names(root: Path):
    found = list(root.rglob("obj.names"))
    if not found:
        return None
    names = [
        ln.strip()
        for ln in found[0].read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    return names or None


def main():
    args = parse_args()
    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(f"--train/--val/--test must sum to 1.0 (got {total})")

    export_root = Path(args.cvat_export)
    dst = Path(args.dst)
    if not export_root.is_dir():
        raise SystemExit(f"CVAT export folder not found: {export_root}")

    images = find_images(export_root)
    if not images:
        raise SystemExit(f"No images found under {export_root}")

    # Create output folders.
    for split in ("train", "val", "test"):
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    rng.shuffle(images)
    n = len(images)
    n_train = int(n * args.train)
    n_val = int(n * args.val)
    splits = {
        "train": images[:n_train],
        "val": images[n_train : n_train + n_val],
        "test": images[n_train + n_val :],
    }

    counts = {"train": [0, 0], "val": [0, 0], "test": [0, 0]}  # [images, with_labels]

    for split, files in splits.items():
        for img in files:
            shutil.copy2(img, dst / "images" / split / img.name)
            label_src = img.with_suffix(".txt")
            label_dst = dst / "labels" / split / (img.stem + ".txt")
            if label_src.exists() and label_src.read_text(encoding="utf-8").strip():
                shutil.copy2(label_src, label_dst)
                counts[split][1] += 1
            else:
                # Good panel: write an empty label so YOLO treats it as background.
                label_dst.write_text("", encoding="utf-8")
            counts[split][0] += 1

    print(f"\nDetection dataset written to: {dst.resolve()}")
    print(f"{'split':<8}{'images':>10}{'with_defect':>14}{'good(empty)':>14}")
    for split, (imgs, with_lbl) in counts.items():
        print(f"{split:<8}{imgs:>10}{with_lbl:>14}{imgs - with_lbl:>14}")

    names = read_class_names(export_root)
    if names:
        print("\nClasses found in obj.names (verify these match data.yaml):")
        for i, name in enumerate(names):
            print(f"  {i}: {name}")
    else:
        print("\nWARNING: obj.names not found — set class names manually in data.yaml.")


if __name__ == "__main__":
    main()
