"""Seed the dashboard with demo data so it can be explored without the real
camera folder or SQL ODS.

It copies a handful of existing dataset images into the watched folder using
the ``<LINE>_<SERIAL>_C<CORNER>.jpg`` naming convention, then runs them through
the normal ingestion pipeline. Product info is mocked; results are real model
predictions.

    python dashboard/demo_seed.py --n 40
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import ingestion


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="How many images to seed")
    ap.add_argument(
        "--src",
        default=str(config.PROJECT_ROOT / "datasets/corner/images"),
        help="Folder of source images to sample from",
    )
    args = ap.parse_args()

    config.ensure_dirs()
    src = Path(args.src)
    images = [
        p for p in src.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if not images:
        raise SystemExit(f"No source images found under {src}")

    # Demo images go into a LOCAL folder -- never the real (network) WATCH_DIR.
    demo_dir = config.DASHBOARD_DIR / "demo_incoming"
    demo_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    rng.shuffle(images)
    picked = images[: args.n]

    dests = []
    for i, img in enumerate(picked):
        location = rng.choice(config.LOCATIONS)
        serial = f"SN{10000 + i}"
        corner = rng.randint(1, 4)
        dest = demo_dir / f"{location}_{serial}_C{corner}{img.suffix.lower()}"
        shutil.copy2(img, dest)
        dests.append(dest)

    print(f"Copied {len(picked)} images into {demo_dir}")
    print("Running ingestion pipeline (this runs the model)…")
    summaries = [ingestion.process_image(d) for d in dests]
    rejects = sum(1 for s in summaries if s.get("decision") == "REJECT")
    print(f"Processed {len(summaries)} images — {rejects} rejected.")
    print("Now launch:  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
