"""Folder-watch ingestion pipeline.

Scans the watched folder for new corner images, runs them through the model,
enriches with product info from the SQL ODS (or mock), writes the result to
the history store, and evaluates alert rules. Idempotent: images already in
the store are skipped, so it is safe to call repeatedly (the dashboard polls
it on each refresh).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import config
import database
import inference
import store

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _rel_key(path: Path) -> str:
    """Unique, stable identifier for an image.

    Nested subfolders can reuse the same filename, so we key off the path
    relative to the watched root (falling back to the name if unrelated).
    """
    try:
        return path.relative_to(config.WATCH_DIR).as_posix()
    except ValueError:
        return path.name


def _list_new_images() -> list[Path]:
    if not config.WATCH_DIR.exists():
        return []
    walker = (
        config.WATCH_DIR.rglob("*")
        if config.WATCH_RECURSIVE
        else config.WATCH_DIR.iterdir()
    )
    files = [f for f in walker if f.suffix.lower() in IMG_EXTS and f.is_file()]

    # Filter out already-processed images in memory (one DB query, not one per
    # file -- important when the share holds tens of thousands of images).
    done = store.processed_keys()
    new = [f for f in files if _rel_key(f) not in done]

    # Newest first. The source uses timestamped subfolders (YYMMDD_HHMMSS), so
    # a reverse lexical sort of the relative path is chronological and avoids a
    # slow network stat() on every file.
    new.sort(key=_rel_key, reverse=True)

    if config.MAX_PER_SCAN > 0:
        new = new[: config.MAX_PER_SCAN]
    return new


def process_image(image_path: Path, conf: float | None = None) -> dict:
    """Full pipeline for a single image. Returns a summary dict."""
    key = _rel_key(image_path)
    result = inference.infer(image_path, conf=conf, key=key)
    info = database.get_product_info(image_path)

    store.add_inspection(
        ts=datetime.now(),
        image_name=key,
        annotated_path=result.annotated_path,
        location=info.location,
        part_id=info.part_id,
        corner=info.corner,
        grinder=info.grinder,
        grind_time=info.grind_time,
        decision=result.decision,
        defect_count=result.defect_count,
        max_conf=result.max_confidence,
        classes=[d.label for d in result.detections],
    )

    fired = []
    if result.has_reject and info.location:
        import alerts

        fired = alerts.check_and_alert(info.location)

    return {
        "image": image_path.name,
        "location": info.location,
        "part_id": info.part_id,
        "corner": info.corner,
        "grinder": info.grinder,
        "grind_time": info.grind_time,
        "decision": result.decision,
        "defects": [d.label for d in result.detections],
        "alerts": fired,
    }


def scan_and_process(conf: float | None = None) -> list[dict]:
    """Process all new images in the watched folder. Returns per-image summaries."""
    store.init_db()
    summaries = []
    for image_path in _list_new_images():
        try:
            summaries.append(process_image(image_path, conf=conf))
        except Exception as exc:  # keep the loop alive on a single bad image
            summaries.append({"image": image_path.name, "error": str(exc)})
    return summaries
