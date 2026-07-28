"""Model loading + inference for the dashboard.

Wraps the trained YOLO ``best.pt`` so the rest of the app only deals with
plain dicts. Produces an annotated (boxed) image for the gallery and a
pass/reject decision based on the configured reject classes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import config


@dataclass
class Detection:
    label: str
    confidence: float


@dataclass
class InspectionResult:
    decision: str  # "PASS" or "REJECT"
    detections: list[Detection] = field(default_factory=list)
    annotated_path: str | None = None

    @property
    def defect_count(self) -> int:
        return len(self.detections)

    @property
    def max_confidence(self) -> float:
        return max((d.confidence for d in self.detections), default=0.0)

    @property
    def has_reject(self) -> bool:
        return self.decision == "REJECT"


_model = None


def load_model():
    """Load the YOLO model once and cache it."""
    global _model
    if _model is None:
        if not Path(config.WEIGHTS).exists():
            raise FileNotFoundError(
                f"Trained weights not found at {config.WEIGHTS}. "
                "Train the model first (scripts/train.py) or set FS50_WEIGHTS."
            )
        from ultralytics import YOLO

        _model = YOLO(str(config.WEIGHTS))
    return _model


def infer(
    image_path: str | Path, conf: float | None = None, key: str | None = None
) -> InspectionResult:
    """Run detection on a single image and return a structured result.

    ``key`` is an optional unique identifier (e.g. the path relative to the
    watched root) used to build a collision-free annotated filename when the
    source has nested subfolders that reuse the same image names.
    """
    import cv2

    conf = config.CONF_THRESHOLD if conf is None else conf
    model = load_model()
    image_path = Path(image_path)

    results = model.predict(
        source=str(image_path),
        imgsz=config.IMG_SIZE,
        conf=conf,
        device=config.DEVICE,
        verbose=False,
    )
    r = results[0]

    detections: list[Detection] = []
    if r.boxes is not None:
        for cls_id, c in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            detections.append(
                Detection(label=r.names[int(cls_id)], confidence=float(c))
            )

    decision = (
        "REJECT"
        if any(d.label in config.REJECT_CLASSES for d in detections)
        else "PASS"
    )

    # Save an annotated copy for the gallery (boxes drawn by YOLO).
    config.ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", key) if key else image_path.stem
    annotated_path = config.ANNOTATED_DIR / f"{safe_stem}_annotated.jpg"
    cv2.imwrite(str(annotated_path), r.plot())

    return InspectionResult(
        decision=decision,
        detections=detections,
        annotated_path=str(annotated_path),
    )
