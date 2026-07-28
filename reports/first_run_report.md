# FS50 Corner Metrology — First Training Run Report

**Project:** AI corner-defect detection for FS50 panels
**Task:** Object detection (YOLO11n, Ultralytics)
**Date of run:** 2026-07-02
**Environment:** Python 3.11, PyTorch 2.12 (CPU), Ultralytics 8.4.78
**Target deployment:** Palantir Foundry application

---

## Slide 1 — Project Overview

- **Goal:** Automatically inspect product corner images and flag defects, replacing/assisting manual QA.
- **How it works:** Each part has 4 corners; 1 image per corner. A YOLO detection model looks at each image and draws a box around any defect it finds. One image can have zero, one, or several defects.
- **Classes (2):**
  | id | class | meaning | decision |
  |----|-------|---------|----------|
  | 0 | `WaterDrops` | water droplet on panel | **PASS** (not a defect) |
  | 1 | `BrokenChips` | chipped / broken corner | **REJECT** (real defect) |
- **Pass/Reject rule (applied in app logic, not the model):** reject only if `BrokenChips` is detected. Water drops or no boxes → good panel.
- **Why keep WaterDrops as a class?** So the model learns to tell drops apart from chips and doesn't confuse the two.

---

## Slide 2 — Dataset

- **Total images:** 177 (annotated in CVAT, exported as YOLO 1.1).
- **Split:**
  | Split | Images | With defect | Good (no boxes) |
  |-------|-------:|------------:|----------------:|
  | Train | 141 | 74 | 67 |
  | Val | 17 | 5 | 12 |
  | Test | 19 | 8 | 11 |
- **Good/background images (~40%)** are included on purpose so the model learns what clean corners look like and doesn't over-flag.
- **Image to show:** `runs/detect/corner/labels.jpg` (class balance + box locations).

> **Key limitation:** the dataset is small — especially the validation set (only **4 BrokenChips** instances). Metrics are therefore indicative, not conclusive.

---

## Slide 3 — Model & Training Setup

- **Model:** YOLO11n ("nano") — smallest/fastest variant, chosen for CPU training. ~2.6M parameters.
- **Pretrained weights:** COCO-pretrained `yolo11n.pt` (transfer learning).
- **Key settings:**
  - Image size: 640×640
  - Epochs: 100 (with early stopping, patience = 20)
  - Batch size: 8
  - Optimizer: AdamW (auto-selected), lr ≈ 0.00167
  - Device: CPU (Intel Core Ultra 7 265U)
- **Run time:** ~0.72 hours for 45 epochs (~55 s/epoch on CPU).

---

## Slide 4 — Training Behavior

- Training **stopped early at epoch 45**; best model was at **epoch 25** (no improvement in the following 20 epochs).
- Losses (box / cls / dfl) decreased steadily — the model learned. Class loss dropped sharply from ~5.8 to ~1.3.
- Validation mAP rose quickly in the first ~15 epochs, then plateaued and fluctuated (expected with a small val set).
- **Images to show:**
  - `runs/detect/corner/results.png` — all loss & metric curves over epochs.
  - `runs/detect/corner/train_batch0.jpg` — example training images with labels.

---

## Slide 5 — Performance (Validation Set)

Best model (`best.pt`) evaluated on the 17-image validation set:

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|----------:|-------:|------:|---------:|
| **All** | 0.947 | 0.639 | 0.758 | 0.433 |
| WaterDrops | 0.983 | 0.778 | 0.812 | 0.566 |
| BrokenChips | 0.911 | 0.500 | 0.704 | 0.300 |

**What the metrics mean (plain language):**
- **Precision** = when the model flags a defect, how often it's correct. Ours is high (0.91–0.98) → **few false alarms**.
- **Recall** = of all real defects, how many the model caught. `BrokenChips` recall = 0.50 → **it currently misses about half the broken chips**.
- **mAP50** = overall detection accuracy at a lenient overlap threshold (higher = better). 0.76 overall is a reasonable first-run baseline.
- **mAP50-95** = stricter accuracy (box must fit tightly). Lower numbers here are normal for a first run.

- **Images to show:**
  - `runs/detect/corner/confusion_matrix.png` — what gets confused with what.
  - `runs/detect/corner/BoxPR_curve.png` — precision/recall tradeoff.
  - `runs/detect/corner/val_batch0_pred.jpg` vs `val_batch0_labels.jpg` — predictions vs ground truth.

---

## Slide 6 — Analysis / Interpretation

- **Strength:** high precision → the model rarely cries wolf. Good for avoiding unnecessary rejections.
- **Main weakness:** **low recall on `BrokenChips` (0.50)** — the class that actually matters for reject decisions. Missing real defects ("escapes") is the most costly error in QA.
- **Root causes:**
  1. **Too few broken-chip examples** — only 4 in validation, limited in training.
  2. **Small, imbalanced dataset** overall → metrics are noisy and the model hasn't seen enough chip variety.
- **Trust level:** treat this run as a **proof that the pipeline works end-to-end**, not as a production-accuracy number.

---

## Slide 7 — Next Steps

**Priority 1 — More & better data (biggest lever):**
- Collect and label **more `BrokenChips`** examples across all 4 corner positions, varied chip sizes, lighting, and parts. Aim for ≥150–300 chip instances.
- Keep the good/water-drop images balanced so the model doesn't over-flag.

**Priority 2 — Honest evaluation:**
- Evaluate on the untouched **test split** for a fairer number:
  `yolo detect val model=runs/detect/corner/weights/best.pt data=data.yaml split=test`
- Visually review misses with `predict.py` to see *where/why* chips are missed.

**Priority 3 — Iterate on training:**
- Retrain after adding data; consider more epochs / higher patience once data grows.
- Optionally test a larger model (YOLO11s) or newer version once the dataset is solid.

**Priority 4 — Path to Palantir Foundry:**
- The deployable artifact is a single file: `best.pt` + the `predict.py` inference logic.
- Confirm Foundry's Python env supports `ultralytics` + CPU `torch`, how images arrive, and whether inference runs as a batch transform or a model endpoint.
- Build a small proof-of-concept in Foundry that loads the weights and scores sample images before scaling up.

---

## Appendix — Where to find the images

All plots are in: `runs/detect/corner/`

| File | Use in report |
|------|---------------|
| `labels.jpg` | Dataset / class balance |
| `results.png` | Training curves (losses + metrics) |
| `train_batch0.jpg` | Example labeled training data |
| `val_batch0_labels.jpg` / `val_batch0_pred.jpg` | Ground truth vs predictions |
| `confusion_matrix.png` / `confusion_matrix_normalized.png` | Class confusion |
| `BoxPR_curve.png`, `BoxP_curve.png`, `BoxR_curve.png`, `BoxF1_curve.png` | Precision/Recall/F1 analysis |
| `weights/best.pt` | The trained model (deployable artifact) |
