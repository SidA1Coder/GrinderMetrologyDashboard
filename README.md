# FS50 Corner Metrology — Corner Defect Detection (YOLO)

Detects defects on **product corner images** using Ultralytics YOLO
(`detect` task). One image can contain **zero, one, or several** defects at
once, so this uses object **detection** (a box per defect) rather than
single-label classification.

## Classes

| id | class            | meaning                                   | action            |
|----|------------------|-------------------------------------------|-------------------|
| 0  | `chipped_corner` | corner chipped / broken                   | **real defect**   |
| 1  | `water_drop`     | water droplet on panel                    | not a concern     |
| 2  | `missed_detection` | improper / blurred / wrong-framed shot  | re-capture image  |

A **good panel** = an image with **no boxes** (no annotations). You do NOT
create a `good` class for detection — "good" is simply the absence of any
defect box. Still include plenty of good images in the dataset so the model
learns what clean corners look like (they act as "background" examples).

> Adjust class names/ids in CVAT and in [data.yaml](data.yaml) so they match
> exactly (same names, same order).

---

## How many images do you need?

Your part has **4 corners**, and you photograph **1 image per corner**. The
model looks at a single corner image and decides what (if anything) is wrong.

What actually drives accuracy in detection is the number of **labeled
instances per class** (how many example boxes of each defect it sees), not
just the number of images. Practical targets:

| Goal                          | images total | per-corner | instances per defect class |
|-------------------------------|-------------:|-----------:|---------------------------:|
| First working prototype       |     ~200–300 |    ~50–75  |   **≥ 50** boxes / class   |
| Solid, deployable model       |    ~600–1000 |  ~150–250  |   **150–300** boxes / class |
| Robust across lighting/parts  |     1500+    |    400+    |   500+ boxes / class       |

Concretely, for a good first model collect roughly:
- **~250 corner images** spread across the 4 corner positions (so the model
  sees each corner geometry), **per part variant** you care about, with
- at least **50–100 real `chipped_corner` examples**, **50–100 `water_drop`
  examples**, **30–50 `missed_detection` examples**, and
- a healthy number of **good corners with no defects** (aim for ~30–50% of the
  set) so it doesn't over-flag.

Tips that matter more than raw count:
- **Variety beats volume:** different lighting, angles, parts, drop sizes,
  chip sizes/positions. 200 varied images > 1000 near-duplicates.
- **Balance the defects:** the rarest defect class sets your real-world
  accuracy. If chips are rare, deliberately collect/seed more chipped parts.
- **Cover all 4 corner positions** for every class so the model generalizes to
  "any corner."
- Hold out a **test set** of images it never trains on for an honest score.

---

## 1. Annotate in CVAT

1. Create a task with a **Rectangle/Box label** for each defect class:
   `chipped_corner`, `water_drop`, `missed_detection`.
2. Draw a tight box around each defect in each image. Multiple boxes per image
   is expected and correct (e.g. a chip *and* two water drops).
3. Leave good corners with **no boxes** — that's how "good" is represented.
4. Export: **Menu → Export task dataset → "YOLO 1.1"**.
   This produces:
   ```
   obj.names                 # class names, one per line
   obj.data
   obj_train_data/
       img001.jpg
       img001.txt            # <cls> <x_center> <y_center> <w> <h> (normalized)
       ...   (images with NO defect simply have an empty/absent .txt)
   ```

---

## 2. Prepare the dataset (split into train/val/test)

Point the script at the CVAT export folder. It splits images+labels into a
YOLO-detection folder layout and writes the class list it found:

```bash
python scripts/prepare_dataset.py --cvat-export path/to/cvat_export --dst datasets/corner
```

Resulting layout:
```
datasets/corner/
  images/{train,val,test}/*.jpg
  labels/{train,val,test}/*.txt
```

Then make sure [data.yaml](data.yaml) class names match `obj.names`.

---

## 3. Train (CPU)

```bash
python scripts/train.py --data data.yaml --epochs 100 --imgsz 640 --model yolo11n.pt
```

- `yolo11n.pt` (nano) is the right choice for CPU.
- Detection needs larger images than classification — keep `--imgsz 640`
  (or 512 if CPU is slow) so small chips/drops remain visible.
- Outputs: `runs/detect/<name>/`; best weights at
  `runs/detect/<name>/weights/best.pt`.

CPU is slow for detection — expect long epochs. If it's painful, reduce
`--imgsz` to 512, lower `--batch`, or train on Google Colab's free GPU.

---

## 4. Predict

```bash
python scripts/predict.py --weights runs/detect/corner/weights/best.pt --source path/to/image_or_folder --save
```

Prints each detected defect with confidence, and (with `--save`) writes
annotated images under `runs/detect/predict/`. No boxes printed = good panel.

---

## 5. Evaluate & iterate

- Check `runs/detect/<name>/`: `results.png`, `confusion_matrix.png`, and
  per-class **mAP50** / precision / recall.
- The class with the lowest recall is where you're missing defects — collect
  and label more of that defect, then retrain.
- Use the held-out `test/` split for the final honest number.

---

## Project layout

```
.
├── README.md
├── requirements.txt
├── data.yaml                 # class names + dataset paths (detection)
├── .gitignore
├── datasets/corner/          # generated images/ + labels/ (gitignored)
├── runs/                     # training/inference outputs (gitignored)
└── scripts/
    ├── prepare_dataset.py     # CVAT YOLO export -> train/val/test split
    ├── train.py               # YOLO detect training (CPU)
    └── predict.py             # YOLO detect inference
```

## Environment

Dependencies are installed in the conda env **`fs50defect`** (Python 3.11,
CPU PyTorch + Ultralytics). Run scripts with that env active:

```bash
conda activate fs50defect
python scripts/train.py --data data.yaml
```
