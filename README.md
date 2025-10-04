
# Mini-Project 1 — Caltech-101 Image Classification

Code, splits, and results for a compact comparison of **classical CV** vs **pretrained deep models** on Caltech‑101 (including the common `BACKGROUND_Google` class).

---

## What’s here

- Classical baselines: **HOG+SVM (RBF)**, **HOG(+HSV color)+Random Forest**
- Deep models: **ResNet‑18**, **EfficientNet‑B0**, **ViT‑B/16** (frozen vs full fine‑tuning in ablations)
- Stratified **70/15/15** train/val/test splits
- Checkpoint selection by **validation macro‑F1**
- Figures: learning curves & confusion matrices
- Reproducible scripts and an aggregate metrics table

---

## Quick results (test set)

| Method | Acc | Macro‑F1 | W‑F1 | Top‑5 |
|---|---:|---:|---:|---:|
| HOG+SVM (RBF) | 0.137 | 0.017 | 0.081 | — |
| RF (HOG+color) | 0.465 | 0.229 | 0.394 | — |
| ResNet‑18 | 0.901 | 0.889 | 0.900 | 0.991 |
| **EfficientNet‑B0** | **0.924** | **0.916** | **0.922** | **0.995** |
| ViT‑B/16 (frozen) | 0.848 | 0.822 | 0.835 | 0.975 |

**Takeaways**
- Pretrained CNNs dominate classical pipelines; color + RF narrows the gap a bit but not enough.
- ViT needs **full fine‑tuning** on this small dataset to match/beat CNNs (see ablations).
- Macro‑F1 is a steadier selection metric than raw accuracy under class imbalance.

---

## Repo layout

```
caltech101_project/
├── data/
│   ├── raw/                      # Caltech‑101 images
│   └── splits/                   # Stratified index files (JSON)
├── src/
│   ├── data_utils.py             # Download/prepare, create splits
│   ├── classical_ml.py           # HOG + SVM
│   ├── classical_rf.py           # HOG(+color) + Random Forest
│   ├── train_torch.py            # ResNet/EfficientNet/ViT training
│   ├── aggregate_results.py      # Summaries → markdown/latex
│   └── utils/                    # Helpers
├── results/
│   ├── hog_svm_rbf/
│   ├── rf_hog_color_128px/
│   ├── resnet18_224_sgd_cosine_ls01/
│   ├── effb0_224_adam_cosine_ls05/
│   ├── vitb16_224_freeze/
│   └── vitb16_224_fullft/
└── report/                       # LaTeX + compiled PDF
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Data

Download Caltech‑101 (public release with `BACKGROUND_Google`) and place under:

```
data/raw/caltech-101/
```

Create stratified splits:

```bash
python src/data_utils.py \
  --make-splits \
  --root data/raw/caltech-101 \
  --out data/splits/caltech101_splits_70_15_15.json \
  --seed 42
```

---

## Train

### Classical baselines

```bash
# HOG + SVM (RBF)
python src/classical_ml.py \
  --root data/raw/caltech-101 \
  --splits data/splits/caltech101_splits_70_15_15.json \
  --out results/hog_svm_rbf

# HOG(+HSV) + Random Forest
python src/classical_rf.py \
  --root data/raw/caltech-101 \
  --splits data/splits/caltech101_splits_70_15_15.json \
  --out results/rf_hog_color_128px
```

### Deep models (PyTorch)

```bash
# ResNet‑18 (224 px, SGD + cosine, label smoothing 0.1)
python src/train_torch.py \
  --model resnet18 --img-size 224 --epochs 10 --batch-size 64 \
  --optimizer sgd --lr 0.01 --momentum 0.9 --cosine --label-smoothing 0.1 \
  --splits data/splits/caltech101_splits_70_15_15.json \
  --out results/resnet18_224_sgd_cosine_ls01

# EfficientNet‑B0 (224 px, Adam + cosine, label smoothing 0.05)
python src/train_torch.py \
  --model efficientnet_b0 --img-size 224 --epochs 10 --batch-size 64 \
  --optimizer adam --lr 3e-4 --cosine --label-smoothing 0.05 \
  --splits data/splits/caltech101_splits_70_15_15.json \
  --out results/effb0_224_adam_cosine_ls05

# ViT‑B/16 — frozen vs full FT
python src/train_torch.py \
  --model vit_b_16 --img-size 224 --epochs 10 --batch-size 64 \
  --freeze-backbone \
  --splits data/splits/caltech101_splits_70_15_15.json \
  --out results/vitb16_224_freeze

python src/train_torch.py \
  --model vit_b_16 --img-size 224 --epochs 10 --batch-size 64 \
  --splits data/splits/caltech101_splits_70_15_15.json \
  --out results/vitb16_224_fullft
```

Notes:
- Inputs normalized with ImageNet mean/std.
- Light augmentation: random resized crop, horizontal flip, mild color jitter.
- **Model selection by validation macro‑F1** (handles class imbalance).

---

## Aggregate & report

```bash
python src/aggregate_results.py
```

- Produces summary `.md` and `.tex` tables.
- Figures (confusion matrices, curves) are saved under each `results/<run_name>/`.
- The LaTeX report (`report/`) pulls from those files.

---

## Reproduce everything (optional)

If you use the convenience script:

```bash
bash run_all.sh
```

---

## License & citation

This repo is for academic coursework. If you use the code/results, please cite:

```
@misc{lucchen_caltech101_2025,
  title  = {Mini-Project 1: Image Classification on Caltech-101},
  author = {Luc Chen},
  year   = {2025},
  note   = {Course project}
}
```

---