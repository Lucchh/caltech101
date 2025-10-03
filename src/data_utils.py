import os, json, argparse
from typing import Tuple
import numpy as np

def _maybe_fix_nested_root(root: str) -> str:
    """
    If the dataset is nested (e.g., data/raw/101_ObjectCategories),
    return that inner path so ImageFolder sees class subfolders.
    """
    candidates = ['101_ObjectCategories', 'caltech-101', 'Caltech101', '101_ObjectCategories_Full']
    for c in candidates:
        p = os.path.join(root, c)
        if os.path.isdir(p):
            return p
    return root

def _collect_targets_from_imagefolder(root: str):
    """
    Loads ImageFolder and returns (dataset, targets array, class names).
    Works with torchvision >= 0.13.
    """
    from torchvision.datasets import ImageFolder
    fixed_root = _maybe_fix_nested_root(root)
    ds = ImageFolder(root=fixed_root)
    if hasattr(ds, 'targets'):
        y = np.array(ds.targets)
    else:
        # Fallback for older versions
        y = np.array([t for _, t in ds.samples])
    return ds, y, ds.classes, fixed_root

def make_splits(root: str,
                out_json: str,
                seed: int = 42,
                train_ratio: float = 0.7,
                val_ratio: float = 0.15,
                test_ratio: float = 0.15):
    """
    Create stratified splits by class using scikit-learn train_test_split.
    Saves a JSON file with absolute root, classes, and indices.
    """
    from sklearn.model_selection import train_test_split

    ds, y, classes, fixed_root = _collect_targets_from_imagefolder(root)
    idx = np.arange(len(y))

    # Train vs temp
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx, y, test_size=(1.0 - train_ratio), stratify=y, random_state=seed
    )

    # Val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, test_size=(1.0 - val_size), stratify=y_temp, random_state=seed
    )

    splits = {
        'root': os.path.abspath(fixed_root),
        'num_classes': len(classes),
        'classes': classes,
        'indices': {
            'train': idx_train.tolist(),
            'val': idx_val.tolist(),
            'test': idx_test.tolist()
        }
    }

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(splits, f, indent=2)

    print(f"Saved splits → {out_json}")
    print(f"Images root detected as → {fixed_root}")
    print(f"#classes: {len(classes)} | #images: {len(y)}")
    print(f"train/val/test sizes: {len(idx_train)}/{len(idx_val)}/{len(idx_test)}")

def main():
    parser = argparse.ArgumentParser(description='Create stratified 70/15/15 splits for Caltech-101')
    parser.add_argument('--root', type=str, required=True, help='Images root (class subfolders or a folder containing them)')
    parser.add_argument('--out', type=str, required=True, help='Where to save JSON with indices')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--make-splits', action='store_true', help='Generate splits')
    args = parser.parse_args()

    if args.make_splits:
        make_splits(args.root, args.out, seed=args.seed)
    else:
        print('Nothing to do. Use --make-splits.')

if __name__ == '__main__':
    main()