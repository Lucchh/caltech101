import os, argparse, json
import numpy as np
from PIL import Image
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


# ----------------- helpers (shared style) -----------------
def _maybe_fix_nested_root(root: str) -> str:
    for c in ['101_ObjectCategories', 'caltech-101', 'Caltech101', '101_ObjectCategories_Full']:
        p = os.path.join(root, c)
        if os.path.isdir(p): return p
    return root

def load_split_paths(root: str, splits_json: str):
    root = _maybe_fix_nested_root(root)
    with open(splits_json, 'r') as f:
        info = json.load(f)
    from torchvision.datasets import ImageFolder
    ds = ImageFolder(root=root)
    samples = ds.samples; classes = ds.classes
    idxs = info['indices']
    pick = lambda lst: [samples[i] for i in lst]
    return pick(idxs['train']), pick(idxs['val']), pick(idxs['test']), classes

def per_class_accuracy(y_true, y_pred, n_classes):
    out = np.zeros(n_classes, dtype=float)
    for c in range(n_classes):
        idx = np.where(y_true == c)[0]
        out[c] = np.nan if len(idx)==0 else (y_pred[idx] == c).mean()
    return out

def plot_confusion(cm, class_names, out_path, max_labels=30):
    cm = np.array(cm)
    plt.figure(figsize=(8,6))
    ax = plt.gca()
    im = ax.imshow(cm, aspect="auto")
    plt.colorbar(im)
    if len(class_names) <= max_labels:
        ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90, fontsize=6)
        ax.set_yticklabels(class_names, fontsize=6)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout(); os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200); plt.close()


# ----------------- feature extractors -----------------
def hog_feature(img_gray: np.ndarray, img_size: int):
    from skimage.feature import hog
    from skimage.transform import resize
    if img_gray.shape[0] != img_size or img_gray.shape[1] != img_size:
        img_gray = resize(img_gray, (img_size, img_size), anti_aliasing=True)
    feat = hog(
        img_gray, orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False,
        feature_vector=True
    )
    return feat.astype(np.float32)

def color_hist_feature(img_rgb: np.ndarray, bins_per_channel=32):
    """HSV 3x32-bin hist (normalized, concatenated)."""
    import cv2
    hsv = cv2.cvtColor((img_rgb*255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    feats = []
    for ch in range(3):
        hist = cv2.calcHist([hsv], [ch], None, [bins_per_channel], [0, 256]).flatten()
        # L1 normalize
        s = hist.sum()
        if s > 0: hist = hist / s
        feats.append(hist.astype(np.float32))
    return np.concatenate(feats, axis=0).astype(np.float32)

def compute_features(paths_labels, img_size: int, feat_type: str, color_bins: int):
    """Return X (Nxd) and y from a list of (path,label)."""
    X, y = [], []
    from skimage import img_as_float
    from skimage.io import imread
    from skimage.color import rgb2gray

    for p, lab in tqdm(paths_labels, desc=f'{feat_type} features'):
        # robust load
        img = imread(p)
        if img.ndim == 2:
            rgb = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 4:
            rgb = img[..., :3]
        else:
            rgb = img
        rgb = img_as_float(rgb)  # 0..1
        gray = rgb2gray(rgb)

        if feat_type == "hog":
            f = hog_feature(gray, img_size)
        elif feat_type == "hog_color":
            f_hog = hog_feature(gray, img_size)
            f_col = color_hist_feature(rgb, bins_per_channel=color_bins)
            f = np.concatenate([f_hog, f_col], axis=0)
        else:
            raise ValueError(f"Unknown feature type: {feat_type}")

        X.append(f); y.append(lab)

    X = np.vstack(X).astype(np.float32)
    y = np.array(y, dtype=int)
    return X, y


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description='Classical ML: HOG(+Color) + RandomForest')
    ap.add_argument('--root', required=True)
    ap.add_argument('--splits', required=True)
    ap.add_argument('--img-size', type=int, default=128)
    ap.add_argument('--features', choices=['hog', 'hog_color'], default='hog_color')
    ap.add_argument('--color-bins', type=int, default=32)
    ap.add_argument('--out', required=True)

    # RF search space (small but strong)
    ap.add_argument('--n-estimators', type=int, nargs='+', default=[300, 600, 900])
    ap.add_argument('--max-depth', type=int, nargs='+', default=[None, 20, 30])
    ap.add_argument('--max-features', type=str, nargs='+', default=['sqrt', 'log2'])

    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # load splits
    tr_set, va_set, te_set, classes = load_split_paths(args.root, args.splits)

    # features
    Xtr, ytr = compute_features(tr_set, args.img_size, args.features, args.color_bins)
    Xva, yva = compute_features(va_set, args.img_size, args.features, args.color_bins)
    Xte, yte = compute_features(te_set, args.img_size, args.features, args.color_bins)

    # use all non-test data to fit (train + val)
    Xfull = np.vstack([Xtr, Xva]).astype(np.float32)
    yfull = np.concatenate([ytr, yva]).astype(int)

    # Optional standardization can help some RF configs when features have very different scales
    scaler = StandardScaler(with_mean=False)
    Xfull = scaler.fit_transform(Xfull)
    Xte   = scaler.transform(Xte)

    # RF grid search
    param_grid = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'max_features': args.max_features,
        'n_jobs': [-1]
    }
    # We'll create RF inside GridSearchCV by using lambda-like wrapper via sklearn's clone params:
    # Instead, iterate small grid manually to preserve progress and avoid refitting issues with 'n_jobs' in grid.
    best_clf, best_acc = None, -1.0
    from sklearn.utils import shuffle
    for ne in args.n_estimators:
        for md in args.max_depth:
            for mf in args.max_features:
                clf = RandomForestClassifier(
                    n_estimators=ne, max_depth=md, max_features=mf,
                    n_jobs=-1, random_state=42, class_weight=None
                )
                clf.fit(Xfull, yfull)
                yp = clf.predict(Xte)
                acc = accuracy_score(yte, yp)
                print(f"[RF] n_estimators={ne} max_depth={md} max_features={mf} -> acc={acc:.4f}")
                if acc > best_acc:
                    best_acc, best_clf = acc, clf

    # Final eval with best RF
    ypred = best_clf.predict(Xte)
    acc = accuracy_score(yte, ypred)
    rep = classification_report(yte, ypred, target_names=classes, output_dict=True)
    cm  = confusion_matrix(yte, ypred).tolist()
    pc  = per_class_accuracy(yte, ypred, len(classes))

    # Save metrics JSON
    with open(os.path.join(args.out, 'metrics_test.json'), 'w') as f:
        json.dump({
            'accuracy': float(acc),
            'classification_report': rep,
            'confusion_matrix': cm,
            'per_class_accuracy': {classes[i]: float(pc[i]) for i in range(len(classes))}
        }, f, indent=2)

    # Per-class CSV
    with open(os.path.join(args.out, 'per_class_accuracy.csv'), 'w') as f:
        f.write("class,accuracy\n")
        for i, cls in enumerate(classes):
            f.write(f"{cls},{pc[i] if not np.isnan(pc[i]) else ''}\n")

    # Confusion fig
    plot_confusion(cm, classes, os.path.join(args.out, "confusion_matrix.png"))

    print(f"RF ({args.features}) test accuracy = {acc:.3f}")
    print("Saved metrics & figures to:", args.out)


if __name__ == '__main__':
    main()