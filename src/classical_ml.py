import os, argparse, json
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

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

def compute_hog_features(paths_labels, img_size: int):
    from skimage.feature import hog
    X, y = [], []
    for p, label in tqdm(paths_labels, desc='HOG features'):
        img = Image.open(p).convert('L').resize((img_size, img_size))
        arr = np.array(img)
        feat = hog(arr, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm='L2-Hys')
        X.append(feat); y.append(label)
    return np.array(X), np.array(y)

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

def main():
    ap = argparse.ArgumentParser(description='Classical ML baseline: HOG + SVM')
    ap.add_argument('--root', required=True)
    ap.add_argument('--splits', required=True)
    ap.add_argument('--img-size', type=int, default=128)
    ap.add_argument('--svm', choices=['linear','rbf'], default='rbf')
    ap.add_argument('--C-grid', type=float, nargs='+', default=[0.01,0.1,1,10])
    ap.add_argument('--gamma-grid', type=float, nargs='+', default=[0.001,0.01,0.1])
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tr_set, va_set, te_set, classes = load_split_paths(args.root, args.splits)

    Xtr, ytr = compute_hog_features(tr_set, args.img_size)
    Xva, yva = compute_hog_features(va_set, args.img_size)
    Xte, yte = compute_hog_features(te_set, args.img_size)

    scaler = StandardScaler(with_mean=False)
    if args.svm == 'linear':
        clf = LinearSVC(dual=False, max_iter=10000)
        pipe = Pipeline([('scaler', scaler), ('clf', clf)])
        param_grid = {'clf__C': args.C_grid}
    else:
        clf = SVC(kernel='rbf', probability=False)
        pipe = Pipeline([('scaler', scaler), ('clf', clf)])
        param_grid = {'clf__C': args.C_grid, 'clf__gamma': args.gamma_grid}

    # Fit on train+val (use all non-test data)
    from numpy import vstack, concatenate
    Xfull, yfull = vstack([Xtr, Xva]), concatenate([ytr, yva])
    gs = GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
    gs.fit(Xfull, yfull)

    ypred = gs.predict(Xte)
    acc = accuracy_score(yte, ypred)
    rep = classification_report(yte, ypred, target_names=classes, output_dict=True)
    cm  = confusion_matrix(yte, ypred).tolist()
    pc  = per_class_accuracy(yte, ypred, len(classes))

    with open(os.path.join(args.out, 'metrics_test.json'), 'w') as f:
        json.dump({'accuracy': acc,
                   'classification_report': rep,
                   'confusion_matrix': cm,
                   'per_class_accuracy': {classes[i]: float(pc[i]) for i in range(len(classes))}},
                  f, indent=2)

    # CSV + confusion matrix figure
    with open(os.path.join(args.out, 'per_class_accuracy.csv'), 'w') as f:
        f.write("class,accuracy\n")
        for i, cls in enumerate(classes):
            f.write(f"{cls},{pc[i] if not np.isnan(pc[i]) else ''}\n")
    plot_confusion(cm, classes, os.path.join(args.out, "confusion_matrix.png"))

    print(f"HOG+SVM test accuracy = {acc:.3f}")
    print("Saved metrics & figures to:", args.out)

if __name__ == '__main__':
    main()