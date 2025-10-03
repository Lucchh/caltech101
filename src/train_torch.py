import os, argparse, json
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_dataloaders(splits_json, img_size=128, batch_size=64, augment=True, num_workers=4):
    with open(splits_json, "r") as f:
        info = json.load(f)
    root = info["root"]; idx = info["indices"]
    normalize = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    if augment:
        tf_train = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2,0.2,0.2,0.1),
            transforms.ToTensor(), normalize
        ])
    else:
        tf_train = transforms.Compose([transforms.Resize((img_size,img_size)), transforms.ToTensor(), normalize])
    tf_eval  = transforms.Compose([transforms.Resize((img_size,img_size)), transforms.ToTensor(), normalize])

    ds_train = datasets.ImageFolder(root, transform=tf_train)
    ds_eval  = datasets.ImageFolder(root, transform=tf_eval)

    tr = DataLoader(Subset(ds_train, idx["train"]), batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    va = DataLoader(Subset(ds_eval,  idx["val"]),   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    te = DataLoader(Subset(ds_eval,  idx["test"]),  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return tr, va, te, ds_train.classes

def build_model(name: str, num_classes: int):
    name = name.lower()
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif name == "vit_b_16":
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")
    return m

def epoch_loop(model, loader, optimizer, criterion, device, train=True):
    if train: model.train()
    else: model.eval()
    total, correct, total_loss = 0, 0, 0.0
    y_true, y_pred = [], []
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, leave=False, desc=("Train" if train else "Eval"), unit="batch"):
            x, y = x.to(device), y.to(device)
            if train: optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            if not train:
                y_true.extend(y.cpu().numpy().tolist())
                y_pred.extend(preds.cpu().numpy().tolist())
    acc = correct / max(total, 1)
    if train: return total_loss/max(total,1), acc
    return total_loss/max(total,1), acc, np.array(y_true), np.array(y_pred)

def per_class_accuracy(y_true, y_pred, n_classes):
    out = np.zeros(n_classes, dtype=float)
    for c in range(n_classes):
        idx = np.where(y_true == c)[0]
        out[c] = np.nan if len(idx)==0 else (y_pred[idx] == c).mean()
    return out

def save_curve(xs, ys, xlabel, ylabel, title, out_path):
    plt.figure(figsize=(6,4))
    plt.plot(xs, ys)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout(); os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200); plt.close()

def save_confusion(cm, class_names, out_path, max_labels=30):
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
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="resnet18", choices=["resnet18","efficientnet_b0","vit_b_16"])
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--no-augment", action="store_true")
    # New training options
    ap.add_argument("--optim", choices=["adam","sgd"], default="adam")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--sched", choices=["none","cosine"], default="cosine")
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr, va, te, classes = get_dataloaders(args.splits, img_size=args.img_size,
                                          batch_size=args.batch_size, augment=(not args.no_augment))
    model = build_model(args.model, len(classes)).to(device)
    if args.freeze:
        for p in model.parameters():
            p.requires_grad = False
        if args.model == "resnet18":
            for p in model.fc.parameters():
                p.requires_grad = True
        elif args.model == "efficientnet_b0":
            for p in model.classifier[-1].parameters():
                p.requires_grad = True
        elif args.model == "vit_b_16":
            for p in model.heads.head.parameters():
                p.requires_grad = True
    if args.optim == "sgd":
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, momentum=args.momentum, weight_decay=1e-4
        )
    else:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=1e-4
        )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if args.sched == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_macro_f1": []}
    best_state, best_score = None, -1.0

    for e in range(1, args.epochs+1):
        tr_loss, tr_acc = epoch_loop(model, tr, optimizer, criterion, device, train=True)
        va_loss, va_acc, yv, pv = epoch_loop(model, va, optimizer, criterion, device, train=False)
        va_macro_f1 = f1_score(yv, pv, average="macro")
        print(f"Epoch {e:02d}: Train acc={tr_acc:.3f}  Val acc={va_acc:.3f}  Val macroF1={va_macro_f1:.3f}")

        history["epoch"].append(e)
        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss);   history["val_acc"].append(va_acc)
        history["val_macro_f1"].append(va_macro_f1)

        if va_macro_f1 > best_score:
            best_score = va_macro_f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if scheduler is not None:
            scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test metrics
    te_loss, te_acc, yt, pt = epoch_loop(model, te, optimizer, criterion, device, train=False)

    # Top-5 accuracy (optional metric for DL)
    with torch.no_grad():
        model.eval()
        topk_correct, total = 0, 0
        for x, y in tqdm(te, leave=False, desc="Top-5", unit="batch"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            k = min(5, logits.size(1))
            topk = torch.topk(logits, k=k, dim=1).indices
            match = (topk == y.view(-1,1)).any(dim=1)
            topk_correct += match.sum().item()
            total += x.size(0)
        top5_acc = topk_correct / max(total, 1)

    # Reports
    report = classification_report(yt, pt, target_names=classes, output_dict=True)
    cm = confusion_matrix(yt, pt).tolist()
    per_cls = per_class_accuracy(yt, pt, len(classes))

    # Save metrics JSON
    with open(os.path.join(args.out, "metrics_test.json"), "w") as f:
        json.dump({
            "accuracy": te_acc,
            "top5_accuracy": top5_acc,
            "classification_report": report,
            "confusion_matrix": cm,
            "per_class_accuracy": {cls: float(per_cls[i]) for i, cls in enumerate(classes)}
        }, f, indent=2)

    # Save per-class accuracy CSV
    with open(os.path.join(args.out, "per_class_accuracy.csv"), "w") as f:
        f.write("class,accuracy\n")
        for i, cls in enumerate(classes):
            f.write(f"{cls},{per_cls[i] if not np.isnan(per_cls[i]) else ''}\n")

    # Save curves
    save_curve(history["epoch"], history["train_loss"], "Epoch", "Loss", "Train Loss", os.path.join(args.out, "train_loss.png"))
    save_curve(history["epoch"], history["val_loss"],   "Epoch", "Loss", "Val Loss",   os.path.join(args.out, "val_loss.png"))
    save_curve(history["epoch"], history["train_acc"],  "Epoch", "Accuracy", "Train Acc", os.path.join(args.out, "train_acc.png"))
    save_curve(history["epoch"], history["val_acc"],    "Epoch", "Accuracy", "Val Acc",   os.path.join(args.out, "val_acc.png"))
    save_curve(history["epoch"], history["val_macro_f1"], "Epoch", "Macro-F1", "Val Macro-F1", os.path.join(args.out, "val_macro_f1.png"))

    # Save confusion matrix heatmap
    save_confusion(cm, classes, os.path.join(args.out, "confusion_matrix.png"))

    # Save summary
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"model": args.model, "img_size": args.img_size, "epochs": args.epochs,
                   "batch_size": args.batch_size, "augment": not args.no_augment}, f, indent=2)

    print(f"Test accuracy = {te_acc:.3f} | Top-5 = {top5_acc:.3f}")
    print("Saved metrics & figures to:", args.out)

if __name__ == "__main__":
    main()