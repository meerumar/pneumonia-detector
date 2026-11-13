"""Train a ResNet18 pneumonia classifier on the Chest X-Ray dataset.

Usage:
    python train.py                      # default settings
    python train.py --epochs 10 --batch_size 16

Outputs:
    models/pneumonia_resnet18.pth        <- trained model (used by the web app)
    outputs/training_curves.png          <- loss & accuracy graphs
    outputs/confusion_matrix.png         <- test-set confusion matrix
    outputs/metrics.json                 <- accuracy, precision, recall, F1, ROC-AUC
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from backend.model_utils import (CLASSES, MODEL_PATH, build_model,
                                 get_eval_transform, get_train_transform)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def set_backbone_trainable(model, trainable: bool):
    """Freeze / unfreeze everything except the final layer (fc)."""
    for name, param in model.named_parameters():
        if not name.startswith("fc"):
            param.requires_grad = trainable


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over the data. Trains if an optimizer is given, otherwise only evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, y_true, y_pred = 0.0, [], []
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            y_true += labels.tolist()
            y_pred += outputs.argmax(1).tolist()
    n = len(y_true)
    return total_loss / n, accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    y_true, y_prob = [], []
    for images, labels in loader:
        probs = torch.softmax(model(images.to(device)), dim=1)[:, 1].cpu()
        y_true += labels.tolist()
        y_prob += probs.tolist()
    return np.array(y_true), np.array(y_prob)


def save_curves(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Validation")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def save_confusion_matrix(cm, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(CLASSES)
    ax.set_yticks([0, 1]); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix (test set)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/chest_xray")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--head_epochs", type=int, default=2, help="epochs with frozen backbone")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for fine-tuning")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not (data_dir / "train").exists() or not (data_dir / "test").exists():
        raise SystemExit(f"Dataset not found in '{data_dir}'. Run: python download_dataset.py")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    Path("outputs").mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)

    # ---- 1. Data: our own 85/15 train/validation split (the dataset's val folder is tiny) ----
    train_full = datasets.ImageFolder(data_dir / "train", get_train_transform())
    val_full = datasets.ImageFolder(data_dir / "train", get_eval_transform())
    test_set = datasets.ImageFolder(data_dir / "test", get_eval_transform())
    assert train_full.classes == CLASSES, f"Unexpected folders: {train_full.classes}"

    idx = torch.randperm(len(train_full)).tolist()
    n_val = int(0.15 * len(idx))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train_loader = DataLoader(Subset(train_full, train_idx), args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(Subset(val_full, val_idx), args.batch_size, num_workers=args.workers)
    test_loader = DataLoader(test_set, args.batch_size, num_workers=args.workers)
    print(f"Train: {len(train_idx)}  Validation: {len(val_idx)}  Test: {len(test_set)}")

    # ---- 2. Class weights: pneumonia images outnumber normal ones ~3:1 ----
    counts = np.bincount([train_full.targets[i] for i in train_idx], minlength=2)
    weights = torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float32).to(device)
    print(f"Class counts {dict(zip(CLASSES, counts))} -> loss weights {weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=weights)

    # ---- 3. Model + two-phase training ----
    model = build_model(pretrained=True).to(device)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_f1, optimizer = -1.0, None

    for epoch in range(1, args.epochs + 1):
        if epoch == 1:                                   # Phase 1: train only the new final layer
            set_backbone_trainable(model, False)
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
        if epoch == args.head_epochs + 1:                # Phase 2: fine-tune the whole network gently
            set_backbone_trainable(model, True)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        tr_loss, tr_acc, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc, va_f1 = run_epoch(model, val_loader, criterion, device)
        for k, v in zip(history, [tr_loss, va_loss, tr_acc, va_acc]):
            history[k].append(v)
        print(f"Epoch {epoch}/{args.epochs}  train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f} f1 {va_f1:.3f}")

        if va_f1 > best_f1:                              # keep the best model only
            best_f1 = va_f1
            torch.save({"model_state": model.state_dict(), "classes": CLASSES}, MODEL_PATH)
            print("  -> saved new best model")

    # ---- 4. Final evaluation on the untouched test set ----
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device)["model_state"])
    y_true, y_prob = predict_all(model, test_loader, device)
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "confusion_matrix": cm.tolist(),
    }
    print("\n=== TEST RESULTS ===")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=3))
    print(json.dumps(metrics, indent=2))

    Path("outputs/metrics.json").write_text(json.dumps(metrics, indent=2))
    save_curves(history, "outputs/training_curves.png")
    save_confusion_matrix(cm, "outputs/confusion_matrix.png")
    print(f"\nModel saved to {MODEL_PATH}\nGraphs saved in the 'outputs' folder.\nNow run:  python run.py")


if __name__ == "__main__":
    main()
