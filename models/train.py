"""
Local Training Script for ResNet-18 on HAM10000.
Run this script to fine-tune the model using your local GPU (if available).

Usage:
    python train.py
"""
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from datasets import load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE    = 64
EPOCHS        = 10
LEARNING_RATE = 1e-4
VAL_SPLIT     = 0.15   # 15% held out for validation

# HAM10000 class order — MUST match DIAGNOSES list in server/app/main.py
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
NUM_CLASSES = len(CLASS_NAMES)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class HAM10000Dataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset     = hf_dataset
        self.transform   = transform
        self.class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item  = self.dataset[idx]
        image = item['image'].convert('RGB')

        # Safely extract label (HF datasets vary in column names)
        if 'dx' in item:
            label_val = item['dx']
        elif 'label' in item:
            label_val = item['label']
        elif 'cell_type' in item:
            label_val = item['cell_type']
        else:
            raise KeyError(f"Label not found. Available keys: {list(item.keys())}")

        label = self.class_to_idx[label_val] if isinstance(label_val, str) else int(label_val)

        if self.transform:
            image = self.transform(image)

        return image, label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_label(item: dict, class_to_idx: dict) -> int:
    """Extract integer class index from a raw HuggingFace dataset item."""
    if 'dx' in item:
        val = item['dx']
    elif 'label' in item:
        val = item['label']
    elif 'cell_type' in item:
        val = item['cell_type']
    else:
        raise KeyError(f"Label column not found. Keys: {list(item.keys())}")
    return class_to_idx[val] if isinstance(val, str) else int(val)


def compute_class_weights_from_hf(hf_dataset) -> torch.Tensor:
    """
    C1 FIX — Compute inverse-frequency class weights by iterating the *raw*
    HuggingFace dataset (dict access, zero image transforms) rather than the
    transform-applying HAM10000Dataset wrapper.  The old approach applied
    resize + color-jitter + ToTensor to every image just to read a label field,
    wasting ~20,000 full image transforms before training started.

    Returns a tensor of shape (NUM_CLASSES,) summing to 1.
    """
    class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    label_counts  = torch.zeros(NUM_CLASSES, dtype=torch.long)

    for item in hf_dataset:            # raw dict — no transforms, no PIL decode
        label_counts[_extract_label(item, class_to_idx)] += 1

    label_counts = label_counts.clamp(min=1)   # guard against unseen classes
    weights      = 1.0 / label_counts.float()
    return weights / weights.sum()             # normalise to sum=1


def build_weighted_sampler(hf_dataset) -> WeightedRandomSampler:
    """
    R3/R4 FIX — Build WeightedRandomSampler in a SINGLE pass over the raw HF
    dataset.  The previous implementation called compute_class_weights_from_hf
    (one full iteration) then iterated the dataset a second time to assign per-
    sample weights.  This merges both into one O(n) pass.

    Each mini-batch is approximately class-balanced across all 7 classes.
    """
    class_to_idx  = {c: i for i, c in enumerate(CLASS_NAMES)}
    label_counts  = torch.zeros(NUM_CLASSES, dtype=torch.long)
    label_indices: list[int] = []

    for item in hf_dataset:                        # single pass, raw dicts
        idx = _extract_label(item, class_to_idx)
        label_counts[idx] += 1
        label_indices.append(idx)

    label_counts  = label_counts.clamp(min=1)
    class_weights = 1.0 / label_counts.float()
    class_weights = class_weights / class_weights.sum()    # normalise

    sample_weights = [class_weights[i].item() for i in label_indices]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def run_epoch(
    model,
    loader,
    criterion,
    device,
    is_train: bool,
    optimizer: Optional[optim.Optimizer] = None,   # C2 FIX — Optional; required only for training
) -> tuple[float, float]:
    """
    Single train or validation epoch.

    C2 FIX — optimizer is now Optional.  Passing it during validation was
    needlessly coupling inference logic to the optimiser.  A ValueError is
    raised if is_train=True but no optimizer is provided.

    Returns:
        (avg_loss, accuracy_pct)
    """
    if is_train and optimizer is None:
        raise ValueError("`optimizer` must be provided when is_train=True")

    model.train(is_train)
    running_loss = 0.0
    correct      = 0
    total        = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss    = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted  = torch.max(outputs, 1)
            total         += labels.size(0)
            correct       += (predicted == labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Data transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load HuggingFace dataset
    print("Downloading/Loading HAM10000 dataset (~3 GB)…")
    hf_data = load_dataset("kuchikihater/HAM10000", split="train")

    # Stratified train/val split (on raw indices — no transforms involved)
    total        = len(hf_data)
    val_size     = int(total * VAL_SPLIT)
    indices      = list(range(total))
    # R3 FIX — fix the seed so the train/val split is reproducible across runs.
    # Without this, every `python train.py` produces a different split, making
    # metric comparison across checkpoints meaningless.
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[val_size:], indices[:val_size]

    train_hf = hf_data.select(train_indices)
    val_hf   = hf_data.select(val_indices)

    # C1 FIX — build sampler from raw HF data (no transform overhead)
    print("Computing class weights and sampler (raw label scan — no image transforms)…")
    sampler = build_weighted_sampler(train_hf)

    train_dataset = HAM10000Dataset(train_hf, transform=train_transform)
    val_dataset   = HAM10000Dataset(val_hf,   transform=val_transform)

    # num_workers=0 on Windows to avoid multiprocessing spawn issues
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,   num_workers=0)

    # Model
    print("Initialising ResNet-18 with ImageNet pre-trained weights…")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model    = model.to(device)

    # C1 FIX — class weights for loss also computed from raw HF data
    class_weights = compute_class_weights_from_hf(train_hf).to(device)
    criterion  = nn.CrossEntropyLoss(weight=class_weights)
    optimizer  = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True,
    )

    # Training loop
    print(f"Starting training for {EPOCHS} epochs…")
    best_val_loss = float('inf')
    save_path     = os.path.join(os.path.dirname(__file__), "resnet18_ham10000.pth")

    for epoch in range(1, EPOCHS + 1):
        # Train
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device,
            is_train=True, optimizer=optimizer,
        )

        # C2 FIX — optimizer NOT passed to validation call
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device,
            is_train=False,              # optimizer defaults to None
        )

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:>2}/{EPOCHS}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.1f}%  |  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.1f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ New best checkpoint saved → {save_path}")

    print("Training complete!")
    print(f"Best validation loss : {best_val_loss:.4f}")
    print(f"Best weights saved at: {save_path}")


if __name__ == "__main__":
    main()
