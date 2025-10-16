"""
Command-line training entry point for squat posture models.

The default configuration loads labeled windows from `data/manually_labeled`,
applies a lightweight augmentation pipeline, and trains the `TemporalCNNGRU`
network defined in `src/models/temporal.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from .augmentations import (
    add_gaussian_noise,
    compose_transforms,
    random_scaling,
    random_time_shift,
)
from .data_loading import (
    DatasetLayout,
    SquatWindowDataset,
    make_dataloader,
    train_val_test_split,
)
from .models import TemporalCNNGRU


def build_datasets(
    data_root: Path,
    apply_augmentation: bool = True,
) -> Tuple[Subset, Subset, Subset]:
    """Load labeled datasets and split into train/val/test subsets."""

    transforms = None
    if apply_augmentation:
        transforms = compose_transforms(
            [
                random_time_shift(max_shift=5),
                random_scaling(0.9, 1.1),
                add_gaussian_noise(std=0.01),
            ]
        )

    labeled_root = DatasetLayout(data_root).manually_labeled
    dataset = SquatWindowDataset(
        labeled_root,
        transforms=transforms,
        drop_columns=("timestamp",),
        target_length=400,
    )
    return train_val_test_split(dataset)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Single training epoch."""

    model.train()
    running_loss = 0.0
    num_samples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        num_samples += inputs.size(0)

    return running_loss / max(num_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> Dict[str, float]:
    """Evaluate the model and return loss/accuracy metrics."""

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = criterion(logits, targets)

        total_loss += loss.item() * inputs.size(0)
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == targets).sum().item()
        total_samples += inputs.size(0)

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }


def run_training(
    data_root: Path,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-3,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """Main training loop, returns the final validation metrics."""

    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_ds, val_ds, test_ds = build_datasets(data_root)

    train_loader = make_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = make_dataloader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = make_dataloader(test_ds, batch_size=batch_size, shuffle=False)

    # Infer sensor dimensionality from the first batch.
    sample_batch, _ = next(iter(train_loader))
    in_channels = sample_batch.shape[1]
    num_classes = 5

    model = TemporalCNNGRU(in_channels=in_channels, num_classes=num_classes).to(device_obj)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device_obj)
        metrics = evaluate(model, val_loader, criterion, device_obj)
        val_loss = metrics["loss"]
        val_acc = metrics["accuracy"]

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3%}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics = evaluate(model, test_loader, criterion, device_obj)
    print(
        f"[Test] loss={final_metrics['loss']:.4f} accuracy={final_metrics['accuracy']:.3%}"
    )
    return final_metrics


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Path to the root `data/` directory.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None, help="Override device id.")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    run_training(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
    )


if __name__ == "__main__":
    main()
