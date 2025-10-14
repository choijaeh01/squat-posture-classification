"""
Utilities for loading and preparing squat posture datasets.

The project expects the following on-disk layout:

data/
├── manually_labeled/
│   ├── class0/
│   ├── … 
│   └── class4/
├── raw/
├── interim/
├── processed/
└── pretrain/
    └── kuhar/

Each `class{idx}` directory inside `manually_labeled` should contain windowed
CSV files (time steps × sensor channels) and optional metadata.json summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


class SquatClass(Enum):
    """Canonical class ids for the squat posture taxonomy."""

    CORRECT = 0
    KNEE_VALGUS = 1
    BUTT_WINK = 2
    EXCESSIVE_LEAN = 3
    PARTIAL_SQUAT = 4

    @classmethod
    def from_path(cls, path: Path) -> "SquatClass":
        """Parse the class id from a `class{idx}` style directory name."""

        try:
            label = int(path.name.replace("class", ""))
        except ValueError as exc:
            raise ValueError(f"Cannot parse class id from path: {path}") from exc
        return cls(label)


SQUAT_CLASS_GUIDE = {
    SquatClass.CORRECT: "정자세 (Correct) — Hips drop below knees with neutral spine.",
    SquatClass.KNEE_VALGUS: "무릎 모임 (Knee Valgus) — Knees cave inward during ascent.",
    SquatClass.BUTT_WINK: "벗 윙크 (Butt Wink) — Pelvis tucks under at the bottom.",
    SquatClass.EXCESSIVE_LEAN: "상체 과다 숙임 (Excessive Lean) — Torso leans toward the floor.",
    SquatClass.PARTIAL_SQUAT: "얕은 스쿼트 (Partial) — Hip crease stays above knee height.",
}


@dataclass(frozen=True)
class DatasetLayout:
    """Convenience container describing the project data directories."""

    root: Path

    @property
    def manually_labeled(self) -> Path:
        return self.root / "manually_labeled"

    @property
    def processed(self) -> Path:
        return self.root / "processed"

    @property
    def interim(self) -> Path:
        return self.root / "interim"

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def kuhar(self) -> Path:
        return self.root / "pretrain" / "kuhar"


class SquatWindowDataset(Dataset):
    """
    PyTorch dataset for windowed CSV files exported from IMU recordings.

    Assumes each CSV lives under `data/manually_labeled/class{idx}/...`.
    Files are loaded via `numpy.loadtxt`, so plain numeric CSV content is
    expected. The dataset returns `(window, label)` tuples where `window`
    is a `torch.Tensor` shaped `(num_channels, num_timesteps)`.
    """

    def __init__(
        self,
        root: Path,
        transforms: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        file_extension: str = ".csv",
    ) -> None:
        self.root = Path(root)
        self.transforms = transforms
        self.file_extension = file_extension

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.root}")

        self._samples: List[Tuple[Path, SquatClass]] = []
        for class_dir in sorted(self.root.glob("class*")):
            if not class_dir.is_dir():
                continue
            label = SquatClass.from_path(class_dir)
            for csv_path in sorted(class_dir.rglob(f"*{self.file_extension}")):
                if csv_path.name.startswith("."):
                    continue
                self._samples.append((csv_path, label))

        if not self._samples:
            raise RuntimeError(
                f"No data files ending with '{self.file_extension}' found in {self.root}"
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        csv_path, label = self._samples[index]
        data = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
        # Convert to (channels, time) layout expected by 1D CNNs.
        tensor = torch.from_numpy(data).T.contiguous()

        if self.transforms is not None:
            tensor = self.transforms(tensor)

        return tensor, label.value


def make_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Thin wrapper around `torch.utils.data.DataLoader` with sensible defaults."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_val_test_split(
    dataset: Dataset,
    ratios: Sequence[float] = (0.7, 0.15, 0.15),
    seed: int = 41,
) -> Tuple[Subset, Subset, Subset]:
    """
    Deterministic split helper that returns three torch `Subset` instances.

    Args:
        dataset: Dataset to split.
        ratios: Fractions that sum to 1.0 (train, val, test).
        seed: Controls shuffling before the split.
    """

    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, received {ratios}")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)

    train_end = int(ratios[0] * len(dataset))
    val_end = train_end + int(ratios[1] * len(dataset))

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    return (
        Subset(dataset, train_indices.tolist()),
        Subset(dataset, val_indices.tolist()),
        Subset(dataset, test_indices.tolist()),
    )


def iter_class_counts(dataset: Dataset) -> Iterable[Tuple[SquatClass, int]]:
    """Yield `(class, count)` pairs for quick dataset sanity checks."""

    counts = {label: 0 for label in SquatClass}
    for _, label in dataset:
        counts[SquatClass(label)] += 1
    return counts.items()
