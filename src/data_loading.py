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
import pandas as pd
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

    Each sample lives under `data/manually_labeled/class{idx}/...`. CSV files
    are read with pandas so header 행이나 문자열 컬럼(예: timestamp)이 있어도 처리할 수
    있으며, 필요하면 `drop_columns`로 명시적으로 제외할 수 있습니다. 반환되는
    텐서는 `(channels, timesteps)` 형태입니다.
    """

    def __init__(
        self,
        root: Path,
        transforms: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        file_extension: str = ".csv",
        drop_columns: Optional[Sequence[str]] = None,
        target_length: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.transforms = transforms
        self.file_extension = file_extension
        self.drop_columns = tuple(drop_columns) if drop_columns is not None else None
        self.target_length = target_length

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
        tensor = self._load_csv(csv_path)

        if self.transforms is not None:
            tensor = self.transforms(tensor)

        return tensor, label.value

    def _load_csv(self, path: Path) -> torch.Tensor:
        """Read a CSV, drop non-numeric columns, and return (channels, time) tensor."""

        df = pd.read_csv(path)
        if self.drop_columns is not None:
            existing = [col for col in self.drop_columns if col in df.columns]
            df = df.drop(columns=existing)

        # 기본적으로 문자열(예: timestamp)을 제외하고 숫자 컬럼만 사용.
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError(f"No numeric columns found in {path}")

        data = numeric_df.to_numpy(dtype=np.float32)

        if self.target_length is not None and data.shape[0] != self.target_length:
            data = _resample_to_length(data, self.target_length)

        return torch.from_numpy(data.T.copy())


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


def _resample_to_length(data: np.ndarray, target_length: int) -> np.ndarray:
    """
    Interpolate a (time, features) array to the desired length.

    Args:
        data: Array shaped `(num_steps, num_features)`.
        target_length: Desired number of timesteps.
    """

    if target_length <= 0:
        raise ValueError("`target_length` must be greater than zero.")

    num_steps = data.shape[0]
    if num_steps == target_length:
        return data

    base_positions = np.linspace(0, num_steps - 1, num_steps, dtype=np.float32)
    target_positions = np.linspace(0, num_steps - 1, target_length, dtype=np.float32)

    resampled = np.empty((target_length, data.shape[1]), dtype=np.float32)
    for feature_idx in range(data.shape[1]):
        resampled[:, feature_idx] = np.interp(
            target_positions, base_positions, data[:, feature_idx]
        )
    return resampled
