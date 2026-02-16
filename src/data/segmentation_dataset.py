"""Segmentation dataset backed by labels_index JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _resolve_path(project_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


class CourtSegmentationDataset(Dataset):
    """Dataset of (image, mask) for semantic court segmentation."""

    def __init__(
        self,
        labels_index_path: Path,
        split: str,
        project_root: Path = Path("."),
        image_size: Tuple[int, int] = (512, 512),  # (H, W)
        horizontal_flip_prob: float = 0.0,
    ) -> None:
        self.labels_index_path = Path(labels_index_path).resolve()
        self.project_root = Path(project_root).resolve()
        self.split = split
        self.image_size = image_size
        self.horizontal_flip_prob = float(horizontal_flip_prob)
        self.rows = self._load_rows()

    def _load_rows(self) -> List[Dict[str, Any]]:
        if not self.labels_index_path.exists():
            raise FileNotFoundError(f"labels_index not found: {self.labels_index_path}")
        rows: List[Dict[str, Any]] = []
        with self.labels_index_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                raw = json.loads(text)
                if raw.get("split") != self.split:
                    continue
                if "frame_path" not in raw or "mask_path" not in raw:
                    raise ValueError(
                        f"labels_index line {line_num} missing frame_path or mask_path"
                    )
                rows.append(raw)
        if not rows:
            raise ValueError(f"No rows found for split='{self.split}' in {self.labels_index_path}")
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        frame_path = _resolve_path(self.project_root, Path(row["frame_path"]))
        mask_path = _resolve_path(self.project_root, Path(row["mask_path"]))

        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {frame_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")
        if mask.ndim == 3:
            mask = mask[..., 0]

        target_h, target_w = self.image_size
        image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        if self.horizontal_flip_prob > 0.0 and np.random.rand() < self.horizontal_flip_prob:
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        image_t = torch.from_numpy(image).float()
        mask_t = torch.from_numpy(mask.astype(np.int64))
        return image_t, mask_t
