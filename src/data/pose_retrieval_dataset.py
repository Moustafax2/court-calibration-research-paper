"""Dataset for siamese template retrieval training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _resolve(project_root: Path, p: Path) -> Path:
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def _load_labels_index(labels_index_path: Path, split: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with labels_index_path.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            row = json.loads(t)
            if row.get("split") != split:
                continue
            mapping[row["frame_path"]] = row["mask_path"]
    return mapping


def _load_assignments(assignments_path: Path, split: str) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    with assignments_path.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            row = json.loads(t)
            if row.get("split") != split:
                continue
            mapping[row["frame_path"]] = int(row["template_id"])
    return mapping


class PoseRetrievalPairDataset(Dataset):
    """
    Produces (anchor_mask, template_mask, label) pairs.
    label=1 for positive pair, label=0 for negative pair.
    """

    def __init__(
        self,
        labels_index_path: Path,
        assignments_path: Path,
        templates_dir: Path,
        split: str,
        project_root: Path = Path("."),
        image_size: Tuple[int, int] = (256, 256),  # (H, W)
        num_classes: int = 4,  # includes background class 0
        hard_negative_prob: float = 0.6,
        augment_anchor_prob: float = 0.4,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.image_size = image_size
        self.templates_dir = Path(templates_dir).resolve()
        self.num_classes = int(num_classes)
        self.hard_negative_prob = float(np.clip(hard_negative_prob, 0.0, 1.0))
        self.augment_anchor_prob = float(np.clip(augment_anchor_prob, 0.0, 1.0))

        labels_map = _load_labels_index(Path(labels_index_path).resolve(), split=split)
        assign_map = _load_assignments(Path(assignments_path).resolve(), split=split)

        common = sorted(set(labels_map.keys()).intersection(assign_map.keys()))
        if not common:
            raise ValueError(
                f"No overlapping frame_path entries for split={split} "
                f"between labels_index and assignments."
            )

        self.samples: List[Tuple[Path, int]] = []
        for frame_path in common:
            mask_path = _resolve(self.project_root, Path(labels_map[frame_path]))
            tid = int(assign_map[frame_path])
            self.samples.append((mask_path, tid))

        self.template_ids = sorted(
            int(p.stem.split("_")[-1]) for p in self.templates_dir.glob("template_*.png")
        )
        if not self.template_ids:
            raise ValueError(f"No template_*.png found in {self.templates_dir}")
        self.template_id_set = set(self.template_ids)

        # Keep only samples with existing template files.
        self.samples = [(m, tid) for (m, tid) in self.samples if tid in self.template_id_set]
        if not self.samples:
            raise ValueError("No valid samples after filtering missing template ids.")

        self.rng = np.random.default_rng(42)
        self.template_mask_cache: Dict[int, np.ndarray] = {
            tid: self._read_mask_ids(self.templates_dir / f"template_{tid:04d}.png")
            for tid in self.template_ids
        }

    def __len__(self) -> int:
        # One positive and one negative for each anchor.
        return len(self.samples) * 2

    def _read_mask_ids(self, path: Path) -> np.ndarray:
        m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if m is None:
            raise FileNotFoundError(f"Could not read mask: {path}")
        if m.ndim == 3:
            m = m[..., 0]
        h, w = self.image_size
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        return m.astype(np.uint8)

    def _to_one_hot(self, mask_ids: np.ndarray) -> np.ndarray:
        oh = np.zeros((self.num_classes, mask_ids.shape[0], mask_ids.shape[1]), dtype=np.float32)
        for c in range(self.num_classes):
            oh[c] = (mask_ids == c).astype(np.float32)
        return oh

    def _augment_anchor(self, mask_ids: np.ndarray) -> np.ndarray:
        if self.rng.random() >= self.augment_anchor_prob:
            return mask_ids
        out = mask_ids.copy()
        # Simple morphology perturbation on non-background mask.
        fg = (out > 0).astype(np.uint8)
        k = int(self.rng.integers(1, 3))
        kernel = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
        if self.rng.random() < 0.5:
            fg = cv2.dilate(fg, kernel, iterations=1)
        else:
            fg = cv2.erode(fg, kernel, iterations=1)
        out = np.where(fg > 0, out, 0).astype(np.uint8)
        return out

    def _iou_multiclass(self, a: np.ndarray, b: np.ndarray) -> float:
        # IoU over non-background classes; mean across present classes.
        ious = []
        for c in range(1, self.num_classes):
            aa = a == c
            bb = b == c
            union = np.logical_or(aa, bb).sum()
            if union == 0:
                continue
            inter = np.logical_and(aa, bb).sum()
            ious.append(float(inter) / float(union))
        if not ious:
            return 0.0
        return float(np.mean(ious))

    def _sample_negative_tid(self, true_tid: int, anchor_ids: np.ndarray) -> int:
        neg_choices = [tid for tid in self.template_ids if tid != true_tid]
        if not neg_choices:
            return true_tid
        if self.rng.random() >= self.hard_negative_prob:
            return int(self.rng.choice(neg_choices))

        # Hard negative: choose most similar wrong template by IoU.
        best_tid = neg_choices[0]
        best_score = -1.0
        for tid in neg_choices:
            score = self._iou_multiclass(anchor_ids, self.template_mask_cache[tid])
            if score > best_score:
                best_score = score
                best_tid = tid
        return int(best_tid)

    def __getitem__(self, idx: int):
        anchor_idx = idx % len(self.samples)
        positive = (idx // len(self.samples)) == 0
        anchor_mask_path, true_tid = self.samples[anchor_idx]
        anchor_ids = self._read_mask_ids(anchor_mask_path)
        anchor_ids = self._augment_anchor(anchor_ids)

        if positive:
            template_tid = true_tid
            label = 1.0
        else:
            template_tid = self._sample_negative_tid(true_tid=true_tid, anchor_ids=anchor_ids)
            label = 0.0

        template_ids = self.template_mask_cache[template_tid]
        anchor = self._to_one_hot(anchor_ids)
        template = self._to_one_hot(template_ids)

        # CHW one-hot tensors
        anchor_t = torch.from_numpy(anchor).float()
        template_t = torch.from_numpy(template).float()
        label_t = torch.tensor(label, dtype=torch.float32)
        return anchor_t, template_t, label_t
