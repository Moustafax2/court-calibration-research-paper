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
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.image_size = image_size
        self.templates_dir = Path(templates_dir).resolve()

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

    def __len__(self) -> int:
        # One positive and one negative for each anchor.
        return len(self.samples) * 2

    def _read_mask(self, path: Path) -> np.ndarray:
        m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if m is None:
            raise FileNotFoundError(f"Could not read mask: {path}")
        if m.ndim == 3:
            m = m[..., 0]
        h, w = self.image_size
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        # Normalize class-id mask into [0,1] single channel.
        m = m.astype(np.float32) / max(1.0, float(np.max(m)))
        return m

    def __getitem__(self, idx: int):
        anchor_idx = idx % len(self.samples)
        positive = (idx // len(self.samples)) == 0
        anchor_mask_path, true_tid = self.samples[anchor_idx]

        if positive:
            template_tid = true_tid
            label = 1.0
        else:
            # random negative template id
            neg_choices = [tid for tid in self.template_ids if tid != true_tid]
            template_tid = int(self.rng.choice(neg_choices)) if neg_choices else true_tid
            label = 0.0

        template_path = self.templates_dir / f"template_{template_tid:04d}.png"
        anchor = self._read_mask(anchor_mask_path)
        template = self._read_mask(template_path)

        # NCHW single-channel tensors
        anchor_t = torch.from_numpy(anchor[None, ...]).float()
        template_t = torch.from_numpy(template[None, ...]).float()
        label_t = torch.tensor(label, dtype=torch.float32)
        return anchor_t, template_t, label_t
