"""Dataset for STN homography refinement training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.camera.pose_dictionary import homography_to_pose_vector


def _resolve(root: Path, p: Path) -> Path:
    if p.is_absolute():
        return p
    return (root / p).resolve()


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            rows.append(json.loads(t))
    return rows


class STNHomographyDataset(Dataset):
    """
    Returns stacked masks and target relative homography vector.

    input: [pred_or_gt_mask, template_mask] (2,H,W)
    target: vec8(H_rel) where H_rel = H_gt @ inv(H_template)
    """

    def __init__(
        self,
        labels_index_path: Path,
        assignments_path: Path,
        manifest_path: Path,
        templates_dir: Path,
        template_homographies_path: Path,
        split: str,
        project_root: Path = Path("."),
        image_size: Tuple[int, int] = (256, 256),  # (H,W)
    ) -> None:
        self.root = Path(project_root).resolve()
        self.image_size = image_size
        self.templates_dir = Path(templates_dir).resolve()

        labels_rows = _load_jsonl(Path(labels_index_path).resolve())
        labels_map: Dict[str, str] = {
            r["frame_path"]: r["mask_path"] for r in labels_rows if r.get("split") == split
        }

        assign_rows = _load_jsonl(Path(assignments_path).resolve())
        assign_map: Dict[str, int] = {
            r["frame_path"]: int(r["template_id"]) for r in assign_rows if r.get("split") == split
        }

        manifest_rows = _load_jsonl(Path(manifest_path).resolve())
        hom_map: Dict[str, np.ndarray] = {}
        for r in manifest_rows:
            if r.get("split") != split:
                continue
            hom_map[r["frame_path"]] = np.asarray(r["homography"], dtype=np.float64)

        with Path(template_homographies_path).resolve().open("r", encoding="utf-8") as f:
            templates_h = json.load(f)
        self.template_h = [np.asarray(h, dtype=np.float64) for h in templates_h]

        common = sorted(set(labels_map.keys()) & set(assign_map.keys()) & set(hom_map.keys()))
        if not common:
            raise ValueError(f"No overlapping rows for split={split} in STN dataset inputs.")

        self.samples: List[Tuple[Path, int, np.ndarray]] = []
        self.targets: List[np.ndarray] = []
        for frame_path in common:
            mask_path = _resolve(self.root, Path(labels_map[frame_path]))
            tid = int(assign_map[frame_path])
            if tid < 0 or tid >= len(self.template_h):
                continue
            h_gt = hom_map[frame_path]
            h_template = self.template_h[tid]
            h_rel = h_gt @ np.linalg.inv(h_template)
            y = homography_to_pose_vector(h_rel).astype(np.float32)
            self.samples.append((mask_path, tid, y))
            self.targets.append(y)
        if not self.samples:
            raise ValueError("No valid STN samples after filtering.")

    def __len__(self) -> int:
        return len(self.samples)

    def _read_mask(self, path: Path) -> np.ndarray:
        m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if m is None:
            raise FileNotFoundError(f"Could not read mask: {path}")
        if m.ndim == 3:
            m = m[..., 0]
        h, w = self.image_size
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        m = m.astype(np.float32)
        m /= max(1.0, float(np.max(m)))
        return m

    def __getitem__(self, idx: int):
        anchor_path, tid, y = self.samples[idx]
        template_path = self.templates_dir / f"template_{tid:04d}.png"
        anchor = self._read_mask(anchor_path)
        template = self._read_mask(template_path)

        x = np.stack([anchor, template], axis=0).astype(np.float32)
        x_t = torch.from_numpy(x).float()
        y_t = torch.from_numpy(y).float()
        return x_t, y_t

    def get_targets_array(self) -> np.ndarray:
        return np.asarray(self.targets, dtype=np.float32)
