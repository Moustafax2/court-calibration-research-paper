"""Pose dictionary training entrypoint (camera pose initialization prep)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from src.camera.pose_dictionary import generate_pose_dictionary
from src.utils.config import load_yaml_config
from src.utils.seed import set_global_seed


def _resolve_from_root(root: Path, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    if p.is_absolute():
        return p.resolve()
    return (root / p).resolve()


def _load_allowed_frame_paths(labels_index_path: Path, split: str) -> set[str]:
    out: set[str] = set()
    with labels_index_path.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            row = json.loads(t)
            if row.get("split") != split:
                continue
            fp = row.get("frame_path")
            if isinstance(fp, str) and fp:
                out.add(fp)
    return out


def train_pose(
    config_path: Path,
    max_samples: int | None = None,
    sample_seed: int | None = None,
) -> Dict[str, Any]:
    cfg = load_yaml_config(config_path)
    seed = int(cfg.get("seed", 42))
    if sample_seed is not None:
        seed = int(sample_seed)
    set_global_seed(seed)

    data_cfg = cfg.get("data", {})
    pose_cfg = cfg.get("pose", {})
    output_cfg = cfg.get("output", {})

    project_root = Path(data_cfg.get("project_root", ".")).resolve()
    manifest_path = _resolve_from_root(project_root, str(data_cfg["manifest"]))
    output_dir = _resolve_from_root(project_root, str(output_cfg.get("dir", "checkpoints/pose")))
    sport = str(data_cfg.get("sport", "basketball"))
    split = str(data_cfg.get("split", "train"))
    allowed_frame_paths: set[str] | None = None
    labels_index_cfg = data_cfg.get("labels_index")
    if labels_index_cfg:
        labels_index_path = _resolve_from_root(project_root, str(labels_index_cfg))
        allowed_frame_paths = _load_allowed_frame_paths(labels_index_path=labels_index_path, split=split)
        if not allowed_frame_paths:
            raise ValueError(
                f"No frame paths found in labels_index={labels_index_path} for split={split}"
            )

    summary = generate_pose_dictionary(
        manifest_path=manifest_path,
        sport=sport,
        output_dir=output_dir,
        project_root=project_root,
        split=split,
        template_size=(
            int(pose_cfg.get("template_width", 960)),
            int(pose_cfg.get("template_height", 540)),
        ),
        min_components=int(pose_cfg.get("min_components", 50)),
        max_components=int(pose_cfg.get("max_components", 260)),
        step=int(pose_cfg.get("step", 10)),
        posterior_threshold=float(pose_cfg.get("posterior_threshold", 0.6)),
        random_state=seed,
        max_samples=(
            int(max_samples)
            if max_samples is not None
            else (
                int(pose_cfg["max_samples"])
                if "max_samples" in pose_cfg and pose_cfg["max_samples"] is not None
                else None
            )
        ),
        allowed_frame_paths=allowed_frame_paths,
        template_source=str(pose_cfg.get("template_source", "medoid")).lower(),
        min_template_fg_ratio=float(pose_cfg.get("min_template_fg_ratio", 0.01)),
    )
    summary["labels_index_filter_used"] = bool(labels_index_cfg)
    if labels_index_cfg:
        summary["labels_index_path"] = str(_resolve_from_root(project_root, str(labels_index_cfg)))
    summary["config_path"] = str(Path(config_path).resolve())
    return summary
