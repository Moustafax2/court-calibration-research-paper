"""Pose dictionary training entrypoint (camera pose initialization prep)."""

from __future__ import annotations

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


def train_pose(config_path: Path) -> Dict[str, Any]:
    cfg = load_yaml_config(config_path)
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)

    data_cfg = cfg.get("data", {})
    pose_cfg = cfg.get("pose", {})
    output_cfg = cfg.get("output", {})

    project_root = Path(data_cfg.get("project_root", ".")).resolve()
    manifest_path = _resolve_from_root(project_root, str(data_cfg["manifest"]))
    output_dir = _resolve_from_root(project_root, str(output_cfg.get("dir", "checkpoints/pose")))
    sport = str(data_cfg.get("sport", "basketball"))

    summary = generate_pose_dictionary(
        manifest_path=manifest_path,
        sport=sport,
        output_dir=output_dir,
        split=str(data_cfg.get("split", "train")),
        template_size=(
            int(pose_cfg.get("template_width", 960)),
            int(pose_cfg.get("template_height", 540)),
        ),
        min_components=int(pose_cfg.get("min_components", 50)),
        max_components=int(pose_cfg.get("max_components", 260)),
        step=int(pose_cfg.get("step", 10)),
        posterior_threshold=float(pose_cfg.get("posterior_threshold", 0.6)),
        random_state=seed,
    )
    summary["config_path"] = str(Path(config_path).resolve())
    return summary
