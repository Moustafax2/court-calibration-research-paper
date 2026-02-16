"""Validation utilities for dataset annotation manifests."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict

from src.data.annotation_schema import FrameAnnotation


def _resolve_frame_path(project_root: Path, frame_path: Path) -> Path:
    if frame_path.is_absolute():
        return frame_path
    return (project_root / frame_path).resolve()


def validate_manifest(
    manifest_path: Path,
    project_root: Path = Path("."),
    check_files: bool = True,
) -> Dict[str, Any]:
    """Validate a JSONL manifest and return summary stats."""
    manifest = Path(manifest_path).resolve()
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    split_counter: Counter[str] = Counter()
    num_rows = 0
    num_valid = 0
    errors: list[str] = []

    with manifest.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            num_rows += 1
            try:
                raw = json.loads(stripped)
                ann = FrameAnnotation.from_dict(raw)
                if check_files:
                    resolved = _resolve_frame_path(project_root, ann.frame_path)
                    if not resolved.exists():
                        raise ValueError(f"frame_path does not exist: {resolved}")
                split_counter[ann.split] += 1
                num_valid += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"line {idx}: {exc}")

    return {
        "manifest_path": str(manifest),
        "num_rows": num_rows,
        "num_valid": num_valid,
        "num_errors": len(errors),
        "split_counts": dict(split_counter),
        "first_errors": errors[:20],
    }
