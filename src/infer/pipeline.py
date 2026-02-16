"""Frame processor abstractions for video inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from src.camera.sports import SportSpec, get_sport_spec


@dataclass
class FrameProcessorResult:
    frame_out: np.ndarray
    metadata: Dict[str, str]


class FrameProcessor:
    """Interface for per-frame calibration/inference processing."""

    def process(self, frame: np.ndarray, frame_idx: int) -> FrameProcessorResult:
        raise NotImplementedError


class NoOpFrameProcessor(FrameProcessor):
    """
    Temporary processor for pipeline wiring.

    Passes frames through unchanged. This replaces the previous fake-drawing
    test logic so we can keep I/O stable while implementing real calibration.
    """

    def process(self, frame: np.ndarray, frame_idx: int) -> FrameProcessorResult:
        return FrameProcessorResult(
            frame_out=frame,
            metadata={"mode": "noop", "frame_idx": str(frame_idx)},
        )


def build_frame_processor(sport: str, ckpt: Path | None) -> Tuple[SportSpec, FrameProcessor]:
    """
    Build a sport-specific frame processor.

    Current behavior:
    - Returns a NoOp processor until calibration models are integrated.
    """
    spec = get_sport_spec(sport)
    _ = ckpt
    return spec, NoOpFrameProcessor()
