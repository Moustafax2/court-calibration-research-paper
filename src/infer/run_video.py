"""Video inference runner."""

from __future__ import annotations

from pathlib import Path

import cv2

from src.infer.pipeline import build_frame_processor


def run_video_pipeline(
    input_path: Path,
    output_path: Path,
    sport: str,
    ckpt: Path | None = None,
    overlay_alpha: float = 0.45,
    device: str | None = None,
    max_frames: int | None = None,
) -> dict:
    """Run the video pipeline with the currently configured frame processor."""
    spec, processor = build_frame_processor(
        sport=sport,
        ckpt=ckpt,
        overlay_alpha=overlay_alpha,
        device=device,
    )

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output writer: {output_path}")

    frames_written = 0
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

            result = processor.process(frame=frame, frame_idx=frame_idx)
            writer.write(result.frame_out)

            frame_idx += 1
            frames_written += 1
    finally:
        cap.release()
        writer.release()

    return {
        "sport": spec.name,
        "processor": processor.__class__.__name__,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "fps": fps,
        "width": width,
        "height": height,
        "frames_written": frames_written,
    }
