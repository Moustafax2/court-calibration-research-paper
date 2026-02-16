#!/usr/bin/env python3
"""Court calibration CLI entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.camera.sports import supported_sports

def cmd_train_seg(args: argparse.Namespace) -> int:
    try:
        from src.train.train_segmentation import train_segmentation
    except ModuleNotFoundError as exc:
        print(f"[train-seg] missing dependency: {exc}")
        print("Install requirements first: pip install -r requirements.txt")
        return 2

    summary = train_segmentation(args.config)
    print("[train-seg] completed")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 0


def cmd_train_pose(args: argparse.Namespace) -> int:
    print(f"[train-pose] config={args.config}")
    print("Not implemented yet.")
    return 0


def cmd_train_stn(args: argparse.Namespace) -> int:
    print(f"[train-stn] config={args.config}")
    print("Not implemented yet.")
    return 0


def cmd_train_e2e(args: argparse.Namespace) -> int:
    print(f"[train-e2e] config={args.config}")
    print("Not implemented yet.")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    print(f"[eval] config={args.config} ckpt={args.ckpt}")
    print("Not implemented yet.")
    return 0


def cmd_infer_video(args: argparse.Namespace) -> int:
    try:
        from src.infer.run_video import run_video_pipeline
    except ModuleNotFoundError as exc:
        print(f"[infer-video] missing dependency: {exc}")
        print("Install requirements first: pip install -r requirements.txt")
        return 2

    summary = run_video_pipeline(
        input_path=args.input,
        output_path=args.output,
        sport=args.sport,
        ckpt=args.ckpt,
        max_frames=args.max_frames,
    )
    print("[infer-video] completed")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    if args.ckpt is not None:
        print(f"  ckpt: {args.ckpt}")
    return 0


def cmd_validate_data(args: argparse.Namespace) -> int:
    from src.data.validate_dataset import validate_manifest

    summary = validate_manifest(
        manifest_path=args.manifest,
        project_root=args.project_root,
        check_files=not args.skip_file_checks,
    )
    print("[validate-data] summary")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 1 if summary["num_errors"] > 0 else 0


def cmd_generate_labels(args: argparse.Namespace) -> int:
    try:
        from src.data.generate_labels import generate_labels_from_manifest
    except ModuleNotFoundError as exc:
        print(f"[generate-labels] missing dependency: {exc}")
        print("Install requirements first: pip install -r requirements.txt")
        return 2

    summary = generate_labels_from_manifest(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        sport=args.sport,
        project_root=args.project_root,
        overwrite=args.overwrite,
    )
    print("[generate-labels] summary")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 1 if summary["num_errors"] > 0 else 0


def _add_common_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config file.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Basketball court calibration pipeline CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_train_seg = subparsers.add_parser("train-seg", help="Train segmentation model")
    _add_common_config_arg(p_train_seg)
    p_train_seg.set_defaults(func=cmd_train_seg)

    p_train_pose = subparsers.add_parser("train-pose", help="Train siamese pose model")
    _add_common_config_arg(p_train_pose)
    p_train_pose.set_defaults(func=cmd_train_pose)

    p_train_stn = subparsers.add_parser("train-stn", help="Train STN refinement model")
    _add_common_config_arg(p_train_stn)
    p_train_stn.set_defaults(func=cmd_train_stn)

    p_train_e2e = subparsers.add_parser("train-e2e", help="Train full end-to-end model")
    _add_common_config_arg(p_train_e2e)
    p_train_e2e.set_defaults(func=cmd_train_e2e)

    p_eval = subparsers.add_parser("eval", help="Evaluate checkpoint")
    _add_common_config_arg(p_eval)
    p_eval.add_argument("--ckpt", type=Path, required=True, help="Checkpoint path")
    p_eval.set_defaults(func=cmd_eval)

    p_infer = subparsers.add_parser("infer-video", help="Run video inference")
    p_infer.add_argument("--input", type=Path, required=True, help="Input video path")
    p_infer.add_argument("--output", type=Path, required=True, help="Output video path")
    p_infer.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Checkpoint path.",
    )
    p_infer.add_argument(
        "--sport",
        type=str,
        default="basketball",
        choices=supported_sports(),
        help="Sport mode",
    )
    p_infer.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap for processed frames.",
    )
    p_infer.set_defaults(func=cmd_infer_video)

    p_validate = subparsers.add_parser(
        "validate-data", help="Validate frame annotation manifest"
    )
    p_validate.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to JSONL annotation manifest.",
    )
    p_validate.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root used to resolve relative frame paths.",
    )
    p_validate.add_argument(
        "--skip-file-checks",
        action="store_true",
        help="Skip existence checks for frame paths.",
    )
    p_validate.set_defaults(func=cmd_validate_data)

    p_gen = subparsers.add_parser(
        "generate-labels",
        help="Generate semantic region labels from a JSONL annotation manifest",
    )
    p_gen.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to JSONL annotation manifest.",
    )
    p_gen.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for generated mask PNGs + labels index.",
    )
    p_gen.add_argument(
        "--sport",
        type=str,
        default="basketball",
        choices=supported_sports(),
        help="Sport mode.",
    )
    p_gen.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root used to resolve relative frame paths.",
    )
    p_gen.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing masks.",
    )
    p_gen.set_defaults(func=cmd_generate_labels)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
